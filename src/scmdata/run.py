"""
ScmRun provides a high level analysis tool for simple climate model relevant
data. It provides a simple interface for reading/writing, subsetting and visualising
model data. ScmRuns are able to hold multiple model runs which aids in analysis of
ensembles of model runs.
"""
import copy
import datetime as dt
import functools
import numbers
import os
import warnings
from logging import getLogger
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import numpy.testing as npt
import pandas as pd
import pint
from dateutil import parser
from openscm_units import unit_registry as ur
from xarray.core.ops import inject_binary_ops

from .errors import MissingRequiredColumnError, NonUniqueMetadataError
from .filters import (
    HIERARCHY_SEPARATOR,
    datetime_match,
    day_match,
    hour_match,
    month_match,
    pattern_match,
    years_match,
)
from .groupby import RunGroupBy
from .netcdf import inject_nc_methods
from .offsets import generate_range, to_offset
from .ops import inject_ops_methods
from .plotting import inject_plotting_methods
from .pyam_compat import IamDataFrame, LongDatetimeIamDataFrame
from .time import _TARGET_DTYPE, TimePoints, TimeseriesConverter
from .units import UnitConverter

_logger = getLogger(__name__)


MetadataType = Dict[str, Union[str, int, float]]


def _read_file(  # pylint: disable=missing-return-doc
    fnames: str, required_cols: Tuple[str], *args: Any, **kwargs: Any
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data to initialize :class:`ScmRun` from a file.

    Parameters
    ----------
    *args
        Passed to :func:`_read_pandas`.
    **kwargs
        Passed to :func:`_read_pandas`.

    Returns
    -------
    :obj:`pd.DataFrame`, :obj:`pd.DataFrame`
        First dataframe is the data. Second dataframe is metadata
    """
    _logger.info("Reading %s", fnames)

    return _format_data(_read_pandas(fnames, *args, **kwargs), required_cols)


def _read_pandas(
    fname: str, *args: Any, lowercase_cols=False, **kwargs: Any
) -> pd.DataFrame:
    """
    Read a file and return a :class:`pd.DataFrame`.

    Parameters
    ----------
    fname
        Path from which to read data

    lowercase_cols
        If True, convert the column names of the file to lowercase

    *args
        Passed to :func:`pd.read_excel` if :obj:`fname` ends with '.xls' or
        '.xslx, otherwise passed to :func:`pd.read_csv`.

    **kwargs
        Passed to :func:`pd.read_excel` if :obj:`fname` ends with '.xls' or
        '.xslx, otherwise passed to :func:`pd.read_csv`.

    Returns
    -------
    :obj:`pd.DataFrame`
        Read data

    Raises
    ------
    OSError
        Path specified by :obj:`fname` does not exist
    """
    if not os.path.exists(fname):
        raise OSError("no data file `{}` found!".format(fname))

    if fname.endswith("xlsx") or fname.endswith("xls"):
        _logger.debug("Assuming excel file")
        xl = pd.ExcelFile(fname)

        if len(xl.sheet_names) > 1 and "sheet_name" not in kwargs:
            kwargs["sheet_name"] = "data"

        df = pd.read_excel(fname, *args, **kwargs)

    else:
        _logger.debug("Reading with pandas read_csv")
        df = pd.read_csv(fname, *args, **kwargs)

    def _to_lower(c):
        if hasattr(c, "lower"):
            return c.lower()
        return c

    if lowercase_cols:
        df.columns = [_to_lower(c) for c in df.columns]

    return df


# pylint doesn't recognise return statements if they include ','
def _format_data(  # pylint: disable=missing-return-doc
    df: Union[pd.DataFrame, pd.Series], required_cols: Tuple[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data to initialize :class:`ScmRun` from :class:`pd.DataFrame` or
    :class:`pd.Series`.

    See docstring of :func:`ScmRun.__init__` for details.

    Parameters
    ----------
    df
        Data to format.

    Returns
    -------
    :obj:`pd.DataFrame`, :obj:`pd.DataFrame`
        First dataframe is the data. Second dataframe is metadata.

    Raises
    ------
    ValueError
        Not all required metadata columns are present or the time axis cannot be
        understood
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # reset the index if meaningful entries are included there
    if list(df.index.names) != [None]:
        df.reset_index(inplace=True)

    if not set(required_cols).issubset(set(df.columns)):
        missing = list(set(required_cols) - set(df.columns))
        raise MissingRequiredColumnError(missing)

    # check whether data in wide or long format
    if "value" in df.columns:
        df, meta = _format_long_data(df, required_cols)
    else:
        df, meta = _format_wide_data(df, required_cols)

    # sort data
    df.sort_index(inplace=True)

    return df, meta


def _format_long_data(df, required_cols):
    # check if time column is given as `year` (int) or `time` (datetime)
    cols = set(df.columns)
    if "year" in cols and "time" not in cols:
        time_col = "year"
    elif "time" in cols and "year" not in cols:
        time_col = "time"
    else:
        msg = "invalid time format, must have either `year` or `time`!"
        raise ValueError(msg)

    required_cols = list(required_cols)
    extra_cols = list(set(cols) - set(required_cols + [time_col, "value"]))
    df = df.pivot_table(columns=required_cols + extra_cols, index=time_col).value
    meta = df.columns.to_frame(index=None)
    df.columns = meta.index

    return df, meta


def _format_wide_data(df, required_cols):
    cols = set(df.columns) - set(required_cols)
    time_cols, extra_cols = False, []
    for i in cols:
        # if in wide format, check if columns are years (int) or datetime
        if isinstance(i, dt.datetime):
            time_cols = True
        else:
            try:
                float(i)
                time_cols = True
            except (ValueError, TypeError):
                try:
                    try:
                        # most common format
                        dt.datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        # this is super slow so avoid if possible
                        parser.parse(str(i))  # if no ValueError, this is datetime
                    time_cols = True
                except ValueError:
                    extra_cols.append(i)  # some other string

    if not time_cols:
        msg = "invalid column format, must contain some time (int, float or datetime) columns!"
        raise ValueError(msg)

    all_cols = set(tuple(required_cols) + tuple(extra_cols))
    all_cols = list(all_cols)

    df_out = df.drop(all_cols, axis="columns").T
    df_out.index.name = "time"
    meta = df[all_cols].set_index(df_out.columns)

    return df_out, meta


def _from_ts(
    df: Any,
    required_cols: Tuple[str],
    index: Any = None,
    **columns: Union[str, bool, float, int, List],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data to initialize :class:`ScmRun` from wide timeseries.

    See docstring of :func:`ScmRun.__init__` for details.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        First dataframe is the data. Second dataframe is metadata

    Raises
    ------
    ValueError
        Not all required columns are present
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if index is not None:
        if isinstance(index, np.ndarray):
            df.index = TimePoints(index).to_index()
        elif isinstance(index, TimePoints):
            df.index = index.to_index()
        else:
            df.index = index

    # format columns to lower-case and check that all required columns exist
    if not set(required_cols).issubset(columns.keys()):
        missing = list(set(required_cols) - set(columns.keys()))
        raise MissingRequiredColumnError(missing)

    df.index.name = "time"

    num_ts = len(df.columns)
    for c_name, col in columns.items():
        col_list = (
            [col] if isinstance(col, str) or not isinstance(col, Iterable) else col
        )

        if len(col_list) == num_ts:
            continue
        if len(col_list) != 1:
            error_msg = (
                "Length of column '{}' is incorrect. It should be length "
                "1 or {}".format(c_name, num_ts)
            )
            raise ValueError(error_msg)
        columns[c_name] = col_list * num_ts

    meta = pd.DataFrame(columns, index=df.columns)

    return df, meta


class BaseScmRun:  # pylint: disable=too-many-public-methods
    """
    Base class of a data container for timeseries data
    """

    required_cols = ("variable", "unit")
    """
    Required metadata columns

    This is the bare minimum columns which are expected. Attempting to create a run
    without the metadata columns specified by :attr:`required_cols` will raise a
    MissingRequiredColumnError
    """

    data_hierarchy_separator = HIERARCHY_SEPARATOR
    """
    str: String used to define different levels in our data hierarchies.

    By default we follow pyam and use "|". In such a case, emissions of |CO2| for
    energy from coal would be "Emissions|CO2|Energy|Coal".
    """

    def __init__(
        self,
        data: Any,
        index: Any = None,
        columns: Optional[Union[Dict[str, list], Dict[str, str]]] = None,
        metadata: Optional[MetadataType] = None,
        copy_data: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize the container with timeseries data.

        Parameters
        ----------
        data: Union[ScmRun, IamDataFrame, pd.DataFrame, np.ndarray, str]
            If a :class:`ScmRun` object is provided, then a new
            :obj:`ScmRun` is created with a copy of the values and metadata from :obj:`data`.

            A :class:`pd.DataFrame` with IAMC-format data columns (the result
            from :func:`ScmRun.timeseries()`) can be provided without any additional
            :obj:`columns` and :obj:`index` information.

            If a numpy array of timeseries data is provided, :obj:`columns` and :obj:`index`
            must also be specified. The shape of the numpy array should be
            ``(n_times, n_series)`` where `n_times` is the number of timesteps and `n_series`
            is the number of time series.

            If a string is passed, data will be attempted to be read from file. Currently,
            reading from CSV, gzipped CSV and Excel formatted files is supported.

        index: np.ndarray
            If :obj:`index` is not ``None``, then the :obj:`index` is used as the timesteps
            for run. All timeseries in the run use the same set of timesteps.

            The values will be attempted to be converted to :class:`np.datetime[s]` values.
            Possible input formats include :

            * :obj:`datetime.datetime`
            * :obj:`int` Start of year
            * :obj:`float` Decimal year
            * :obj:`str` Uses :func:`dateutil.parser`. Slow and should be avoided if possible

            If :obj:`index` is ``None``, than the time index will be obtained from the
            :obj:`data` if possible.

        columns
            If None, ScmRun will attempt to infer the values from the source.
            Otherwise, use this dict to write the metadata for each timeseries in data.
            For each metadata key (e.g. "model", "scenario"), an array of values (one
            per time series) is expected. Alternatively, providing a list of length 1
            applies the same value to all timeseries in data. For example, if you had
            three timeseries from 'rcp26' for 3 different models 'model', 'model2' and
            'model3', the column dict would look like either 'col_1' or 'col_2':

            .. code:: python

                >>> col_1 = {
                    "scenario": ["rcp26"],
                    "model": ["model1", "model2", "model3"],
                    "region": ["unspecified"],
                    "variable": ["unspecified"],
                    "unit": ["unspecified"]
                }
                >>> col_2 = {
                    "scenario": ["rcp26", "rcp26", "rcp26"],
                    "model": ["model1", "model2", "model3"],
                    "region": ["unspecified"],
                    "variable": ["unspecified"],
                    "unit": ["unspecified"]
                }
                >>> assert pd.testing.assert_frame_equal(
                    ScmRun(d, columns=col_1).meta,
                    ScmRun(d, columns=col_2).meta
                )

        metadata:
            Optional dictionary of metadata for instance as a whole.

            This can be used to store information such as the longer-form information
            about a particular dataset, for example, dataset description or DOIs.

            Defaults to an empty :obj:`dict` if no default metadata are provided.

        copy_data: bool
            If True, an explicit copy of data is performed.

            .. note::
                The copy can be very expensive on large timeseries and should only be needed
                in cases where the original data is manipulated.

        **kwargs:
            Additional parameters passed to :func:`_read_file` to read files

        Raises
        ------
        ValueError
            * If you try to load from multiple files at once. If you wish to do this, please use :func:`scmdata.run.run_append` instead.
            * Not specifying :obj:`index` and :obj:`columns` if :obj:`data` is a :obj:`numpy.ndarray`

        :obj:`scmdata.errors.MissingRequiredColumn`
            If metadata for :attr:`required_cols` is not found

        TypeError
            Timeseries cannot be read from :obj:`data`
        """
        if isinstance(data, ScmRun):
            self._df = data._df.copy() if copy_data else data._df
            self._meta = data._meta
            self._time_points = TimePoints(data.time_points.values)
            if metadata is None:
                metadata = data.metadata.copy()
        else:
            if copy_data and hasattr(data, "copy"):
                data = data.copy()
            self._init_timeseries(data, index, columns, copy_data=copy_data, **kwargs)

        if self._duplicated_meta():
            raise NonUniqueMetadataError(self.meta)

        self.metadata = metadata.copy() if metadata is not None else {}

    def _init_timeseries(
        self,
        data,
        index: Any = None,
        columns: Optional[Dict[str, list]] = None,
        copy_data=False,
        **kwargs: Any,
    ):
        if isinstance(data, np.ndarray):
            if columns is None:
                raise ValueError("`columns` argument is required")
            if index is None:
                raise ValueError("`index` argument is required")

        if columns is not None:
            (_df, _meta) = _from_ts(
                data, index=index, required_cols=self.required_cols, **columns
            )
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            (_df, _meta) = _format_data(data, self.required_cols)
        elif (IamDataFrame is not None) and isinstance(data, IamDataFrame):
            (_df, _meta) = _format_data(
                data.data.copy() if copy_data else data.data, self.required_cols
            )
        else:
            if not isinstance(data, str):
                if isinstance(data, list) and isinstance(data[0], str):
                    raise ValueError(
                        "Initialising from multiple files not supported, "
                        "use `scmdata.run.ScmRun.append()`"
                    )
                error_msg = "Cannot load {} from {}".format(type(self), type(data))
                raise TypeError(error_msg)

            (_df, _meta) = _read_file(data, required_cols=self.required_cols, **kwargs)

        self._time_points = TimePoints(_df.index.values)

        _df = _df.astype(float)
        self._df = _df
        self._df.index = self._time_points.to_index()
        self._meta = pd.MultiIndex.from_frame(_meta.astype("category"))

    def copy(self):
        """
        Return a :func:`copy.deepcopy` of self.

        Also creates copies the underlying Timeseries data

        Returns
        -------
        :obj:`ScmRun`
            :func:`copy.deepcopy` of ``self``
        """
        ret = copy.copy(self)
        ret._df = self._df.copy()
        ret._meta = self._meta.copy()
        ret.metadata = copy.copy(self.metadata)

        return ret

    def __len__(self) -> int:
        """
        Get the number of timeseries.
        """
        return self._df.shape[1]

    def __getitem__(self, key: Any) -> Any:
        """
        Get item of self with helpful direct access.

        Provides direct access to "time", "year" as well as the columns in :attr:`meta`.
        If key is anything else, the key will be applied to :attr:`_data`.
        """
        _key_check = (
            [key] if isinstance(key, str) or not isinstance(key, Iterable) else key
        )
        if key == "time":
            return pd.Series(self._time_points.to_index(), dtype="object")
        if key == "year":
            return pd.Series(self._time_points.years())
        if set(_key_check).issubset(self.meta_attributes):
            try:
                return self._meta_column(key).astype(
                    self._meta_column(key).cat.categories.dtype
                )
            except ValueError:
                return self._meta_column(key).astype(float)

        raise KeyError("[{}] is not in metadata".format(key))

    def __setitem__(
        self, key: str, value: Union[np.ndarray, list, int, float, str]
    ) -> Any:
        """
        Update metadata

        Notes
        -----
        If the meta values changes are applied to a filtered subset, the change will be reflected
        in the original :obj:`ScmRun` object.

        .. code:: python

            >>> df
            <scmdata.ScmRun (timeseries: 3, timepoints: 3)>
            Time:
                Start: 2005-01-01T00:00:00
                End: 2015-01-01T00:00:00
            Meta:
                   model     scenario region             variable   unit climate_model
                0  a_iam   a_scenario  World       Primary Energy  EJ/yr       a_model
                1  a_iam   a_scenario  World  Primary Energy|Coal  EJ/yr       a_model
                2  a_iam  a_scenario2  World       Primary Energy  EJ/yr       a_model
            >>> df["climate_model"] = ["a_model", "a_model", "b_model"]
            >>> df
            <scmdata.ScmRun (timeseries: 3, timepoints: 3)>
            Time:
                Start: 2005-01-01T00:00:00
                End: 2015-01-01T00:00:00
            Meta:
                   model     scenario region             variable   unit climate_model
                0  a_iam   a_scenario  World       Primary Energy  EJ/yr       a_model
                1  a_iam   a_scenario  World  Primary Energy|Coal  EJ/yr       a_model
                2  a_iam  a_scenario2  World       Primary Energy  EJ/yr       b_model
            >>> df2 = df.filter(variable="Primary Energy")
            >>> df2["pe_only"] = True
            >>> df2
            <scmdata.ScmRun (timeseries: 2, timepoints: 3)>
            Time:
                Start: 2005-01-01T00:00:00
                End: 2015-01-01T00:00:00
            Meta:
                   model     scenario region             variable   unit climate_model pe_only
                0  a_iam   a_scenario  World       Primary Energy  EJ/yr       a_model    True
                2  a_iam  a_scenario2  World       Primary Energy  EJ/yr       b_model    True
            >>> df
            <scmdata.ScmRun (timeseries: 3, timepoints: 3)>
            Time:
                Start: 2005-01-01T00:00:00
                End: 2015-01-01T00:00:00
            Meta:
                   model     scenario region             variable   unit climate_model pe_only
                0  a_iam   a_scenario  World       Primary Energy  EJ/yr       a_model    True
                1  a_iam   a_scenario  World  Primary Energy|Coal  EJ/yr       a_model     NaN
                2  a_iam  a_scenario2  World       Primary Energy  EJ/yr       b_model    True

        Parameters
        ----------
        key
            Column name

        value
            Values to write

            If a list of values is provided, then the length of that :obj:`value` must be the same as the number of timeseries

        Raises
        ------
        ValueError
            If the length of :obj:`meta` is inconsistent with the number of timeseries
        """
        meta = np.atleast_1d(value)
        if key == "time":
            self._time_points = TimePoints(meta)
            self._df.index = self._time_points.to_index()
        else:
            if len(meta) == 1:
                new_meta = self._meta.to_frame()
                new_meta[key] = meta[0]
                self._meta = pd.MultiIndex.from_frame(new_meta.astype("category"))
            elif len(meta) == len(self):
                new_meta_index = self._meta.to_frame(index=False)
                new_meta_index[key] = pd.Series(meta, dtype="category")
                self._meta = pd.MultiIndex.from_frame(new_meta_index)
            else:
                raise ValueError(
                    "Invalid length for metadata, `{}`, must be 1 or equal to the number of timeseries, `{}`".format(
                        len(meta), len(self)
                    )
                )

        if self._duplicated_meta():
            raise NonUniqueMetadataError(self.meta)

    def __repr__(self):
        def _indent(s):
            lines = ["\t" + line for line in s.split("\n")]
            return "\n".join(lines)

        meta_str = _indent(self.meta.__repr__())
        time_str = [
            "Start: {}".format(self.time_points.values[0]),
            "End: {}".format(self.time_points.values[-1]),
        ]
        time_str = _indent("\n".join(time_str))
        return "<scmdata.ScmRun (timeseries: {}, timepoints: {})>\nTime:\n{}\nMeta:\n{}".format(
            len(self), len(self.time_points), time_str, meta_str
        )

    @staticmethod
    def _binary_op(
        f: Callable[..., Any], reflexive=False, **kwargs,
    ) -> Callable[..., "ScmRun"]:
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, ScmRun):
                return NotImplemented

            is_scalar = isinstance(other, (numbers.Number, pint.Quantity))
            if not is_scalar:
                other_ndim = len(other.shape)
                if other_ndim == 1:
                    if other.shape[0] != self.shape[1]:
                        raise ValueError(
                            "only vectors with the same number of timesteps "
                            "as self ({}) are supported".format(self.shape[1])
                        )
                else:
                    raise ValueError(
                        "operations with {}d data are not supported".format(other_ndim)
                    )

            def _perform_op(df):
                if isinstance(other, pint.Quantity):
                    try:
                        data = df.values * ur(df.get_unique_meta("unit", True))
                        use_pint = True
                    except KeyError:
                        # let Pint assume dimensionless and raise an error as
                        # necessary
                        data = df.values
                        use_pint = False
                else:
                    data = df.values
                    use_pint = False

                res = []
                for v in data:
                    if not reflexive:
                        res.append(f(v, other))
                    else:
                        res.append(f(other, v))
                res = np.vstack(res)

                if use_pint:
                    df._df.values[:] = res.magnitude.T
                    df["unit"] = str(res.units)
                else:
                    df._df.values[:] = res.T
                return df

            return self.copy().groupby("unit").map(_perform_op)

        return func

    def drop_meta(self, columns: Union[list, str], inplace: Optional[bool] = False):
        """
        Drop meta columns out of the Run

        Parameters
        ----------
        columns
            The column or columns to drop
        inplace
            If True, do operation inplace and return None.

        Raises
        ------
        KeyError
            If any of the columns do not exist in the meta :class:`DataFrame`
        """
        if inplace:
            ret = self
        else:
            ret = self.copy()

        if isinstance(columns, str):
            columns = [columns]

        existing_cols = ret.meta_attributes
        for c in columns:
            if c not in existing_cols:
                raise KeyError(c)
            if c in self.required_cols:
                raise MissingRequiredColumnError([c])
        for c in columns:
            ret._meta = ret._meta.droplevel(c)

        if ret._duplicated_meta():
            raise NonUniqueMetadataError(ret.meta)

        if not inplace:
            return ret

    @property
    def meta_attributes(self):
        """
        Get a list of all meta keys

        Returns
        -------
        list
            Sorted list of meta keys
        """
        return sorted(list(self._meta.names))

    @property
    def time_points(self):
        """
        Time points of the data

        Returns
        -------
        :obj:`scmdata.time.TimePoints`
        """
        return self._time_points

    def timeseries(
        self, meta=None, check_duplicated=True, time_axis=None, drop_all_nan_times=False
    ):
        """
        Return the data with metadata as a :obj:`pd.DataFrame`.

        Parameters
        ----------
        meta : list[str]
            The list of meta columns that will be included in the output's
            MultiIndex. If None (default), then all metadata will be used.

        check_duplicated : bool
            If True, an exception is raised if any of the timeseries have
            duplicated metadata

        time_axis : {None, "year", "year-month", "days since 1970-01-01", "seconds since 1970-01-01"}
            Time axis to use for the output's columns. If ``None``,
            :class:`datetime.datetime` objects will be used. If ``"year"``, the
            year of each time point  will be used. If ``"year-month"``, the year
            plus (month - 0.5) / 12  will be used. If
            ``"days since 1970-01-01"``, the number of days since 1st Jan 1970
            will be used (calculated using the :mod:`datetime` module). If
            ``"seconds since 1970-01-01"``, the number of seconds  since 1st Jan
            1970 will be used (calculated using the :mod:`datetime` module).

        drop_all_nan_times : bool
            Should time points which contain only nan values be dropped? This operation is applied
            after any transforms introduced by the value of ``time_axis``.

        Returns
        -------
        :obj:`pd.DataFrame`
            DataFrame with datetimes as columns and timeseries as rows.
            Metadata is in the index.

        Raises
        ------
        :class:`NonUniqueMetadataError`
            If the metadata are not unique between timeseries and
            ``check_duplicated`` is ``True``

        NotImplementedError
            The value of `time_axis` is not recognised

        ValueError
            The value of `time_axis` would result in columns which aren't unique
        """
        df = self._df.T
        _meta = self.meta if meta is None else self.meta[meta]

        if check_duplicated and self._duplicated_meta(meta=_meta):
            raise NonUniqueMetadataError(_meta)

        if time_axis is None:
            columns = self._time_points.to_index()
        elif time_axis == "year":
            columns = self._time_points.years()
        elif time_axis == "year-month":
            columns = (
                self._time_points.years() + (self._time_points.months() - 0.5) / 12
            )
        elif time_axis == "days since 1970-01-01":

            def calc_days(x):
                ref = np.array(["1970-01-01"], dtype=_TARGET_DTYPE)[0]

                return (x - ref).astype("timedelta64[D]")

            columns = calc_days(self._time_points.values).astype(int)

        elif time_axis == "seconds since 1970-01-01":

            def calc_seconds(x):
                ref = np.array(["1970-01-01"], dtype=_TARGET_DTYPE)[0]

                return x - ref

            columns = calc_seconds(self._time_points.values).astype(int)

        else:
            raise NotImplementedError("time_axis = '{}'".format(time_axis))

        if len(np.unique(columns)) != len(columns):
            raise ValueError(
                "Ambiguous time values with time_axis = '{}'".format(time_axis)
            )

        df.columns = pd.Index(columns, name="time")
        df.index = pd.MultiIndex.from_frame(_meta)

        if drop_all_nan_times:
            df = df.dropna(how="all", axis="columns")

        return df

    def _duplicated_meta(self, meta=None):
        _meta = self._meta if meta is None else meta

        return _meta.duplicated().any()

    def long_data(self, time_axis=None):
        """
        Return data in long form, particularly useful for plotting with seaborn

        Parameters
        ----------
        time_axis : {None, "year", "year-month", "days since 1970-01-01", "seconds since 1970-01-01"}
            Time axis to use for the output's columns. If `None`,
            :class:`datetime.datetime` objects will be used. If `"year"`, the
            year of each time point  will be used. If `"year-month", the year
            plus (month - 0.5) / 12  will be used. If
            `"days since 1970-01-01"`, the number of days  since 1st Jan 1970
            will be used (calculated using the ``datetime``  module). If
            `"seconds since 1970-01-01"`, the number of seconds  since 1st Jan
            1970 will be used (calculated using the ``datetime`` module).

        Returns
        -------
        :obj:`pd.DataFrame`
            :obj:`pd.DataFrame` containing the data in 'long form' (i.e. one observation per row).
        """
        out = self.timeseries(time_axis=time_axis).stack()
        out.name = "value"
        out = out.to_frame().reset_index()

        return out

    @property
    def shape(self) -> tuple:
        """
        Get the shape of the underlying data as ``(num_timeseries, num_timesteps)``

        Returns
        -------
        tuple of int
        """
        return self._df.T.shape

    @property
    def values(self) -> np.ndarray:
        """
        Timeseries values without metadata

        The values are returned such that each row is a different
        timeseries being a row and each column is a different time (although
        no time information is included as a plain :obj:`np.ndarray` is
        returned).

        Returns
        -------
        np.ndarray
            The array in the same shape as :py:obj:`ScmRun.shape`, that is
            ``(num_timeseries, num_timesteps)``.
        """
        return self._df.values.T

    @property
    def empty(self) -> bool:
        """
        Indicate whether :obj:`ScmRun` is empty i.e. contains no data

        Returns
        -------
        bool
            If :obj:`ScmRun` is empty, return ``True``, if not return ``False``
        """
        return np.equal(len(self), 0)

    @property
    def meta(self) -> pd.DataFrame:
        """
        Metadata
        """
        df = pd.DataFrame(
            self._meta.to_list(), columns=self._meta.names, index=self._df.columns
        )

        return df[sorted(df.columns)]

    def _meta_column(self, col) -> pd.Series:
        out = self._meta.get_level_values(col)
        return pd.Series(out, name=col, index=self._df.columns)

    def filter(
        self,
        keep: bool = True,
        inplace: bool = False,
        log_if_empty: bool = True,
        **kwargs: Any,
    ):
        """
        Return a filtered ScmRun (i.e., a subset of the data).

        .. code:: python

            >>> df
            <scmdata.ScmRun (timeseries: 3, timepoints: 3)>
            Time:
                Start: 2005-01-01T00:00:00
                End: 2015-01-01T00:00:00
            Meta:
                   model     scenario region             variable   unit climate_model
                0  a_iam   a_scenario  World       Primary Energy  EJ/yr       a_model
                1  a_iam   a_scenario  World  Primary Energy|Coal  EJ/yr       a_model
                2  a_iam  a_scenario2  World       Primary Energy  EJ/yr       a_model
                [3 rows x 7 columns]

            >>> df.filter(scenario="a_scenario")
            <scmdata.ScmRun (timeseries: 2, timepoints: 3)>
            Time:
                Start: 2005-01-01T00:00:00
                End: 2015-01-01T00:00:00
            Meta:
                   model     scenario region             variable   unit climate_model
                0  a_iam   a_scenario  World       Primary Energy  EJ/yr       a_model
                1  a_iam   a_scenario  World  Primary Energy|Coal  EJ/yr       a_model
                [2 rows x 7 columns]

            >>> df.filter(scenario="a_scenario", keep=False)
            <scmdata.ScmRun (timeseries: 1, timepoints: 3)>
            Time:
                Start: 2005-01-01T00:00:00
                End: 2015-01-01T00:00:00
            Meta:
                   model     scenario region             variable   unit climate_model
                2  a_iam  a_scenario2  World       Primary Energy  EJ/yr       a_model
                [1 rows x 7 columns]

            >>> df.filter(level=1)
            <scmdata.ScmRun (timeseries: 2, timepoints: 3)>
            Time:
                Start: 2005-01-01T00:00:00
                End: 2015-01-01T00:00:00
            Meta:
                   model     scenario region             variable   unit climate_model
                0  a_iam   a_scenario  World       Primary Energy  EJ/yr       a_model
                2  a_iam  a_scenario2  World       Primary Energy  EJ/yr       a_model
                [2 rows x 7 columns]

            >>> df.filter(year=range(2000, 2011))
            <scmdata.ScmRun (timeseries: 3, timepoints: 2)>
            Time:
                Start: 2005-01-01T00:00:00
                End: 2010-01-01T00:00:00
            Meta:
                   model     scenario region             variable   unit climate_model
                0  a_iam   a_scenario  World       Primary Energy  EJ/yr       a_model
                1  a_iam   a_scenario  World  Primary Energy|Coal  EJ/yr       a_model
                2  a_iam  a_scenario2  World       Primary Energy  EJ/yr       a_model
                [2 rows x 7 columns]

        Parameters
        ----------
        keep
            If True, keep all timeseries satisfying the filters, otherwise drop all the
            timeseries satisfying the filters

        inplace
            If True, do operation inplace and return None

        log_if_empty
            If ``True``, log a warning level message if the result is empty.

        **kwargs
            Argument names are keys with which to filter, values are used to do the
            filtering. Filtering can be done on:

            - all metadata columns with strings, "*" can be used as a wildcard in search
              strings

            - 'level': the maximum "depth" of IAM variables (number of hierarchy levels,
              excluding the strings given in the 'variable' argument)

            - 'time': takes a :class:`datetime.datetime` or list of
              :class:`datetime.datetime`'s
              TODO: default to np.datetime64

            - 'year', 'month', 'day', hour': takes an :class:`int` or list of
              :class:`int`'s ('month' and 'day' also accept :class:`str` or list of
              :class:`str`)

            If ``regexp=True`` is included in :obj:`kwargs` then the pseudo-regexp
            syntax in :func:`pattern_match` is disabled.

        Returns
        -------
        :obj:`ScmRun`
            If not ``inplace``, return a new instance with the filtered data.
        """
        ret = copy.copy(self) if not inplace else self

        if len(ret):
            _keep_times, _keep_rows = self._apply_filters(kwargs)
            if not keep and sum(~_keep_rows) and sum(~_keep_times):
                raise ValueError(
                    "If keep==False, filtering cannot be performed on the temporal axis "
                    "and with metadata at the same time"
                )

            reduce_times = (~_keep_times).sum() > 0
            reduce_rows = (~_keep_rows).sum() > 0

            if not keep:
                if reduce_times:
                    _keep_times = ~_keep_times
                if reduce_rows:
                    _keep_rows = ~_keep_rows
                if not reduce_rows and not reduce_times:
                    _keep_times = _keep_times * False
                    _keep_rows = _keep_rows * False

            ret._df = ret._df.loc[_keep_times, _keep_rows]
            ret._meta = ret._meta[_keep_rows]
            ret["time"] = self.time_points.values[_keep_times]

        if log_if_empty and ret.empty:
            _logger.warning("Filtered ScmRun is empty!", stack_info=True)

        if not inplace:
            return ret

        return None

    # pylint doesn't recognise ',' in returns type definition
    def _apply_filters(  # pylint: disable=missing-return-doc
        self, filters: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine rows to keep in data for given set of filters.

        Parameters
        ----------
        filters
            Dictionary of filters ``({col: values}})``; uses a pseudo-regexp syntax by
            default but if ``filters["regexp"]`` is ``True``, regexp is used directly.

        Returns
        -------
        :obj:`np.ndarray` of :class:`bool`, :obj:`np.ndarray` of :class:`bool`
            Two boolean :class:`np.ndarray`'s. The first contains the columns to keep
            (i.e. which time points to keep). The second contains the rows to keep (i.e.
            which metadata matched the filters).

        Raises
        ------
        ValueError
            Filtering cannot be performed on requested column
        """
        regexp = filters.pop("regexp", False)
        keep_ts = np.array([True] * len(self.time_points))
        keep_meta = np.array([True] * len(self))

        # filter by columns and list of values
        for col, values in filters.items():
            if col in self._meta.names:
                if col == "variable":
                    level = filters["level"] if "level" in filters else None
                else:
                    level = None

                keep_meta &= pattern_match(
                    self._meta.get_level_values(col),
                    values,
                    level=level,
                    regexp=regexp,
                    separator=self.data_hierarchy_separator,
                )

            elif col == "level":
                if "variable" not in filters.keys():
                    keep_meta &= pattern_match(
                        self._meta.get_level_values("variable"),
                        "*",
                        level=values,
                        regexp=regexp,
                        separator=self.data_hierarchy_separator,
                    )
                # else do nothing as level handled in variable filtering

            elif col == "year":
                keep_ts &= years_match(self._time_points.years(), values)

            elif col == "month":
                keep_ts &= month_match(self._time_points.months(), values)

            elif col == "day":
                keep_ts &= self._day_match(values)

            elif col == "hour":
                keep_ts &= hour_match(self._time_points.hours(), values)

            elif col == "time":
                keep_ts &= datetime_match(self._time_points.values, values)

            else:
                raise ValueError("filter by `{}` not supported".format(col))

        return keep_ts, keep_meta

    def _day_match(self, values):
        if isinstance(values, str):
            wday = True
        elif isinstance(values, list) and isinstance(values[0], str):
            wday = True
        else:
            wday = False

        if wday:
            days = self._time_points.weekdays()
        else:  # ints or list of ints
            days = self._time_points.days()

        return day_match(days, values)

    def head(self, *args, **kwargs):
        """
        Return head of :func:`self.timeseries()`.

        Parameters
        ----------
        *args
            Passed to :func:`self.timeseries().head()`

        **kwargs
            Passed to :func:`self.timeseries().head()`

        Returns
        -------
        :obj:`pd.DataFrame`
            Tail of :func:`self.timeseries()`
        """
        return self.timeseries().head(*args, **kwargs)

    def tail(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return tail of :func:`self.timeseries()`.

        Parameters
        ----------
        *args
            Passed to :func:`self.timeseries().tail()`

        **kwargs
            Passed to :func:`self.timeseries().tail()`

        Returns
        -------
        :obj:`pd.DataFrame`
            Tail of :func:`self.timeseries()`
        """
        return self.timeseries().tail(*args, **kwargs)

    def get_unique_meta(
        self, meta: str, no_duplicates: Optional[bool] = False,
    ) -> Union[List[Any], Any]:
        """
        Get unique values in a metadata column.

        Parameters
        ----------
        meta
            Column to retrieve metadata for

        no_duplicates
            Should I raise an error if there is more than one unique value in the
            metadata column?

        Raises
        ------
        ValueError
            There is more than one unique value in the metadata column and
            ``no_duplicates`` is ``True``.

        Returns
        -------
        [List[Any], Any]
            List of unique metadata values. If ``no_duplicates`` is ``True`` the
            metadata value will be returned (rather than a list).
        """
        vals = self._meta.get_level_values(meta).unique().to_list()
        if no_duplicates:
            if len(vals) != 1:
                raise ValueError(
                    "`{}` column is not unique (found values: {})".format(meta, vals)
                )

            return vals[0]

        return vals

    def interpolate(
        self,
        target_times: Union[np.ndarray, List[Union[dt.datetime, int]]],
        interpolation_type: str = "linear",
        extrapolation_type: str = "linear",
    ):
        """
        Interpolate the dataframe onto a new time frame.

        Parameters
        ----------
        target_times
            Time grid onto which to interpolate
        interpolation_type: str
            Interpolation type. Options are 'linear'
        extrapolation_type: str or None
            Extrapolation type. Options are None, 'linear' or 'constant'

        Returns
        -------
        :obj:`ScmRun`
            A new :class:`ScmRun` containing the data interpolated onto the
            :obj:`target_times` grid
        """
        # pylint: disable=protected-access

        target_times = np.asarray(target_times, dtype="datetime64[s]")

        res = self.copy()

        target_times = TimePoints(target_times)

        timeseries_converter = TimeseriesConverter(
            self.time_points.values,
            target_times.values,
            interpolation_type=interpolation_type,
            extrapolation_type=extrapolation_type,
        )
        target_data = np.zeros((len(target_times), len(res)))

        # TODO: Extend TimeseriesConverter to handle 2d inputs
        for i in range(len(res)):
            target_data[:, i] = timeseries_converter.convert_from(
                res._df.iloc[:, i].values
            )
        res._df = pd.DataFrame(
            target_data, columns=res._df.columns, index=target_times.to_index()
        )
        res._time_points = target_times

        return res

    def resample(self, rule: str = "AS", **kwargs: Any):
        """
        Resample the time index of the timeseries data onto a custom grid.

        This helper function allows for values to be easily interpolated onto annual or
        monthly timesteps using the rules='AS' or 'MS' respectively. Internally, the
        interpolate function performs the regridding.

        Parameters
        ----------
        rule
            See the pandas `user guide
            <http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
            for a list of options. Note that Business-related offsets such as
            "BusinessDay" are not supported.

        **kwargs
            Other arguments to pass through to :func:`interpolate`

        Returns
        -------
        :obj:`ScmRun`
            New :class:`ScmRun` instance on a new time index

        Examples
        --------
        Resample a dataframe to annual values

        >>> scm_df = ScmRun(
        ...     pd.Series([1, 2, 10], index=(2000, 2001, 2009)),
        ...     columns={
        ...         "model": ["a_iam"],
        ...         "scenario": ["a_scenario"],
        ...         "region": ["World"],
        ...         "variable": ["Primary Energy"],
        ...         "unit": ["EJ/y"],
        ...     }
        ... )
        >>> scm_df.timeseries().T
        model             a_iam
        scenario     a_scenario
        region            World
        variable Primary Energy
        unit               EJ/y
        year
        2000                  1
        2010                 10

        An annual timeseries can be the created by interpolating to the start of years
        using the rule 'AS'.

        >>> res = scm_df.resample('AS')
        >>> res.timeseries().T
        model                        a_iam
        scenario                a_scenario
        region                       World
        variable            Primary Energy
        unit                          EJ/y
        time
        2000-01-01 00:00:00       1.000000
        2001-01-01 00:00:00       2.001825
        2002-01-01 00:00:00       3.000912
        2003-01-01 00:00:00       4.000000
        2004-01-01 00:00:00       4.999088
        2005-01-01 00:00:00       6.000912
        2006-01-01 00:00:00       7.000000
        2007-01-01 00:00:00       7.999088
        2008-01-01 00:00:00       8.998175
        2009-01-01 00:00:00      10.00000

        >>> m_df = scm_df.resample('MS')
        >>> m_df.timeseries().T
        model                        a_iam
        scenario                a_scenario
        region                       World
        variable            Primary Energy
        unit                          EJ/y
        time
        2000-01-01 00:00:00       1.000000
        2000-02-01 00:00:00       1.084854
        2000-03-01 00:00:00       1.164234
        2000-04-01 00:00:00       1.249088
        2000-05-01 00:00:00       1.331204
        2000-06-01 00:00:00       1.416058
        2000-07-01 00:00:00       1.498175
        2000-08-01 00:00:00       1.583029
        2000-09-01 00:00:00       1.667883
                                    ...
        2008-05-01 00:00:00       9.329380
        2008-06-01 00:00:00       9.414234
        2008-07-01 00:00:00       9.496350
        2008-08-01 00:00:00       9.581204
        2008-09-01 00:00:00       9.666058
        2008-10-01 00:00:00       9.748175
        2008-11-01 00:00:00       9.833029
        2008-12-01 00:00:00       9.915146
        2009-01-01 00:00:00      10.000000
        [109 rows x 1 columns]


        Note that the values do not fall exactly on integer values as not all years are
        exactly the same length.

        References
        ----------
        See the pandas documentation for
        `resample <http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.resample.html>`
        for more information about possible arguments.
        """
        orig_dts = self["time"]
        target_dts = generate_range(
            orig_dts.iloc[0], orig_dts.iloc[-1], to_offset(rule)
        )
        return self.interpolate(list(target_dts), **kwargs)

    def time_mean(self, rule: str):
        """
        Take time mean of self

        Note that this method will not copy the ``metadata`` attribute to the returned
        value.

        Parameters
        ----------
        rule : ["AC", "AS", "A"]
            How to take the time mean. The names reflect the pandas
            `user guide <http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
            where they can, but only the options
            given above are supported. For clarity, if ``rule`` is ``'AC'``, then the
            mean is an annual mean i.e. each time point in the result is the mean of
            all values for that particular year. If ``rule`` is ``'AS'``, then the
            mean is an annual mean centred on the beginning of the year i.e. each time
            point in the result is the mean of all values from July 1st in the
            previous year to June 30 in the given year. If ``rule`` is ``'A'``, then
            the mean is an annual mean centred on the end of the year i.e. each time
            point in the result is the mean of all values from July 1st of the given
            year to June 30 in the next year.

        Returns
        -------
        :obj:`ScmRun`
            The time mean of ``self``.
        """
        if rule == "AS":

            def group_annual_mean_beginning_of_year(x):
                if x.month <= 6:
                    return x.year
                return x.year + 1

            ts_resampled = (
                self.timeseries()
                .T.groupby(group_annual_mean_beginning_of_year)
                .mean()
                .T
            )
            ts_resampled.columns = ts_resampled.columns.map(
                lambda x: dt.datetime(x, 1, 1)
            )
            return type(self)(ts_resampled)

        if rule == "AC":

            def group_annual_mean(x):
                return x.year

            ts_resampled = self.timeseries().T.groupby(group_annual_mean).mean().T
            ts_resampled.columns = ts_resampled.columns.map(
                lambda x: dt.datetime(x, 7, 1)
            )
            return type(self)(ts_resampled)

        if rule == "A":

            def group_annual_mean_end_of_year(x):
                if x.month >= 7:
                    return x.year
                return x.year - 1

            ts_resampled = (
                self.timeseries().T.groupby(group_annual_mean_end_of_year).mean().T
            )
            ts_resampled.columns = ts_resampled.columns.map(
                lambda x: dt.datetime(x, 12, 31)
            )
            return type(self)(ts_resampled)

        raise ValueError("`rule` = `{}` is not supported".format(rule))

    def process_over(
        self,
        cols: Union[str, List[str]],
        operation: str,
        na_override=-1e6,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Process the data over the input columns.

        Parameters
        ----------
        cols
            Columns to perform the operation on. The timeseries will be grouped by all
            other columns in :attr:`meta`.

        operation : ['median', 'mean', 'quantile']
            The operation to perform. This uses the equivalent pandas function. Note
            that quantile means the value of the data at a given point in the cumulative
            distribution of values at each point in the timeseries, for each timeseries
            once the groupby is applied. As a result, using ``q=0.5`` is is the same as
            taking the median and not the same as taking the mean/average.

        na_override: [int, float]
            Convert any nan value in the timeseries meta to this value during processsing.
            The meta values converted back to nan's before the dataframe is returned. This
            should not need to be changed unless the existing metadata clashes with the
            default na_override value.

            This functionality is disabled if na_override is None, but may result incorrect
            results if the timeseries meta includes any nan's.

        **kwargs
            Keyword arguments to pass to the pandas operation

        Returns
        -------
        :obj:`pd.DataFrame`
            The quantiles of the timeseries, grouped by all columns in :attr:`meta`
            other than :obj:`cols`

        Raises
        ------
        ValueError
            If the operation is not one of ['median', 'mean', 'quantile']

            If the value of na_override clashes with any existing metadata
        """
        cols = [cols] if isinstance(cols, str) else cols
        ts = self.timeseries()
        if na_override is not None:
            ts_idx = ts.index.to_frame()
            if ts_idx[ts_idx == na_override].any().any():
                raise ValueError(
                    "na_override clashes with existing meta: {}".format(na_override)
                )
            ts.index = pd.MultiIndex.from_frame(ts_idx.fillna(na_override))
        group_cols = list(set(ts.index.names) - set(cols))
        grouper = ts.groupby(group_cols)

        if operation == "median":
            res = grouper.median(**kwargs)
        elif operation == "mean":
            res = grouper.mean(**kwargs)
        elif operation == "quantile":
            res = grouper.quantile(**kwargs)
        else:
            raise ValueError("operation must be one of ['median', 'mean', 'quantile']")

        if na_override is not None:
            idx_df = res.index.to_frame()
            idx_df[idx_df == na_override] = np.nan
            res.index = pd.MultiIndex.from_frame(idx_df)

        return res

    def quantiles_over(
        self,
        cols: Union[str, List[str]],
        quantiles: Union[str, List[float]],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Calculate quantiles of the data over the input columns.

        Parameters
        ----------
        cols
            Columns to perform the operation on. The timeseries will be grouped by all
            other columns in :attr:`meta`.

        quantiles
            The quantiles to calculate. This should be a list of quantiles to calculate
            (quantile values between 0 and 1). ``quantiles`` can also include the strings
            "median" or "mean" if these values are to be calculated.

        **kwargs
            Passed to :meth:`~ScmRun.process_over`.

        Returns
        -------
        :obj:`pd.DataFrame`
            The quantiles of the timeseries, grouped by all columns in :attr:`meta`
            other than :obj:`cols`. Each calculated quantile is given a label which is
            stored in the ``quantile`` column within the output index.

        Raises
        ------
        TypeError
            ``operation`` is included in ``kwargs``. The operation is inferred from ``quantiles``.
        """
        if "operation" in kwargs:
            raise TypeError(
                "quantiles_over() does not take the keyword argument 'operation', the operations "
                "are inferred from the 'quantiles' argument"
            )

        out = []
        for quant in quantiles:
            if quant == "median":
                quantile_df = self.process_over(cols, "median")
            elif quant == "mean":
                quantile_df = self.process_over(cols, "mean")
            else:
                quantile_df = self.process_over(cols, "quantile", q=quant)

            quantile_df["quantile"] = quant

            out.append(quantile_df)

        out = pd.concat(out).set_index("quantile", append=True)

        return out

    def groupby(self, *group):
        """
        Group the object by unique metadata

        Enables iteration over groups of data. For example, to iterate over each scenario in the object

        .. code:: python

            >>> for group in df.groupby("scenario"):
            >>>    print(group)
            <scmdata.ScmRun (timeseries: 2, timepoints: 3)>
            Time:
                Start: 2005-01-01T00:00:00
                End: 2015-01-01T00:00:00
            Meta:
                   model    scenario region             variable   unit climate_model
                0  a_iam  a_scenario  World       Primary Energy  EJ/yr       a_model
                1  a_iam  a_scenario  World  Primary Energy|Coal  EJ/yr       a_model
            <scmdata.ScmRun (timeseries: 1, timepoints: 3)>
            Time:
                Start: 2005-01-01T00:00:00
                End: 2015-01-01T00:00:00
            Meta:
                   model     scenario region        variable   unit climate_model
                2  a_iam  a_scenario2  World  Primary Energy  EJ/yr       a_model

        Parameters
        ----------
        group: str or list of str
            Columns to group by

        Returns
        -------
        :obj:`RunGroupBy`
            See the documentation for :class:`RunGroupBy` for more information

        """
        if len(group) == 1 and not isinstance(group[0], str):
            group = tuple(group[0])
        return RunGroupBy(self, group)

    def convert_unit(
        self,
        unit: str,
        context: Optional[str] = None,
        inplace: bool = False,
        **kwargs: Any,
    ):
        """
        Convert the units of a selection of timeseries.

        Uses :class:`scmdata.units.UnitConverter` to perform the conversion.

        Parameters
        ----------
        unit
            Unit to convert to. This must be recognised by
            :class:`~openscm.units.UnitConverter`.

        context
            Context to use for the conversion i.e. which metric to apply when performing
            CO2-equivalent calculations. If ``None``, no metric will be applied and
            CO2-equivalent calculations will raise :class:`DimensionalityError`.

        inplace
            If True, apply the conversion inplace and return None

        **kwargs
            Extra arguments which are passed to :meth:`~ScmRun.filter` to
            limit the timeseries which are attempted to be converted. Defaults to
            selecting the entire ScmRun, which will likely fail.

        Returns
        -------
        :obj:`ScmRun`
            If :obj:`inplace` is not ``False``, a new :class:`ScmRun` instance
            with the converted units.

        Notes
        -----
        If ``context`` is not ``None``, then the context used for the conversion will
        be checked against any existing metadata and, if the conversion is valid,
        stored in the output's metadata.

        Raises
        ------
        ValueError
            ``"unit_context"`` is already included in ``self``'s :meth:`meta_attributes`
            and it does not match ``context`` for the variables to be converted.
        """
        # pylint: disable=protected-access
        if inplace:
            ret = self
        else:
            ret = self.copy()

        to_convert_filtered = ret.filter(**kwargs, log_if_empty=False)
        to_not_convert_filtered = ret.filter(**kwargs, keep=False, log_if_empty=False)

        already_correct_unit = to_convert_filtered.filter(unit=unit, log_if_empty=False)
        if (
            "unit_context" in already_correct_unit.meta_attributes
            and not already_correct_unit.empty
        ):
            self._check_unit_context(already_correct_unit, context)

        to_convert = to_convert_filtered.filter(
            unit=unit, log_if_empty=False, keep=False
        )
        to_not_convert = run_append([to_not_convert_filtered, already_correct_unit,])

        if "unit_context" in to_convert.meta_attributes and not to_convert.empty:
            self._check_unit_context(to_convert, context)

        if context is not None:
            to_convert["unit_context"] = context

        if "unit_context" not in to_not_convert.meta_attributes and context is not None:
            to_not_convert["unit_context"] = None

        def apply_units(group):
            orig_unit = group.get_unique_meta("unit", no_duplicates=True)
            uc = UnitConverter(orig_unit, unit, context=context)

            group._df.values[:] = uc.convert_from(group._df.values)
            group["unit"] = unit

            return group

        ret = to_convert
        if not to_convert.empty:
            ret = ret.groupby("unit").map(apply_units)

        ret = run_append([ret, to_not_convert], inplace=inplace)
        if not inplace:
            return ret

    @staticmethod
    def _check_unit_context(dat, context):
        unit_context = dat.get_unique_meta("unit_context")

        # check if contexts don't match, unless the context is nan
        non_matching_contexts = len(unit_context) > 1 or unit_context[0] != context
        if isinstance(unit_context[0], float):
            non_matching_contexts &= not np.isnan(unit_context[0])

        if non_matching_contexts:
            raise ValueError(
                "Existing unit conversion context(s), `{}`, doesn't match input "
                "context, `{}`, drop `unit_context` metadata before doing "
                "conversion".format(unit_context, context)
            )

    def relative_to_ref_period_mean(self, append_str=None, **kwargs):
        """
        Return the timeseries relative to a given reference period mean.

        The reference period mean is subtracted from all values in the input timeseries.

        Parameters
        ----------
        append_str
            Deprecated

        **kwargs
            Arguments to pass to :func:`filter` to determine the data to be included in
            the reference time period. See the docs of :func:`filter` for valid options.

        Returns
        -------
        :obj:`ScmRun`
            New object containing the timeseries, adjusted to the reference period mean.
            The reference period year bounds are stored in the meta columns
            ``"reference_period_start_year"`` and ``"reference_period_end_year"``.

        Raises
        ------
        NotImplementedError
            ``append_str`` is not ``None``
        """
        if append_str is not None:
            raise NotImplementedError("`append_str` is deprecated")

        ts = self.timeseries()
        # mypy confused by `inplace` default
        ref_data = self.filter(**kwargs)
        ref_period_mean = ref_data.timeseries().mean(axis="columns")  # type: ignore

        res = ts.sub(ref_period_mean, axis="rows")
        res.reset_index(inplace=True)

        res["reference_period_start_year"] = ref_data["year"].min()
        res["reference_period_end_year"] = ref_data["year"].max()

        return type(self)(res)

    def append(
        self,
        other,
        inplace: bool = False,
        duplicate_msg: Union[str, bool] = True,
        metadata: Optional[MetadataType] = None,
        **kwargs: Any,
    ):
        """
        Append additional data to the current dataframe.

        For details, see :func:`run_append`.

        Parameters
        ----------
        other
            Data (in format which can be cast to :class:`ScmRun`) to append

        inplace
            If ``True``, append data in place and return ``None``. Otherwise, return a
            new :class:`ScmRun` instance with the appended data.

        duplicate_msg
            If ``True``, raise a :class:`scmdata.errors.NonUniqueMetadataError` error so the user
            can see the duplicate timeseries. If ``False``, take the average
            and do not raise a warning or error. If ``"warn"``, raise a
            warning if duplicate data is detected.

        metadata
            If not ``None``, override the metadata of the resulting :obj:`ScmRun` with ``metadata``.
            Otherwise, the metadata for the runs are merged. In the case where there are duplicate
            metadata keys, the values from the first run are used.

        **kwargs
            Keywords to pass to :func:`ScmRun.__init__` when reading
            :obj:`other`

        Returns
        -------
        :obj:`ScmRun`
            If not :obj:`inplace`, return a new :class:`ScmRun` instance
            containing the result of the append.

        Raises
        ------
        NonUniqueMetadataError
            If the appending results in timeseries with duplicate metadata and :attr:`duplicate_msg` is ``True``

        """
        if not isinstance(other, ScmRun):
            other = self.__class__(other, **kwargs)

        return run_append(
            [self, other],
            inplace=inplace,
            duplicate_msg=duplicate_msg,
            metadata=metadata,
        )

    def to_iamdataframe(self) -> LongDatetimeIamDataFrame:  # pragma: no cover
        """
        Convert to a :class:`LongDatetimeIamDataFrame` instance.

        :class:`LongDatetimeIamDataFrame` is a subclass of :class:`pyam.IamDataFrame`.
        We use :class:`LongDatetimeIamDataFrame` to ensure all times can be handled, see
        docstring of :class:`LongDatetimeIamDataFrame` for details.

        Returns
        -------
        :class:`LongDatetimeIamDataFrame`
            :class:`LongDatetimeIamDataFrame` instance containing the same data.

        Raises
        ------
        ImportError
            If `pyam <https://github.com/IAMconsortium/pyam>`_ is not installed
        """
        if LongDatetimeIamDataFrame is None:
            raise ImportError(
                "pyam is not installed. Features involving IamDataFrame are unavailable"
            )

        return LongDatetimeIamDataFrame(self.timeseries())

    def to_csv(self, fname: str, **kwargs: Any) -> None:
        """
        Write timeseries data to a csv file

        Parameters
        ----------
        fname
            Path to write the file into
        """
        self.timeseries().reset_index().to_csv(fname, **kwargs, index=False)

    def reduce(self, func, dim=None, axis=None, **kwargs):
        """
        Apply a function along a given axis

        This is to provide the GroupBy functionality in :func:`ScmRun.groupby` and is not generally called directly.

        This implementation is very bare-bones - no reduction along the time time dimension is allowed and only the `dim`
        parameter is used.

        Parameters
        ----------
        func: function
        dim : str
            Ignored
        axis : int
            The dimension along which the function is applied. The only valid value is 0 which corresponds to the along the
            time-series dimension.
        kwargs
            Other parameters passed to `func`

        Returns
        -------
        :obj:`ScmRun`

        Raises
        ------
        ValueError
            If a dimension other than None is provided

        NotImplementedError
            If `axis` is anything other than 0


        """
        if dim is not None:
            raise ValueError("ScmRun.reduce does not handle dim. Use axis instead")

        input_data = self.values

        if axis is None or axis == 1:
            raise NotImplementedError(
                "Cannot currently reduce along the time dimension"
            )

        if axis is not None:
            data = func(input_data, axis=axis, **kwargs)
        else:
            data = func(input_data, **kwargs)

        if getattr(data, "shape", ()) == self.shape:
            return type(self)(
                data, index=self.time_points, columns=self.meta.to_dict("list")
            )
        else:
            removed_axes = range(2) if axis is None else np.atleast_1d(axis) % 2
            index = self.time_points
            meta = self.meta.to_dict("list")
            if 0 in removed_axes and len(meta):
                # Reduced the timeseries
                m = self.meta
                n_unique = m.nunique(axis=0)
                m = m.drop(columns=n_unique[n_unique > 1].index).drop_duplicates()
                if len(m) != 1:  # pragma: no cover
                    raise AssertionError(m)

                meta = m.to_dict("list")

            if 1 in removed_axes:
                raise NotImplementedError  # pragma: no cover

            return type(self)(data, index=index, columns=meta)


def _merge_metadata(metadata):
    res = metadata[0].copy()

    for m in metadata[1:]:
        for k, v in m.items():
            if k not in res:
                res[k] = v
    return res


def run_append(
    runs,
    inplace: bool = False,
    duplicate_msg: Union[str, bool] = True,
    metadata: Optional[MetadataType] = None,
):
    """
    Append together many objects.

    When appending many objects, it may be more efficient to call this routine once with
    a list of :class:`ScmRun`'s, than using :func:`ScmRun.append` multiple times.

    If timeseries with duplicate metadata are found, the timeseries are appended and values
    falling on the same timestep are averaged if :obj:`duplicate_msg` is not "return". If
    :obj:`duplicate_msg` is "return", then the result will contain the duplicated timeseries
    for further inspection.

    .. code:: python

        >>> res = base.append(other, duplicate_msg="return")
        <scmdata.ScmRun (timeseries: 5, timepoints: 3)>
        Time:
            Start: 2005-01-01T00:00:00
            End: 2015-06-12T00:00:00
        Meta:
                  scenario             variable  model climate_model region   unit
            0   a_scenario       Primary Energy  a_iam       a_model  World  EJ/yr
            1   a_scenario  Primary Energy|Coal  a_iam       a_model  World  EJ/yr
            2  a_scenario2       Primary Energy  a_iam       a_model  World  EJ/yr
            3  a_scenario3       Primary Energy  a_iam       a_model  World  EJ/yr
            4   a_scenario       Primary Energy  a_iam       a_model  World  EJ/yr
        >>> ts = res.timeseries(check_duplicated=False)
        >>> ts[ts.index.duplicated(keep=False)]
        time                                                        2005-01-01  ...  2015-06-12
        scenario   variable       model climate_model region unit               ...
        a_scenario Primary Energy a_iam a_model       World  EJ/yr         1.0  ...         7.0
                                                             EJ/yr        -1.0  ...         1.0


    Parameters
    ----------
    runs: list of :class:`ScmRun`
        The runs to append. Values will be attempted to be cast to :class:`ScmRun`.

    inplace
        If ``True``, then the operation updates the first item in :obj:`runs` and returns
        ``None``.

    duplicate_msg
        If ``True``, raise a ``NonUniqueMetadataError`` error so the user can
        see the duplicate timeseries. If ``False``, take the average and do
        not raise a warning or error. If ``"warn"``, raise a warning if
        duplicate data is detected.

    metadata
        If not ``None``, override the metadata of the resulting :obj:`ScmRun` with ``metadata``.
        Otherwise, the metadata for the runs are merged. In the case where there are duplicate
        metadata keys, the values from the first run are used.

    Returns
    -------
    :obj:`ScmRun`
        If not :obj:`inplace`, the return value is the object containing the merged
        data. The resultant class will be determined by the type of the first object.

    Raises
    ------
    TypeError
        If :obj:`inplace` is ``True`` but the first element in :obj:`dfs` is not an
        instance of :class:`ScmRun`

        ``runs`` argument is not a list

    ValueError
        :obj:`duplicate_msg` option is not recognised.

        No runs are provided to be appended
    """
    if not isinstance(runs, list):
        raise TypeError("runs is not a list")

    if not len(runs):
        raise ValueError("No runs to append")

    if inplace:
        if not isinstance(runs[0], ScmRun):
            raise TypeError("Can only append inplace to an ScmRun")
        ret = runs[0]
    else:
        ret = runs[0].copy()

    to_join_dfs = []
    to_join_metas = []
    overlapping_times = False

    ind = range(ret._df.shape[1])
    ret._df.columns = ind
    ret._meta.index = ind

    min_idx = ret._df.shape[1]
    for run in runs[1:]:
        run_to_join_df = run._df

        max_idx = min_idx + run_to_join_df.shape[1]
        ind = range(min_idx, max_idx)
        min_idx = max_idx

        run_to_join_df.columns = ind
        run_to_join_meta = run._meta.to_frame()
        run_to_join_meta.index = ind

        # check everything still makes sense
        npt.assert_array_equal(run_to_join_meta.index, run_to_join_df.columns)

        # check for overlap
        idx_to_check = run_to_join_df.index
        if not overlapping_times and (
            idx_to_check.isin(ret._df.index).any()
            or any([idx_to_check.isin(df.index).any() for df in to_join_dfs])
        ):
            overlapping_times = True

        to_join_dfs.append(run_to_join_df)
        to_join_metas.append(run_to_join_meta)

    ret._df = pd.concat([ret._df] + to_join_dfs, axis="columns").sort_index()
    ret._time_points = TimePoints(ret._df.index.values)
    ret._df.index = ret._time_points.to_index()
    ret._meta = pd.MultiIndex.from_frame(
        pd.concat([ret._meta.to_frame()] + to_join_metas).astype("category")
    )

    if ret._duplicated_meta():
        if overlapping_times and duplicate_msg:
            _handle_potential_duplicates_in_append(ret, duplicate_msg)

        ts = ret.timeseries(check_duplicated=False)
        orig_ts_index = ts.index
        nan_cols = pd.isna(orig_ts_index.to_frame()).any()
        orig_dtypes = orig_ts_index.to_frame().dtypes

        # Convert index to str
        ts.index = pd.MultiIndex.from_frame(
            ts.index.to_frame().astype(str).reset_index(drop=True)
        )

        deduped_ts = ts.groupby(ts.index, as_index=True).mean()

        ret._df = deduped_ts.reset_index(drop=True).T

        new_meta = pd.DataFrame.from_records(
            deduped_ts.index.values, columns=ts.index.names
        )

        # Convert back from str
        for c in nan_cols[nan_cols].index:
            new_meta[c].replace("nan", np.nan, inplace=True)
        for c, dtype in orig_dtypes.iteritems():
            new_meta[c] = new_meta[c].astype(dtype)

        ret._meta = pd.MultiIndex.from_frame(new_meta.astype("category"))

    if metadata is not None:
        ret.metadata = metadata
    else:
        ret.metadata = _merge_metadata([r.metadata for r in runs])

    if not inplace:
        return ret


def _handle_potential_duplicates_in_append(data, duplicate_msg):
    if duplicate_msg == "warn":
        warn_msg = (
            "Duplicate time points detected, the output will be the average of "
            "the duplicates.  Set `duplicate_msg=False` to silence this message."
        )
        warnings.warn(warn_msg)
        return None

    if duplicate_msg and not isinstance(duplicate_msg, str):
        raise NonUniqueMetadataError(data.meta)

    raise ValueError("Unrecognised value for duplicate_msg")


inject_binary_ops(BaseScmRun)
inject_nc_methods(BaseScmRun)
inject_plotting_methods(BaseScmRun)
inject_ops_methods(BaseScmRun)


class ScmRun(BaseScmRun):
    """
    Data container for holding one or many time-series of SCM data.
    """

    required_cols = ("model", "scenario", "region", "variable", "unit")
    """
    Minimum metadata columns required by an ScmRun.

    If an application requires a different set of required metadata, this
    can be specified by overriding :attr:`required_cols` on a custom class
    inheriting :class:`scmdata.run.BaseScmRun`. Note that at a minimum,
    ("variable", "unit") columns are required.
    """
