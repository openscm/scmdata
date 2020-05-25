"""
Time period handling and interpolation

A large portion of this module was originally from openscm. Thanks to the original author, Sven Willner
"""

from datetime import datetime

import cftime
import numpy as np
import pandas as pd
from dateutil import parser

_TARGET_TYPE = np.int64
_TARGET_DTYPE = "datetime64[s]"


class InsufficientDataError(Exception):
    """
    Insufficient data is available to interpolate/extrapolate
    """

    pass


def _float_year_to_datetime(inp: float) -> np.datetime64:
    year = int(inp)
    fractional_part = inp - year
    return np.datetime64(  # pylint: disable=too-many-function-args
        year - 1970, "Y"
    ) + np.timedelta64(  # pylint: disable=too-many-function-args
        int(
            (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).total_seconds()
            * fractional_part
        ),
        "s",
    )


_ufunc_float_year_to_datetime = np.frompyfunc(_float_year_to_datetime, 1, 1)
_ufunc_str_to_datetime = np.frompyfunc(parser.parse, 1, 1)


def _parse_datetime(inp: np.ndarray) -> np.ndarray:
    try:
        return _ufunc_float_year_to_datetime(inp.astype(float))
    except (TypeError, ValueError):
        return _ufunc_str_to_datetime(inp)


def _format_datetime(dts: np.ndarray) -> np.ndarray:
    """
    Convert an array to an array of :class:`np.datetime64`.

    Parameters
    ----------
    dts
        Input to attempt to convert

    Returns
    -------
    :class:`np.ndarray` of :class:`np.datetime64`
        Converted array

    Raises
    ------
    ValueError
        If one of the values in :obj:`dts` cannot be converted to :class:`np.datetime64`
    """
    if len(dts) <= 0:  # pylint: disable=len-as-condition
        return np.array([], dtype=_TARGET_DTYPE)

    dtype = dts.dtype.type
    if dts.dtype.kind == "O":
        dtype = np.dtype(type(dts[0])).type
    if issubclass(dtype, np.datetime64):
        return np.asarray(dts, dtype=_TARGET_DTYPE)
    if issubclass(dtype, np.floating):
        return _ufunc_float_year_to_datetime(dts).astype(_TARGET_DTYPE)
    if issubclass(dtype, np.integer):
        return (np.asarray(dts) - 1970).astype("datetime64[Y]").astype(_TARGET_DTYPE)
    if issubclass(dtype, str):
        return _parse_datetime(dts).astype(_TARGET_DTYPE)
    return np.asarray(dts, dtype=_TARGET_DTYPE)


class TimePoints:
    """
    Handles time points by wrapping :class:`np.ndarray` of :class:`np.datetime64`..
    """

    def __init__(self, values):
        """
        Initialize.

        Parameters
        ----------
        values
            Time points array to handle
        """
        self._values = _format_datetime(np.asarray(values))

    def __len__(self) -> int:
        """
        Get the number of time points.
        """
        return len(self._values)

    @property
    def values(self) -> np.ndarray:
        """
        Time points
        """
        return self._values

    def to_index(self) -> pd.Index:
        """
        Get time points as :class:`pd.Index`.

        Returns
        -------
        :class:`pd.Index`
            :class:`pd.Index` of :class:`np.dtype` :class:`object` with name ``"time"``
            made from the time points represented as :class:`datetime.datetime`.
        """
        return pd.Index(self._values.astype(object), dtype=object, name="time")

    def as_cftime(self) -> list:
        """
        Get as cftime datetimes

        Returns
        -------
        list of cftime.datetime
        """
        return [
            cftime.datetime(*dt.timetuple()[:6]) for dt in self._values.astype(object)
        ]

    def years(self) -> np.ndarray:
        """
        Get year of each time point.

        Returns
        -------
        :obj:`np.array` of :obj:`int`
            Year of each time point
        """
        return np.vectorize(getattr)(self._values.astype(object), "year")

    def months(self) -> np.ndarray:
        """
        Get month of each time point.

        Returns
        -------
        :obj:`np.array` of :obj:`int`
            Month of each time point
        """
        return np.vectorize(getattr)(self._values.astype(object), "month")

    def days(self) -> np.ndarray:
        """
        Get day of each time point.

        Returns
        -------
        :obj:`np.array` of :obj:`int`
            Day of each time point
        """
        return np.vectorize(getattr)(self._values.astype(object), "day")

    def hours(self) -> np.ndarray:
        """
        Get hour of each time point.

        Returns
        -------
        :obj:`np.array` of :obj:`int`
            Hour of each time point
        """
        return np.vectorize(getattr)(self._values.astype(object), "hour")

    def weekdays(self) -> np.ndarray:
        """
        Get weekday of each time point.

        Returns
        -------
        :obj:`np.array` of :obj:`int`
            Day of the week of each time point
        """
        return np.vectorize(datetime.weekday)(self._values.astype(object))


class TimeseriesConverter:
    """
    Interpolator used to convert data between different time bases

    This is a modified version originally in :mod:`openscm.time.TimeseriesConverter`.
    The integral preserving interpolation was removed as it is outside the scope of
    this package.

    Parameters
    ----------
    source_time_points: np.ndarray
        Source timeseries time points
    target_time_points: np.ndarray
        Target timeseries time points
    interpolation_type : {"linear"}
        Interpolation type. Options are 'linear'
    extrapolation_type : {"linear", "constant", None}
        Extrapolation type. Options are None, 'linear' or 'constant'

    Raises
    ------
    InsufficientDataError
        Timeseries too short to extrapolate
    """

    def __init__(
        self,
        source_time_points: np.ndarray,
        target_time_points: np.ndarray,
        interpolation_type="linear",
        extrapolation_type="linear",
    ):

        self.source = (
            np.array(source_time_points)
            .astype(_TARGET_DTYPE)
            .astype(_TARGET_TYPE, copy=True)
        )
        self.target = (
            np.array(target_time_points)
            .astype(_TARGET_DTYPE)
            .astype(_TARGET_TYPE, copy=True)
        )
        self.interpolation_type = interpolation_type
        self.extrapolation_type = extrapolation_type

        if not self.points_are_compatible(self.source, self.target):
            error_msg = (
                "Target time points are outside the source time points, use an "
                "extrapolation type other than None"
            )
            raise InsufficientDataError(error_msg)

    def points_are_compatible(self, source: np.ndarray, target: np.ndarray) -> bool:
        """
        Are the two sets of time points compatible i.e. can I convert between the two?

        Parameters
        ----------
        source
            Source timeseries time points
        target
            Target timeseries time points

        Returns
        -------
        bool
            Can I convert between the time points?
        """
        if self.extrapolation_type is None and (
            source[0] > target[0] or source[-1] < target[-1]
        ):
            return False

        return True

    def _get_scipy_extrapolation_args(self, values: np.ndarray):
        if self.extrapolation_type == "linear":
            return {"fill_value": "extrapolate"}
        if self.extrapolation_type == "constant":
            return {"fill_value": (values[0], values[-1]), "bounds_error": False}
        # TODO: add cubic support
        return {}

    def _get_scipy_interpolation_arg(self) -> str:
        if self.interpolation_type == "linear":
            return "linear"
        # TODO: add cubic support
        raise NotImplementedError

    def _convert(
        self,
        values: np.ndarray,
        source_time_points: np.ndarray,
        target_time_points: np.ndarray,
    ) -> np.ndarray:
        """
        Wrap :func:`_convert_unsafe` to provide proper error handling.

        Any nan values are removed from :obj:`source` before interpolation

        Parameters
        ----------
        values
            Array of data to convert
        source_time_points
            Source timeseries time points
        target_time_points
            Target timeseries time points

        Raises
        ------
        InsufficientDataError
            Length of the time series is too short to convert
        InsufficientDataError
            Target time points are outside the source time points and
            :attr:`extrapolation_type` is 'NONE'
        ImportError
            Optional dependency scipy has not been installed

        Returns
        -------
        np.ndarray
            Converted time period average data for timeseries :obj:`values`
        """
        values = np.asarray(values)
        # Check for nans
        nan_mask = np.isnan(values)
        if nan_mask.sum():
            values = values[~nan_mask]
            source_time_points = source_time_points[~nan_mask]

        if len(values) < 3:
            raise InsufficientDataError

        try:
            return self._convert_unsafe(values, source_time_points, target_time_points)
        except Exception:  # pragma: no cover # emergency valve
            print("numpy interpolation failed...")
            raise

    def _convert_unsafe(
        self,
        values: np.ndarray,
        source_time_points: np.ndarray,
        target_time_points: np.ndarray,
    ) -> np.ndarray:
        # Lazy-load scipy.interpolate
        from scipy import interpolate

        res_point = interpolate.interp1d(
            source_time_points.astype(_TARGET_TYPE),
            values,
            kind=self._get_scipy_interpolation_arg(),
            **self._get_scipy_extrapolation_args(values),
        )

        return res_point(target_time_points.astype(_TARGET_TYPE))

    def convert_from(self, values: np.ndarray) -> np.ndarray:
        """
        Convert value **from** source timeseries time points to target timeseries time
        points.

        Parameters
        ----------
        values: np.ndarray
            Value

        Returns
        -------
        np.ndarray
            Converted data for timeseries :obj:`values` into the target timebase
        """
        return self._convert(values, self.source, self.target)

    def convert_to(self, values: np.ndarray) -> np.ndarray:
        """
        Convert value from target timeseries time points **to** source timeseries time
        points.

        Parameters
        ----------
        values: np.ndarray
            Value

        Returns
        -------
        np.ndarray
            Converted data for timeseries :obj:`values` into the source timebase
        """
        return self._convert(values, self.target, self.source)
