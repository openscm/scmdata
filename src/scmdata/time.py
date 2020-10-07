"""
Time period handling and interpolation

A large portion of this module was originally from openscm. Thanks to the original author, Sven Willner
"""

from datetime import datetime

import cftime
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
from xarray import CFTimeIndex
from dateutil import parser

_TARGET_TYPE = np.int64
_TARGET_DTYPE = "datetime64[s]"
STANDARD_CALENDARS = {"standard", "gregorian", "proleptic_gregorian"}
"""
Over the span of a ``datetime64[ns]`` these calendars are all equivalent
"""

_CFTIME_CALENDARS = {
    "standard": cftime.datetime,
    "360_day": cftime.Datetime360Day,
    "gregorian": cftime.DatetimeGregorian,
    "proleptic_gregorian": cftime.DatetimeProlepticGregorian,
    "noleap": cftime.DatetimeNoLeap,
    "julian": cftime.DatetimeJulian,
}


class InsufficientDataError(Exception):
    """
    Insufficient data is available to interpolate/extrapolate
    """

    pass


def _float_year_to_datetime(inp: float) -> np.datetime64:
    year = int(inp)

    if year < 0:
        raise OutOfBoundsDatetime("Cannot connect negative decimal year")

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


def _str_to_cftime(inp: str, calendar: str):
    cls = _CFTIME_CALENDARS[calendar]

    negative_year = False
    if inp.startswith("-"):
        negative_year = True
        inp = inp[1:]

    assert len(inp) == 19

    y = int(inp[:4])
    if negative_year:
        y = -y
    mon = int(inp[5:7])
    d = int(inp[8:10])
    h = int(inp[11:13])
    m = int(inp[14:16])
    s = int(inp[17:])

    return cls(y, mon, d, h, m, s)


_ufunc_float_year_to_datetime = np.frompyfunc(_float_year_to_datetime, 1, 1)
_ufunc_str_to_datetime = np.frompyfunc(np.datetime64, 1, 1)
_ufunc_str_to_datetime_parser = np.frompyfunc(parser.parse, 1, 1)
_ufunc_str_to_cftime = np.frompyfunc(_str_to_cftime, 2, 1)


def _parse_datetime(inp: np.ndarray) -> np.ndarray:
    try:
        return _ufunc_float_year_to_datetime(inp.astype(float))
    except (TypeError, ValueError):
        try:
            return _ufunc_str_to_datetime(inp)
        except ValueError:
            return _ufunc_str_to_datetime_parser(inp)


def _format_datetime(dts) -> np.ndarray:
    """
    Convert a list of times to numpy datetimes

    This truncates the datetimes to have second resolution

    Parameters
    ----------
    dts : np.array or list
        Input to attempt to convert

    Returns
    -------
    :class:`np.ndarray` with dtype :class:`np.datetime64[s]`
        Converted array

    Raises
    ------
    ValueError
        If one of the values in :obj:`dts` cannot be converted to :class:`np.datetime64`
    """

    dts = np.asarray(dts)

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


def _to_cftimes(np_dates, calendar):
    # This would be faster, but results in calendar issues
    # return cftime.num2date(
    #     np_dates.astype(int), "seconds since 1970-01-01", calendar=calendar
    # )

    if calendar not in _CFTIME_CALENDARS:
        raise ValueError("Unknown calendar: {}".format(calendar))

    return _ufunc_str_to_cftime(np.datetime_as_string(np_dates), calendar)


def decode_datetimes_to_index(dates, calendar=None, use_cftime=None):
    """
    Decodes a list of dates to an index

    Uses a :class:`pandas.DatetimeIndex` where possible. When a non-standard calendar is
    used or for dates before year 1678 or after year 2262, a dates are converted
    to :module:`cftime` datetimes and a :class:`xarray.CFTimeIndex()` is used.

    A wide formats for dates is supported. The following are all equivalent:

    * str ("2000" or "2000-01-01")
    * int (2000)
    * decimal years (2000.0)
    * python datetimes (``datetime.datetime(2000, 1, 1)``)
    * cftime datetimes (``cftime.datetime(2000, 1, 1)``)
    * numpy datetimes (``np.datetime64("2000-01-01", "Y")``)

    Parameters
    ----------
    dates
        Dates to be converted

    calendar: str
        Describes the calendar used by in the time calculations. All the values
        currently defined in the [CF metadata convention](http://cfconventions.org)
        and are implemented in [cftime](https://unidata.github.io/cftime)

        Valid calendars ``'standard', 'gregorian', 'proleptic_gregorian', 'noleap', '360_day’, 'julian'``.
        Default is ``'standard'``, which is a mixed Julian/Gregorian calendar.

        If a calendar other than ``'standard', 'gregorian'`` or ``'proleptic_gregorian'``
        is selected, then dates will be attempted to converted to ``cftime``'s

    use_cftime: bool
        If None (default), then try and determine the appropriate time index to use.
        Attempts to use a :class:`pandas.DatetimeIndex`, but falls back to
        :class:`xarray.CFTimeIndex` if the conversion fails.

        If True, dates are explicitly converted to `cftime`'s and a
        :class:`xarray.CFTimeIndex` is returned.

        If False, a :class:`pandas.DatetimeIndex` will always be returned (if
        possible). In this case a :class:`pandas.errors.OutOfBoundsDatetime`
        is raised if a date falls before year 1678 or after year 2262.

    Returns
    -------
    :class:`pandas.DatetimeIndex` or :class:`xarray.CFTimeIndex`

        The return type depends on the value of calendar and the dates provided

    Raises
    ------
    :class:`pandas.errors.OutOfBoundsDatetime`
        ``use_cftime == False`` and date before year 1678 or after year 2262 is
        provided

    ValueError
        ``use_cftime == False`` and a non-standard calendar is requested
    """
    dates = np.asarray(dates)
    dates = _format_datetime(dates)

    if calendar is None:
        calendar = "standard"

    if use_cftime is None:
        try:
            if calendar not in STANDARD_CALENDARS:
                raise ValueError(
                    "Cannot use pandas indexes with a non-standard calendar"
                )
            index = pd.DatetimeIndex(dates)
        except (OutOfBoundsDatetime, ValueError):
            index = CFTimeIndex(_to_cftimes(dates, calendar))
    elif use_cftime:
        index = CFTimeIndex(_to_cftimes(dates, calendar))
    else:
        if calendar not in STANDARD_CALENDARS:
            raise ValueError("Cannot use pandas indexes with a non-standard calendar")
        index = pd.DatetimeIndex(dates)

    index.name = "time"
    return index


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
        return CFTimeIndex(self.as_cftime(), name="time")

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
