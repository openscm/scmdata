import pytest
from scmdata.time import _format_datetime, decode_datetimes_to_index, _CFTIME_CALENDARS
import datetime as dt
import numpy as np

from xarray import CFTimeIndex
from pandas import DatetimeIndex
from pandas.errors import OutOfBoundsDatetime
import pandas.testing as pdt
import pandas as pd
import cftime

input_type = pytest.mark.parametrize(
    "input_type",
    [
        "int-year",
        "decimal-year",
        "str-year",
        "str-year-month-day",
        "numpy-ns",
        "numpy-s",
        "numpy-y",
        "datetime",
        "cftime",
    ],
)


def convert_input(dates_as_int, input_type):
    if input_type == "int-year":
        return [int(d) for d in dates_as_int]
    elif input_type == "decimal-year":
        if min(dates_as_int) < 0:
            pytest.skip("datetime out of range")
        return [float(d) for d in dates_as_int]
    elif input_type == "str-year":
        return [str(d) for d in dates_as_int]
    elif input_type == "str-year-month-day":
        return [str(d) + "-01-01" for d in dates_as_int]
    elif input_type == "numpy-ns":
        if min(dates_as_int) <= 1678 or max(dates_as_int) >= 2262:
            pytest.skip("datetime out of range")
        return np.asarray([str(d) + "-01-01" for d in dates_as_int]).astype(
            "datetime64[ns]"
        )
    elif input_type == "numpy-s":
        return np.asarray([str(d) + "-01-01" for d in dates_as_int]).astype(
            "datetime64[s]"
        )
    elif input_type == "numpy-y":
        return np.asarray([str(d) + "-01-01" for d in dates_as_int]).astype(
            "datetime64[Y]"
        )
    elif input_type == "pandas":
        try:
            return [pd.Timestamp(dt.datetime(d, 1, 1)) for d in dates_as_int]
        except OutOfBoundsDatetime:
            pytest.skip("datetime out of range")
        return [np.datestr(d) + "-01-01" for d in dates_as_int]
    elif input_type == "datetime":
        try:
            return [dt.datetime(d, 1, 1) for d in dates_as_int]
        except ValueError:
            pytest.skip("datetime out of range")
    elif input_type == "cftime":
        return [cftime.datetime(d, 1, 1) for d in dates_as_int]


@input_type
def test_format(input_type):
    dates = [2000, 2010, 2020]

    inp_dates = convert_input(dates, input_type)
    res = _format_datetime(inp_dates)
    exp = np.asarray(["2000-01-01", "2010-01-01", "2020-01-01"]).astype("datetime64[s]")

    np.testing.assert_array_equal(res, exp)


@input_type
def test_format_wide_range(input_type):
    dates = [-100, 0, 1000, 5000]

    inp_dates = convert_input(dates, input_type)
    res = _format_datetime(inp_dates)
    exp = np.asarray(["-100-01-01", "0-01-01", "1000-01-01", "5000-01-01"]).astype(
        "datetime64[s]"
    )

    np.testing.assert_array_equal(res, exp)


@pytest.mark.parametrize("use_cftime", [True, None])
@input_type
def test_to_cftime_index(input_type, use_cftime):
    years = [-1000, 1000, 2000, 3000]
    inp_dates = convert_input(years, input_type)

    res = decode_datetimes_to_index(inp_dates, use_cftime=use_cftime)

    cftime_dts = [cftime.datetime(y, 1, 1) for y in years]
    exp = CFTimeIndex(cftime_dts, name="time")

    assert isinstance(res, CFTimeIndex)
    assert all(res.year == years)
    pdt.assert_index_equal(res, exp)


@pytest.mark.parametrize("calendar", _CFTIME_CALENDARS.keys())
@pytest.mark.parametrize("use_cftime", [True, None])
@input_type
def test_decode_index_with_calendar(input_type, calendar, use_cftime):
    years = [-1000, 1000, 2000, 3000]
    inp_dates = convert_input(years, input_type)

    res = decode_datetimes_to_index(inp_dates, calendar=calendar, use_cftime=use_cftime)

    cls = _CFTIME_CALENDARS[calendar]
    cftime_dts = [cls(y, 1, 1) for y in years]
    exp = CFTimeIndex(cftime_dts, name="time")

    assert isinstance(res, CFTimeIndex)
    assert all(res.year == years)
    pdt.assert_index_equal(res, exp)


def test_decode_index_with_invalid_calendar():
    years = [-1000, 1000, 2000, 3000]

    with pytest.raises(ValueError, match="Unknown calendar: not-a-cal"):
        decode_datetimes_to_index(years, calendar="not-a-cal")


def test_decode_index_with_nonstandard_calendar():
    years = [-1000, 1000, 2000, 3000]

    res = decode_datetimes_to_index(years, calendar="360_day")
    assert isinstance(res, CFTimeIndex)

    with pytest.raises(
        ValueError, match="Cannot use pandas indexes with a non-standard calendar"
    ):
        decode_datetimes_to_index(years, calendar="360_day", use_cftime=False)


@pytest.mark.parametrize("use_cftime", [False, None])
@input_type
def test_to_pd_index(input_type, use_cftime):
    years = [2000, 2050, 2100]
    inp_dates = convert_input(years, input_type)

    res = decode_datetimes_to_index(inp_dates, use_cftime=use_cftime)

    exp = DatetimeIndex([str(y) for y in years], name="time")

    # Pandas datetimes are coerced to ns
    assert res.values.dtype == "datetime64[ns]"
    assert all(res.year == years)
    pdt.assert_index_equal(res, exp)


@input_type
def test_to_pd_index_with_overflow(input_type):
    years = [1500, 2050, 2100]
    inp_dates = convert_input(years, input_type)

    with pytest.raises(OutOfBoundsDatetime):
        decode_datetimes_to_index(inp_dates, use_cftime=False)
