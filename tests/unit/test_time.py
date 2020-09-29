import pytest
from scmdata.time import _format_datetime, decode_datetimes_to_index
import datetime as dt
import numpy as np

from xarray import CFTimeIndex
from pandas import DatetimeIndex
import cftime

input_type = pytest.mark.parametrize(
    "input_type",
    [
        "int-year",
        "decimal-year",
        "str-year",
        "str-year-month-day",
        "numpy",
        "datetime",
        "cftime",
    ],
)


def convert_input(dates_as_int, input_type):
    if input_type == "int-year":
        return [int(d) for d in dates_as_int]
    elif input_type == "decimal-year":
        return [float(d) for d in dates_as_int]
    elif input_type == "str-year":
        return [str(d) for d in dates_as_int]
    elif input_type == "str-year-month-day":
        return [str(d) + "-01-01" for d in dates_as_int]
    elif input_type == "numpy":
        return [str(d) + "-01-01" for d in dates_as_int]
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


def test_format_weird():
    inp = np.asarray(["-1000"])
    res = _format_datetime(inp)
    exp = np.asarray(["-1000-01-01"]).astype("datetime64[s]")

    np.testing.assert_array_equal(res, exp)


def test_to_cftime_index():
    inp = np.asarray(["-1000-01-01", "1000-01-01", "2000-01-01", "2000-01-01"]).astype(
        "datetime64[s]"
    )

    res = decode_datetimes_to_index(
        ["-1000-01-01", "1000-01-01", "2000-01-01", "2000-01-01"]
    )

    exp = CFTimeIndex(inp)

    assert all(idx.year == [-1000, 1000, 2000, 3000])


def test_to_pd_index():
    inp = np.asarray(["-1000-01-01", "1000-01-01", "2000-01-01", "2000-01-01"]).astype(
        "datetime64[s]"
    )

    idx = CFTimeIndex(inp)

    assert all(idx.year == [-1000, 1000, 2000, 3000])
