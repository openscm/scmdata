import re
from datetime import datetime

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from scmdata.time import TimePoints
from scmdata.timeseries import TimeSeries


@pytest.mark.parametrize("data", ([1, 2, 3], (1, 2, 3), np.array([1, 2, 3])))
@pytest.mark.parametrize("time", ([10, 2010, 5010], (10, 2010, 5010), np.array([10, 2010, 5010]), [datetime(y, 1, 1) for y in [10, 2010, 5010]], np.array([10, 2010, 5010], dtype="datetime64[s]")))
def test_timeseries_init_array_like(data, time):
    res = TimeSeries(data, time)

    npt.assert_array_equal(res.values, data)
    assert isinstance(res.time_points, TimePoints)

    exp_axis = TimePoints(time)
    npt.assert_array_equal(res.time_points.values, exp_axis.values)


def test_timeseries_init_xarray():
    raw_data = [-1, -2, -3]
    time_points = [20, 2020, 5050]
    data = xr.DataArray(raw_data, coords=[("time", time_points)])

    res = TimeSeries(data)

    npt.assert_array_equal(res.values, raw_data)
    assert isinstance(res.time_points, TimePoints)

    exp_axis = np.array(["{:04d}-01-01".format(y) for y in time_points], dtype="datetime64[s]")
    npt.assert_array_equal(res.time_points.values, exp_axis)


@pytest.mark.parametrize("data", (
    [[1, 2], [2, 4]],
    np.array([[1, 2], [2, 4]]),
    xr.DataArray([[1, 2], [2, 4]], coords=[("time", [2000, 2001]), ("lat", [-45, 45])]),
))
def test_timeseries_init_2d_data(data):
    with pytest.raises(ValueError, match="data must be 1d"):
        TimeSeries(data)


def test_timeseries_init_xarray_time_is_not_none():
    raw_data = [-1, -2, -3]
    time_points = [20, 2020, 5050]
    data = xr.DataArray(raw_data, coords=[("time", time_points)])
    error_msg = (
        "If data is an :obj:`xr.DataArray` instance, time must be `None`"
    )
    with pytest.raises(TypeError, match=re.escape(error_msg)):
        res = TimeSeries(data, time=time_points)


@pytest.mark.parametrize("coord_name", ("times", "lat", "Time", "t"))
def test_timeseries_init_xarray_no_time_coord(coord_name):
    raw_data = [-1, -2, -3]
    time_points = [20, 2020, 5050]
    data = xr.DataArray(raw_data, coords=[(coord_name, time_points)])
    error_msg = (
        "If data is an :obj:`xr.DataArray` instance, its only dimension must "
        "be named `'time'`"
    )
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        res = TimeSeries(data)


@pytest.mark.parametrize("data", ([1, 2, 3], (1, 2, 3), np.array([1, 2, 3])))
def test_timeseries_init_no_time_list(data):
    error_msg = (
        "If data is not an :obj:`xr.DataArray` instance, `time` must not be "
        "`None`"
    )
    with pytest.raises(TypeError, match=re.escape(error_msg)):
        TimeSeries(data)


@pytest.mark.parametrize("data", ([1, 2, 3], (1, 2, 3), np.array([1, 2, 3])))
def test_timeseries_init_time_and_coords(data):
    error_msg = (
        "If ``data`` is not an :obj:`xr.DataArray`, `coords` must not be "
        "supplied via `kwargs` because it will be automatically filled with "
        "the value of `time`."
    )
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        TimeSeries(data, time=[2010, 2020, 2030], coords={"lat": [45, 0, -45]})


@pytest.fixture(scope="function")
def ts():
    times = np.asarray(
        [datetime(2000, 1, 1), datetime(2001, 1, 1), datetime(2002, 1, 1),]
    )
    return TimeSeries([1, 2, 3], time=times)


@pytest.mark.parametrize("inplace", [True, False])
def test_timeseries_add(ts, inplace):
    if inplace:
        ts += 2
        ts2 = ts
    else:
        ts2 = ts + 2

    npt.assert_allclose(ts2.values, [3, 4, 5])


@pytest.mark.parametrize("inplace", [True, False])
def test_timeseries_sub(ts, inplace):
    if inplace:
        ts -= 2
        ts2 = ts
    else:
        ts2 = ts - 2

    npt.assert_allclose(ts2.values, [-1, 0, 1])


@pytest.mark.parametrize("inplace", [True, False])
def test_timeseries_mul(ts, inplace):
    if inplace:
        ts *= 2
        ts2 = ts
    else:
        ts2 = ts * 2

    npt.assert_allclose(ts2.values, [2, 4, 6])
