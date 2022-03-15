import re
from datetime import datetime

import cftime
import numpy as np
import numpy.testing as npt
import pint.errors
import pytest
import xarray as xr
from openscm_units import unit_registry as ur

from scmdata.time import TimePoints
from scmdata.timeseries import TimeSeries


@pytest.fixture(scope="function")
def ts():
    times = np.asarray(
        [datetime(2000, 1, 1), datetime(2001, 1, 1), datetime(2002, 1, 1)]
    )
    return TimeSeries([1, 2, 3], time=times)


@pytest.mark.parametrize("data", ([1, 2, 3], (1, 2, 3), np.array([1, 2, 3])))
@pytest.mark.parametrize(
    "time",
    (
        [10, 2010, 5010],
        (10, 2010, 5010),
        np.array([10, 2010, 5010]),
        [datetime(y, 1, 1) for y in [10, 2010, 5010]],
        np.array([10, 2010, 5010], dtype="datetime64[s]"),
    ),
)
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

    exp_axis = np.array(
        ["{:04d}-01-01".format(y) for y in time_points], dtype="datetime64[s]"
    )
    npt.assert_array_equal(res.time_points.values, exp_axis)


@pytest.mark.parametrize(
    "data",
    (
        [[1, 2], [2, 4]],
        np.array([[1, 2], [2, 4]]),
        xr.DataArray(
            [[1, 2], [2, 4]], coords=[("time", [2000, 2001]), ("lat", [-45, 45])]
        ),
    ),
)
def test_timeseries_init_2d_data(data):
    with pytest.raises(ValueError, match="data must be 1d"):
        TimeSeries(data)


def test_timeseries_init_xarray_time_is_not_none():
    raw_data = [-1, -2, -3]
    time_points = [20, 2020, 5050]
    data = xr.DataArray(raw_data, coords=[("time", time_points)])
    error_msg = "If data is an :class:`xarray.DataArray` instance, time must be `None`"
    with pytest.raises(TypeError, match=re.escape(error_msg)):
        TimeSeries(data, time=time_points)


@pytest.mark.parametrize("coord_name", ("times", "lat", "Time", "t"))
def test_timeseries_init_xarray_no_time_coord(coord_name):
    raw_data = [-1, -2, -3]
    time_points = [20, 2020, 5050]
    data = xr.DataArray(raw_data, coords=[(coord_name, time_points)])
    error_msg = (
        "If data is an :class:`xarray.DataArray` instance, its only dimension must "
        "be named `'time'`"
    )
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        TimeSeries(data)


@pytest.mark.parametrize("data", ([1, 2, 3], (1, 2, 3), np.array([1, 2, 3])))
def test_timeseries_init_no_time_list(data):
    error_msg = (
        "If data is not an :class:`xarray.DataArray` instance, `time` must not be "
        "`None`"
    )
    with pytest.raises(TypeError, match=re.escape(error_msg)):
        TimeSeries(data)


@pytest.mark.parametrize("data", ([1, 2, 3], (1, 2, 3), np.array([1, 2, 3])))
def test_timeseries_init_time_and_coords(data):
    error_msg = (
        "If ``data`` is not an :class:`xarray.DataArray`, `coords` must not be "
        "supplied via `kwargs` because it will be automatically filled with "
        "the value of `time`."
    )
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        TimeSeries(data, time=[2010, 2020, 2030], coords={"lat": [45, 0, -45]})


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


@pytest.fixture(scope="function")
def ts_gtc_per_yr_units():
    times = np.asarray(
        [datetime(2000, 1, 1), datetime(2001, 1, 1), datetime(2002, 1, 1)]
    )
    return TimeSeries([1, 2, 3], time=times, attrs={"unit": "GtC / yr"})


@pytest.mark.parametrize("inplace", [True, False])
def test_timeseries_add_pint_scalar(ts_gtc_per_yr_units, inplace):
    to_add = 2 * ur("MtC / yr")
    if inplace:
        ts_gtc_per_yr_units += to_add
        ts2 = ts_gtc_per_yr_units
    else:
        ts2 = ts_gtc_per_yr_units + to_add

    npt.assert_allclose(ts2.values, [1.002, 2.002, 3.002])
    assert ts2.meta["unit"] == "gigatC / a"


@pytest.mark.parametrize("inplace", [True, False])
def test_timeseries_sub_pint_scalar(ts_gtc_per_yr_units, inplace):
    to_sub = 2 * ur("MtC / yr")
    if inplace:
        ts_gtc_per_yr_units -= to_sub
        ts2 = ts_gtc_per_yr_units
    else:
        ts2 = ts_gtc_per_yr_units - to_sub

    npt.assert_allclose(ts2.values, [0.998, 1.998, 2.998])
    assert ts2.meta["unit"] == "gigatC / a"


@pytest.mark.parametrize("inplace", [True, False])
def test_timeseries_mul_pint_scalar(ts_gtc_per_yr_units, inplace):
    to_mul = 2 * ur("yr")
    if inplace:
        ts_gtc_per_yr_units *= to_mul
        ts2 = ts_gtc_per_yr_units
    else:
        ts2 = ts_gtc_per_yr_units * to_mul

    npt.assert_allclose(ts2.values, [2, 4, 6])
    assert ts2.meta["unit"] == "gigatC"


@pytest.mark.parametrize("inplace", [True, False])
def test_timeseries_div_pint_scalar(ts_gtc_per_yr_units, inplace):
    to_div = 2 * ur("yr**-1")
    if inplace:
        ts_gtc_per_yr_units /= to_div
        ts2 = ts_gtc_per_yr_units
    else:
        ts2 = ts_gtc_per_yr_units / to_div

    npt.assert_allclose(ts2.values, [1 / 2, 2 / 2, 3 / 2])
    assert ts2.meta["unit"] == "gigatC"


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("op", ["add", "sub"])
def test_timeseries_add_sub_pint_scalar_no_ts_units(ts, inplace, op):
    scalar = 2 * ur("GtC / yr")

    error_msg = re.escape("Cannot convert from 'dimensionless' to 'gigatC / a'")
    with pytest.raises(pint.errors.DimensionalityError, match=error_msg):
        if inplace:
            if op == "add":
                ts += scalar
            elif op == "sub":
                ts -= scalar
            else:
                raise NotImplementedError(op)
        else:
            if op == "add":
                ts + scalar
            elif op == "sub":
                ts - scalar
            else:
                raise NotImplementedError(op)


@pytest.mark.parametrize("inplace", [True, False])
def test_timeseries_mul_pint_scalar_no_units(ts, inplace):
    scalar = 2 * ur("GtC / yr")
    if inplace:
        ts *= scalar
        ts2 = ts

    else:
        ts2 = ts * scalar

    # operation works because units of base assumed to be dimensionless
    npt.assert_allclose(ts2.values, [2, 4, 6])
    assert ts2.meta["unit"] == "gigatC / a"


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("op", ["div"])
def test_timeseries_div_pint_scalar_no_units(ts, inplace, op):
    scalar = 2 * ur("GtC / yr")
    if inplace:
        if op == "div":
            ts /= scalar
            ts2 = ts
        else:
            raise NotImplementedError(op)
    else:
        if op == "div":
            ts2 = ts / scalar
        else:
            raise NotImplementedError(op)

    # operation works because units of base assumed to be dimensionless
    npt.assert_allclose(ts2.values, [1 / 2, 1, 3 / 2])
    assert ts2.meta["unit"] == "a / gigatC"


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("op", ["add", "sub"])
def test_timeseries_add_sub_pint_scalar_invalid_units(ts_gtc_per_yr_units, inplace, op):
    other_unit = "{} / yr".format(ts_gtc_per_yr_units.meta["unit"])
    scalar = 2 * ur(other_unit)

    error_msg = re.escape(
        "Cannot convert from 'gigatC / a' ([carbon] * [mass] / [time]) to 'gigatC / "
        "a ** 2' ([carbon] * [mass] / [time] ** 2)"
    )
    with pytest.raises(pint.errors.DimensionalityError, match=error_msg):
        if inplace:
            if op == "add":
                ts_gtc_per_yr_units += scalar
            elif op == "sub":
                ts_gtc_per_yr_units -= scalar
            else:
                raise NotImplementedError(op)
        else:
            if op == "add":
                ts_gtc_per_yr_units + scalar
            elif op == "sub":
                ts_gtc_per_yr_units - scalar
            else:
                raise NotImplementedError(op)


@pytest.mark.parametrize("inplace", [True, False])
def test_timeseries_add_pint_vector(ts_gtc_per_yr_units, inplace):
    to_add = np.arange(3) * ur("MtC / yr")
    if inplace:
        ts_gtc_per_yr_units += to_add
        ts2 = ts_gtc_per_yr_units
    else:
        ts2 = ts_gtc_per_yr_units + to_add

    npt.assert_allclose(ts2.values, [1, 2.001, 3.002])
    assert ts2.meta["unit"] == "gigatC / a"


@pytest.mark.parametrize("inplace", [True, False])
def test_timeseries_sub_pint_vector(ts_gtc_per_yr_units, inplace):
    to_sub = np.arange(3) * ur("MtC / yr")
    if inplace:
        ts_gtc_per_yr_units -= to_sub
        ts2 = ts_gtc_per_yr_units
    else:
        ts2 = ts_gtc_per_yr_units - to_sub

    npt.assert_allclose(ts2.values, [1, 1.999, 2.998])
    assert ts2.meta["unit"] == "gigatC / a"


@pytest.mark.parametrize("inplace", [True, False])
def test_timeseries_mul_pint_vector(ts_gtc_per_yr_units, inplace):
    to_mul = np.arange(3) * ur("yr")
    if inplace:
        ts_gtc_per_yr_units *= to_mul
        ts2 = ts_gtc_per_yr_units
    else:
        ts2 = ts_gtc_per_yr_units * to_mul

    npt.assert_allclose(ts2.values, [0, 2, 6])
    assert ts2.meta["unit"] == "gigatC"


@pytest.mark.parametrize("inplace", [True, False])
def test_timeseries_div_pint_vector(ts_gtc_per_yr_units, inplace):
    to_div = np.arange(3) * ur("yr**-1")
    if inplace:
        ts_gtc_per_yr_units /= to_div
        ts2 = ts_gtc_per_yr_units
    else:
        ts2 = ts_gtc_per_yr_units / to_div

    npt.assert_allclose(ts2.values, [np.inf, 2 / 1, 3 / 2])
    assert ts2.meta["unit"] == "gigatC"


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("op", ["add", "sub"])
def test_timeseries_add_sub_pint_vector_no_ts_units(ts, inplace, op):
    vector = np.arange(3) * ur("GtC / yr")

    error_msg = re.escape("Cannot convert from 'dimensionless' to 'gigatC / a'")
    with pytest.raises(pint.errors.DimensionalityError, match=error_msg):
        if inplace:
            if op == "add":
                ts += vector
            elif op == "sub":
                ts -= vector
            else:
                raise NotImplementedError(op)
        else:
            if op == "add":
                ts + vector
            elif op == "sub":
                ts - vector
            else:
                raise NotImplementedError(op)


@pytest.mark.parametrize("inplace", [True, False])
def test_timeseries_mul_pint_vector_no_units(ts, inplace):
    vector = np.arange(3) * ur("GtC / yr")
    if inplace:
        ts *= vector
        ts2 = ts

    else:
        ts2 = ts * vector

    # operation works because units of base assumed to be dimensionless
    npt.assert_allclose(ts2.values, [0, 2, 6])
    assert ts2.meta["unit"] == "gigatC / a"


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("op", ["div"])
def test_timeseries_div_pint_vector_no_units(ts, inplace, op):
    vector = np.arange(3) * ur("GtC / yr")
    if inplace:
        if op == "div":
            ts /= vector
            ts2 = ts
        else:
            raise NotImplementedError(op)
    else:
        if op == "div":
            ts2 = ts / vector
        else:
            raise NotImplementedError(op)

    # operation works because units of base assumed to be dimensionless
    npt.assert_allclose(ts2.values, [np.inf, 2, 3 / 2])
    assert ts2.meta["unit"] == "a / gigatC"


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("op", ["add", "sub"])
def test_timeseries_add_sub_pint_vector_invalid_units(ts_gtc_per_yr_units, inplace, op):
    other_unit = "{} / yr".format(ts_gtc_per_yr_units.meta["unit"])
    vector = np.arange(3) * ur(other_unit)

    error_msg = re.escape(
        "Cannot convert from 'gigatC / a' ([carbon] * [mass] / [time]) to 'gigatC / "
        "a ** 2' ([carbon] * [mass] / [time] ** 2)"
    )
    with pytest.raises(pint.errors.DimensionalityError, match=error_msg):
        if inplace:
            if op == "add":
                ts_gtc_per_yr_units += vector
            elif op == "sub":
                ts_gtc_per_yr_units -= vector
            else:
                raise NotImplementedError(op)
        else:
            if op == "add":
                ts_gtc_per_yr_units + vector
            elif op == "sub":
                ts_gtc_per_yr_units - vector
            else:
                raise NotImplementedError(op)


def test_interpolate(combo):
    ts = TimeSeries(combo.source_values, time=combo.source)

    res = ts.interpolate(
        combo.target,
        interpolation_type=combo.interpolation_type,
        extrapolation_type=combo.extrapolation_type,
    )

    npt.assert_array_almost_equal(res.values.squeeze(), combo.target_values)


@pytest.mark.parametrize(
    "dt", [datetime, cftime.datetime, cftime.DatetimeNoLeap, cftime.Datetime360Day]
)
def test_extrapolation_long(dt):
    source = np.arange(800, 1000)
    source_times = [dt(y, 1, 1) for y in source]

    ts = TimeSeries(source, time=source_times)

    target = np.arange(800, 1100)
    res = ts.interpolate(
        [dt(y, 1, 1) for y in target],
        extrapolation_type="linear",
    )

    # Interpolating annually using seconds is not identical to just assuming years
    npt.assert_array_almost_equal(res.values.squeeze(), target, decimal=0)


@pytest.mark.parametrize(
    "dt", [datetime, cftime.datetime, cftime.DatetimeNoLeap, cftime.Datetime360Day]
)
def test_extrapolation_nan(dt):
    source = np.arange(2000, 2005, dtype=float)
    source_times = [dt(int(y), 1, 1) for y in source]
    source[-2:] = np.nan

    ts = TimeSeries(source, time=source_times)

    target = np.arange(2000, 2010)
    res = ts.interpolate(
        [dt(int(y), 1, 1) for y in target],
        extrapolation_type="linear",
    )

    npt.assert_array_almost_equal(res.values.squeeze(), target, decimal=2)


def test_copy(ts):
    orig = ts
    copy = ts.copy()

    assert id(orig) != id(copy)
    assert id(orig._data) != id(copy._data)
    assert id(orig._data.attrs) != id(copy._data.attrs)
