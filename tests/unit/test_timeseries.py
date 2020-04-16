from datetime import datetime

import numpy as np
import numpy.testing as npt
import pytest

from scmdata.timeseries import TimeSeries


@pytest.fixture(scope="function")
def ts():
    times = np.asarray(
        [datetime(2000, 1, 1), datetime(2001, 1, 1), datetime(2002, 1, 1),]
    )
    return TimeSeries([1, 2, 3], coords=[("time", times)])


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
