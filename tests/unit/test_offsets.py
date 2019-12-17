from datetime import datetime
from types import GeneratorType

import numpy as np
import pytest
from pandas.tseries.offsets import DateOffset, NaT

from scmdata.offsets import apply_dt, generate_range, to_offset


@pytest.mark.parametrize(
    "offset_rule",
    ["B", "C", "BM", "BMS", "CBM", "CBMS", "BQ", "BSS", "BA", "BAS", "RE", "BH", "CBH"],
)
def test_invalid_offsets(offset_rule):
    with pytest.raises(ValueError):
        to_offset(offset_rule)


def test_annual_start():
    offset = to_offset("AS")
    assert offset.__class__.__name__ == "LongYearBegin"

    dt = datetime(2001, 2, 12)

    res = offset.apply(dt)
    assert isinstance(res, datetime)
    assert res.year == 2002
    assert res.month == 1
    assert res.day == 1

    res = offset.rollback(dt)
    assert res.year == 2001
    assert res.month == 1
    assert res.day == 1

    res = offset.rollforward(dt)
    assert res.year == 2002
    assert res.month == 1
    assert res.day == 1


def test_month_start():
    offset = to_offset("MS")
    assert offset.__class__.__name__ == "LongMonthBegin"

    dt = datetime(2001, 2, 12)

    res = offset.apply(dt)
    assert isinstance(res, datetime)
    assert res.year == 2001
    assert res.month == 3
    assert res.day == 1

    res = offset.rollback(dt)
    assert res.year == 2001
    assert res.month == 2
    assert res.day == 1

    res = offset.rollforward(dt)
    assert res.year == 2001
    assert res.month == 3
    assert res.day == 1


@pytest.mark.parametrize(
    "start,end",
    (
        [datetime(2000, 2, 12), datetime(2001, 2, 12)],
        [datetime(2000, 2, 12), datetime(3001, 2, 12)],
        [datetime(1000, 2, 12), datetime(2001, 2, 12)],
    ),
)
def test_generate_range(start, end):
    offset = to_offset("AS")
    start = start
    end = end

    res = generate_range(start, end, offset)
    assert isinstance(res, GeneratorType)

    dts = list(res)

    exp = [datetime(y, 1, 1) for y in range(start.year, end.year + 2)]

    assert dts == exp


@pytest.mark.parametrize("inp", [np.nan, NaT])
def test_nan_apply_dt_errors(inp):
    offset = DateOffset()
    test_func = apply_dt(lambda _, x: x)
    assert test_func(offset, inp) is NaT


def test_nan_apply_dt_normalize():
    offset = DateOffset(normalize=True)
    test_func = apply_dt(lambda _, x: x)
    assert test_func(offset, datetime(2000, 1, 1, 13, 10, 1)) == datetime(
        2000, 1, 1, 0, 0, 0
    )
