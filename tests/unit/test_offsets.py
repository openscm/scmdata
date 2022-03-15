import cftime
import pytest
from cftime import datetime

from scmdata.offsets import generate_range, to_offset


@pytest.mark.parametrize(
    "offset_rule",
    [
        "B",
        "C",
        "BM",
        "BMS",
        "CBM",
        "CBMS",
        "BQ",
        "BSS",
        "BA",
        "BAS",
        "RE",
        "BH",
        "CBH",
        "R",
        "REQ",
    ],
)
def test_invalid_offsets(offset_rule):
    with pytest.raises(ValueError):
        to_offset(offset_rule)


def test_annual_start():
    offset = to_offset("AS")

    dt = datetime(2001, 2, 12)

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

    dt = datetime(2001, 2, 12)

    res = offset.rollback(dt)
    assert res.year == 2001
    assert res.month == 2
    assert res.day == 1

    res = offset.rollforward(dt)
    assert res.year == 2001
    assert res.month == 3
    assert res.day == 1


@pytest.mark.parametrize(
    "output_cls",
    (
        None,
        cftime.DatetimeGregorian,
        cftime.Datetime360Day,
    ),
)
@pytest.mark.parametrize(
    "start,end",
    (
        [datetime(2000, 2, 12), datetime(2001, 2, 12)],
        [datetime(2000, 2, 12), datetime(3001, 2, 12)],
        [datetime(1000, 2, 12), datetime(2001, 2, 12)],
        [datetime(2000, 2, 12), datetime(2001, 2, 12)],
    ),
)
def test_generate_range(start, end, output_cls):
    offset = to_offset("AS")
    start = start
    end = end

    if output_cls is None:
        res = generate_range(start, end, offset)
        output_cls = cftime.DatetimeGregorian
    else:
        res = generate_range(start, end, offset, date_cls=output_cls)

    exp = [output_cls(y, 1, 1) for y in range(start.year, end.year + 2)]

    assert list(res) == exp


def test_generate_range_on_edges():
    offset = to_offset("AS")
    start = datetime(2000, 1, 1)
    end = datetime(2100, 1, 1)

    res = generate_range(start, end, offset)
    exp = [datetime(y, 1, 1) for y in range(start.year, end.year + 1)]

    assert list(res) == exp
