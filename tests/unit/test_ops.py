# tests to write:
# - operations align properly (i.e. use more than one timeseries in tests)
# - what happens if the subtraction doesn't align properly (either some rows in one ScmRun that aren't in the other or metadata is not unique once the ``op_cols`` are removed)
# - multiple ``op_cols`` also works
import pytest

from scmdata.run import ScmRun
from scmdata.testing import assert_scmdf_almost_equal


def get_single_ts(data=[1, 2, 3], index=[1, 2, 3], variable="Emissions|CO2",
    scenario="scen", model="mod", unit="GtC / yr",
    region="World", **kwargs):

    return ScmRun(
        data=[1, 2, 3],
        index=[1, 2, 3],
        columns={
            "variable": variable,
            "scenario": scenario,
            "model": model,
            "unit": unit,
            "region": region,
            **kwargs,
        }
    )

@pytest.fixture
def base_single_scmrun():
    return get_single_ts(variable="Emissions|CO2")

@pytest.fixture
def other_single_scmrun():
    return get_single_ts(variable="Emissions|CO2|Fossil")


def test_subtract_single_timeseries(base_single_scmrun, other_single_scmrun):
    res = base_single_scmrun.subtract(other_single_scmrun, op_cols={"variable": "Emissions|CO2|AFOLU"})

    base_ts = base_single_scmrun.timeseries().reset_index("variable", drop=True)
    other_ts = other_single_scmrun.timeseries().reset_index("variable", drop=True)

    exp_ts = base_ts - other_ts
    exp_ts["variable"] = "Emissions|CO2|AFOLU"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


def test_add_single_timeseries(base_single_scmrun, other_single_scmrun):
    res = base_single_scmrun.add(other_single_scmrun, op_cols={"variable": "Emissions|CO2|AFOLU"})

    base_ts = base_single_scmrun.timeseries().reset_index("variable", drop=True)
    other_ts = other_single_scmrun.timeseries().reset_index("variable", drop=True)

    exp_ts = base_ts + other_ts
    exp_ts["variable"] = "Emissions|CO2|AFOLU"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


def test_multiply_single_timeseries(base_single_scmrun, other_single_scmrun):
    res = base_single_scmrun.multiply(other_single_scmrun, op_cols={"variable": "Emissions|CO2|AFOLU"})

    base_ts = base_single_scmrun.timeseries().reset_index("variable", drop=True)
    other_ts = other_single_scmrun.timeseries().reset_index("variable", drop=True)

    exp_ts = base_ts * other_ts
    exp_ts["variable"] = "Emissions|CO2|AFOLU"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


def test_divide_single_timeseries(base_single_scmrun, other_single_scmrun):
    res = base_single_scmrun.divide(other_single_scmrun, op_cols={"variable": "Emissions|CO2|AFOLU"})

    base_ts = base_single_scmrun.timeseries().reset_index("variable", drop=True)
    other_ts = other_single_scmrun.timeseries().reset_index("variable", drop=True)

    exp_ts = base_ts / other_ts
    exp_ts["variable"] = "Emissions|CO2|AFOLU"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)
