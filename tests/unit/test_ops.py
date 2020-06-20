# tests to write:
# - operations align properly (i.e. use more than one timeseries in tests)
# - what happens if the subtraction doesn't align properly (either some rows in one ScmRun that aren't in the other or metadata is not unique once the ``op_cols`` are removed)
# - multiple ``op_cols`` also works
import pytest

from scmdata.run import ScmRun
from scmdata.testing import assert_scmdf_almost_equal


def get_ts(data, index, **kwargs):
    return ScmRun(data=data, index=index, columns=kwargs)


def get_single_ts(data=[1, 2, 3], index=[1, 2, 3], variable="Emissions|CO2",
    scenario="scen", model="mod", unit="GtC / yr",
    region="World", **kwargs):

    return get_ts(
        data=data,
        index=index,
        variable=variable,
        scenario=scenario,
        model=model,
        unit=unit,
        region=region,
        **kwargs
    )


def get_multiple_ts(data=[1, 2, 3], index=[1, 2, 3], variable="Emissions|CO2",
    scenario="scen", model="mod", unit="GtC / yr",
    region="World", **kwargs):

    return get_ts(
        data=data,
        index=index,
        variable=variable,
        scenario=scenario,
        model=model,
        unit=unit,
        region=region,
        **kwargs
    )


@pytest.fixture
def base_single_scmrun():
    return get_single_ts(variable="Emissions|CO2")


@pytest.fixture
def other_single_scmrun():
    return get_single_ts(variable="Emissions|CO2|Fossil")


@pytest.fixture
def base_multiple_scmrun():
    return get_single_ts(variable="Emissions|CO2")


@pytest.fixture
def other_multiple_scmrun():
    return get_single_ts(variable="Emissions|CO2|Fossil")


OPS_MARK = pytest.mark.parametrize("op", ("add", "subtract", "multiply", "divide"))


def perform_op(base, other, op, reset_index):
    base_ts = base.timeseries().reset_index(reset_index, drop=True)
    other_ts = other.timeseries().reset_index(reset_index, drop=True)

    if op == "add":
        exp_ts = base_ts + other_ts

    elif op == "subtract":
        exp_ts = base_ts - other_ts

    elif op == "multiply":
        exp_ts = base_ts * other_ts

    elif op == "divide":
        exp_ts = base_ts / other_ts

    else:
        raise NotImplementedError(op)

    return exp_ts


@OPS_MARK
def test_single_timeseries(op, base_single_scmrun, other_single_scmrun):
    res = getattr(base_single_scmrun, op)(other_single_scmrun, op_cols={"variable": "Emissions|CO2|AFOLU"})

    exp_ts = perform_op(
        base_single_scmrun,
        other_single_scmrun,
        op,
        "variable"
    )
    exp_ts["variable"] = "Emissions|CO2|AFOLU"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


@OPS_MARK
def test_multiple_timeseries(op, base_multiple_scmrun, other_multiple_scmrun):
    res = getattr(base_multiple_scmrun, op)(other_multiple_scmrun, op_cols={"scenario": "one to two"})

    exp_ts = perform_op(
        base_multiple_scmrun,
        other_multiple_scmrun,
        op,
        "scenario"
    )
    exp_ts["scenario"] = "one to two"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)
