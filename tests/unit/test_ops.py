import re

import numpy as np
import pytest
from openscm_units import unit_registry

from scmdata.run import ScmRun
from scmdata.testing import assert_scmdf_almost_equal


def get_ts(data, index, **kwargs):
    return ScmRun(data=data, index=index, columns=kwargs)


def get_single_ts(
    data=[1, 2, 3],
    index=[1, 2, 3],
    variable="Emissions|CO2",
    scenario="scen",
    model="mod",
    unit="GtC / yr",
    region="World",
    **kwargs
):

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


def get_multiple_ts(
    data=np.array([[1, 2, 3], [10, 20, 30]]).T,
    index=[2020, 2030, 2040],
    variable=["Emissions|CO2", "Emissions|CH4"],
    scenario="scen",
    model="mod",
    unit=["GtC / yr", "MtCH4 / yr"],
    region="World",
    **kwargs
):
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
    return get_single_ts(data=[-1, 0, 5], variable="Emissions|CO2|Fossil")


@pytest.fixture
def base_multiple_scmrun():
    return get_multiple_ts(scenario="Scenario A")


@pytest.fixture
def other_multiple_scmrun():
    return get_multiple_ts(
        data=np.array([[-1, 0, 3.2], [11.1, 20, 32.3]]).T, scenario="Scenario B"
    )


def convert_to_pint_name(unit):
    return str(unit_registry(unit).units)


OPS_MARK = pytest.mark.parametrize("op", ("add", "subtract"))


def perform_op(base, other, op, reset_index):
    base_ts = base.timeseries().reset_index(reset_index, drop=True)
    other_ts = other.timeseries().reset_index(reset_index, drop=True)

    if op == "add":
        exp_ts = base_ts + other_ts

    elif op == "subtract":
        exp_ts = base_ts - other_ts

    else:
        raise NotImplementedError(op)

    return exp_ts.reset_index()


@OPS_MARK
def test_single_timeseries(op, base_single_scmrun, other_single_scmrun):
    res = getattr(base_single_scmrun, op)(
        other_single_scmrun, op_cols={"variable": "Emissions|CO2|AFOLU"}
    )

    exp_ts = perform_op(base_single_scmrun, other_single_scmrun, op, "variable")
    exp_ts["variable"] = "Emissions|CO2|AFOLU"

    # when we add pint back in this happens
    # if op in ["add", "subtract"]:
    #     exp_ts["unit"] = "gigatC / a"

    # once we add multiply and divide in, turn these on
    # elif op == "multiply":
    #     exp_ts["unit"] = "gigatC ** 2 / a ** 2"

    # elif op == "divide":
    #     exp_ts["unit"] = "dimensionless"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


@OPS_MARK
def test_multiple_timeseries(op, base_multiple_scmrun, other_multiple_scmrun):
    res = getattr(base_multiple_scmrun, op)(
        other_multiple_scmrun, op_cols={"scenario": "A to B"}
    )

    exp_ts = perform_op(base_multiple_scmrun, other_multiple_scmrun, op, "scenario")
    exp_ts["scenario"] = "A to B"

    # when we add pint back in this happens
    # if op in ["add", "subtract"]:
    #     exp_ts["unit"] = exp_ts["unit"].apply(convert_to_pint_name).values

    # elif op == "multiply":
    #     exp_ts["unit"] = (
    #         exp_ts["unit"]
    #         .apply(lambda x: convert_to_pint_name("({})**2".format(x)))
    #         .values
    #     )

    # elif op == "divide":
    #     exp_ts["unit"] = "dimensionless"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


def test_missing_series_error():
    base = get_multiple_ts(region=["World|R5LAM", "World|R5REF"])
    other = get_multiple_ts(region=["World|R5LAM", "World|R5OECD"])

    error_msg = re.escape(
        "No equivalent in `other` for "
        "[('scenario', 'scen'), ('model', 'mod'), ('unit', 'MtCH4 / yr'), "
        "('region', 'World|R5REF')]"
    )
    with pytest.raises(KeyError, match=error_msg):
        base.add(other, op_cols={"variable": "Warming plus Cumulative emissions CO2"})


def test_different_unit_error():
    base = get_single_ts(variable="Surface Temperature", unit="K")
    other = get_single_ts(variable="Cumulative Emissions|CO2", unit="GtC")

    error_msg = re.escape(
        "No equivalent in `other` for "
        "[('scenario', 'scen'), ('model', 'mod'), ('unit', 'K'), "
        "('region', 'World')]"
    )
    with pytest.raises(KeyError, match=error_msg):
        base.add(other, op_cols={"variable": "Warming plus Cumulative emissions CO2"})


def test_multiple_ops_cols():
    base = get_single_ts(variable="Surface Temperature", unit="K")
    other = get_single_ts(variable="Cumulative Emissions|CO2", unit="GtC")

    res = base.add(
        other,
        op_cols={
            "variable": "Warming plus Cumulative emissions CO2",
            "unit": "nonsense",
        },
    )

    exp_ts = perform_op(base, other, "add", ["variable", "unit"])
    exp_ts["variable"] = "Warming plus Cumulative emissions CO2"
    exp_ts["unit"] = "nonsense"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)
