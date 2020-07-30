import re

import numpy as np
import pandas as pd
import pint_pandas
import pytest
from openscm_units import unit_registry
from pint.errors import DimensionalityError

from scmdata.run import ScmRun
from scmdata.testing import assert_scmdf_almost_equal


pint_pandas.PintType.ureg = unit_registry


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


OPS_MARK = pytest.mark.parametrize("op", ("add", "subtract", "multiply", "divide"))


def perform_op(base, other, op, reset_index):
    base_ts = base.timeseries().reset_index(reset_index, drop=True)
    other_ts = other.timeseries().reset_index(reset_index, drop=True)

    if op == "add":
        exp_ts = base_ts + other_ts

    elif op == "subtract":
        exp_ts = base_ts - other_ts

    elif op == "divide":
        exp_ts = base_ts / other_ts

    elif op == "multiply":
        exp_ts = base_ts * other_ts

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

    if op in ["add", "subtract"]:
        exp_ts["unit"] = "gigatC / a"

    elif op == "multiply":
        exp_ts["unit"] = "gigatC ** 2 / a ** 2"

    elif op == "divide":
        exp_ts["unit"] = "dimensionless"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


@OPS_MARK
def test_multiple_timeseries(op, base_multiple_scmrun, other_multiple_scmrun):
    res = getattr(base_multiple_scmrun, op)(
        other_multiple_scmrun, op_cols={"scenario": "A to B"}
    )

    exp_ts = perform_op(base_multiple_scmrun, other_multiple_scmrun, op, "scenario")
    exp_ts["scenario"] = "A to B"

    if op in ["add", "subtract"]:
        exp_ts["unit"] = exp_ts["unit"].apply(convert_to_pint_name).values

    elif op == "multiply":
        exp_ts["unit"] = (
            exp_ts["unit"]
            .apply(lambda x: convert_to_pint_name("({})**2".format(x)))
            .values
        )

    elif op == "divide":
        exp_ts["unit"] = "dimensionless"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


def test_missing_series_error():
    base = get_multiple_ts(region=["World|R5LAM", "World|R5REF"])
    other = get_multiple_ts(region=["World|R5LAM", "World|R5OECD"])

    error_msg = re.escape(
        "No equivalent in `other` for "
        "[('scenario', 'scen'), ('model', 'mod'), ('region', 'World|R5REF')]"
    )
    with pytest.raises(KeyError, match=error_msg):
        base.add(other, op_cols={"variable": "Warming plus Cumulative emissions CO2"})


def test_different_unit_error():
    base = get_single_ts(variable="Surface Temperature", unit="K")
    other = get_single_ts(variable="Cumulative Emissions|CO2", unit="GtC")

    error_msg = re.escape(
        "Cannot convert from 'kelvin' ([temperature]) to "
        "'gigatC' ([carbon] * [mass])"
    )
    with pytest.raises(DimensionalityError, match=error_msg):
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


def test_warming_per_gt():
    base = get_single_ts(variable="Surface Temperature", unit="K")
    other = get_single_ts(variable="Cumulative Emissions|CO2", unit="GtC")

    res = base.divide(
        other, op_cols={"variable": "Warming per Cumulative emissions CO2"}
    )

    exp_ts = perform_op(base, other, "divide", ["variable", "unit"])
    exp_ts["variable"] = "Warming per Cumulative emissions CO2"
    exp_ts["unit"] = "kelvin / gigatC"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


def perform_scalar_op(base, scalar, op):
    base_ts = base.timeseries().T
    unit_level = base_ts.columns.names.index("unit")
    base_ts = base_ts.pint.quantify(level=unit_level)

    out = []
    for _, series in base_ts.iteritems():
        if op == "add":
            op_series = series + scalar

        elif op == "subtract":
            op_series = series - scalar

        elif op == "divide":
            op_series = series / scalar

        elif op == "divide_inverse":
            op_series = scalar / series

        elif op == "multiply":
            op_series = series * scalar

        elif op == "multiply_inverse":
            op_series = scalar * series

        else:
            raise NotImplementedError(op)

        out.append(op_series)

    out = pd.concat(out, axis="columns")
    out.columns.names = base_ts.columns.names
    out = out.pint.dequantify().T

    return out


@OPS_MARK
def test_scalar_ops_pint(op):
    scalar = 1 * unit_registry("MtC / yr")
    start = get_multiple_ts(
        variable="Emissions|CO2", unit="GtC / yr", scenario=["scen_a", "scen_b"]
    )

    exp_ts = perform_scalar_op(start, scalar, op)
    exp = ScmRun(exp_ts)

    if op in ["add", "subtract"]:
        exp["unit"] = "gigatC / a"

    elif op == "multiply":
        exp["unit"] = "gigatC * megatC / a ** 2"

    elif op == "divide":
        exp["unit"] = "gigatC / megatC"

    if op == "add":
        res = start + scalar

    elif op == "subtract":
        res = start - scalar

    elif op == "divide":
        res = start / scalar

    elif op == "multiply":
        res = start * scalar

    else:
        raise NotImplementedError(op)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


@pytest.mark.xfail(reason="pint doesn't recognise ScmRun")
def test_scalar_divide_pint_by_run():
    scalar = 1 * unit_registry("MtC / yr")
    start = get_multiple_ts(
        variable="Emissions|CO2", unit="GtC / yr", scenario=["scen_a", "scen_b"]
    )

    exp_ts = perform_scalar_op(start, scalar, "divide_inverse")
    exp = ScmRun(exp_ts)

    exp["unit"] = "megatC / gigatC"

    res = scalar / start

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


@pytest.mark.xfail(reason="pint doesn't recognise ScmRun")
def test_scalar_multiply_pint_by_run():
    scalar = 1 * unit_registry("MtC / yr")
    start = get_multiple_ts(
        variable="Emissions|CO2", unit="GtC / yr", scenario=["scen_a", "scen_b"]
    )

    exp_ts = perform_scalar_op(start, scalar, "multiply_inverse")
    exp = ScmRun(exp_ts)

    exp["unit"] = "megatC * gigatC / a**2"

    res = scalar * start

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


@pytest.mark.parametrize("op", ["add", "subtract"])
def test_scalar_ops_pint_wrong_unit(op):
    scalar = 1 * unit_registry("Mt CH4 / yr")
    start = get_multiple_ts(
        variable="Emissions|CO2", unit="GtC / yr", scenario=["scen_a", "scen_b"]
    )

    error_msg = re.escape(
        "Cannot convert from 'gigatC / a' ([carbon] * [mass] / [time]) to 'CH4 * megametric_ton / a' ([mass] * [methane] / [time])"
    )
    with pytest.raises(DimensionalityError, match=error_msg):
        if op == "add":
            start + scalar

        elif op == "subtract":
            start - scalar

        else:
            raise NotImplementedError(op)


def perform_scalar_op_float_int(base, scalar, op):
    base_ts = base.timeseries()

    if op == "add":
        base_ts = base_ts + scalar

    elif op == "subtract":
        base_ts = base_ts - scalar

    elif op == "divide":
        base_ts = base_ts / scalar

    elif op == "multiply":
        base_ts = base_ts * scalar

    else:
        raise NotImplementedError(op)

    return base_ts


@OPS_MARK
@pytest.mark.parametrize("scalar", (1, 1.0))
def test_scalar_ops_float_int(op, scalar):
    start = get_multiple_ts(
        variable="Emissions|CO2", unit="GtC / yr", scenario=["scen_a", "scen_b"]
    )

    exp_ts = perform_scalar_op_float_int(start, scalar, op)
    exp = ScmRun(exp_ts)

    if op == "add":
        res = start + scalar

    elif op == "subtract":
        res = start - scalar

    elif op == "divide":
        res = start / scalar

    elif op == "multiply":
        res = start * scalar

    else:
        raise NotImplementedError(op)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


@OPS_MARK
def test_wrong_shape_ops(op):
    start = get_multiple_ts(
        variable="Emissions|CO2", unit="GtC / yr", scenario=["scen_a", "scen_b"]
    )

    other = np.arange(np.prod(start.shape)).reshape(start.shape)[:, :-1]

    error_msg = re.escape(
        "operands could not be broadcast together with shapes (3,) (2,2)"
    )
    with pytest.raises(ValueError, match=error_msg):
        if op == "add":
            start + other

        elif op == "subtract":
            start - other

        elif op == "divide":
            start / other

        elif op == "multiply":
            start * other

        else:
            raise NotImplementedError(op)


@OPS_MARK
def test_wrong_length_ops(op):
    start = get_multiple_ts(
        variable="Emissions|CO2", unit="GtC / yr", scenario=["scen_a", "scen_b"]
    )

    other = np.arange(np.prod(start.shape)).reshape(start.shape)[:-1, :]

    error_msg = re.escape("Incorrect length")
    with pytest.raises(ValueError, match=error_msg):
        if op == "add":
            start + other

        elif op == "subtract":
            start - other

        elif op == "divide":
            start / other

        elif op == "multiply":
            start * other

        else:
            raise NotImplementedError(op)
