# tests to write:
# - operations align properly (i.e. use more than one timeseries in tests)
# - what happens if the subtraction doesn't align properly (either some rows in one ScmRun that aren't in the other or metadata is not unique once the ``op_cols`` are removed)
# - multiple ``op_cols`` also works
from scmdata.run import ScmRun
from scmdata.testing import assert_scmdf_almost_equal


def test_subtract_single_timeseries():
    common_cols = {"scenario": "scen", "model": "mod", "unit": "GtC / yr", "region": "World"}

    base = ScmRun(data=[1, 2, 3], index=[1, 2, 3], columns={"variable": "Emissions|CO2", **common_cols})
    other = ScmRun(data=[3, 2, 1], index=[1, 2, 3], columns={"variable": "Emissions|CO2|Fossil", **common_cols})

    res = base.subtract(other, op_cols={"variable": "Emissions|CO2|AFOLU"})

    base_ts = base.timeseries().reset_index("variable", drop=True)
    other_ts = other.timeseries().reset_index("variable", drop=True)

    exp_ts = base_ts - other_ts
    exp_ts["variable"] = "Emissions|CO2|AFOLU"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


def test_add_single_timeseries():
    common_cols = {"scenario": "scen", "model": "mod", "unit": "GtC / yr", "region": "World"}

    base = ScmRun(data=[1, 2, 3], index=[1, 2, 3], columns={"variable": "Emissions|CO2", **common_cols})
    other = ScmRun(data=[3, 2, 1], index=[1, 2, 3], columns={"variable": "Emissions|CO2|Fossil", **common_cols})

    res = base.add(other, op_cols={"variable": "Emissions|CO2|AFOLU"})

    base_ts = base.timeseries().reset_index("variable", drop=True)
    other_ts = other.timeseries().reset_index("variable", drop=True)

    exp_ts = base_ts + other_ts
    exp_ts["variable"] = "Emissions|CO2|AFOLU"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)
