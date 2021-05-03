import xarray as xr


def test_to_xarray(scm_run):
    res = scm_run.to_xarray(dimensions=("region", "scenario"))

    assert isinstance(res, xr.Dataset)
    assert set(res.data_vars) == set(scm_run.get_unique_meta("variable"))

    assert False, "check dimensions as expected"
    assert False, "check extras as expected"
    assert False, "check units as expected"
    assert False, "check metadata as expected"

# Tests to write:
# - dimensions handling
# - extras handling
# - weird variable name handling
