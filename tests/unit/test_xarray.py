import pytest
import xarray as xr


@pytest.mark.parametrize("dimensions,expected_dimensions", (
    (("region", "scenario", "time"), ("region", "scenario", "time")),
    (("time", "region", "scenario"), ("time", "region", "scenario")),
    (("region", "time", "scenario"), ("region", "time", "scenario")),
    (("region", "scenario"), ("region", "scenario", "time")),
    (("scenario", "region"), ("scenario", "region", "time")),
    (("scenario",), ("scenario", "time")),
))
def test_to_xarray_dimension_order(scm_run, dimensions, expected_dimensions):
    res = scm_run.to_xarray(dimensions=dimensions)

    assert isinstance(res, xr.Dataset)
    assert set(res.data_vars) == set(scm_run.get_unique_meta("variable"))

    for variable_name, data_var in res.data_vars.items():
        assert data_var.dims == expected_dimensions
        # no extras
        assert not set(data_var.coords) - set(data_var.dims)

        unit = scm_run.filter(variable=variable_name).get_unique_meta("unit", True)
        assert data_var.units == unit

    # all other metadata should be in attrs
    for meta_col in set(scm_run.meta.columns) - set(dimensions) - {"variable", "unit"}:
        meta_val = scm_run.get_unique_meta(meta_col, True)
        assert res.attrs["scmdata_metadata_{}".format(meta_col)] == meta_val

# Tests to write:
# - dimensions handling
# - extras handling
# - weird variable name handling
# - multiple units for given variable
