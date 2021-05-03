import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr


def do_basic_to_xarray_checks(res, start_run, dimensions, extras):
    assert isinstance(res, xr.Dataset)
    assert set(res.data_vars) == set(start_run.get_unique_meta("variable"))

    for variable_name, data_var in res.data_vars.items():
        assert data_var.dims == dimensions

        unit = start_run.filter(variable=variable_name).get_unique_meta("unit", True)
        assert data_var.units == unit

        # check a couple of data points to make sure the translation is correct
        for idx in [0, -1]:
            xarray_spot = data_var.isel({v: idx for v in dimensions})
            fkwargs = {k: [v.values.tolist()] for k, v in xarray_spot.coords.items()}
            fkwargs["variable"] = variable_name

            start_run_spot = start_run.filter(**fkwargs)
            if np.isnan(xarray_spot):
                assert start_run_spot.empty
            else:
                start_run_vals = float(start_run_spot.values.squeeze())
                npt.assert_array_equal(xarray_spot.values, start_run_vals)

    # all other metadata should be in attrs
    for meta_col in set(start_run.meta.columns) - set(dimensions) - set(extras) - {"variable", "unit"}:
        meta_val = start_run.get_unique_meta(meta_col, True)
        assert res.attrs["scmdata_metadata_{}".format(meta_col)] == meta_val


@pytest.mark.parametrize("dimensions,expected_dimensions", (
    (("region", "scenario", "time"), ("region", "scenario", "time")),
    (("time", "region", "scenario"), ("time", "region", "scenario")),
    (("region", "time", "scenario"), ("region", "time", "scenario")),
    (("region", "scenario"), ("region", "scenario", "time")),
    (("scenario", "region"), ("scenario", "region", "time")),
    (("scenario",), ("scenario", "time")),
))
def test_to_xarray(scm_run, dimensions, expected_dimensions):
    res = scm_run.to_xarray(dimensions=dimensions)

    do_basic_to_xarray_checks(res, scm_run, expected_dimensions, (),)

    # no extras
    assert not set(res.coords) - set(res.dims)


@pytest.mark.parametrize("extras", (
    ("model",),
    ("climate_model",),
    ("climate_model", "model"),
))
def test_to_xarray_extras(scm_run, extras):
    dimensions = ("scenario", "region", "time")
    res = scm_run.to_xarray(dimensions=dimensions, extras=extras)

    do_basic_to_xarray_checks(res, scm_run, dimensions, extras)

    assert set(extras) == set(res.coords) - set(res.dims)

    scm_run_meta = scm_run.meta
    for extra_col in extras:
        xarray_vals = res[extra_col].values
        extra_dims = res[extra_col].dims
        assert len(extra_dims) == 1
        extra_dims = extra_dims[0]
        xarray_coords = res[extra_col][extra_dims].values

        for xarray_extra_val, extra_xarray_coord in zip(xarray_vals, xarray_coords):
            scm_run_extra_val = scm_run_meta[scm_run_meta[extra_dims] == extra_xarray_coord][extra_col].unique().tolist()
            assert len(scm_run_extra_val) == 1
            scm_run_extra_val = scm_run_extra_val[0]

            assert scm_run_extra_val == xarray_extra_val

# Tests to write:
# - dimensions handling
# - extras handling
# - weird variable name handling
# - multiple units for given variable
# - overlapping dimensions and extras
# - underdefined dimensions and extras
