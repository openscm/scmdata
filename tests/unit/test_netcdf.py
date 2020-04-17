import tempfile
from os.path import exists, join

import netCDF4 as nc
import numpy.testing as npt
import pytest

from scmdata.netcdf import nc_to_run, run_to_nc
from scmdata.testing import assert_scmdf_almost_equal


def test_run_to_nc(scm_data):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        run_to_nc(scm_data, out_fname, dimensions=("scenario",))

        assert exists(out_fname)

        ds = nc.Dataset(out_fname)

        assert ds.dimensions["time"].size == len(scm_data.time_points)
        assert ds.dimensions["scenario"].size == 2

        assert ds.variables["scenario"][0] == "a_scenario"
        assert ds.variables["scenario"][1] == "a_scenario2"

        npt.assert_allclose(
            ds.variables["primary_energy"][0, :],
            scm_data.filter(variable="Primary Energy", scenario="a_scenario").values[0],
        )
        npt.assert_allclose(
            ds.variables["primary_energy"][1, :],
            scm_data.filter(variable="Primary Energy", scenario="a_scenario2").values[
                0
            ],
        )
        npt.assert_allclose(
            ds.variables["primary_energy__coal"][0, :],
            scm_data.filter(
                variable="Primary Energy|Coal", scenario="a_scenario"
            ).values[0],
        )


def test_nc_to_run(scm_data):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        run_to_nc(scm_data, out_fname, dimensions=("scenario",))

        assert exists(out_fname)

        df = nc_to_run(scm_data.__class__, out_fname)
        assert isinstance(df, scm_data.__class__)

        assert_scmdf_almost_equal(scm_data, df, check_ts_names=False)


def test_run_to_nc_with_extras(scm_data):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")

        # make an extra column which maps 1:1 with scenario
        unique_scenarios = scm_data["scenario"].unique().tolist()
        run_id = scm_data["scenario"].apply(lambda x: unique_scenarios.index(x))

        scm_data.set_meta(run_id, "run_id")
        run_to_nc(scm_data, out_fname, dimensions=("scenario",), extras=("run_id",))

        assert exists(out_fname)

        ds = nc.Dataset(out_fname)

        assert ds.dimensions["time"].size == len(scm_data.time_points)
        assert ds.dimensions["scenario"].size == 2

        assert ds.variables["scenario"][0] == "a_scenario"
        assert ds.variables["scenario"][1] == "a_scenario2"

        assert ds.variables["run_id"]._is_metadata
        for i, run_id in enumerate(ds.variables["run_id"]):
            assert run_id == unique_scenarios.index(ds.variables["scenario"][i])

        npt.assert_allclose(
            ds.variables["primary_energy"][0, :],
            scm_data.filter(variable="Primary Energy", scenario="a_scenario").values[0],
        )
        assert not ds.variables["primary_energy"]._is_metadata
        npt.assert_allclose(
            ds.variables["primary_energy"][1, :],
            scm_data.filter(variable="Primary Energy", scenario="a_scenario2").values[
                0
            ],
        )
        npt.assert_allclose(
            ds.variables["primary_energy__coal"][0, :],
            scm_data.filter(
                variable="Primary Energy|Coal", scenario="a_scenario"
            ).values[0],
        )
        assert not ds.variables["primary_energy__coal"]._is_metadata


def test_nc_to_run_with_extras(scm_data):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        scm_data.set_meta([1, 1, 2], "run_id")
        run_to_nc(scm_data, out_fname, dimensions=("scenario",), extras=("run_id",))

        assert exists(out_fname)

        df = nc_to_run(scm_data.__class__, out_fname)
        assert isinstance(df, scm_data.__class__)

        assert_scmdf_almost_equal(scm_data, df, check_ts_names=False)


def test_nc_to_run_with_extras_non_unique_for_dimension(scm_data):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        scm_data.set_meta([1, 2, 1], "run_id")

        error_msg = "metadata for run_id is not unique for requested dimensions"
        with pytest.raises(ValueError, match=error_msg):
            run_to_nc(scm_data, out_fname, dimensions=("scenario",), extras=("run_id",))


def test_nc_methods(scm_data):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        scm_data.to_nc(out_fname, dimensions=("scenario",))

        assert exists(out_fname)

        # Same as ScmRun.from_nc(out_fname)
        df = scm_data.__class__.from_nc(out_fname)

        assert isinstance(df, scm_data.__class__)
        assert_scmdf_almost_equal(scm_data, df, check_ts_names=False)
