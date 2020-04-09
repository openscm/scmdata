import tempfile
from os.path import exists, join

import netCDF4 as nc
import numpy.testing as npt
import pandas.testing as pdt
from scmdata.testing import assert_scmdf_almost_equal
from scmdata.netcdf import nc_to_run, run_to_nc


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


def test_nc_methods(scm_data):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        scm_data.to_nc(out_fname, dimensions=("scenario",))

        assert exists(out_fname)

        # Same as ScmRun.from_nc(out_fname)
        df = scm_data.__class__.from_nc(out_fname)

        assert isinstance(df, scm_data.__class__)
        assert_scmdf_almost_equal(scm_data, df, check_ts_names=False)
