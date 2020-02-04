import tempfile
from os.path import join, exists

import netCDF4 as nc
import numpy.testing as npt
from scmdata.rw import df_to_nc, nc_to_df
import pandas.testing as pdt


def test_to_nc(test_scm_df):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        df_to_nc(test_scm_df, out_fname, dimensions=("scenario",))

        assert exists(out_fname)

        ds = nc.Dataset(out_fname)

        assert ds.dimensions["time"].size == len(test_scm_df.time_points)
        assert ds.dimensions["scenario"].size == 2

        assert ds.variables["scenario"][0] == "a_scenario"
        assert ds.variables["scenario"][1] == "a_scenario2"

        npt.assert_allclose(ds.variables["primary_energy"][0, :],
                            test_scm_df.filter(variable="Primary Energy", scenario="a_scenario").values[0])
        npt.assert_allclose(ds.variables["primary_energy"][1, :],
                            test_scm_df.filter(variable="Primary Energy", scenario="a_scenario2").values[0])
        npt.assert_allclose(ds.variables["primary_energy__coal"][0, :],
                            test_scm_df.filter(variable="Primary Energy|Coal", scenario="a_scenario").values[0])


def test_to_df(test_scm_df):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        df_to_nc(test_scm_df, out_fname, dimensions=("scenario",))

        assert exists(out_fname)

        df = nc_to_df(out_fname)
        pdt.assert_frame_equal(test_scm_df.timeseries(), df.timeseries(), check_like=True)
