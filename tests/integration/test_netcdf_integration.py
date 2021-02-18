import tempfile
from os.path import join

import numpy as np
import xarray as xr

from scmdata import ScmRun
from scmdata.netcdf import run_to_nc
from scmdata.testing import assert_scmdf_almost_equal


def test_run_to_nc_read_with_xarray(scm_run):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        run_to_nc(scm_run, out_fname, dimensions=("scenario",))

        xr_dataset = xr.load_dataset(out_fname)

    # check that time is read back in as datetime
    assert xr_dataset["time"].dtype == "<M8[ns]"
    assert not isinstance(xr_dataset["time"].values[0], np.float64)


def test_read_legacy_datetimes_nc(scm_run, test_data_path):
    old_datetimes_run = ScmRun.from_nc(join(test_data_path, "legacy_datetimes.nc"))

    assert_scmdf_almost_equal(old_datetimes_run, scm_run, check_ts_names=False)
