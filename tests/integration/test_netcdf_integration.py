import os
import tempfile
from os.path import join

import numpy as np
import xarray as xr

from scmdata.netcdf import run_to_nc


def test_run_to_nc_read_with_xarray(scm_run):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        run_to_nc(scm_run, out_fname, dimensions=("scenario",))

        xr_dataset = xr.load_dataset(out_fname)

    # check that time is read back in as datetime
    assert xr_dataset["time"].dtype == "<M8[ns]"
    assert not isinstance(xr_dataset["time"].values[0], np.float64)


def test_run_to_nc_xarray_kwarg_passing(scm_run, tmpdir):
    out_fname = join(tmpdir, "out.nc")
    run_to_nc(scm_run, out_fname, dimensions=("scenario",))

    out_fname_compressed = join(tmpdir, "out_shrunk.nc")
    # actually using compression here makes the file bigger because the data
    # is so small
    comp = dict(dtype="int32")
    encoding = {var: comp for var in scm_run.get_unique_meta("variable")}
    run_to_nc(
        scm_run,
        out_fname_compressed,
        dimensions=("scenario",),
        encoding=encoding,
        engine="netcdf4",
        format="NETCDF4",
    )

    assert os.stat(out_fname).st_size > os.stat(out_fname_compressed).st_size
