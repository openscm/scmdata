import logging
import re
import tempfile
from os.path import exists, join
from unittest.mock import patch

import netCDF4 as nc
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from scmdata import ScmRun
from scmdata.netcdf import nc_to_run, run_to_nc
from scmdata.testing import assert_scmdf_almost_equal


def test_run_to_nc(scm_run):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        run_to_nc(scm_run, out_fname, dimensions=("scenario",))

        assert exists(out_fname)

        ds = nc.Dataset(out_fname)

        assert ds.dimensions["time"].size == scm_run.shape[1]
        assert ds.dimensions["scenario"].size == 2

        assert ds.variables["scenario"][0] == "a_scenario"
        assert ds.variables["scenario"][1] == "a_scenario2"

        npt.assert_allclose(
            ds.variables["Primary_Energy"][0, :],
            scm_run.filter(variable="Primary Energy", scenario="a_scenario").values[0],
        )
        npt.assert_allclose(
            ds.variables["Primary_Energy"][1, :],
            scm_run.filter(variable="Primary Energy", scenario="a_scenario2").values[0],
        )
        npt.assert_allclose(
            ds.variables["Primary_Energy__Coal"][0, :],
            scm_run.filter(
                variable="Primary Energy|Coal", scenario="a_scenario"
            ).values[0],
        )


@pytest.mark.parametrize(
    "v", ["primary energy", "Primary Energy", "Primary Energy|Coal|Test",],
)
def test_run_to_nc_case(scm_run, v):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        scm_run = scm_run.filter(variable="Primary Energy")
        scm_run["variable"] = v

        run_to_nc(scm_run, out_fname, dimensions=("scenario",))
        res = nc_to_run(scm_run.__class__, out_fname)

        assert res.get_unique_meta("variable", True) == v


def test_run_to_nc_4d(scm_run, tmpdir):
    df = scm_run.timeseries().reset_index()
    df["climate_model"] = "base_m"
    df["run_id"] = 1

    big_df = [df]
    for climate_model in ["abc_m", "def_m", "ghi_m"]:
        for run_id in range(10):
            new_df = df.copy()
            new_df["run_id"] = run_id
            new_df["climate_model"] = climate_model

            big_df.append(new_df)

    scm_run = scm_run.__class__(pd.concat(big_df).reset_index(drop=True))

    out_fname = join(tmpdir, "out.nc")
    run_to_nc(scm_run, out_fname, dimensions=("scenario", "climate_model", "run_id"))

    assert exists(out_fname)

    ds = nc.Dataset(out_fname)

    assert ds.dimensions["time"].size == len(scm_run.time_points)
    assert ds.dimensions["scenario"].size == 2
    assert ds.dimensions["climate_model"].size == 4
    assert ds.dimensions["run_id"].size == 10

    assert ds.variables["scenario"][0] == "a_scenario"
    assert ds.variables["scenario"][1] == "a_scenario2"
    assert ds.variables["climate_model"][0] == "abc_m"
    assert ds.variables["climate_model"][1] == "base_m"
    assert ds.variables["climate_model"][2] == "def_m"
    assert ds.variables["climate_model"][3] == "ghi_m"
    npt.assert_array_equal(ds.variables["run_id"][:], range(10))

    assert ds.variables["Primary_Energy"].shape == (2, 4, 10, 3)
    assert ds.variables["Primary_Energy__Coal"].shape == (2, 4, 10, 3)


def test_run_to_nc_nan_dimension_error(scm_run, tmpdir):
    scm_run["run_id"] = np.nan

    out_fname = join(tmpdir, "out.nc")
    with pytest.raises(AssertionError, match="nan in dimension: `run_id`"):
        run_to_nc(scm_run, out_fname, dimensions=("scenario", "run_id"))


@pytest.mark.parametrize("dimensions", (("scenario",), ("scenario", "time")))
def test_nc_to_run(scm_run, dimensions):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        run_to_nc(scm_run, out_fname, dimensions=dimensions)

        assert exists(out_fname)

        df = nc_to_run(scm_run.__class__, out_fname)
        assert isinstance(df, scm_run.__class__)

        assert_scmdf_almost_equal(scm_run, df, check_ts_names=False)


def test_nc_to_run_4d(scm_run):
    df = scm_run.timeseries()
    val_cols = df.columns.tolist()
    df = df.reset_index()

    df["climate_model"] = "base_m"
    df["run_id"] = 1
    df.loc[:, val_cols] = np.random.rand(df.shape[0], len(val_cols))

    big_df = [df]
    for climate_model in ["abc_m", "def_m", "ghi_m"]:
        for run_id in range(10):
            new_df = df.copy()
            new_df["run_id"] = run_id
            new_df["climate_model"] = climate_model
            new_df.loc[:, val_cols] = np.random.rand(df.shape[0], len(val_cols))

            big_df.append(new_df)

    scm_run = scm_run.__class__(pd.concat(big_df).reset_index(drop=True))

    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        run_to_nc(
            scm_run, out_fname, dimensions=("scenario", "climate_model", "run_id")
        )

        assert exists(out_fname)

        df = nc_to_run(scm_run.__class__, out_fname)
        assert isinstance(df, scm_run.__class__)

        assert_scmdf_almost_equal(scm_run, df, check_ts_names=False)


def test_nc_to_run_non_unique_for_dimension(scm_run):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")

        error_msg = "['region'] dimensions are not unique for variable Primary Energy"
        with pytest.raises(ValueError, match=re.escape(error_msg)):
            run_to_nc(scm_run, out_fname, dimensions=("region",))


def test_nc_to_run_non_unique_meta(scm_run):
    scm_run["climate_model"] = ["b_model", "a_model", "a_model"]

    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")

        error_msg = (
            "metadata for climate_model is not unique for variable Primary Energy"
        )
        with pytest.raises(ValueError, match=error_msg):
            run_to_nc(scm_run, out_fname, dimensions=("scenario",))


@pytest.mark.parametrize("dtype", (int, float, str))
def test_run_to_nc_with_extras(scm_run, dtype):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")

        # make an extra column which maps 1:1 with scenario
        unique_scenarios = scm_run["scenario"].unique().tolist()
        run_id = (
            scm_run["scenario"].apply(lambda x: unique_scenarios.index(x)).astype(dtype)
        )

        scm_run["run_id"] = run_id
        run_to_nc(scm_run, out_fname, dimensions=("scenario",), extras=("run_id",))

        assert exists(out_fname)

        ds = nc.Dataset(out_fname)

        assert ds.dimensions["time"].size == len(scm_run.time_points)
        assert ds.dimensions["scenario"].size == 2

        assert ds.variables["scenario"][0] == "a_scenario"
        assert ds.variables["scenario"][1] == "a_scenario2"

        assert ds.variables["run_id"]._is_metadata
        for i, run_id in enumerate(ds.variables["run_id"]):
            exp_val = dtype(unique_scenarios.index(ds.variables["scenario"][i]))
            assert run_id == exp_val

        npt.assert_allclose(
            ds.variables["Primary_Energy"][0, :],
            scm_run.filter(variable="Primary Energy", scenario="a_scenario").values[0],
        )
        assert not ds.variables["Primary_Energy"]._is_metadata
        npt.assert_allclose(
            ds.variables["Primary_Energy"][1, :],
            scm_run.filter(variable="Primary Energy", scenario="a_scenario2").values[0],
        )
        npt.assert_allclose(
            ds.variables["Primary_Energy__Coal"][0, :],
            scm_run.filter(
                variable="Primary Energy|Coal", scenario="a_scenario"
            ).values[0],
        )
        assert not ds.variables["Primary_Energy__Coal"]._is_metadata


def test_nc_to_run_with_extras(scm_run):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        scm_run["run_id"] = [1, 1, 2]
        run_to_nc(scm_run, out_fname, dimensions=("scenario",), extras=("run_id",))

        assert exists(out_fname)

        df = nc_to_run(scm_run.__class__, out_fname)
        assert isinstance(df, scm_run.__class__)

        assert_scmdf_almost_equal(scm_run, df, check_ts_names=False)


def test_nc_to_run_without_dimensions(scm_run):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        scm_run = scm_run.filter(scenario="a_scenario2")
        scm_run["run_id"] = [2]
        run_to_nc(scm_run, out_fname, dimensions=(), extras=("run_id",))

        assert exists(out_fname)

        df = nc_to_run(scm_run.__class__, out_fname)
        assert isinstance(df, scm_run.__class__)

        assert_scmdf_almost_equal(scm_run, df, check_ts_names=False)


def test_nc_to_run_with_extras_non_unique_for_dimension(scm_run):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        scm_run["run_id"] = [1, 2, 1]

        error_msg = "metadata for run_id is not unique for requested dimensions"
        with pytest.raises(ValueError, match=error_msg):
            run_to_nc(scm_run, out_fname, dimensions=("scenario",), extras=("run_id",))


def test_nc_methods(scm_run):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        scm_run.to_nc(out_fname, dimensions=("scenario",))

        assert exists(out_fname)

        # Same as ScmRun.from_nc(out_fname)
        df = scm_run.__class__.from_nc(out_fname)

        assert isinstance(df, scm_run.__class__)
        assert_scmdf_almost_equal(scm_run, df, check_ts_names=False)


@patch("scmdata.netcdf.has_netcdf", False)
def test_no_netcdf(scm_run):
    with pytest.raises(
        ImportError, match="netcdf4 is not installed. Run 'pip install netcdf4'"
    ):
        run_to_nc(scm_run.__class__, "ignored")

    with pytest.raises(
        ImportError, match="netcdf4 is not installed. Run 'pip install netcdf4'"
    ):
        nc_to_run(scm_run, "ignored")


def test_nc_read_failure(scm_run, test_data_path, caplog):
    test_fname = join(
        test_data_path, "netcdf-scm_tas_Amon_bcc-csm1-1_rcp26_r1i1p1_209001-211012.nc"
    )

    with pytest.raises(Exception):
        nc_to_run(scm_run.__class__, test_fname)

    assert caplog.record_tuples[0][0] == "scmdata.netcdf"
    assert caplog.record_tuples[0][1] == logging.ERROR
    assert caplog.record_tuples[0][2] == "Failed reading netcdf file: {}".format(
        test_fname
    )


@pytest.mark.parametrize(
    "mdata",
    (
        {},
        {"test": "value"},
        {
            "test_int": 1,
            "test_float": 1.234,
            "test_array": np.asarray([1, 2, 3]),
            "test_list": [1, 2, 3],
        },
    ),
)
def test_nc_with_metadata(scm_run, mdata):
    def _cmp(a, b):
        if isinstance(a, (list, np.ndarray)):
            return (a == b).all()
        else:
            return a == b

    scm_run.metadata = mdata.copy()
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")

        run_to_nc(scm_run, out_fname, dimensions=("scenario",))

        ds = nc.Dataset(out_fname)

        nc_attrs = {a: ds.getncattr(a) for a in ds.ncattrs()}
        for k, v in mdata.items():
            assert k in nc_attrs
            assert _cmp(nc_attrs[k], v)
        assert "created_at" in nc_attrs

        df = nc_to_run(scm_run.__class__, out_fname)

        for k, v in mdata.items():
            assert k in df.metadata
            assert _cmp(df.metadata[k], v)
        assert "created_at" in df.metadata


@pytest.mark.parametrize(
    "mdata", ({"test_fails": {"something": "else"},}, {"test_fails": {1, 2, 3}},),
)
def test_nc_with_metadata_fails(scm_run, mdata):
    scm_run.metadata = mdata.copy()
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")

        msg = "illegal data type for attribute b'test_fails'"
        with pytest.raises(TypeError, match=msg):
            run_to_nc(scm_run, out_fname, dimensions=("scenario",))


def test_run_to_nc_required_cols_in_extras():
    start = ScmRun(
        np.arange(6).reshape(3, 2),
        index=[2010, 2020, 2030],
        columns={
            "variable": "Surface Temperature",
            "unit": "K",
            "model": ["model_a", "model_b"],
            "scenario": ["scen_a", "scen_b"],
            "region": "World",
        },
    )

    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        start.to_nc(out_fname, dimensions=("scenario",), extras=("model",))

        loaded = ScmRun.from_nc(out_fname)

    assert_scmdf_almost_equal(start, loaded, check_ts_names=False)


def test_error_run_to_nc_required_cols_in_extras_duplicated():
    start = ScmRun(
        np.arange(6).reshape(3, 2),
        index=[2010, 2020, 2030],
        columns={
            "variable": "Surface Temperature",
            "unit": "K",
            "model": ["model_a", "model_b"],
            "scenario": "scen_a",
            "region": "World",
        },
    )

    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        msg = "metadata for model is not unique for requested dimensions"
        with pytest.raises(ValueError, match=msg):
            start.to_nc(out_fname, extras=("model",))


def test_read_legacy_datetimes_nc(scm_run, test_data_path):
    old_datetimes_run = ScmRun.from_nc(join(test_data_path, "legacy_datetimes.nc"))

    assert_scmdf_almost_equal(old_datetimes_run, scm_run, check_ts_names=False)
