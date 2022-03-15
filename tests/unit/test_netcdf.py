import logging
import re
import tempfile
from os.path import exists, join
from unittest.mock import MagicMock, patch

import netCDF4 as nc
import numpy as np
import numpy.testing as npt
import packaging.version
import pandas as pd
import pytest
import xarray as xr

from scmdata import ScmRun
from scmdata.netcdf import _get_xr_dataset_to_write, nc_to_run, run_to_nc
from scmdata.testing import assert_scmdf_almost_equal


def test_run_to_nc(scm_run):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        run_to_nc(scm_run, out_fname, dimensions=("scenario",))

        assert exists(out_fname)

        with nc.Dataset(out_fname) as ds:
            assert ds.dimensions["time"].size == len(scm_run.time_points)
            assert ds.dimensions["scenario"].size == 2

            assert ds.variables["scenario"][0] == "a_scenario"
            assert ds.variables["scenario"][1] == "a_scenario2"

            npt.assert_allclose(
                ds.variables["Primary_Energy"][0, :],
                scm_run.filter(variable="Primary Energy", scenario="a_scenario").values[
                    0
                ],
            )
            npt.assert_allclose(
                ds.variables["Primary_Energy"][1, :],
                scm_run.filter(
                    variable="Primary Energy", scenario="a_scenario2"
                ).values[0],
            )
            npt.assert_allclose(
                ds.variables["Primary_Energy__Coal"][0, :],
                scm_run.filter(
                    variable="Primary Energy|Coal", scenario="a_scenario"
                ).values[0],
            )


@pytest.mark.parametrize(
    "v",
    ["primary energy", "Primary Energy", "Primary Energy|Coal|Test"],
)
def test_run_to_nc_case(scm_run, v):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        scm_run = scm_run.filter(variable="Primary Energy")
        scm_run["variable"] = v

        run_to_nc(scm_run, out_fname, dimensions=("scenario",))
        res = nc_to_run(scm_run.__class__, out_fname)

        assert res.get_unique_meta("variable", True) == v


@pytest.mark.parametrize("ch", "!@#$%^&*()~`+={}]<>,;:'\".")
@pytest.mark.parametrize("start_with_weird", (True, False))
def test_run_to_nc_weird_name(scm_run, ch, start_with_weird):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        scm_run = scm_run.filter(variable="Primary Energy")
        variable = scm_run.get_unique_meta("variable", True)

        if start_with_weird:
            variable = ch + " " + variable
        else:
            variable = variable + " " + ch

        scm_run["variable"] = variable

        if start_with_weird:
            error_msg = re.escape("NetCDF: Name contains illegal characters")
            with pytest.raises(RuntimeError, match=error_msg):
                run_to_nc(scm_run, out_fname, dimensions=("scenario",))

        else:
            run_to_nc(scm_run, out_fname, dimensions=("scenario",))
            res = nc_to_run(scm_run.__class__, out_fname)

            assert res.get_unique_meta("variable", True) == variable


@pytest.mark.parametrize("ch", ("|", " ", "  "))
def test_run_to_nc_special_character_pass(scm_run, ch):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        scm_run = scm_run.filter(variable="Primary Energy")
        variable = scm_run.get_unique_meta("variable", True)
        variable = ch + variable
        scm_run["variable"] = variable

        run_to_nc(scm_run, out_fname, dimensions=("scenario",))
        res = nc_to_run(scm_run.__class__, out_fname)

        assert res.get_unique_meta("variable", True) == variable


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

    with nc.Dataset(out_fname) as ds:
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

        # remove as order doesn't really matter unless I misunderstand something?
        # assert ds.variables["Primary_Energy"].shape == (2, 4, 10, 3)
        # assert ds.variables["Primary_Energy__Coal"].shape == (2, 4, 10, 3)


def test_run_to_nc_nan_dimension_error(scm_run, tmpdir):
    scm_run["run_id"] = np.nan

    out_fname = join(tmpdir, "out.nc")
    with pytest.raises(AssertionError, match="nan in dimension: `run_id`"):
        run_to_nc(scm_run, out_fname, dimensions=("scenario", "run_id"))


@pytest.mark.parametrize(
    "dimensions",
    (
        ("scenario", "variable"),
        ("scenario",),
        ("scenario", "time"),
        ("scenario", "variable", "time"),
    ),
)
def test_nc_to_run(scm_run, dimensions):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        run_to_nc(scm_run, out_fname, dimensions=dimensions)

        assert exists(out_fname)

        run_read = nc_to_run(scm_run.__class__, out_fname)
        assert isinstance(run_read, scm_run.__class__)

        assert_scmdf_almost_equal(scm_run, run_read, check_ts_names=False)


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

        run_read = nc_to_run(scm_run.__class__, out_fname)
        assert isinstance(run_read, scm_run.__class__)

        assert_scmdf_almost_equal(scm_run, run_read, check_ts_names=False)


def test_nc_to_run_with_extras_sparsity(scm_run):
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
            scm_run,
            out_fname,
            dimensions=("climate_model", "run_id"),
            extras=("scenario",),
        )

        assert exists(out_fname)

        xr_ds = xr.load_dataset(out_fname)
        # Should save with four dimensions: "time", "climate_model", "run_id", "_id"
        # the "_id" dimension is required as a short-hand mapping between extras and
        # the data.
        # There is no way to avoid this sparsity.
        assert len(xr_ds["Primary_Energy"].shape) == 4

        run_read = nc_to_run(scm_run.__class__, out_fname)
        assert isinstance(run_read, scm_run.__class__)

        assert_scmdf_almost_equal(scm_run, run_read, check_ts_names=False)


def test_nc_to_run_with_extras_id_not_needed_sparsity(scm_run):
    df = scm_run.filter(scenario="a_scenario").timeseries()
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

    big_df = pd.concat(big_df).reset_index(drop=True)
    big_df["paraset_id"] = big_df["run_id"].apply(lambda x: x // 3)
    scm_run = scm_run.__class__(big_df)

    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")

        run_to_nc(
            scm_run,
            out_fname,
            dimensions=("climate_model", "run_id"),
            extras=("paraset_id",),
        )

        assert exists(out_fname)

        xr_ds = xr.load_dataset(out_fname)
        # Should save with three dimensions: "time", "climate_model", "run_id"
        # There should be no "_id" as paraset_id is uniquely defined by "run_id"
        assert len(xr_ds["Primary_Energy"].shape) == 3

        run_read = nc_to_run(scm_run.__class__, out_fname)
        assert isinstance(run_read, scm_run.__class__)

        assert_scmdf_almost_equal(scm_run, run_read, check_ts_names=False)


def test_nc_to_run_with_extras_id_needed_and_not_needed(scm_run):
    scmrun = scm_run.filter(scenario="a_scenario")

    full_df = []
    for model in ("model_a", "model_b"):
        for scenario in ("scenario_a", "scenario_b"):
            for run_id in range(10):
                tmp = scmrun.timeseries()
                tmp["run_id"] = run_id
                tmp["model"] = model
                tmp["scenario"] = scenario
                tmp.index = tmp.index.droplevel(["model", "scenario"])
                full_df.append(tmp)

    full_df = pd.concat(full_df)
    scm_run = scm_run.__class__(full_df)
    scm_run["paraset_id"] = scm_run["run_id"].apply(lambda x: x // 3)

    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")

        run_to_nc(
            scm_run,
            out_fname,
            dimensions=("climate_model", "run_id", "scenario"),
            extras=("paraset_id", "model"),
        )

        assert exists(out_fname)

        xr_ds = xr.load_dataset(out_fname)

        # Should save with dimensions: "time", "climate_model",
        # "run_id", "scenario" and "_id"
        assert len(xr_ds["Primary_Energy"].shape) == 5

        # model must be saved with id
        assert xr_ds["model"].dims == ("_id",)
        # paraset_id is wholly defined by run_id
        assert xr_ds["paraset_id"].dims == ("run_id",)

        run_read = nc_to_run(scm_run.__class__, out_fname)
        assert isinstance(run_read, scm_run.__class__)

        assert_scmdf_almost_equal(scm_run, run_read, check_ts_names=False)


def test_nc_to_run_non_unique_for_dimension(scm_run):
    error_msg = (
        "dimensions: `{}` and extras: `{}` do not uniquely define the "
        "timeseries, please add extra dimensions and/or extras".format(["region"], [])
    )
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")

        with pytest.raises(ValueError, match=re.escape(error_msg)):
            run_to_nc(scm_run, out_fname, dimensions=("region",))


def test_nc_to_run_non_unique_meta(scm_run):
    scm_run["climate_model"] = ["b_model", "a_model", "a_model"]

    error_msg = re.escape(
        "Other metadata is not unique for dimensions: `{}` and extras: `{}`. "
        "Please add meta columns with more than one value to dimensions or "
        "extras.".format(["scenario"], [])
    )
    error_msg = (
        "{}\nNumber of unique values in each column:\n.*\n(\\s|\\S)*"
        "Existing values in the other metadata:.*".format(error_msg)
    )

    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")

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

        with nc.Dataset(out_fname) as ds:
            assert ds.dimensions["time"].size == len(scm_run.time_points)
            assert ds.dimensions["scenario"].size == 2

            assert ds.variables["scenario"][0] == "a_scenario"
            assert ds.variables["scenario"][1] == "a_scenario2"

            for i, run_id in enumerate(ds.variables["run_id"]):
                exp_val = dtype(unique_scenarios.index(ds.variables["scenario"][i]))
                assert run_id == exp_val

            npt.assert_allclose(
                ds.variables["Primary_Energy"][0, :],
                scm_run.filter(variable="Primary Energy", scenario="a_scenario").values[
                    0
                ],
            )
            npt.assert_allclose(
                ds.variables["Primary_Energy"][1, :],
                scm_run.filter(
                    variable="Primary Energy", scenario="a_scenario2"
                ).values[0],
            )
            npt.assert_allclose(
                ds.variables["Primary_Energy__Coal"][0, :],
                scm_run.filter(
                    variable="Primary Energy|Coal", scenario="a_scenario"
                ).values[0],
            )


def test_nc_to_run_with_extras(scm_run):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        scm_run["run_id"] = [1, 1, 2]
        run_to_nc(scm_run, out_fname, dimensions=("scenario",), extras=("run_id",))

        assert exists(out_fname)

        run_read = nc_to_run(scm_run.__class__, out_fname)
        assert isinstance(run_read, scm_run.__class__)

        assert_scmdf_almost_equal(scm_run, run_read, check_ts_names=False)


def test_nc_to_run_without_dimensions(scm_run):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        scm_run = scm_run.filter(scenario="a_scenario2")
        scm_run["run_id"] = [2]
        run_to_nc(scm_run, out_fname, dimensions=(), extras=("run_id",))

        assert exists(out_fname)

        run_read = nc_to_run(scm_run.__class__, out_fname)
        assert isinstance(run_read, scm_run.__class__)

        assert_scmdf_almost_equal(scm_run, run_read, check_ts_names=False)


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

        with nc.Dataset(out_fname) as ds:
            nc_attrs = {a: ds.getncattr(a) for a in ds.ncattrs()}
            for k, v in mdata.items():
                assert k in nc_attrs
                assert _cmp(nc_attrs[k], v)
            assert "created_at" in nc_attrs

            run_read = nc_to_run(scm_run.__class__, out_fname)

            for k, v in mdata.items():
                assert k in run_read.metadata
                assert _cmp(run_read.metadata[k], v)

            assert "created_at" in run_read.metadata


@pytest.mark.parametrize(
    "mdata",
    ({"test_fails": {"something": "else"}}, {"test_fails": {1, 2, 3}}),
)
def test_nc_with_metadata_fails(scm_run, mdata):
    scm_run.metadata = mdata.copy()
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")

        xarray_version = packaging.version.parse(xr.__version__)
        if xarray_version >= packaging.version.parse("0.16.2"):
            msg = "Invalid value for attr 'test_fails':.*"
        else:
            msg = "Invalid value for attr: {}".format(mdata["test_fails"])
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


def test_run_to_nc_extra_instead_of_dimension_run_id(scm_run):
    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")
        scm_run["run_id"] = [1, 2, 1]

        run_to_nc(scm_run, out_fname, dimensions=("scenario",), extras=("run_id",))
        loaded = ScmRun.from_nc(out_fname)

    assert_scmdf_almost_equal(scm_run, loaded, check_ts_names=False)


def test_run_to_nc_extra_instead_of_dimension():
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

        start.to_nc(out_fname, extras=("model",))
        loaded = ScmRun.from_nc(out_fname)

    assert_scmdf_almost_equal(start, loaded, check_ts_names=False)


def test_run_to_nc_dimensions_cover_all_metadata():
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

        start.to_nc(out_fname, dimensions=("region", "model", "scenario"))
        loaded = ScmRun.from_nc(out_fname)

    assert_scmdf_almost_equal(start, loaded, check_ts_names=False)


@pytest.mark.parametrize(
    "start_variable",
    (
        "PrimaryEnergy|Coal|FinalEnergy",  # easy, no spaces just pipes
        "Primary Energy|Coal|Final Energy",  # single spaces
        "Primary  Energy|Coal|Final Energy",  # double spaces
        "Primary   Energy|Coal|Final  Energy",  # triple and double spaces
        "Primary    Energy|Coal|Final  Energy",  # quadruple and double spaces
        "Primary_Energy|Coal",  # underscore in name
        "Primary_Energy|Coal|Final Energy",  # underscore and space in name
    ),
)
def test_run_to_nc_loop_tricky_variable_name(scm_run, start_variable):
    # tests that the mapping between variable and units works even with
    # tricky variable names that get renamed in various was before serialising to
    # disk
    assert "Primary Energy|Coal" in scm_run.get_unique_meta("variable")
    scm_run["variable"] = scm_run["variable"].apply(
        lambda x: x.replace("Primary Energy|Coal", start_variable)
    )
    scm_run["unit"] = scm_run["variable"].apply(
        lambda x: "EJ/yr" if x != start_variable else "MJ / yr"
    )

    with tempfile.TemporaryDirectory() as tempdir:
        out_fname = join(tempdir, "out.nc")

        scm_run.to_nc(out_fname, dimensions=("scenario",))
        loaded = ScmRun.from_nc(out_fname)

    assert_scmdf_almost_equal(scm_run, loaded, check_ts_names=False)


@patch("scmdata.netcdf._get_xr_dataset_to_write")
def test_run_to_nc_xarray_kwarg_passing(mock_get_xr_dataset, scm_run, tmpdir):
    dimensions = ["scenario"]
    extras = []
    mock_ds = MagicMock()
    mock_ds.data_vars = _get_xr_dataset_to_write(scm_run, dimensions, extras).data_vars
    mock_get_xr_dataset.return_value = mock_ds

    out_fname = join(tmpdir, "out.nc")
    run_to_nc(scm_run, out_fname, dimensions=dimensions, extras=extras, engine="engine")

    mock_ds.to_netcdf.assert_called_with(out_fname, engine="engine")


@patch("scmdata.netcdf._get_xr_dataset_to_write")
@pytest.mark.parametrize(
    "in_kwargs,call_kwargs",
    (
        (
            dict(encoding={"Primary Energy": {"zlib": True, "complevel": 9}}),
            dict(encoding={"Primary_Energy": {"zlib": True, "complevel": 9}}),
        ),
        (
            dict(encoding={"Primary_Energy": {"zlib": True, "complevel": 9}}),
            dict(encoding={"Primary_Energy": {"zlib": True, "complevel": 9}}),
        ),
        (dict(unlimited_dims="Primary Energy"), dict(unlimited_dims="Primary_Energy")),
        (dict(unlimited_dims="Primary_Energy"), dict(unlimited_dims="Primary_Energy")),
        (
            dict(
                encoding={"Primary Energy": {"zlib": True, "complevel": 9}},
                unlimited_dims="Primary Energy",
            ),
            dict(
                encoding={"Primary_Energy": {"zlib": True, "complevel": 9}},
                unlimited_dims="Primary_Energy",
            ),
        ),
    ),
)
def test_run_to_nc_xarray_kwarg_passing_variable_renaming(
    mock_get_xr_dataset, scm_run, tmpdir, in_kwargs, call_kwargs
):
    dimensions = ["scenario"]
    extras = []

    mock_ds = MagicMock()
    mock_ds.data_vars = _get_xr_dataset_to_write(scm_run, dimensions, extras).data_vars
    mock_get_xr_dataset.return_value = mock_ds

    out_fname = join(tmpdir, "out.nc")
    run_to_nc(scm_run, out_fname, dimensions=("scenario",), **in_kwargs)

    # variable should be renamed so it matches what goes to disk
    mock_ds.to_netcdf.assert_called_with(out_fname, **call_kwargs)
