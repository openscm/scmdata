import os.path
import re
import tempfile
from glob import glob
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scmdata import ScmRun, run_append
from scmdata.database import DatabaseBackend, NetCDFBackend, ScmDatabase
from scmdata.errors import NonUniqueMetadataError
from scmdata.testing import assert_scmdf_almost_equal


@pytest.fixture()
def start_scmrun():
    return ScmRun(
        np.arange(6).reshape(3, 2),
        [2010, 2020, 2030],
        columns={
            "scenario": "scenario",
            "model": "model",
            "climate_model": ["cmodel_a", "cmodel_b"],
            "variable": "variable",
            "unit": "unit",
            "region": "region",
            "ensemble_member": 0,
        },
    )


class DummyBackend(DatabaseBackend):
    def __init__(self, **kwargs):
        super(DummyBackend, self).__init__(**kwargs)
        self.keys = {}

    def get(self, filters, ext="*.nc"):
        return self.keys  # not doing any filtering...

    def save(self, sr):
        key = os.path.join(
            *[sr.get_unique_meta(k, True) for k in self.kwargs["levels"]]
        )
        self.keys[key] = sr
        return key

    def load(self, key):
        return self.keys[key]


@pytest.fixture()
def tdb(tmpdir):
    return ScmDatabase(tmpdir)


@pytest.fixture(scope="function")
def tdb_with_data(tmpdir, start_scmrun):
    db = ScmDatabase(tmpdir, levels=["climate_model", "variable"])
    db.save(start_scmrun)

    return db


def test_database_init_and_repr():
    tdb = ScmDatabase("root_dir")
    assert tdb.root_dir == "root_dir"
    assert "root_dir: root_dir" in str(tdb)


@pytest.mark.parametrize("levels", [("scenario",), ("scenario", "model")])
def test_database_passes_config(levels):
    tdb = ScmDatabase("root_dir", levels=levels, backend_config={"test": "example"})
    assert tdb._backend.kwargs["levels"] == levels
    assert tdb._backend.kwargs["test"] == "example"


@pytest.mark.parametrize("cfg_name", ["levels", "root_dir"])
def test_database_invalid_config(cfg_name):
    msg = "backend_config cannot contain key of `{}`".format(cfg_name)
    with pytest.raises(ValueError, match=msg):
        ScmDatabase("root_dir", backend_config={cfg_name: "test"})


def test_database_custom_backend():
    backend = DummyBackend()
    tdb = ScmDatabase("root_dir", backend=backend)

    assert tdb._backend == backend


def test_database_custom_backend_invalid():
    class WrongBackend(dict):
        pass

    backend = WrongBackend()
    msg = "Backend should be an instance of scmdata.database.DatabaseBackend"
    with pytest.raises(ValueError, match=msg):
        ScmDatabase("root_dir", backend=backend)


def test_database_custom_backend_missing():
    msg = "Unknown database backend: other"
    with pytest.raises(ValueError, match=msg):
        ScmDatabase("root_dir", backend="other")


class TestNetCDFBackend:
    @pytest.mark.parametrize(
        "levels,inp,exp_tail",
        (
            (
                ["climate_model", "variable", "region", "scenario"],
                {
                    "climate_model": "cm",
                    "variable": "v",
                    "region": "r",
                    "scenario": "s",
                    "ensemble_member": "em",
                },
                os.path.join("cm", "v", "r", "s", "cm__v__r__s.nc"),
            ),
            (
                ["climate_model", "variable", "region", "scenario"],
                {
                    "climate_model": "cm_a",
                    "variable": "v",
                    "region": "r",
                    "scenario": "s",
                    "ensemble_member": "em",
                },
                os.path.join("cm_a", "v", "r", "s", "cm_a__v__r__s.nc"),
            ),
            (
                ["climate_model", "variable", "region", "scenario", "ensemble_member"],
                {
                    "climate_model": "cm",
                    "variable": "v",
                    "region": "r",
                    "scenario": "s",
                    "ensemble_member": "em",
                },
                os.path.join("cm", "v", "r", "s", "em", "cm__v__r__s__em.nc"),
            ),
            (
                ["climate_model", "ensemble_member"],
                {
                    "climate_model": "cm",
                    "variable": "v",
                    "region": "r",
                    "scenario": "s",
                    "ensemble_member": 1,
                },
                os.path.join("cm", "1", "cm__1.nc"),
            ),
            (
                ["climate_model", "variable", "region", "scenario"],
                {
                    "climate_model": "cm",
                    "variable": "v",
                    "region": "r",
                    "scenario": "s",
                },
                os.path.join("cm", "v", "r", "s", "cm__v__r__s.nc"),
            ),
            (
                ["climate_model", "variable", "region", "scenario", "ensemble_member"],
                {
                    "climate_model": "MAGICC 7.1.0",
                    "variable": "Emissions|CO2",
                    "region": "World|R5.2OECD90",
                    "scenario": "1pctCO2-bgc",
                    "ensemble_member": "001",
                },
                os.path.join(
                    "MAGICC-7.1.0",
                    "Emissions-CO2",
                    "World-R5.2OECD90",
                    "1pctCO2-bgc",
                    "001",
                    "MAGICC-7.1.0__Emissions-CO2__World-R5.2OECD90__1pctCO2-bgc__001.nc",
                ),
            ),
            (
                ["climate_model", "variable", "region", "scenario"],
                {
                    "climate_model": "MAGICC7.1.0",
                    "variable": "Emissions|CO2",
                    "region": "World|R5.2OECD90",
                    "scenario": "1pctCO2-bgc",
                },
                os.path.join(
                    "MAGICC7.1.0",
                    "Emissions-CO2",
                    "World-R5.2OECD90",
                    "1pctCO2-bgc",
                    "MAGICC7.1.0__Emissions-CO2__World-R5.2OECD90__1pctCO2-bgc.nc",
                ),
            ),
        ),
    )
    def test_get_out_filepath(self, levels, inp, exp_tail):
        root_dir = os.path.join(f"{os.sep}tmp", "example")
        backend = NetCDFBackend(levels=levels, root_dir=root_dir)
        res = backend._get_out_filepath(**inp)
        exp = os.path.join(root_dir, exp_tail)

        assert res == exp

    def test_get_out_filepath_not_all_values(self):
        backend = NetCDFBackend(levels=["climate_model"], root_dir="")
        with pytest.raises(KeyError, match=": climate_model"):
            backend._get_out_filepath(other="test")

    def test_netcdf_save_missing_meta(self, tdb, start_scmrun):
        with tempfile.TemporaryDirectory() as tempdir:
            backend = NetCDFBackend(levels=tdb.levels, root_dir=tempdir)

            start_scmrun["variable"] = ["variable_a", "variable_b"]
            run = start_scmrun.drop_meta("climate_model")

            # TODO: make missing key exceptions consistent
            msg = "Level climate_model not found"
            with pytest.raises(KeyError, match=msg):
                backend.save(run)

    def test_netcdf_save_duplicate_meta(self, tdb, start_scmrun):
        with tempfile.TemporaryDirectory() as tempdir:
            backend = NetCDFBackend(levels=("climate_model",), root_dir=tempdir)
            msg = re.escape(
                "`climate_model` column is not unique (found values: ['cmodel_a', 'cmodel_b'])"
            )
            with pytest.raises(ValueError, match=msg):
                backend.save(start_scmrun)

    @patch.object(ScmRun, "to_nc")
    def test_netcdf_save(self, mock_to_nc, tdb, start_scmrun):
        with tempfile.TemporaryDirectory() as tempdir:
            backend = NetCDFBackend(levels=tdb.levels, root_dir=tempdir)

            with patch.object(backend, "_get_out_filepath") as mock_get_out_filepath:
                out_fname = os.path.join(tempdir, "test-level", "out.nc")
                mock_get_out_filepath.return_value = out_fname
                inp_scmrun = start_scmrun.filter(climate_model="cmodel_a")

                backend.save(inp_scmrun)

                mock_get_out_filepath.assert_called_once()
                mock_get_out_filepath.assert_called_with(
                    climate_model=inp_scmrun.get_unique_meta(
                        "climate_model", no_duplicates=True
                    ),
                    variable=inp_scmrun.get_unique_meta("variable", no_duplicates=True),
                    region=inp_scmrun.get_unique_meta("region", no_duplicates=True),
                    scenario=inp_scmrun.get_unique_meta("scenario", no_duplicates=True),
                )

                assert os.path.exists(os.path.dirname(out_fname))
                mock_to_nc.assert_called_once()
                inp_scmrun.to_nc.assert_called_with(out_fname, dimensions=[])

    @patch("scmdata.database.ensure_dir_exists")
    @patch.object(ScmRun, "to_nc")
    def test_netcdf_save_non_unique_meta(
        self, mock_to_nc, mock_ensure_dir_exists, tdb, start_scmrun
    ):
        with tempfile.TemporaryDirectory() as tempdir:
            backend = NetCDFBackend(levels=tdb.levels, root_dir=tempdir)

            with patch.object(backend, "_get_out_filepath") as mock_get_out_filepath:
                out_fname = os.path.join(tempdir, "test-level", "out.nc")
                mock_get_out_filepath.return_value = out_fname

                inp_scmrun = start_scmrun
                inp_scmrun["ensemble_member"] = [0, 1]
                inp_scmrun["climate_model"] = "cmodel_a"

                backend.save(inp_scmrun)

                mock_get_out_filepath.assert_called_once()
                mock_get_out_filepath.assert_called_with(
                    climate_model=inp_scmrun.get_unique_meta(
                        "climate_model", no_duplicates=True
                    ),
                    variable=inp_scmrun.get_unique_meta("variable", no_duplicates=True),
                    region=inp_scmrun.get_unique_meta("region", no_duplicates=True),
                    scenario=inp_scmrun.get_unique_meta("scenario", no_duplicates=True),
                )

                mock_ensure_dir_exists.assert_called_once()
                mock_ensure_dir_exists.assert_called_with(out_fname)

                mock_to_nc.assert_called_once()
                mock_to_nc.assert_called_with(out_fname, dimensions=["ensemble_member"])


def test_save_to_database_single_file_non_unique_levels(tdb, start_scmrun):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "`climate_model` column is not unique (found values: ['cmodel_a', 'cmodel_b'])"
        ),
    ):
        tdb._backend.save(start_scmrun)


def test_database_save(tdb, start_scmrun):
    tdb._backend = DummyBackend(levels=tdb.levels)
    tdb.save(start_scmrun)

    expected_calls = len(
        list(
            start_scmrun.groupby(
                ["climate_model", "variable", "region", "scenario", "ensemble_member"]
            )
        )
    )
    assert len(tdb._backend.keys) == expected_calls


@pytest.mark.parametrize("ch", "!@#$%^&*()~`+={}]<>,;:'\" .")
def test_database_save_weird(tdb, start_scmrun, ch):
    weird_var_name = "variable" + ch + "test"
    start_scmrun["variable"] = [weird_var_name, "other"]
    tdb.save(start_scmrun)

    assert len(start_scmrun.filter(variable=weird_var_name))
    assert_scmdf_almost_equal(
        tdb.load(variable=weird_var_name), start_scmrun.filter(variable=weird_var_name)
    )

    replace_ch = "-" if ch not in "." else ch
    exp = pd.DataFrame(
        [
            ["cmodel_a", "variable" + replace_ch + "test", "region", "scenario"],
            ["cmodel_b", "other", "region", "scenario"],
        ],
        columns=tdb.levels,
    )

    pd.testing.assert_frame_equal(tdb.available_data(), exp)


# TODO: Extend to use "weird" characters at both start and end of name once #188 is resolved
@pytest.mark.parametrize("ch", "!@#$%^&*()~`+={}]<>,;:'\" .")
def test_database_save_weird_end(tdb, start_scmrun, ch):
    replace_ch = "-"
    weird_var_name = "variable" + ch
    expected_var_name = "variable" + replace_ch

    start_scmrun["variable"] = [weird_var_name, "other"]

    # Edge case to support windows
    if ch == ".":
        with pytest.raises(ValueError, match="Metadata cannot end in a '.'"):
            tdb.save(start_scmrun)
        return

    tdb.save(start_scmrun)

    assert len(start_scmrun.filter(variable=weird_var_name))
    assert_scmdf_almost_equal(
        tdb.load(variable=weird_var_name), start_scmrun.filter(variable=weird_var_name)
    )

    exp = pd.DataFrame(
        [
            ["cmodel_a", expected_var_name, "region", "scenario"],
            ["cmodel_b", "other", "region", "scenario"],
        ],
        columns=tdb.levels,
    )

    pd.testing.assert_frame_equal(tdb.available_data(), exp)


@pytest.mark.parametrize("ch", "[?/")
def test_database_save_weird_unsupported(tdb, start_scmrun, ch):
    weird_var_name = "variable" + ch
    start_scmrun["variable"] = [weird_var_name, "other"]
    with pytest.raises(Exception):
        tdb.save(start_scmrun)


# / cannot be in variable metadata, but work otherwise
def test_database_save_weird_slash(tdb, start_scmrun):
    weird_name = "cmodel/test"
    start_scmrun["climate_model"] = [weird_name, "other"]

    tdb.save(start_scmrun)

    assert len(start_scmrun.filter(climate_model=weird_name))
    assert_scmdf_almost_equal(
        tdb.load(climate_model=weird_name),
        start_scmrun.filter(climate_model=weird_name),
    )


def test_database_loaded(tdb_with_data):
    assert os.path.exists(
        os.path.join(
            tdb_with_data._root_dir, "cmodel_a", "variable", "cmodel_a__variable.nc"
        )
    )
    assert os.path.exists(
        os.path.join(
            tdb_with_data._root_dir, "cmodel_b", "variable", "cmodel_b__variable.nc"
        )
    )

    out_names = glob(
        os.path.join(
            tdb_with_data._root_dir,
            "**",
            "*.nc",
        ),
        recursive=True,
    )
    assert len(out_names) == 2


@pytest.mark.parametrize(
    "filter",
    [
        {},
        {"climate_model": "cmodel_a"},
        {"climate_model": "cmodel_b"},
        {"climate_model": "cmodel_a", "variable": "variable"},
        {"climate_model": ["cmodel_a", "cmodel_c"]},
    ],
)
def test_database_load_data(tdb_with_data, start_scmrun, filter):
    loaded_ts = tdb_with_data.load(**filter)
    assert_scmdf_almost_equal(
        loaded_ts, start_scmrun.filter(**filter), check_ts_names=False
    )


def test_database_load_data_extras(tdb_with_data):
    with pytest.raises(ValueError, match="Unknown level: extra"):
        tdb_with_data.load(extra="other")


@pytest.mark.parametrize(
    "filter",
    [
        {"variable": "other"},
        {"climate_model": "cmodel_c"},
        {"climate_model": "cmodel_a", "variable": "other"},
    ],
)
def test_database_load_data_missing(tdb_with_data, filter):
    with pytest.raises(ValueError, match="No runs to append"):
        tdb_with_data.load(**filter)


def test_database_overwriting(tdb_with_data, start_scmrun):
    start_scmrun_2 = start_scmrun.copy()
    start_scmrun_2["ensemble_member"] = 1

    # The target file will already exist so should merge files
    tdb_with_data.save(start_scmrun_2)

    out_names = glob(
        os.path.join(
            tdb_with_data._root_dir,
            "**",
            "*.nc",
        ),
        recursive=True,
    )
    assert len(out_names) == 2

    loaded_ts = tdb_with_data.load(climate_model="cmodel_a")
    assert_scmdf_almost_equal(
        loaded_ts,
        run_append(
            [
                start_scmrun.filter(climate_model="cmodel_a"),
                start_scmrun_2.filter(climate_model="cmodel_a"),
            ]
        ),
        check_ts_names=False,
    )

    loaded_ts = tdb_with_data.load()
    assert_scmdf_almost_equal(
        loaded_ts, run_append([start_scmrun, start_scmrun_2]), check_ts_names=False
    )


def test_database_save_duplicates(tdb_with_data, start_scmrun):
    with pytest.raises(NonUniqueMetadataError):
        tdb_with_data.save(start_scmrun.filter(climate_model="cmodel_a"))


def test_database_delete(tdb_with_data):
    out_names = glob(
        os.path.join(
            tdb_with_data._root_dir,
            "**",
            "*.nc",
        ),
        recursive=True,
    )
    assert len(out_names) == 2

    tdb_with_data.delete(climate_model="cmodel_a")
    out_names = glob(
        os.path.join(
            tdb_with_data._root_dir,
            "**",
            "*.nc",
        ),
        recursive=True,
    )
    assert len(out_names) == 1
    assert out_names[0].endswith("cmodel_b__variable.nc")


def test_database_delete_unknown(tdb_with_data):
    with pytest.raises(ValueError, match="Unknown level: extra"):
        tdb_with_data.delete(extra="other")


def test_database_available_data(tdb_with_data, start_scmrun):
    res = tdb_with_data.available_data()
    exp = pd.DataFrame(
        [["cmodel_a", "variable"], ["cmodel_b", "variable"]],
        columns=tdb_with_data.levels,
    )

    pd.testing.assert_frame_equal(res, exp)

    start_scmrun["variable"] = "a_variable"
    tdb_with_data.save(start_scmrun)

    exp = pd.DataFrame(
        [
            ["cmodel_a", "a_variable"],
            ["cmodel_a", "variable"],
            ["cmodel_b", "a_variable"],
            ["cmodel_b", "variable"],
        ],
        columns=tdb_with_data.levels,
    )
    pd.testing.assert_frame_equal(tdb_with_data.available_data(), exp)

    start_scmrun["variable"] = "z_variable"
    tdb_with_data.save(start_scmrun)

    exp = pd.DataFrame(
        [
            ["cmodel_a", "a_variable"],
            ["cmodel_a", "variable"],
            ["cmodel_a", "z_variable"],
            ["cmodel_b", "a_variable"],
            ["cmodel_b", "variable"],
            ["cmodel_b", "z_variable"],
        ],
        columns=tdb_with_data.levels,
    )
    pd.testing.assert_frame_equal(tdb_with_data.available_data(), exp)
