import os.path
import re
from glob import glob
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scmdata import ScmRun, run_append
from scmdata.database import ScmDatabase
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
    assert tdb._root_dir == "root_dir"
    assert "root_dir: root_dir" in str(tdb)


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
            {"climate_model": "cm", "variable": "v", "region": "r", "scenario": "s"},
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
def test_get_out_filepath(levels, inp, exp_tail, tdb):
    tdb.levels = levels
    res = tdb._get_out_filepath(**inp)
    exp = os.path.join(tdb._root_dir, exp_tail)

    assert res == exp


@patch("scmdata.database.ensure_dir_exists")
@patch.object(ScmDatabase, "_get_out_filepath")
@patch.object(ScmRun, "to_nc")
def test_save_to_database_single_file(
    mock_to_nc, mock_get_out_filepath, mock_ensure_dir_exists, tdb, start_scmrun
):
    tout_file = "test_out.nc"
    mock_get_out_filepath.return_value = tout_file
    inp_scmrun = start_scmrun.filter(climate_model="cmodel_a")

    tdb._save_to_database_single_file(inp_scmrun)

    mock_get_out_filepath.assert_called_once()
    mock_get_out_filepath.assert_called_with(
        climate_model=inp_scmrun.get_unique_meta("climate_model", no_duplicates=True),
        variable=inp_scmrun.get_unique_meta("variable", no_duplicates=True),
        region=inp_scmrun.get_unique_meta("region", no_duplicates=True),
        scenario=inp_scmrun.get_unique_meta("scenario", no_duplicates=True),
    )

    mock_ensure_dir_exists.assert_called_once()
    mock_ensure_dir_exists.assert_called_with(tout_file)

    mock_to_nc.assert_called_once()
    mock_to_nc.assert_called_with(tout_file, dimensions=[])


@patch("scmdata.database.ensure_dir_exists")
@patch.object(ScmDatabase, "_get_out_filepath")
@patch.object(ScmRun, "to_nc")
def test_save_to_database_single_file_non_unique_meta(
    mock_to_nc, mock_get_out_filepath, mock_ensure_dir_exists, tdb, start_scmrun
):
    tout_file = "test_out.nc"
    mock_get_out_filepath.return_value = tout_file
    inp_scmrun = start_scmrun
    inp_scmrun["ensemble_member"] = [0, 1]
    inp_scmrun["climate_model"] = "cmodel_a"

    tdb._save_to_database_single_file(inp_scmrun)

    mock_get_out_filepath.assert_called_once()
    mock_get_out_filepath.assert_called_with(
        climate_model=inp_scmrun.get_unique_meta("climate_model", no_duplicates=True),
        variable=inp_scmrun.get_unique_meta("variable", no_duplicates=True),
        region=inp_scmrun.get_unique_meta("region", no_duplicates=True),
        scenario=inp_scmrun.get_unique_meta("scenario", no_duplicates=True),
    )

    mock_ensure_dir_exists.assert_called_once()
    mock_ensure_dir_exists.assert_called_with(tout_file)

    mock_to_nc.assert_called_once()
    mock_to_nc.assert_called_with(tout_file, dimensions=["ensemble_member"])


def test_save_to_database_single_file_non_unique_levels(tdb, start_scmrun):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "`climate_model` column is not unique (found values: ['cmodel_a', 'cmodel_b'])"
        ),
    ):
        tdb._save_to_database_single_file(start_scmrun)


@patch.object(ScmDatabase, "_save_to_database_single_file")
def test_database_save(mock_save_to_database_single_file, tdb, start_scmrun):
    tdb.save(start_scmrun)

    expected_calls = len(
        list(
            start_scmrun.groupby(
                ["climate_model", "variable", "region", "scenario", "ensemble_member"]
            )
        )
    )
    assert mock_save_to_database_single_file.call_count == expected_calls


@pytest.mark.parametrize("ch", "!@#$%^&*()~`+={}]<>,;:'\" .")
def test_database_save_weird(tdb, start_scmrun, ch):
    weird_var_name = "variable" + ch
    start_scmrun["variable"] = [weird_var_name, "other"]
    tdb.save(start_scmrun)

    assert len(start_scmrun.filter(variable=weird_var_name))
    assert_scmdf_almost_equal(
        tdb.load(variable=weird_var_name), start_scmrun.filter(variable=weird_var_name)
    )

    replace_ch = "-" if ch not in ".*" else ch
    exp = pd.DataFrame(
        [
            ["cmodel_a", "variable" + replace_ch, "region", "scenario"],
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


def test_database_loaded(tdb_with_data):
    assert os.path.exists(
        os.path.join(
            tdb_with_data._root_dir, "cmodel_a", "variable", "cmodel_a_variable.nc"
        )
    )
    assert os.path.exists(
        os.path.join(
            tdb_with_data._root_dir, "cmodel_b", "variable", "cmodel_b_variable.nc"
        )
    )

    out_names = glob(
        os.path.join(tdb_with_data._root_dir, "**", "*.nc",), recursive=True
    )
    assert len(out_names) == 2


@pytest.mark.parametrize(
    "filter",
    [
        {},
        {"climate_model": "cmodel_a"},
        {"climate_model": "cmodel_b"},
        {"climate_model": "cmodel_a", "variable": "variable"},
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
        os.path.join(tdb_with_data._root_dir, "**", "*.nc",), recursive=True
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
        os.path.join(tdb_with_data._root_dir, "**", "*.nc",), recursive=True
    )
    assert len(out_names) == 2

    tdb_with_data.delete(climate_model="cmodel_a")
    out_names = glob(
        os.path.join(tdb_with_data._root_dir, "**", "*.nc",), recursive=True
    )
    assert len(out_names) == 1
    assert out_names[0].endswith("cmodel_b_variable.nc")


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
