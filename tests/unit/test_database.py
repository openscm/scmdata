import os.path
import re
from unittest.mock import patch

import numpy as np
import pytest
from scmdata import ScmRun

from scmdata.database import SCMDatabase

MOCK_ROOT_DIR_NAME = os.path.join("/mock", "root", "dir")


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
def tdb():
    return SCMDatabase(MOCK_ROOT_DIR_NAME)


def test_database_init_and_repr():
    tdb = SCMDatabase("root_dir")
    assert tdb._root_dir == "root_dir"
    assert "root_dir: root_dir" in str(tdb)


@pytest.mark.parametrize(
    "inp,exp_tail",
    (
        (
            {
                "climate_model": "cm",
                "variable": "v",
                "region": "r",
                "scenario": "s",
                "ensemble_member": "em",
            },
            os.path.join("cm", "v", "r", "s", "cm_v_r_s_em.nc"),
        ),
        (
            {"climate_model": "cm", "variable": "v", "region": "r", "scenario": "s"},
            os.path.join("cm", "v", "r", "s", "cm_v_r_s.nc"),
        ),
        (
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
                "MAGICC-7.1.0_Emissions-CO2_World-R5.2OECD90_1pctCO2-bgc_001.nc",
            ),
        ),
        (
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
                "MAGICC7.1.0_Emissions-CO2_World-R5.2OECD90_1pctCO2-bgc.nc",
            ),
        ),
    ),
)
def test_get_out_filepath(inp, exp_tail, tdb):
    res = tdb.get_out_filepath(**inp)
    exp = os.path.join(tdb._root_dir, exp_tail)

    assert res == exp


@patch("scmdata.database.ensure_dir_exists")
@patch.object(SCMDatabase, "get_out_filepath")
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
        inp_scmrun.get_unique_meta("climate_model", no_duplicates=True),
        inp_scmrun.get_unique_meta("variable", no_duplicates=True),
        inp_scmrun.get_unique_meta("region", no_duplicates=True),
        inp_scmrun.get_unique_meta("scenario", no_duplicates=True),
        inp_scmrun.get_unique_meta("ensemble_member", no_duplicates=True),
    )

    mock_ensure_dir_exists.assert_called_once()
    mock_ensure_dir_exists.assert_called_with(tout_file)

    mock_to_nc.assert_called_once()
    mock_to_nc.assert_called_with(tout_file)


def test_save_to_database_single_file_non_unique_meta(tdb, start_scmrun):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "`climate_model` column is not unique (found values: ['cmodel_a', 'cmodel_b'])"
        ),
    ):
        tdb._save_to_database_single_file(start_scmrun)


def test_save_to_database_single_file_no_ensemble_member(tdb, start_scmrun):
    with pytest.raises(KeyError, match=re.escape("Level ensemble_member not found")):
        tdb._save_to_database_single_file(
            start_scmrun.filter(climate_model="cmodel_a").drop_meta(
                "ensemble_member", inplace=False
            )
        )


@patch.object(SCMDatabase, "_save_to_database_single_file")
def test_save_to_database(mock_save_to_database_single_file, tdb, start_scmrun):
    tdb.save_to_database(start_scmrun)

    expected_calls = len(
        list(
            start_scmrun.groupby(
                ["climate_model", "variable", "region", "scenario", "ensemble_member"]
            )
        )
    )
    assert mock_save_to_database_single_file.call_count == expected_calls
