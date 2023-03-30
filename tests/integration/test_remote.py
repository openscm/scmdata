import pandas as pd

import scmdata
from scmdata.remote import (
    RemoteDataset,
    _read_api_facets,
    _read_api_meta,
    _read_api_timeseries,
)

NDCS_URL = "https://api.climateresource.com.au/ndcs/v1"


def test_read_ndcs():
    res = _read_api_timeseries(
        NDCS_URL,
        **{
            "version": "14Feb2022b_CR",
            "variable": "Emissions|Total *",
            "hot_air": "exclude",
            "category": "Current",
        },
    )

    assert len(res)
    assert isinstance(res, scmdata.ScmRun)


def test_read_facets():
    res = _read_api_facets(
        NDCS_URL,
        **{
            "version": "14Feb2022b_CR",
            "variable": "Emissions|Total *",
            "hot_air": "exclude",
            "category": "Current",
        },
    )

    assert (res.columns == ["name", "value", "count"]).all()
    assert len(res)


def test_read_meta():
    res = _read_api_meta(
        NDCS_URL,
        version="14Feb2022b_CR",
        variable="Emissions|Total *",
    )

    assert isinstance(res, pd.DataFrame)
    assert "category" in res
    assert "Emissions|Total GHG excl. LULUCF" in res["variable"].tolist()


def test_remote_dataset_real():
    ds = RemoteDataset(NDCS_URL)

    assert "USA" in ds.get_unique_meta("region")

    ds = ds.filter(region="AUS")
    assert "USA" not in ds.get_unique_meta("region")
    assert "AUS" in ds.get_unique_meta("region")
    ds_meta = ds.meta()

    run = ds.query()
    assert isinstance(run, scmdata.ScmRun)
    pd.testing.assert_frame_equal(run.meta, ds_meta)

    # We should be able to use other ScmRun funcs
    res = ds.process_over("variable", "mean")
    assert isinstance(res, pd.DataFrame)
