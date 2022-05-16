import os
import pandas as pd

import scmdata
from scmdata.remote import (
    read_api_facets,
    read_api_timeseries,
    read_api_meta,
    RemoteDataset,
)


NDCS_URL = "https://api.climateresource.com.au/ndcs/v1"


def test_read_ndcs():
    res = read_api_timeseries(
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
    res = read_api_facets(
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
    res = read_api_meta(
        NDCS_URL,
        version="14Feb2022b_CR",
        variable="Emissions|Total *",
    )

    assert isinstance(res, pd.DataFrame)
    assert "category" in res
    assert "Emissions|Total GHG excl. LULUCF" in res["variable"].tolist()


class MockRemoteDataset(RemoteDataset):
    # replaces remote queries with static dataset

    def __init__(self, *args, **kwargs):
        super(MockRemoteDataset, self).__init__(*args, **kwargs)
        self._data_queries = []
        self._meta_queries = []

    def _get_data(self):
        from conftest import TEST_DATA

        fname = os.path.join(TEST_DATA, "sr15", "sr15-output.csv")
        return scmdata.ScmRun(fname).filter(**self.filters)

    def query(self) -> scmdata.ScmRun:
        self._data_queries.append(self.filters)
        return self._get_data()

    def meta(self) -> pd.DataFrame:
        self._meta_queries.append(self.filters)
        return self._get_data().meta


def test_remote_dataset_filtering():
    ds = MockRemoteDataset(NDCS_URL)

    filtered_ds = ds.filter(variable="Population")
    assert filtered_ds.filters == {"variable": "Population"}

    # returns a new object
    assert id(filtered_ds) != id(ds)

    # Can also filter on creation
    ds = MockRemoteDataset(NDCS_URL, {"variable": "Population"})
    assert ds.filters == {"variable": "Population"}


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
    # ds.lineplot()
