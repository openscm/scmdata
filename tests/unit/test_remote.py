import os
import pandas as pd
import pytest

import scmdata
from scmdata.remote import (
    _read_api_facets,
    _read_api_timeseries,
    _read_api_meta,
    RemoteDataset,
)
from scmdata.errors import NonUniqueMetadataError, RemoteQueryError

NDCS_URL = "https://api.climateresource.com.au/ndcs/v1"


class MockRemoteDataset(RemoteDataset):
    # replaces remote queries with static dataset
    _data_queries = []
    _meta_queries = []
    _side_effect = None

    def _clear(self):
        MockRemoteDataset._data_queries.clear()
        MockRemoteDataset._meta_queries.clear()

    def _get_data(self, filters):
        from conftest import TEST_DATA

        if self._side_effect:
            raise self._side_effect

        fname = os.path.join(TEST_DATA, "sr15", "sr15-output.csv")
        return scmdata.ScmRun(fname).filter(**filters)

    def query(self) -> scmdata.ScmRun:
        MockRemoteDataset._data_queries.append(self.filters)
        return self._get_data(self.filters)

    def meta(self) -> pd.DataFrame:
        MockRemoteDataset._meta_queries.append(self.filters)
        return self._get_data(self.filters).meta

    def _read_api_info(self):
        self._meta_cols = self.meta().columns.tolist()


def test_remote_dataset_filtering():
    ds = MockRemoteDataset(NDCS_URL)

    filtered_ds = ds.filter(variable="Population")
    assert filtered_ds.filters == {"variable": "Population"}

    # returns a new object
    assert id(filtered_ds) != id(ds)

    # Can also filter on creation
    ds = MockRemoteDataset(NDCS_URL, {"variable": "Population"})
    assert ds.filters == {"variable": "Population"}


def test_remote_query():
    ds = MockRemoteDataset(NDCS_URL)
    ds._clear()

    res = ds.query()
    res_filtered = ds.filter(variable="Emissions|CO2").query()

    assert isinstance(res, scmdata.ScmRun)
    assert isinstance(res_filtered, scmdata.ScmRun)
    assert MockRemoteDataset._data_queries == [{}, {"variable": "Emissions|CO2"}]


def test_remote_query_with_extras():
    ds = MockRemoteDataset(NDCS_URL)
    ds._clear()

    res = ds.query()
    res_filtered = ds.filter(variable="Emissions|CO2").query()

    assert isinstance(res, scmdata.ScmRun)
    assert isinstance(res_filtered, scmdata.ScmRun)
    assert MockRemoteDataset._data_queries == [{}, {"variable": "Emissions|CO2"}]


def test_remote_get_unique_meta():
    ds = MockRemoteDataset(NDCS_URL)

    variables = ds.get_unique_meta("variable")
    assert isinstance(variables, list)
    assert len(variables)

    with pytest.raises(KeyError):
        ds.get_unique_meta("unknown")

    with pytest.raises(ValueError):
        ds.get_unique_meta("variable", True)

    single = ds.filter(variable="Temperature|Global Mean").get_unique_meta("unit", True)
    assert single == "°C"

    ds._side_effect = RemoteQueryError("Something went wrong", "opps")
    with pytest.raises(RemoteQueryError, match="Something went wrong: opps"):
        ds.get_unique_meta("variable")
