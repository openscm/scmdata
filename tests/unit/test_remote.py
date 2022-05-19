import logging
import os
import pandas as pd
import pytest
import re

import scmdata
from scmdata.remote import (
    _read_api_facets,
    _read_api_timeseries,
    _read_api_meta,
    RemoteDataset,
)
from unittest.mock import patch, Mock
from scmdata.errors import NonUniqueMetadataError, RemoteQueryError

NDCS_URL = "https://api.climateresource.com.au/ndcs/v1/"


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


@pytest.fixture()
def remote_ds():
    ds = MockRemoteDataset(NDCS_URL)
    ds._clear()

    return ds


def test_remote_dataset_filtering(remote_ds):
    filtered_ds = remote_ds.filter(variable="Population")
    assert filtered_ds.filters == {"variable": "Population"}

    # returns a new object
    assert id(filtered_ds) != id(remote_ds)

    # Can also filter on creation
    ds = MockRemoteDataset(NDCS_URL, {"variable": "Population"})
    assert ds.filters == {"variable": "Population"}


def test_remote_query(remote_ds):
    res = remote_ds.query()
    res_filtered = remote_ds.filter(variable="Emissions|CO2").query()

    assert isinstance(res, scmdata.ScmRun)
    assert isinstance(res_filtered, scmdata.ScmRun)
    assert MockRemoteDataset._data_queries == [{}, {"variable": "Emissions|CO2"}]


@patch("scmdata.remote._read_api_timeseries")
def test_remote_query_mocked(mock_timeseries, caplog):
    caplog.set_level(logging.INFO)

    ds = RemoteDataset(NDCS_URL).filter(variable="test")
    ds.filter_options = Mock(return_value=["variable"])
    res = ds.query()

    assert res == mock_timeseries.return_value
    assert res.source == ds

    assert len(caplog.messages) == 1
    mock_timeseries.assert_called_with(NDCS_URL, **{"variable": "test"})


@patch("scmdata.remote._read_api_timeseries")
@patch("scmdata.remote._read_api_facets")
def test_remote_query_with_extras(mock_facets, mock_timeseries, caplog):
    caplog.set_level(logging.WARNING)
    ds = RemoteDataset(NDCS_URL, filters=dict(variable="Emissions|CO2", other="test"))
    ds.filter_options = Mock(return_value=["variable"])

    msg = "Could not filter dataset by ['other']"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ds.query(raise_on_error=True)

    assert len(caplog.messages) == 0

    ds.query()

    assert len(caplog.messages) == 1
    assert re.search(re.escape(msg + ". Ignoring"), caplog.text)


def test_remote_filter_with_keep(caplog, remote_ds):
    initial_filters = {"variable": "Temperature|Global Mean"}
    ds = remote_ds.filter(**initial_filters)
    res = ds.filter(keep=False, scenario="ADVANCE_INDC")
    assert len(caplog.messages) == 1
    assert (
        caplog.messages[0]
        == "'keep' is not handled by the API. Querying data and performing filtering locally"
    )

    assert "ADVANCE_INDC" in ds.get_unique_meta("scenario")
    assert "ADVANCE_INDC" not in res.get_unique_meta("scenario")
    assert MockRemoteDataset._data_queries == [
        initial_filters
    ]  # Query didn't include scenario, but preserves initial query


def test_remote_filter_with_duplicate(remote_ds):
    remote_ds = remote_ds.filter(variable="a")

    with pytest.raises(ValueError, match="Already filtering by variable"):
        remote_ds.filter(variable="test")


def test_remote_get_unique_meta(remote_ds):
    variables = remote_ds.get_unique_meta("variable")
    assert isinstance(variables, list)
    assert len(variables)

    with pytest.raises(KeyError):
        remote_ds.get_unique_meta("unknown")

    with pytest.raises(ValueError):
        remote_ds.get_unique_meta("variable", True)

    single = remote_ds.filter(variable="Temperature|Global Mean").get_unique_meta(
        "unit", True
    )
    assert single == "°C"

    remote_ds._side_effect = RemoteQueryError("Something went wrong", "opps")
    with pytest.raises(RemoteQueryError, match="Something went wrong: opps"):
        remote_ds.get_unique_meta("variable")


def test_filter_options(remote_ds):
    res = remote_ds.filter_options()
    assert res == [*remote_ds.meta().columns.tolist(), "year.min", "year.max"]


@pytest.mark.parametrize(
    "base_url", ("https://api.example.com/v1", "https://api.example.com/v1/")
)
def test_remote_url(remote_ds, base_url):
    remote_ds.base_url = base_url
    res = remote_ds.filter(
        **{
            "variable": "test",
            "scenario": "other",
        }
    ).url()
    assert res == "https://api.example.com/v1/timeseries?variable=test&scenario=other"
