import pandas as pd
import requests
import urllib.parse
from typing import List

import scmdata
from scmdata import ScmRun
import io

from scmdata.errors import RemoteQueryError


def _make_request(method, url, params) -> requests.Response:
    try:
        resp = requests.request(method, url, params=params)
        resp.raise_for_status()

        return resp
    except requests.exceptions.ConnectionError as err:
        # connection failure or DNS error
        raise RemoteQueryError("Failed to connect", error=err)
    except requests.exceptions.Timeout as err:
        # Failed to get a response from the API
        raise RemoteQueryError("Connection timeout", error=err)
    except requests.exceptions.HTTPError as err:
        # Handles non-200 status codes
        raise RemoteQueryError("Client error", error=err)
    except requests.exceptions.RequestException as err:
        raise RemoteQueryError("Unknown error occurred when fetching data", error=err)


def read_api_timeseries(url: str, **filters):
    """
    Fetch data from a Timeseries API

    Parameters
    ----------
    url


    filters

    Raises
    ------
    RemoteQueryError
        Any

    Returns
    -------
    Data matching query
    """
    timeseries_url = urllib.parse.urljoin(url, "timeseries")
    filters["format"] = "csv"  # CSV format is faster to parse compared to json

    resp = _make_request("get", timeseries_url, filters)

    df = pd.read_csv(io.StringIO(resp.text))
    return ScmRun(df)


def read_api_facets(url, **filters):
    timeseries_url = urllib.parse.urljoin(url, "facets")

    resp = _make_request("get", timeseries_url, filters)

    data = resp.json()
    items = []
    for name in data:
        for item in data[name]:
            items.append({"name": name, **item})
    return pd.DataFrame(items)[["name", "value", "count"]]


def read_api_meta(url, **filters):
    timeseries_url = urllib.parse.urljoin(url, "meta")

    resp = _make_request("get", timeseries_url, filters)

    data = resp.json()
    return pd.DataFrame(data["meta"])


class RemoteDataset:
    def __init__(self, base_url: str, filters=None):
        self.base_url = base_url
        self.filters = filters or {}

    def meta(self) -> pd.DataFrame:
        return read_api_meta(self.base_url, **self.filters)

    def get_unique_meta(self, col: str) -> List:
        # TODO: handle single item kwarg
        return self.meta()[col].unique()

    def query(self) -> scmdata.ScmRun:
        return read_api_timeseries(self.base_url, **self.filters)

    def filter(self, **filters):
        new_filters = {**self.filters}
        for k in filters:
            if k in self.filters:
                raise ValueError(f"Already filtering by {k}")
            new_filters[k] = filters[k]

        return RemoteDataset(base_url=self.base_url, filters=new_filters)
