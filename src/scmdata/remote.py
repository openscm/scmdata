import pandas as pd
import requests
import urllib.parse
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
    filters["format"] = "csv"

    resp = _make_request("get", timeseries_url, filters)

    data = resp.json()
    items = []
    for name in data:
        for item in data[name]:
            items.append({"name": name, **item})
    return pd.DataFrame(items)[["name", "value", "count"]]
