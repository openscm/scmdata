import pandas as pd
import requests
import urllib.parse
from scmdata import ScmRun
import io


def read_api_timeseries(url, **filters):
    timeseries_url = urllib.parse.urljoin(url, "timeseries")
    filters["format"] = "csv"
    try:
        resp = requests.get(timeseries_url, params=filters)
        resp.raise_for_status()
    except Exception as e:
        raise ValueError(f"Could not fetch data: {str(e)}")

    df = pd.read_csv(io.StringIO(resp.text))
    return ScmRun(df)


def read_api_facets(url, **filters):
    timeseries_url = urllib.parse.urljoin(url, "facets")
    filters["format"] = "csv"
    try:
        resp = requests.get(timeseries_url, params=filters)
        resp.raise_for_status()
    except Exception as e:
        raise ValueError(f"Could not fetch data: {str(e)}")

    data = resp.json()
    items = []
    for name in data:
        for item in data[name]:
            items.append({"name": name, **item})
    return pd.DataFrame(items)[["name", "value", "count"]]
