import pandas as pd
import requests
import urllib.parse
from typing import List, Optional
from functools import lru_cache

import scmdata
from scmdata import ScmRun
import io

from scmdata.errors import RemoteQueryError

import logging

logger = logging.getLogger(__name__)


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


def _read_api_timeseries(url: str, **filters):
    timeseries_url = urllib.parse.urljoin(url, "timeseries")
    filters["format"] = "csv"  # CSV format is faster to parse compared to json

    resp = _make_request("get", timeseries_url, filters)

    df = pd.read_csv(io.StringIO(resp.text))
    return ScmRun(df)


@lru_cache(32)
def _read_api_facets(url, **filters):
    timeseries_url = urllib.parse.urljoin(url, "facets")

    resp = _make_request("get", timeseries_url, filters)

    data = resp.json()
    items = []
    for name in data:
        for item in data[name]:
            items.append({"name": name, **item})
    return pd.DataFrame(items)[["name", "value", "count"]]


def _read_api_meta(url, **filters):
    timeseries_url = urllib.parse.urljoin(url, "meta")

    resp = _make_request("get", timeseries_url, filters)

    data = resp.json()
    return pd.DataFrame(data["meta"])


class RemoteDataset:
    def __init__(self, base_url: str, filters=None):
        # Ensure the url is terminated with a '/'
        self.base_url = base_url.rstrip("/") + "/"
        self.filters = filters or {}
        self._meta_cols = None

    def _read_api_info(self):
        facets = _read_api_facets(self.base_url)
        self._meta_cols = list(facets.keys())

    def __getattr__(self, item):
        # Proxy ScmRun functions
        if hasattr(ScmRun, item):
            return getattr(self.query(), item)

    def url(self) -> str:
        opts = self.filter_options()
        filters = {k: self.filters[k] for k in self.filters.keys() if k in opts}
        if len(filters):
            query_params = "?" + urllib.parse.urlencode(filters)
        else:
            query_params = ""

        return urllib.parse.urljoin(self.base_url, "timeseries") + query_params

    def meta(self) -> pd.DataFrame:
        """
        Fetch metadata about the filtered dataset from the API
        Returns
        -------
        The meta data for each row. This is the equivalent to :func:`scmdata.ScmRun.meta`
        """
        logger.info(
            f"Fetching remote meta from {self.base_url} matching {self.filters}"
        )
        return _read_api_meta(self.base_url, **self.filters)

    def get_unique_meta(
        self,
        col: str,
        no_duplicates: Optional[bool] = False,
    ) -> List:
        """
        Get unique values in a metadata column.

        This performs a remote query to the API server

        Parameters
        ----------
        col
            Column to retrieve metadata for

        no_duplicates:
            Should I raise an error if there is more than one unique value in the
            metadata column?

        Raises
        ------
        ValueError
            There is more than one unique value in the metadata column and
            ``no_duplicates`` is ``True``.

        KeyError
            If a ``meta`` column does not exist in the run's metadata

        RemoteQueryError
            Something went wrong when querying the API

        Returns
        -------
        [List[Any], Any]
            List of unique metadata values. If ``no_duplicates`` is ``True`` the
            metadata value will be returned (rather than a list).

        """
        vals = self.meta()[col].unique().tolist()
        if no_duplicates:
            if len(vals) != 1:
                raise ValueError(
                    "`{}` column is not unique (found values: {})".format(col, vals)
                )

            return vals[0]
        return vals

    def filter_options(self) -> List[str]:
        if self._meta_cols is None:
            self._read_api_info()

        extra_filters = ["year.min", "year.max"]
        return [*self._meta_cols, *extra_filters]

    def query(self, raise_on_error=False) -> scmdata.ScmRun:
        """
        Fetch timeseries from the API

        The resulting data will follow any applied filters (see :func:`filters`).

        Raises
        ------
        RemoteQueryError
            Something went wrong when querying the API

        Returns
        -------
        :class:`scmdata.ScmRun`
        """
        logger.info(
            f"Fetching remote timeseries from {self.base_url} matching {self.filters}"
        )

        opts = self.filter_options()
        filter_keys = self.filters.keys()
        filters = {k: self.filters[k] for k in filter_keys if k in opts}

        extra_filters = [k for k in filter_keys if k not in opts]
        if len(extra_filters):

            msg = f"Could not filter dataset by {extra_filters}"
            if raise_on_error:
                raise ValueError(msg)
            logger.warning(msg + ". Ignoring")
        run = _read_api_timeseries(self.base_url, **filters)

        run.source = self

        return run

    def filter(self, **filters):
        if not filters.get("keep", True):
            logger.warning(
                "'keep' is not handled by the API. Querying data and performing filtering locally"
            )
            return self.query().filter(**filters)

        new_filters = {**self.filters}
        for k in filters:
            if k in self.filters:
                raise ValueError(f"Already filtering by {k}")
            new_filters[k] = filters[k]

        return self.__class__(base_url=self.base_url, filters=new_filters)
