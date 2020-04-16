"""
TimeSeries handling

Functionality for handling and storing individual time-series
"""

import copy
import datetime as dt
import functools
from typing import Any, Callable, List, Union

import numpy as np
import xarray as xr
from xarray.core.ops import inject_binary_ops

from scmdata.time import TimeseriesConverter


class _Counter:
    def __init__(self):
        self.count = 0

    def __call__(self):
        val = self.count
        self.count += 1

        return val

    def reset(self):
        self.count = 0


get_default_name = _Counter()


class TimeSeries:
    """
    A 1D time-series with metadata

    Proxies a xarray.DataArray with a single time dimension
    """

    def __init__(self, data, **kwargs):
        values = np.asarray(data)

        if values.ndim != 1:
            raise ValueError("TimeSeries must be 1d")
        if isinstance(data, xr.DataArray):
            self._data = data
        else:
            # Auto incrementing name
            if "name" not in kwargs:
                kwargs["name"] = get_default_name()

            self._data = xr.DataArray(values, **kwargs)

    def __repr__(self):
        return self._data.__repr__()

    @property
    def name(self):
        """
        Timeseries name

        If no name was provided this will be an automatically incrementing number
        """
        return self._data.name

    def __len__(self):
        """
        Length of the time-series (number of time points)

        Returns
        -------
        int
        """
        return len(self._data["time"])

    def copy(self):
        """
        Create a deep copy of the timeseries.

        Any further modifications to the :obj:`Timeseries` returned copy will not be reflecting
        in the current :obj:`Timeseries`

        Returns
        -------
        :obj:`Timeseries`
        """
        return copy.deepcopy(self)

    @property
    def metadata(self):
        """
        Metadata associated with the timeseries

        Returns
        -------
        dict
        """
        return self._data.attrs

    @property
    def values(self):
        """
        Get the data as a numpy array

        Returns
        -------
        :obj`np.ndarray`
        """
        return self._data.values

    def __getitem__(self, item):
        res = self._data.__getitem__(item)
        if res.ndim == 0:
            return res
        return TimeSeries(res)

    def __setitem__(self, key, value):
        self._data.__setitem__(key, value)

    @staticmethod
    def _binary_op(
        f: Callable[..., Any], reflexive=False, **kwargs,
    ) -> Callable[..., "TimeSeries"]:
        @functools.wraps(f)
        def func(self, other):
            other_data = getattr(other, "_data", other)

            ts = (
                f(self._data, other_data)
                if not reflexive
                else f(other_data, self._data)
            )
            ts.attrs = self._data.attrs
            return TimeSeries(ts)

        return func

    @staticmethod
    def _inplace_binary_op(f: Callable) -> Callable[..., "TimeSeries"]:
        @functools.wraps(f)
        def func(self, other):
            other_data = getattr(other, "_data", other)
            f(self._data, other_data)
            return self

        return func

    def reindex(self, time, **kwargs):
        """
        Update the time dimension, filling in the missing values with NaN's

        This is different to interpolating to fill in the missing values. Uses `xarray.DataArray.reindex` to perform the
        reindexing

        Parameters
        ----------
        time : `obj`:np.ndarray
            Time values to reindex the data to. Should be np 'datetime64` values
        **kwargs
            Additional arguments passed to xarray's DataArray.reindex function

        Returns
        -------
        A new TimeSeries, with the new time dimension

        References
        ----------
        http://xarray.pydata.org/en/stable/generated/xarray.DataArray.reindex_like.html#xarray.DataArray.reindex_like
        """
        return TimeSeries(self._data.reindex({"time": time}, **kwargs))

    def interpolate(
        self,
        target_times: Union[np.ndarray, List[Union[dt.datetime, int]]],
        interpolation_type: str = "linear",
        extrapolation_type: str = "linear",
    ):
        """
        Interpolate the timeseries onto a new time axis

        Parameters
        ----------
        target_times
            Time grid onto which to interpolate
        interpolation_type: str
            Interpolation type. Options are 'linear'
        extrapolation_type: str or None
            Extrapolation type. Options are None, 'linear' or 'constant'
        Returns
        -------

        """
        target_times = np.asarray(target_times, dtype="datetime64[s]")
        timeseries_converter = TimeseriesConverter(
            self._data["time"].values,
            target_times,
            interpolation_type=interpolation_type,
            extrapolation_type=extrapolation_type,
        )
        import cftime

        cftime_dts = [
            cftime.datetime(*dt.timetuple()[:6]) for dt in target_times.astype(object)
        ]
        d = self._data.reindex({"time": cftime_dts})
        d[:] = timeseries_converter.convert_from(self._data.values)

        return TimeSeries(d)


inject_binary_ops(TimeSeries)
