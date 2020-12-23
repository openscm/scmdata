"""
TimeSeries handling

Functionality for handling and storing individual time-series
"""

import copy
import datetime as dt
import functools
from typing import Any, Callable, List, Union

import numpy as np
import pint
import xarray as xr
from openscm_units import unit_registry as ur
from xarray.core.ops import inject_binary_ops

from .time import TimePoints, TimeseriesConverter


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

    Proxies an xarray.DataArray with a single time dimension
    """

    def __init__(self, data, time=None, **kwargs):
        """
        Initialise a :obj:`TimeSeries` instance

        Parameters
        ----------
        data : array_like
            Data to be held by the :obj:`TimeSeries` instance. ``data`` must
            be one-dimensional. If ``data`` is an :obj:`xr.DataArray`
            instance, its single dimension must be ``"time"``. If ``data``
            is not an :obj:`xr.DataArray`, then ``time`` must also be supplied.

        time : array_like or None
            Only used if ``data`` is not an :obj:`xr.DataArray`. These become
            the time axis of ``self._data``.

        **kwargs
            Only used if `data`` is not an :obj:`xr.DataArray`. Passed to the
            :obj:`xr.DataArray` constructor.

        Raises
        ------
        ValueError
            ``data`` is not one-dimensional

        TypeError
            ``data`` is an :obj:`xr.DataArray` and ``time is not None``

        ValueError
            ``data`` is an :obj:`xr.DataArray` and its dimension is not named
            ``"time"``.

        TypeError
            ``data`` is not an :obj:`xr.DataArray` and ``time is None``

        ValueError
            ``data`` is not an :obj:`xr.DataArray` and ``coords`` is supplied
            via ``**kwargs``
        """
        values = np.asarray(data)

        if values.ndim != 1:
            raise ValueError("data must be 1d")

        if isinstance(data, xr.DataArray):
            if time is not None:
                raise TypeError(
                    "If data is an :obj:`xr.DataArray` instance, time must be " "`None`"
                )

            if data.dims != ("time",):
                raise ValueError(
                    "If data is an :obj:`xr.DataArray` instance, its only "
                    "dimension must be named `'time'`"
                )
            self._data = data

        else:
            if time is None:
                raise TypeError(
                    "If data is not an :obj:`xr.DataArray` instance, `time` "
                    "must not be `None`"
                )

            if "coords" in kwargs:
                raise ValueError(
                    "If ``data`` is not an :obj:`xr.DataArray`, `coords` must "
                    "not be supplied via `kwargs` because it will be "
                    "automatically filled with the value of `time`."
                )

            # Auto incrementing name
            if "name" not in kwargs:
                kwargs["name"] = get_default_name()

            if isinstance(time, tuple):
                time = list(time)

            self._data = xr.DataArray(values, coords=[("time", time)], **kwargs)

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
    def meta(self):
        """
        Metadata associated with the timeseries

        Returns
        -------
        dict
        """
        return self._data.attrs

    @property
    def time_points(self):
        """
        Time points of the data

        Returns
        -------
        :obj:`np.ndarray`
        """
        return TimePoints(self._data.coords["time"].values)

    @property
    def values(self):
        """
        Get the data as a numpy array

        Returns
        -------
        :obj:`np.ndarray`
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

            if isinstance(other, pint.Quantity):
                try:
                    self_data = self._data * ur(self.meta["unit"])
                except KeyError:
                    # let Pint assume dimensionless and raise an error as
                    # necessary
                    self_data = self._data
            else:
                self_data = self._data

            ts = f(self_data, other_data) if not reflexive else f(other_data, self_data)
            ts.attrs = self._data.attrs
            if isinstance(other, pint.Quantity):
                ts.attrs["unit"] = str(ts.data.units)
                ts.data = ts.data.magnitude

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

        This is different to interpolating to fill in the missing values. Uses
        `xarray.DataArray.reindex` to perform the reindexing

        Parameters
        ----------
        time : `obj`:np.ndarray
            Time values to reindex the data to. Should be ``np.datetime64``
            values

        **kwargs
            Additional arguments passed to xarray's DataArray.reindex function

        Returns
        -------
        :obj:`TimeSeries`
            A new TimeSeries with the new time dimension

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
        :obj:`TimeSeries`
            A new TimeSeries with the new time dimension
        """
        target_times = TimePoints(target_times)
        timeseries_converter = TimeseriesConverter(
            self.time_points.values,
            target_times.values,
            interpolation_type=interpolation_type,
            extrapolation_type=extrapolation_type,
        )

        ts = self.reindex(target_times.as_cftime())
        ts._data[:] = timeseries_converter.convert_from(self._data.values)

        return ts


inject_binary_ops(TimeSeries)
