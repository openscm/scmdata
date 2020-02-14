import copy
import datetime as dt
import functools
from typing import List, Union, Callable, Any

import numpy as np
import xarray as xr
from scmdata.time import TimeseriesConverter
from xarray.core.ops import inject_binary_ops


class Counter:
    """
    An auto-incrementing counter.

    """
    def __init__(self):
        """
        Initialise.

        Attributes
        ----------
        count : int
            How many times has the counter been called? Set to zero upon
            initialisation.
        """
        self.count = 0

    def __call__(self):
        val = self.count
        self.count += 1

        return val

    def reset(self):
        self.count = 0


unique_name_generator = Counter()
""":class:`Counter` instance, returns a unique name each time it's called"""


class TimeSeries:
    """
    A 1D timeseries with metadata

    Proxies an :class:`xarray.DataArray` with a single time dimension
    """

    def __init__(self, data, **kwargs):
        """
        Initiliase.

        Parameters
        ----------
        data : array_like
            Timeseries data to wrap.

        **kwargs
            If ``data`` is not an :class:`xarray.DataArray` instance, passed when converting ``data`` to an :class:`xarray.DataArray` instance.

        Attributes
        ----------
        _data : :class:`xr.DataArray`
            Timeseries data.
        """
        values = np.asarray(data)

        if values.ndim != 1:
            raise ValueError("TimeSeries must be 1d")

        if isinstance(data, xr.DataArray):
            self._data = data
        else:
            if "name" not in kwargs: # use auto incrementing name
                kwargs["name"] = unique_name_generator()

            self._data = xr.DataArray(values, **kwargs)

    def __repr__(self):
        return self._data.__repr__()

    @property
    def name(self):
        return self._data.name

    def __len__(self):
        return len(self._data["time"])

    def copy(self):
        return copy.deepcopy(self)

    @property
    def meta(self):  # TODO: rename meta to distinguish from pyam.IamDataFrame's 'meta' table, which has a completely different meaning?
        return self._data.attrs

    @property
    def values(self):
        return self._data.values

    def __getitem__(self, item):
        res = self._data.__getitem__(item)
        if res.ndim == 0:
            return res
        return TimeSeries(res)

    def __setitem__(self, key, value):
        self._data.__setitem__(key, value)

    @staticmethod
    def _binary_op(  # TODO: check why type hints are here
            f: Callable[..., Any],
            reflexive=False,
            **kwargs,
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
            # TODO: check if we should be taking name here too?
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
        Reindex the time dimension.

        Uses :func:`xarray.DataArray.reindex` to perform the reindexing. Note that
        unlike interpolation, reindexing does not fill in missing values. The fill
        value can be controlled via the ``kwargs``.

        Parameters
        ----------
        time : array_like
            Time points onto which to reindex.

        **kwargs
            Additional arguments passed to xarray's DataArray.reindex function

        Returns
        -------
        :class:`Timeseries`
            New TimeSeries with the reindexed time dimension

        References
        ----------
        http://xarray.pydata.org/en/stable/generated/xarray.DataArray.reindex.html
        """
        # TODO: check whether we need to copy attributes and name too?
        return TimeSeries(self._data.reindex({"time": time}, **kwargs))

    def interpolate(self,
                    target_times: Union[np.ndarray, List[Union[dt.datetime, int]]],
                    interpolation_type: str = "linear",
                    extrapolation_type: str = "linear",
                    ):
        """
        Interpolate the timeseries onto a new timebase

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

        d = self._data.reindex({"time": target_times})
        d[:] = timeseries_converter.convert_from(self._data.values)

        return TimeSeries(d)


inject_binary_ops(TimeSeries)
