import copy

import xarray as xr


class TimeSeries:
    """
    A 1D timeseries with metadata

    Proxies a xarray.DataArray with a single time dimension
    """
    def __init__(self, data, **kwargs):
        if data.ndim != 1:
            raise ValueError("TimeSeries must be 1d")
        if isinstance(data, xr.DataArray):
            self._data = data
        else:
            self._data = xr.DataArray(data, **kwargs)

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
    def meta(self):
        return self._data.attrs

    def __getitem__(self, item):
        res = self._data.__getitem__(item)
        if res.ndim == 0:
            return res
        return TimeSeries(res)

    def __setitem__(self, key, value):
        self._data.__setitem__(key, value)

    def reindex(self, time, **kwargs):
        """
        Updates the time dimension, filling in the missing values with NaN's

        This is different to interpolating to fill in the missing values. Uses `xarray.DataArray.reindex` to perform the
        reindexing

        Parameters
        ----------
        time : `obj`:np.ndarray
        kwargs
            Additional arguments passed to xarray's DataArray.reindex function

        Returns
        -------
        A new TimeSeries, with the new time dimension

        References
        ----------
        http://xarray.pydata.org/en/stable/generated/xarray.DataArray.reindex_like.html#xarray.DataArray.reindex_like
        """
        return TimeSeries(self._data.reindex({"time": time}, **kwargs))