import xarray as xr
import pandas as pd


class TimeSeries:
    def __init__(self, data, **kwargs):
        if data.ndim != 1:
            raise ValueError("TimeSeries must be 1d")
        self._data = xr.DataArray(data, **kwargs)

    def __repr__(self):
        return self._data.__repr__()

    def __len__(self):
        return len(self._data["time"])

    def as_series(self):
        return pd.Series()

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
