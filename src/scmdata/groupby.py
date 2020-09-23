"""
Functionality for grouping and filtering ScmRun objects
"""
import warnings
from collections.abc import Iterable

import numpy as np
from xarray.core import ops
from xarray.core.common import ImplementsArrayReduce


def _maybe_wrap_array(original, new_array):
    """
    Wrap a transformed array with ``__array_wrap__`` if it can be done safely.

    This lets us treat arbitrary functions that take and return ndarray objects
    like ufuncs, as long as they return an array with the same shape.
    """
    # in case func lost array's metadata
    if isinstance(new_array, np.ndarray) and new_array.shape == original.shape:
        return original.__array_wrap__(new_array)
    else:
        return new_array


class _GroupBy(ImplementsArrayReduce):
    def __init__(self, meta, groups, na_fill_value=-10000):
        m = meta.reset_index(drop=True)
        self.na_fill_value = float(na_fill_value)

        # Work around the bad handling of NaN values in groupbys
        if any([np.issubdtype(m[c].dtype, np.number) for c in m]):
            if (meta == na_fill_value).any(axis=None):
                raise ValueError(
                    "na_fill_value conflicts with data value. Choose a na_fill_value not in meta"
                )
            else:
                m = m.fillna(na_fill_value)

        self._grouper = m.groupby(list(groups), group_keys=True)

    def _iter_grouped(self):
        def _try_fill_value(v):
            try:
                if float(v) == float(self.na_fill_value):
                    return np.nan
            except ValueError:
                pass
            return v

        for indices in self._grouper.groups:
            if not isinstance(indices, Iterable) or isinstance(indices, str):
                indices = [indices]

            indices = [_try_fill_value(v) for v in indices]
            res = self.run.filter(**{k: v for k, v in zip(self.group_keys, indices)})
            if not len(res):
                raise ValueError(
                    "Empty group for {}".format(list(zip(self.group_keys, indices)))
                )
            yield res

    def __iter__(self):
        return self._iter_grouped()


class RunGroupBy(_GroupBy):
    """
    GroupBy object specialized to grouping ScmRun objects
    """

    def __init__(self, run, groups):
        self.run = run
        self.group_keys = groups
        super().__init__(run.meta, groups)

    def map(self, func, *args, **kwargs):
        """
        Apply a function to each group and append the results

        `func` is called like `func(ar, *args, **kwargs)` for each :obj:`ScmRun` ``ar``
        in this group. If the result of this function call is None, than it is
        excluded from the results.

        The results are appended together using :func:`run_append`. The function
        can change the size of the input :obj:`ScmRun` as long as :func:`run_append`
        can be applied to all results.

        Examples
        --------
        .. code:: python

            >>> def write_csv(arr):
            ...     variable = arr.get_unique_meta("variable")
            ...     arr.to_csv("out-{}.csv".format(variable)
            >>> df.groupby("variable").map(write_csv)

        Parameters
        ----------
        func : function
            Callable to apply to each timeseries.

        ``*args``
            Positional arguments passed to `func`.

        ``**kwargs``
            Used to call `func(ar, **kwargs)` for each array `ar`.

        Returns
        -------
        applied : :obj:`ScmRun`
            The result of splitting, applying and combining this array.
        """
        grouped = self._iter_grouped()
        applied = [
            _maybe_wrap_array(arr, func(arr, *args, **kwargs)) for arr in grouped
        ]
        return self._combine(applied)

    def _combine(self, applied):
        """
        Recombine the applied objects like the original.
        """
        from scmdata.run import run_append

        # Remove all None values
        applied = [df for df in applied if df is not None]

        if len(applied) == 0:
            return None
        else:
            return run_append(applied)

    def reduce(self, func, dim=None, axis=None, **kwargs):
        """
        Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : `...`, str or sequence of str, optional
            Not used in this implementation
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `func` is calculated over all dimension for each group item.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : :obj:`ScmRun`
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is not None and dim != "time":
            raise ValueError("Only reduction along the time dimension is supported")

        def reduce_array(ar):
            return ar.reduce(func, dim, axis, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return self.map(reduce_array)


ops.inject_reduce_methods(RunGroupBy)
