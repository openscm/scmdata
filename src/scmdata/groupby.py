import numpy as np
from xarray.core import ops
from xarray.core.common import ImplementsArrayReduce


def maybe_wrap_array(original, new_array):
    """Wrap a transformed array with __array_wrap__ is it can be done safely.

    This lets us treat arbitrary functions that take and return ndarray objects
    like ufuncs, as long as they return an array with the same shape.
    """
    # in case func lost array's metadata
    if isinstance(new_array, np.ndarray) and new_array.shape == original.shape:
        return original.__array_wrap__(new_array)
    else:
        return new_array


class GroupBy(ImplementsArrayReduce):
    def __init__(self, meta, groups):
        self._grouper = meta.groupby(groups, group_keys=True)

    def _iter_grouped(self):
        for indices in self._grouper.groups:
            indices = np.atleast_1d(indices)
            yield self.run.filter(**{k: v for k, v in zip(self.group_keys, indices)})


class RunGroupBy(GroupBy):
    """GroupBy object specialized to grouping ScmRun objects
    """

    def __init__(self, run, groups):
        self.run = run
        self.group_keys = groups
        super().__init__(run.meta, groups)

    def map(self, func, args=(), **kwargs):
        """Apply a function to each time in the group and concatenate them
        together into a new array.

        `func` is called like `func(ar, *args, **kwargs)` for each array `ar`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the array. The rule is:

        1. If the dimension along which the group coordinate is defined is
           still in the first grouped array after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : function
            Callable to apply to each timeseries.
        
        ``*args`` : tuple, optional
            Positional arguments passed to `func`.
        ``**kwargs``
            Used to call `func(ar, **kwargs)` for each array `ar`.

        Returns
        -------
        applied : DataArray or DataArray
            The result of splitting, applying and combining this array.
        """
        grouped = self._iter_grouped()
        applied = [maybe_wrap_array(arr, func(arr, *args, **kwargs)) for arr in grouped]
        return self._combine(applied)

    def _combine(self, applied):
        """Recombine the applied objects like the original."""
        from scmdata.run import df_append

        # Remove all None values
        applied = [df for df in applied if df is not None]

        if len(applied) == 0:
            return None
        else:
            return df_append(applied)

    def reduce(
            self, func, dim=None, axis=None, keep_attrs=None, **kwargs
    ):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : `...`, str or sequence of str, optional
            Dimension(s) over which to apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `func` is calculated over all dimension for each group item.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is not None:
            assert dim is "time", "Only reduction along the time dimension is supported"

        def reduce_array(ar):
            return ar.reduce(func, dim, axis, keep_attrs=keep_attrs, **kwargs)

        return self.map(reduce_array)


ops.inject_reduce_methods(RunGroupBy)
