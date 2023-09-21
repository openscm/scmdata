"""
Functionality for grouping and filtering ScmRun objects
"""
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Callable, Generic, Iterator, Optional, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from xarray.core import ops
from xarray.core.common import ImplementsArrayReduce

from scmdata._typing import MetadataValue
from scmdata.run import GenericRun

if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
    from typing_extensions import Concatenate, ParamSpec

    P = ParamSpec("P")


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


class RunGroupBy(ImplementsArrayReduce, Generic[GenericRun]):
    """
    GroupBy object specialized to grouping ScmRun objects
    """

    def __init__(
        self, run: "GenericRun", groups: "Iterable[str]", na_fill_value: float = -10000
    ):
        self.run = run
        self.group_keys = groups

        m = run.meta.reset_index(drop=True)
        self.na_fill_value = float(na_fill_value)

        # Work around the bad handling of NaN values in groupbys
        if any([np.issubdtype(m[c].dtype, np.number) for c in m]):
            if (m == na_fill_value).any(axis=None):
                raise ValueError(
                    "na_fill_value conflicts with data value. Choose a na_fill_value "
                    "not in meta"
                )
            else:
                m = m.fillna(na_fill_value)

        self._grouper: "DataFrameGroupBy" = m.groupby(list(groups), group_keys=True)

    def _iter_grouped(self) -> "Iterator[GenericRun]":
        def _try_fill_value(v: MetadataValue) -> MetadataValue:
            try:
                if float(v) == float(self.na_fill_value):
                    return np.nan
            except ValueError:
                pass
            return v

        groups: Iterable[
            Union[MetadataValue, tuple[MetadataValue, ...]]
        ] = self._grouper.groups
        for indices in groups:
            if not isinstance(indices, Iterable) or isinstance(indices, str):
                indices_clean: tuple[MetadataValue, ...] = (indices,)
            else:
                indices_clean = indices

            indices_clean = tuple(_try_fill_value(v) for v in indices_clean)
            filter_kwargs = {k: v for k, v in zip(self.group_keys, indices_clean)}
            res = self.run.filter(**filter_kwargs)  # type: ignore
            if not len(res):
                raise ValueError(
                    f"Empty group for {list(zip(self.group_keys, indices_clean))}"
                )
            yield res

    def __iter__(self) -> Iterator[GenericRun]:
        """
        Iterate over the groups
        """
        return self._iter_grouped()

    def apply(
        self,
        func: "Callable[Concatenate[GenericRun, P], Union[GenericRun, pd.DataFrame, None]]",
        *args: "P.args",
        **kwargs: "P.kwargs",
    ) -> "GenericRun":
        """
        Apply a function to each group and append the results

        `func` is called like `func(ar, *args, **kwargs)` for each :class:`ScmRun <scmdata.run.ScmRun>` ``ar``
        in this group. If the result of this function call is None, than it is
        excluded from the results.

        The results are appended together using :func:`run_append`. The function
        can change the size of the input :class:`ScmRun <scmdata.run.ScmRun>` as long as :func:`run_append`
        can be applied to all results.

        Examples
        --------
        .. code:: python

            >>> def write_csv(arr: scmdata.ScmRun) -> None:
            ...     variable = arr.get_unique_meta("variable")
            ...     arr.to_csv("out-{}.csv".format(variable))
            >>> df.groupby("variable").apply(write_csv)

        Parameters
        ----------
        func
            Callable to apply to each timeseries.

        ``*args``
            Positional arguments passed to `func`.

        ``**kwargs``
            Used to call `func(ar, **kwargs)` for each array `ar`.

        Returns
        -------
        applied : :class:`ScmRun <scmdata.run.ScmRun>`
            The result of splitting, applying and combining this array.
        """
        grouped = self._iter_grouped()
        applied = [
            _maybe_wrap_array(arr, func(arr, *args, **kwargs)) for arr in grouped
        ]
        return self._combine(applied)

    def apply_parallel(
        self,
        func: "Callable[Concatenate[GenericRun, P], Union[GenericRun, pd.DataFrame, None]]",
        n_jobs: int = 1,
        backend: str = "loky",
        *args: "P.args",
        **kwargs: "P.kwargs",
    ) -> "GenericRun":
        """
        Apply a function to each group in parallel and append the results

        Provides the same functionality as :func:`~apply` except that :mod:`joblib` is used to apply
        `func` to each group in parallel. This can be slower than using :func:`~apply` for small
        numbers of groups or in the case where `func` is fast as there is overhead setting up the
        processing pool.

        See Also
        --------
        :func:`~apply`

        Parameters
        ----------
        func
            Callable to apply to each timeseries.

        n_jobs
            Number of jobs to run in parallel (defaults to a single job which is useful for
            debugging purposes). If `-1` all CPUs are used.

        backend
            Backend used for parallelisation. Defaults to 'loky' which uses separate processes for
            each worker.

            See :class:`joblib.Parallel` for a more complete description of the available
            options.

        ``*args``
            Positional arguments passed to `func`.

        ``**kwargs``
            Used to call `func(ar, **kwargs)` for each array `ar`.

        Returns
        -------
        applied : :class:`ScmRun <scmdata.run.ScmRun>`
            The result of splitting, applying and combining this array.
        """
        try:
            import joblib  # type: ignore
        except ImportError as e:
            raise ImportError(
                "joblib is not installed. Run 'pip install joblib'"
            ) from e

        grouped = self._iter_grouped()
        applied: "list[Union[GenericRun, pd.DataFrame, None]]" = joblib.Parallel(
            n_jobs=n_jobs, backend=backend
        )(joblib.delayed(func)(arr, *args, **kwargs) for arr in grouped)
        return self._combine(applied)

    def map(self, func, *args, **kwargs):
        """
        Apply a function to each group and append the results

        .. deprecated:: 0.14.2
            :func:`map` will be removed in scmdata 1.0.0, it is renamed to :func:`apply`
            with identical functionality.

        See Also
        --------
        :func:`apply`
        """
        warnings.warn("Use RunGroupby.apply instead", DeprecationWarning)
        return self.apply(func, *args, **kwargs)

    def _combine(
        self, applied: "Sequence[Union[GenericRun, pd.DataFrame, None]]"
    ) -> "GenericRun":
        """
        Recombine the applied objects like the original.
        """
        from scmdata.run import run_append

        # Remove all None values
        applied_clean = [df for df in applied if df is not None]

        if len(applied_clean) == 0:
            return self.run.__class__()
        else:
            return run_append(applied_clean)

    def reduce(
        self,
        func: "Callable[Concatenate[NDArray[np.float_], P], NDArray[np.float_]]",
        dim: Optional[Union[str, Iterable[str]]] = None,
        axis: Optional[Union[str, Iterable[int]]] = None,
        *args: "P.args",
        **kwargs: "P.kwargs",
    ) -> "GenericRun":
        """
        Reduce the items in this group by applying `func` along some dimension(s).

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
        reduced : :class:`ScmRun <scmdata.run.ScmRun>`
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is not None and dim != "time":
            raise ValueError("Only reduction along the time dimension is supported")

        def reduce_array(ar):
            return ar.reduce(func, dim, axis, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return self.apply(reduce_array)


ops.inject_reduce_methods(RunGroupBy)
