"""
Functionality for grouping and filtering ScmRun objects
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Generic, Iterator, TypeVar, Union

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from xarray.core.common import ImplementsArrayReduce

from scmdata._typing import MetadataValue
from scmdata.run import BaseScmRun, GenericRun

# Hack central, a better way to do this would be nice
xr_version_split = xr.__version__.split(".")
if int(xr_version_split[0]) < 2025 or xr_version_split[1] == "01":  # noqa: PLR2004
    from xarray.core import ops
else:
    from xarray.computation import ops

if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
    from typing_extensions import Concatenate, ParamSpec

    P = ParamSpec("P")
    Q = ParamSpec("Q")
    RunLike = TypeVar("RunLike", bound=BaseScmRun)
    ApplyCallableReturnType = Union[RunLike, pd.DataFrame, None]
    ApplyCallable = Callable[Concatenate[RunLike, Q], ApplyCallableReturnType[RunLike]]
    ParallelProcessor = Callable[
        Concatenate[
            ApplyCallable[RunLike, Q],
            Iterable[RunLike],
            Q,
        ],
        Iterable[ApplyCallableReturnType[RunLike]],
    ]


class RunGroupBy(ImplementsArrayReduce, Generic[GenericRun]):
    """
    GroupBy object specialized to grouping ScmRun objects
    """

    def __init__(
        self, run: GenericRun, groups: Iterable[str], na_fill_value: float = -10000
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

        self._grouper: DataFrameGroupBy = m.groupby(list(groups), group_keys=True)

    def _iter_grouped(self) -> Iterator[GenericRun]:
        def _try_fill_value(v: MetadataValue) -> MetadataValue:
            try:
                if float(v) == float(self.na_fill_value):
                    return np.nan
            except ValueError:
                pass
            return v

        groups: Iterable[
            MetadataValue | tuple[MetadataValue, ...]
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
        func: Callable[Concatenate[GenericRun, P], GenericRun | (pd.DataFrame | None)],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> GenericRun:
        """
        Apply a function to each group and append the results

        `func` is called like `func(ar, *args, **kwargs)` for each :class:`ScmRun <scmdata.run.ScmRun>`
        group. If the result of this function call is ``None``, than it is
        excluded from the results.

        The results are appended together using :func:`run_append`. The function
        can change the size of the input :class:`ScmRun <scmdata.run.ScmRun>`
        as long as :func:`run_append` can be applied to all results.

        Examples
        --------
        .. code:: python

            >>> from scmdata import ScmRun
            >>> def show_var_and_convert_unit(arr: scmdata.ScmRun) -> None:
            ...     variable = arr.get_unique_meta("variable", True)
            ...     unit = arr.get_unique_meta("unit", True)
            ...     print(f"{variable}'s original unit was {unit}")
            ...
            ...     return arr.convert_unit("MtC")

            >>> df = ScmRun(
            ...     data=[[1, 2], [3, 4]],
            ...     index=[2010, 2020],
            ...     columns={
            ...         "variable": ["v1", "v2"],
            ...         "model": "model",
            ...         "scenario": "scenario",
            ...         "region": "World",
            ...         "unit": ["tC", "GtC"],
            ...     },
            ... )
            >>> df.groupby("variable").apply(show_var_and_convert_unit)
            v1's original unit was tC
            v2's original unit was GtC
            <ScmRun (timeseries: 2, timepoints: 2)>
            Time:
                Start: 2010-01-01T00:00:00
                End: 2020-01-01T00:00:00
            Meta:
                   model region  scenario unit variable
                0  model  World  scenario  MtC       v1
                1  model  World  scenario  MtC       v2

        Parameters
        ----------
        func
            Callable to apply to each group.

        *args
            Positional arguments passed to `func`.

        **kwargs
            Keyword arguments passed to `func`.

        Returns
        -------
            The result of applying and combining.
        """
        grouped = self._iter_grouped()
        applied = [func(arr, *args, **kwargs) for arr in grouped]
        return self._combine(applied)

    def apply_parallel(
        self,
        func: ApplyCallable[GenericRun, P],
        parallel_processor: ParallelProcessor[GenericRun, P] | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> GenericRun:
        """
        Apply a function to each group in parallel and append the results

        Provides the same functionality as :func:`~apply` except that parallel processing can be
        used via the ``parallel_processor`` argument. By default, :mod:`joblib` is used to apply
        `func` to each group in parallel. This can be slower than using :func:`~apply` for small
        numbers of groups or in the case where `func` is fast as there is overhead setting up the
        processing pool.

        See Also
        --------
        :func:`~apply`

        Parameters
        ----------
        func
            Callable to apply to each group.

        parallel_processor
            Parallel processor to use to process the groups. If not provided,
            a default joblib parallel processor is used (for details, see
             :func:`get_joblib_parallel_processor`).

        *args
            Positional arguments passed to `func`.

        **kwargs
            Keyword arguments passed to `func`.

        Returns
        -------
            The result of applying and combining.
        """
        if parallel_processor is None:
            parallel_processor = get_joblib_parallel_processor()

        grouped = self._iter_grouped()
        applied = parallel_processor(func, grouped, *args, **kwargs)

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
        self, applied: Iterable[GenericRun | (pd.DataFrame | None)]
    ) -> GenericRun:
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
        func: Callable[Concatenate[NDArray[np.float64], P], NDArray[np.float64]],
        dim: str | Iterable[str] | None = None,
        axis: str | Iterable[int] | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> GenericRun:
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


def get_joblib_parallel_processor(
    n_jobs: int = -1,
    backend: str = "loky",
    *args: Any,
    **kwargs: Any,
) -> ParallelProcessor[RunLike, Q]:
    """
    Get parallel processor using :mod:`joblib` as the backend.

    Parameters
    ----------
    n_jobs
        Number of jobs to run in parallel. If `-1` all CPUs are used.

    backend
        Backend used for parallelisation. Defaults to 'loky' which uses separate processes for
        each worker.
        See :class:`joblib.Parallel` for a more complete description of the available
        options.

    *args
        Passed to initialiser of :class:`joblib.Parallel`

    **kwargs
        Passed to initialiser of :class:`joblib.Parallel`

    Returns
    -------
        Function that can be used for parallel processing in
        :meth:`RunGroupBy.apply_parallel`
    """
    try:
        import joblib
    except ImportError as e:  # pragma: no cover
        raise ImportError("joblib is not installed. Run 'pip install joblib'") from e

    processor = joblib.Parallel(*args, n_jobs=n_jobs, backend=backend, **kwargs)

    def joblib_parallel_processor(
        func: ApplyCallable[RunLike, Q],
        groups: Iterable[RunLike],
        /,
        *args: Q.args,
        **kwargs: Q.kwargs,
    ) -> Iterable[ApplyCallableReturnType[RunLike]]:
        prepped_groups = (
            joblib.delayed(func)(group, *args, **kwargs) for group in groups
        )
        applied = processor(prepped_groups)

        return applied

    return joblib_parallel_processor


ops.inject_reduce_methods(RunGroupBy)
