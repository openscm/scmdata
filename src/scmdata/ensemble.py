"""
:class:`ScmEnsemble` provides a container for holding data from different model runs.
"""

import warnings

import pandas as pd

from scmdata.plotting import inject_plotting_methods
from scmdata.run import BaseScmRun
from scmdata.timeseries import _Counter

get_default_name = _Counter()


class ScmEnsemble:
    """
    Container for holding multiple :obj:`ScmRun` objects

    Each run in the ensemble is assigned a ``run_id``.

    """

    meta_col = "run_id"
    """
    str: Name of meta containing a unique identifer for each run
    
    If no value is provided, then an implicit value is create. This ensures that every timeseries
    has unique metadata
    """

    def __init__(self, runs=None, run_ids=None):
        """

        Parameters
        ----------
        runs : list of ScmRun
        """

        self._counter = _Counter()

        if runs is None:
            runs = []

        self._runs = runs
        if run_ids:
            if len(run_ids) != len(runs):
                raise ValueError("length of run_ids must equal the length of runs")
            self._run_ids = run_ids[:]
        else:
            self._run_ids = [self._counter() for _ in runs]

    def __len__(self):
        return self.num_timeseries

    @property
    def num_timeseries(self):
        if not self.num_runs:
            return 0
        return sum([len(r) for r in self._runs])

    @property
    def num_runs(self):
        return len(self._runs)

    @property
    def run_ids(self):
        return self._run_ids[:]

    def __repr__(self):
        return "<scmdata.ScmEnsemble (runs: {}, timeseries: {})>".format(
            len(self), sum([len(r) for r in self._runs])
        )

    def __iter__(self):
        return zip(self._run_ids, self._runs)

    def __getitem__(self, item):
        # Convert to multiindex?

        if item in ["time", "year"]:
            # Drop duplicate years
            res = (
                pd.concat([r[item] for r in self._runs])
                .drop_duplicates()
                .reset_index(drop=True)
            )
        else:
            items = []
            for r in self._runs:
                try:
                    items.append(r[item])
                except KeyError:
                    continue
            if not len(items):
                raise KeyError("[{}] is not in metadata".format(item))

            res = pd.concat(items)

        return res

    def get_unique_meta(self, meta, no_duplicates=False):
        """
        Get unique values in a metadata column.

        Parameters
        ----------
        meta
            Column to retrieve metadata for

        no_duplicates
            Should I raise an error if there is more than one unique value in the
            metadata column?

        Raises
        ------
        ValueError
            There is more than one unique value in the metadata column and
            ``no_duplicates`` is ``True``.

        Returns
        -------
        [List[Any], Any]
            List of unique metadata values. If ``no_duplicates`` is ``True`` the
            metadata value will be returned (rather than a list).
        """
        vals = self[meta].unique().tolist()
        if no_duplicates:
            if len(vals) != 1:
                raise ValueError(
                    "`{}` column is not unique (found values: {})".format(meta, vals)
                )

            return vals[0]

        return vals

    def copy(self, deep=True):
        if deep:
            runs = [r.copy() for r in self.runs]
        else:
            runs = self.runs
        return ScmEnsemble(runs, self._run_ids)

    @property
    def runs(self):
        # copy so not directly modifiable
        return self._runs[:]

    def filter(self, inplace=False, keep=True, **kwargs):
        if inplace:
            for r in self._runs:
                r.filter(inplace=True, keep=keep, **kwargs)
        else:
            return ScmEnsemble(
                [r.filter(inplace=False, keep=keep, **kwargs) for r in self.runs]
            )

    def timeseries(self, **kwargs):
        if not len(self.runs):
            return pd.DataFrame()

        def _get_timeseries(run_id, r):
            df = r.timeseries(**kwargs)
            if self.meta_col in df.index.names:
                warnings.warn("Overriding {} meta column")
            df[self.meta_col] = run_id
            df = df.set_index("run_id", append=True)

            # reorder columns so they are in alphabetical order
            df.index = df.index.reorder_levels(sorted(df.index.names))
            return df

        return pd.concat([_get_timeseries(run_id, r) for run_id, r in self])

    def append(
        self, run, inplace=False,
    ):
        return ensemble_append([self, run], inplace=inplace)


def ensemble_append(ensemble_or_runs, inplace=False):
    if not isinstance(ensemble_or_runs, list):
        raise TypeError("ensemble_or_runs is not a list")

    if not len(ensemble_or_runs):
        raise ValueError("Nothing to append")

    first = ensemble_or_runs[0]
    if inplace:
        if not isinstance(first, ScmEnsemble):
            raise TypeError("Can only append inplace to an ScmEnsemble")
        ret = first
    else:
        if isinstance(first, ScmEnsemble):
            ret = first.copy()
        elif isinstance(first, BaseScmRun):
            ret = ScmEnsemble([first])
        else:
            raise TypeError("Cannot handle appending type {}".format(type(first)))

    for run in ensemble_or_runs[1:]:
        if isinstance(run, ScmEnsemble):
            for r_id, r in run:
                ret._runs.append(r)
                ret._run_ids.append(r_id or ret._counter())
        elif isinstance(run, BaseScmRun):
            ret._runs.append(run)
            ret._run_ids.append(ret._counter())
        else:
            raise TypeError("Cannot handle appending type {}".format(type(run)))

    if not inplace:
        return ret


inject_plotting_methods(ScmEnsemble)
