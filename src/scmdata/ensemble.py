"""
:class:`ScmEnsemble` provides a container for holding data from different model runs.
"""

import warnings

import pandas as pd

from scmdata.timeseries import _Counter

get_default_name = _Counter()


class ScmEnsemble:
    """
    Container for holding multiple :obj:`ScmRun` objects

    """

    meta_col = "run_id"
    """
    str: Name of meta containing a unique identifer for each run
    
    If no value is provided, then an implicit value is create. This ensures that every timeseries
    has unique metadata
    """

    def __init__(self, runs=None):
        """

        """

        if runs is None:
            runs = []
        self._runs = runs
        self._run_ids = [get_default_name() for _ in runs]

    def __len__(self):
        # Should this be sum (len(r) for r in runs)
        return len(self._runs)

    @property
    def num_timeseries(self):
        if not len(self):
            return 0
        return sum([len(r) for r in self._runs])

    def __repr__(self):
        return "<scmdata.ScmEnsemble (runs: {}, timeseries: {})>".format(
            len(self), sum([len(r) for r in self._runs])
        )

    def __iter__(self):
        return zip(self._run_ids, self._runs)

    @property
    def runs(self):
        # copy so not directly modifiable
        return self._runs[:]

    def filter(self, inplace=False, keep=True, **kwargs):
        runs = [r.filter(inplace=inplace, keep=keep, **kwargs) for r in self.runs]

        if inplace:
            self._runs = runs
        else:
            return ScmEnsemble(runs)

    def timeseries(self, **kwargs):
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

    def append(self, run, run_id=None):
        self._runs.append(run)
        self._run_ids.append(run_id or get_default_name())
