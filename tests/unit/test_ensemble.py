import datetime as dt

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from scmdata.ensemble import ScmEnsemble


@pytest.fixture(scope="function")
def ensemble(test_scm_run):
    return ScmEnsemble(
        [
            test_scm_run,
            test_scm_run.filter(variable="Primary Energy").interpolate(
                [dt.datetime(y, 1, 1) for y in range(2010, 2026, 5)]
            ),
        ]
    )


def test_ensemble(test_scm_run):
    ensemble = ScmEnsemble([test_scm_run, test_scm_run])

    assert len(ensemble) == 2

    assert ensemble.num_timeseries == len(test_scm_run) * 2


def test_ensemble_num_timeseries(ensemble):
    assert ScmEnsemble().num_timeseries == 0

    assert ensemble.num_timeseries == sum([len(r) for r in ensemble.runs])
    assert ensemble.num_timeseries > 0


def test_ensemble_timeseries(ensemble):
    res = ensemble.timeseries()

    npt.assert_array_almost_equal(ensemble.runs[0]["year"], [2005, 2010, 2015])
    npt.assert_array_almost_equal(ensemble.runs[1]["year"], [2010, 2015, 2020, 2025])
    npt.assert_array_almost_equal(res.columns.year, [2005, 2010, 2015, 2020, 2025])

    # index should be in order
    assert res.index.names == sorted(res.index.names)

    def _get_timeseries(idx, r):
        ts = r.timeseries()
        ts["run_id"] = idx
        ts = ts.set_index("run_id", append=True)

        return ts

    exp = pd.concat([_get_timeseries(run_id, r) for run_id, r in ensemble])
    exp.index = exp.index.reorder_levels(sorted(exp.index.names))

    pd.testing.assert_frame_equal(res, exp)

    assert np.all(np.isnan(exp.xs(0, level="run_id").loc[:, "2020-01-01":"2025-01-01"]))

    # Check that subset is the same
    pdt.assert_frame_equal(
        ensemble.runs[0].timeseries().reset_index(),
        res.xs(0, level="run_id").dropna(how="all", axis="columns").reset_index(),
        check_like=True,  # To remove once #97 is resolved
    )


def test_ensemble_timeseries_empty():
    res = ScmEnsemble().timeseries()
    assert res.empty


@pytest.mark.parametrize("inplace", [True, False])
def test_ensemble_filter_basic(test_scm_ensemble, inplace):
    res = test_scm_ensemble.filter(scenario="a_scenario2")

    if inplace:
        assert res is None
        res = test_scm_ensemble
    else:
        assert id(res) != id(test_scm_ensemble)
        assert len(res) != id(test_scm_ensemble)
