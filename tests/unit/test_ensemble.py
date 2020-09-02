import pytest
from scmdata.ensemble import ScmEnsemble
import datetime as dt
import pandas as pd


@pytest.fixture(scope="function")
def ensemble(test_scm_run):
    return ScmEnsemble(
        [
            test_scm_run,
            test_scm_run.filter(variable="Primary Energy").interpolate(
                [dt.datetime(y, 1, 1) for y in range(2005, 2021, 5)]
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

    def _get_timeseries(r):
        ts = r.timeseries()

        return ts

    exp = pd.concat([_get_timeseries() for r in ensemble.runs])
    exp
    exp.index = exp.index.reorder_levels(exp.levels.name)

    pd.testing.assert_frame_equal(res, exp)


@pytest.mark.parametrize("inplace", [True, False])
def test_ensemble_filter_basic(test_scm_ensemble, inplace):
    res = test_scm_ensemble.filter(scenario="a_scenario2")

    if inplace:
        assert res is None
        res = test_scm_ensemble
    else:
        assert id(res) != id(test_scm_ensemble)
        assert len(res) != id(test_scm_ensemble)

    res
