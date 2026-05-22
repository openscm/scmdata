import numpy as np
import pytest

from scmdata import ScmRun
from scmdata.testing import assert_scmdf_almost_equal

group_tests = pytest.mark.parametrize(
    "g",
    (
        ("variable",),
        ("variable", "scenario"),
        ("variable", "region"),
        ("model",),
    ),
)


@group_tests
@pytest.mark.parametrize("parallel", [True, False])
def test_groupby(scm_run, g, parallel):
    # Check that the metadata for each group is unique for the dimensions being grouped
    # together
    def func(df):
        sub_df = df.meta[list(g)]

        for c in g:
            assert len(sub_df[c].unique()) == 1
        return df

    if parallel:
        res = scm_run.groupby(*g).apply_parallel(func)
    else:
        res = scm_run.groupby(*g).apply(func)

    assert isinstance(res, ScmRun)
    assert_scmdf_almost_equal(res, scm_run, check_ts_names=False)


@group_tests
def test_groupby_all_except(scm_run, g):
    # Check that the metadata for each group is unique for the dimensions not
    # being grouped together
    def func(df):
        grouped = list(set(df.meta) - set(g))
        sub_df = df.meta[grouped]

        for c in grouped:
            assert len(sub_df[c].unique()) == 1
        return df

    res = scm_run.groupby_all_except(*g).apply(func)

    assert isinstance(res, ScmRun)
    assert_scmdf_almost_equal(res, scm_run, check_ts_names=False)


@group_tests
def test_get_meta_columns_except(scm_run, g):
    res = scm_run.get_meta_columns_except(g)

    assert isinstance(res, list)
    assert res == sorted(tuple(set(scm_run.meta.columns) - set(g)))


def test_groupby_return_none(scm_run):
    def func(df):
        if df.get_unique_meta("variable", no_duplicates=True) == "Primary Energy":
            return df
        else:
            return None

    res = scm_run.groupby("variable").apply(func)

    assert "Primary Energy|Coal" in scm_run["variable"].values
    assert "Primary Energy|Coal" not in res["variable"].values


def test_groupby_return_none_all(scm_run):
    def func(_):
        return None

    assert scm_run.groupby("variable").apply(func).empty


def test_groupby_integer_metadata_single_grouper(scm_run):
    def increment_ensemble_member(scmrun):
        scmrun["ensemble_member"] += 10

        return scmrun

    scm_run["ensemble_member"] = range(scm_run.shape[0])

    res = scm_run.groupby("ensemble_member").apply(increment_ensemble_member)

    assert (res["ensemble_member"] == scm_run["ensemble_member"] + 10).all()


def test_map_deprecated(scm_run):
    def func(r):
        return r * 2

    with pytest.warns(DeprecationWarning, match="Use RunGroupby.apply instead"):
        res = scm_run.groupby("scenario").map(func)

    assert_scmdf_almost_equal(
        res, scm_run * 2, allow_unordered=True, check_ts_names=False
    )


def test_groupby_integer_metadata():
    def increment_ensemble_member(scmrun):
        scmrun["ensemble_member"] += 10

        return scmrun

    start = ScmRun(
        data=[[1, 2], [0, 1]],
        index=[2010, 2020],
        columns={
            "model": "model",
            "scenario": "scenario",
            "variable": "variable",
            "unit": "unit",
            "region": "region",
            "ensemble_member": [0, 1],
        },
    )

    res = start.groupby(["variable", "region", "scenario", "ensemble_member"]).apply(
        increment_ensemble_member
    )

    assert (res["ensemble_member"] == start["ensemble_member"] + 10).all()


def test_groupby_nan_metadata():
    def increment_ensemble_member(scmrun):
        scmrun["ensemble_member"] += 10
        scmrun["has_nan"] = np.isnan(scmrun.get_unique_meta("ensemble_member"))

        return scmrun

    start = ScmRun(
        data=[[1, 2], [0, 1]],
        index=[2010, 2020],
        columns={
            "model": "model",
            "scenario": "scenario",
            "variable": "variable",
            "unit": "unit",
            "region": ["region", "regionb"],
            "ensemble_member": [0, np.nan],
        },
    )

    res = start.groupby(["ensemble_member"]).apply(increment_ensemble_member)
    exp = start.copy()
    exp["ensemble_member"] += 10
    exp["has_nan"] = [False, True]

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


def test_groupby_with_string_extension_dtype():
    """
    Regression test for issue #318.

    Under pandas 3.x, string-valued meta columns of a DataFrame round-
    tripped through MultiIndex can come back as ``pandas.StringDtype``
    rather than ``object``. ``RunGroupBy.__init__`` previously called
    ``np.issubdtype(StringDtype, np.number)`` directly, which numpy
    2.x rejects with ``TypeError: Cannot interpret <StringDtype(...)>``.
    The fix routes that check through ``_is_numeric_dtype``, which
    returns ``False`` for any dtype numpy cannot classify.

    The test exercises ``ScmRun.groupby`` directly to keep the
    regression surface tight; ``convert_unit`` (the path that
    originally tripped this for downstream users) has unrelated
    pandas 3.x issues that are not in scope here.
    """
    import pandas as pd

    run = ScmRun(
        pd.DataFrame(
            [[1.0, 2.0]],
            index=pd.MultiIndex.from_tuples(
                [("FaIR", "ssp245", "m", "World", "Emissions|CO2", "GtC/yr", 0)],
                names=[
                    "climate_model",
                    "scenario",
                    "model",
                    "region",
                    "variable",
                    "unit",
                    "run_id",
                ],
            ),
            columns=[2010, 2020],
        )
    )

    # Sanity check: under pandas 3.x at least one meta column should be
    # StringDtype after the MultiIndex round-trip. On older stacks they
    # are plain object; the test still passes (the fix is a no-op on
    # numpy 1.x where np.issubdtype handled the inputs anyway).
    meta = run.meta.reset_index(drop=True)
    dtypes = [str(meta[c].dtype) for c in meta]

    # Pre-fix, ScmRun.groupby raised TypeError when any meta column was
    # StringDtype. Post-fix it should return a single group.
    groups = list(run.groupby("variable"))
    assert len(groups) == 1
    assert groups[0].get_unique_meta("variable", no_duplicates=True) == "Emissions|CO2"
    assert "string" in dtypes or "str" in dtypes or "object" in dtypes
