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
def test_groupby(scm_run, g):
    # Check that the metadata for each group is unique for the dimensions being grouped
    # together
    def func(df):
        sub_df = df.meta[list(g)]

        for c in g:
            assert len(sub_df[c].unique()) == 1
        return df

    res = scm_run.groupby(*g).map(func)

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

    res = scm_run.groupby_all_except(*g).map(func)

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

    res = scm_run.groupby("variable").map(func)

    assert "Primary Energy|Coal" in scm_run["variable"].values
    assert "Primary Energy|Coal" not in res["variable"].values


def test_groupby_return_none_all(scm_run):
    def func(_):
        return None

    assert scm_run.groupby("variable").map(func) is None


def test_groupby_integer_metadata_single_grouper(scm_run):
    def increment_ensemble_member(scmrun):
        scmrun["ensemble_member"] += 10

        return scmrun

    scm_run["ensemble_member"] = range(scm_run.shape[0])

    res = scm_run.groupby("ensemble_member").map(increment_ensemble_member)

    assert (res["ensemble_member"] == scm_run["ensemble_member"] + 10).all()


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

    res = start.groupby(["variable", "region", "scenario", "ensemble_member"]).map(
        increment_ensemble_member
    )

    assert (res["ensemble_member"] == start["ensemble_member"] + 10).all()
