import pytest

from scmdata.testing import assert_scmdf_almost_equal


@pytest.mark.parametrize(
    "g", (("variable",), ("variable", "scenario"), ("variable", "region"), ("model",),)
)
def test_groupby(test_scm_run, g):
    # Check that the metadata for each group is unique for the dimensions being grouped together
    def func(df):
        sub_df = df.meta[list(g)]

        for c in g:
            assert len(sub_df[c].unique()) == 1
        return df

    res = test_scm_run.groupby(*g).map(func)

    assert_scmdf_almost_equal(res, test_scm_run, allow_unordered=True, check_ts_names=True)


def test_groupby_return_none(test_scm_run):
    def func(df):
        if df.get_unique_meta("variable", no_duplicates=True) == "Primary Energy":
            return df
        else:
            return None

    res = test_scm_run.groupby("variable").map(func)

    assert "Primary Energy|Coal" in test_scm_run["variable"].values
    assert "Primary Energy|Coal" not in res["variable"].values


def test_groupby_return_none_all(test_scm_run):
    def func(_):
        return None

    assert test_scm_run.groupby("variable").map(func) is None
