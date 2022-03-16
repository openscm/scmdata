import copy
import datetime as dt
import logging
import os
import re
import warnings
from datetime import datetime
from unittest.mock import patch

import cftime
import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt
from packaging.version import parse
from pandas.errors import UnsupportedFunctionCall
from pint.errors import DimensionalityError, UndefinedUnitError

from scmdata.errors import (
    DuplicateTimesError,
    InsufficientDataError,
    MissingRequiredColumnError,
    NonUniqueMetadataError,
)
from scmdata.run import BaseScmRun, ScmRun, run_append
from scmdata.testing import (
    _check_pandas_less_110,
    _check_pandas_less_120,
    assert_scmdf_almost_equal,
)


@pytest.fixture
def scm_run_interpolated(scm_run):
    return scm_run.interpolate(
        [
            dt.datetime(y, 1, 1)
            for y in range(scm_run["year"].min(), scm_run["year"].max() + 1)
        ]
    )


def test_init_df_year_converted_to_datetime(test_pd_df):
    res = ScmRun(test_pd_df)
    assert (res["year"].unique() == [2005, 2010, 2015]).all()
    assert (
        res["time"].unique()
        == [dt.datetime(2005, 1, 1), dt.datetime(2010, 1, 1), dt.datetime(2015, 1, 1)]
    ).all()


@pytest.mark.parametrize(
    "in_format",
    [
        "pd.Series",
        "year_col",
        "year_col_index",
        "time_col",
        "time_col_index",
        "time_col_str_simple",
        "time_col_str_complex",
        "time_col_reversed",
        "str_times",
    ],
)
def test_init_df_formats(test_pd_run_df, in_format):
    if in_format == "pd.Series":
        idx = ["climate_model", "model", "scenario", "region", "variable", "unit"]
        test_init = test_pd_run_df.melt(id_vars=idx, var_name="year").set_index(
            idx + ["year"]
        )["value"]
    elif in_format == "year_col":
        idx = ["climate_model", "model", "scenario", "region", "variable", "unit"]
        test_init = test_pd_run_df.melt(id_vars=idx, var_name="year")
    elif in_format == "year_col_index":
        idx = ["climate_model", "model", "scenario", "region", "variable", "unit"]
        test_init = test_pd_run_df.melt(id_vars=idx, var_name="year").set_index(
            idx + ["year"]
        )
    elif in_format == "time_col":
        idx = ["climate_model", "model", "scenario", "region", "variable", "unit"]
        test_init = test_pd_run_df.melt(id_vars=idx, var_name="year")
        test_init["time"] = test_init["year"].apply(lambda x: dt.datetime(x, 1, 1))
        test_init = test_init.drop("year", axis="columns")
    elif in_format == "time_col_index":
        idx = ["climate_model", "model", "scenario", "region", "variable", "unit"]
        test_init = test_pd_run_df.melt(id_vars=idx, var_name="year")
        test_init["time"] = test_init["year"].apply(lambda x: dt.datetime(x, 1, 1))
        test_init = test_init.drop("year", axis="columns")
        test_init = test_init.set_index(idx + ["time"])
    elif in_format == "time_col_str_simple":
        idx = ["climate_model", "model", "scenario", "region", "variable", "unit"]
        test_init = test_pd_run_df.melt(id_vars=idx, var_name="year")
        test_init["time"] = test_init["year"].apply(
            lambda x: "{}-1-1 00:00:00".format(x)
        )
        test_init = test_init.drop("year", axis="columns")
    elif in_format == "time_col_str_complex":
        idx = ["climate_model", "model", "scenario", "region", "variable", "unit"]
        test_init = test_pd_run_df.melt(id_vars=idx, var_name="year")
        test_init["time"] = test_init["year"].apply(lambda x: "{}/1/1".format(x))
        test_init = test_init.drop("year", axis="columns")
    elif in_format == "time_col_reversed":
        test_init = test_pd_run_df[test_pd_run_df.columns[::-1]]
    elif in_format == "str_times":
        test_init = test_pd_run_df.copy()
        test_init.columns = test_init.columns.map(
            lambda x: "{}/1/1".format(x) if isinstance(x, int) else x
        )

    res = ScmRun(test_init)
    assert (res["year"].unique() == [2005, 2010, 2015]).all()
    assert (
        res["time"].unique()
        == [dt.datetime(2005, 1, 1), dt.datetime(2010, 1, 1), dt.datetime(2015, 1, 1)]
    ).all()
    assert "Start: 2005" in res.__repr__()
    assert "End: 2015" in res.__repr__()

    res_df = res.timeseries()
    res_df.columns = res_df.columns.map(lambda x: x.year)
    res_df = res_df.reset_index()

    pd.testing.assert_frame_equal(
        res_df[test_pd_run_df.columns.tolist()],
        test_pd_run_df,
        check_like=True,
    )


def test_init_df_missing_time_axis_error(test_pd_df):
    idx = ["climate_model", "model", "scenario", "region", "variable", "unit"]
    test_init = test_pd_df.melt(id_vars=idx, var_name="year")
    test_init = test_init.drop("year", axis="columns")
    error_msg = re.escape("invalid time format, must have either `year` or `time`!")
    with pytest.raises(ValueError, match=error_msg):
        ScmRun(test_init)


def test_init_df_missing_time_columns_error(test_pd_df):
    test_init = test_pd_df.copy()
    test_init = test_init.drop(
        test_init.columns[test_init.columns.map(lambda x: isinstance(x, int))],
        axis="columns",
    )
    error_msg = re.escape(
        "invalid column format, must contain some time (int, float or datetime) "
        "columns!"
    )
    with pytest.raises(ValueError, match=error_msg):
        ScmRun(test_init)


def test_init_df_missing_col_error(test_pd_df):
    test_pd_df = test_pd_df.drop("model", axis="columns")
    error_msg = re.escape("Missing required columns `['model']`!")
    with pytest.raises(MissingRequiredColumnError, match=error_msg):
        ScmRun(test_pd_df)


def test_init_ts_missing_col_error(test_ts):
    error_msg = re.escape("Missing required columns `['model']`!")
    with pytest.raises(MissingRequiredColumnError, match=error_msg):
        ScmRun(
            test_ts,
            columns={
                "climate_model": ["a_model"],
                "scenario": ["a_scenario", "a_scenario", "a_scenario2"],
                "region": ["World"],
                "variable": ["Primary Energy", "Primary Energy|Coal", "Primary Energy"],
                "unit": ["EJ/yr"],
            },
            index=[2005, 2010, 2015],
        )


def test_init_required_cols(test_pd_df):
    class MyRun(BaseScmRun):
        required_cols = ("climate_model", "variable", "unit")

    del test_pd_df["model"]

    assert all([c in test_pd_df.columns for c in MyRun.required_cols])
    MyRun(test_pd_df)

    del test_pd_df["climate_model"]

    assert not all([c in test_pd_df.columns for c in MyRun.required_cols])
    error_msg = re.escape("Missing required columns `['climate_model']`!")
    with pytest.raises(
        MissingRequiredColumnError,
        match=error_msg,
    ):
        MyRun(test_pd_df)


def test_init_multiple_file_error():
    error_msg = re.escape(
        "Initialising from multiple files not supported, use "
        "`scmdata.run.ScmRun.append()`"
    )
    with pytest.raises(ValueError, match=error_msg):
        ScmRun(["file_1", "filepath_2"])


def test_init_unrecognised_type_error():
    fail_type = {"dict": "key"}
    error_msg = re.escape("Cannot load {} from {}".format(str(ScmRun), type(fail_type)))
    with pytest.raises(TypeError, match=error_msg):
        ScmRun(fail_type)


def test_init_remote_files():

    remote_file = (
        "https://rcmip-protocols-au.s3-ap-southeast-2.amazonaws.com"
        "/v5.1.0/rcmip-emissions-annual-means-v5-1-0.csv"
    )

    run = ScmRun(remote_file, lowercase_cols=True)
    assert not run.empty


def test_init_ts_col_string(test_ts):
    res = ScmRun(
        test_ts,
        columns={
            "model": "an_iam",
            "climate_model": "a_model",
            "scenario": ["a_scenario", "a_scenario", "a_scenario2"],
            "region": "World",
            "variable": ["Primary Energy", "Primary Energy|Coal", "Primary Energy"],
            "unit": "EJ/yr",
        },
        index=[2005, 2010, 2015],
    )
    npt.assert_array_equal(res["model"].unique(), "an_iam")
    npt.assert_array_equal(res["climate_model"].unique(), "a_model")
    npt.assert_array_equal(res["region"].unique(), "World")
    npt.assert_array_equal(res["unit"].unique(), "EJ/yr")


@pytest.mark.parametrize("fail_setting", [["a_iam", "a_iam"]])
def test_init_ts_col_wrong_length_error(test_ts, fail_setting):
    correct_scenarios = ["a_scenario", "a_scenario", "a_scenario2"]
    error_msg = re.escape(
        "Length of column 'model' is incorrect. It should be length 1 or {}".format(
            len(correct_scenarios)
        )
    )
    with pytest.raises(ValueError, match=error_msg):
        ScmRun(
            test_ts,
            columns={
                "model": fail_setting,
                "climate_model": ["a_model"],
                "scenario": correct_scenarios,
                "region": ["World"],
                "variable": ["Primary Energy", "Primary Energy|Coal", "Primary Energy"],
                "unit": ["EJ/yr"],
            },
            index=[2005, 2010, 2015],
        )


def get_test_pd_df_with_datetime_columns(tpdf):
    return tpdf.rename(
        {
            2005.0: dt.datetime(2005, 1, 1),
            2010.0: dt.datetime(2010, 1, 1),
            2015.0: dt.datetime(2015, 1, 1),
        },
        axis="columns",
    )


def test_init_with_ts(test_ts, test_pd_df):
    df = ScmRun(
        test_ts,
        columns={
            "model": ["a_iam"],
            "climate_model": ["a_model"],
            "scenario": ["a_scenario", "a_scenario", "a_scenario2"],
            "region": ["World"],
            "variable": ["Primary Energy", "Primary Energy|Coal", "Primary Energy"],
            "unit": ["EJ/yr"],
        },
        index=[2005, 2010, 2015],
    )

    tdf = get_test_pd_df_with_datetime_columns(test_pd_df)
    pd.testing.assert_frame_equal(df.timeseries().reset_index(), tdf, check_like=True)

    b = ScmRun(test_pd_df)

    assert_scmdf_almost_equal(df, b, check_ts_names=False)


def test_init_with_scmdf(test_scm_run_datetimes, test_scm_datetime_run):
    df = ScmRun(
        test_scm_run_datetimes,
    )

    assert_scmdf_almost_equal(df, test_scm_datetime_run, check_ts_names=False)


@pytest.mark.parametrize(
    "years", [["2005.0", "2010.0", "2015.0"], ["2005", "2010", "2015"]]
)
def test_init_with_years_as_str(test_pd_df, years):
    df = copy.deepcopy(
        test_pd_df
    )  # This needs to be a deep copy so it doesn't break the other tests
    cols = copy.deepcopy(test_pd_df.columns.values)
    cols[-3:] = years
    df.columns = cols

    df = ScmRun(df)

    obs = df.time_points.values

    exp = np.array(
        [dt.datetime(2005, 1, 1), dt.datetime(2010, 1, 1), dt.datetime(2015, 1, 1)],
        dtype="datetime64[s]",
    )
    assert (obs == exp).all()


def test_init_with_year_columns(test_pd_df):
    df = ScmRun(test_pd_df)
    tdf = get_test_pd_df_with_datetime_columns(test_pd_df)
    pd.testing.assert_frame_equal(df.timeseries().reset_index(), tdf, check_like=True)


def test_init_with_decimal_years():
    inp_array = [2.0, 1.2, 7.9]
    d = pd.Series(inp_array, index=[1765.0, 1765.083, 1765.167])
    cols = {
        "model": ["a_model"],
        "scenario": ["a_scenario"],
        "region": ["World"],
        "variable": ["Primary Energy"],
        "unit": ["EJ/yr"],
    }

    res = ScmRun(d, columns=cols)
    assert (
        res["time"].unique()
        == [
            dt.datetime(1765, 1, 1, 0, 0),
            dt.datetime(1765, 1, 31, 7, 4, 48),
            dt.datetime(1765, 3, 2, 22, 55, 11),
        ]
    ).all()
    npt.assert_array_equal(res.values[0], inp_array)


def test_init_df_from_timeseries(test_scm_df_mulitple):
    df = ScmRun(test_scm_df_mulitple.timeseries())

    assert_scmdf_almost_equal(df, test_scm_df_mulitple, check_ts_names=False)


def test_init_df_with_extra_col(test_pd_df):
    tdf = test_pd_df.copy()

    extra_col = "test value"
    extra_value = "scm_model"
    tdf[extra_col] = extra_value

    df = ScmRun(tdf)

    tdf = get_test_pd_df_with_datetime_columns(tdf)
    assert extra_col in df.meta
    pd.testing.assert_frame_equal(df.timeseries().reset_index(), tdf, check_like=True)


def test_init_df_without_required_arguments(test_run_ts):
    with pytest.raises(ValueError, match="`columns` argument is required"):
        ScmRun(test_run_ts, index=[2000, 20005, 2010], columns=None)
    with pytest.raises(ValueError, match="`index` argument is required"):
        ScmRun(test_run_ts, index=None, columns={"variable": "test"})


def test_init_iam(test_iam_df, test_pd_df):
    a = ScmRun(test_iam_df)
    b = ScmRun(test_pd_df)

    assert_scmdf_almost_equal(a, b, check_ts_names=False)


def test_init_self(test_iam_df):
    a = ScmRun(test_iam_df)
    b = ScmRun(a)

    assert_scmdf_almost_equal(a, b)


def test_init_with_metadata(scm_run):
    expected_metadata = {"test": "example"}
    b = ScmRun(scm_run.timeseries(), metadata=expected_metadata)

    # Data should be copied
    assert id(b.metadata) != id(expected_metadata)
    assert b.metadata == expected_metadata


def test_init_self_with_metadata(scm_run):
    scm_run.metadata["test"] = "example"

    b = ScmRun(scm_run)
    assert id(scm_run.metadata) != id(b.metadata)
    assert scm_run.metadata == b.metadata

    c = ScmRun(scm_run, metadata={"test": "other"})
    assert c.metadata == {"test": "other"}


def _check_copy(a, b, copy_data):
    if copy_data:
        assert id(a.values.base) != id(b.values.base)
    else:
        assert id(a.values.base) == id(b.values.base)


@pytest.mark.parametrize("copy_data", [True, False])
def test_init_with_copy_run(copy_data, scm_run):
    res = ScmRun(scm_run, copy_data=copy_data)

    assert id(res) != id(scm_run)
    _check_copy(res._df, scm_run._df, copy_data)


@pytest.mark.parametrize("copy_data", [True, False])
def test_init_with_copy_dataframe(copy_data, test_pd_df):
    res = ScmRun(test_pd_df, copy_data=copy_data)

    # an incoming pandas DF no longer references the original
    _check_copy(res._df, test_pd_df, True)


def test_init_duplicate_columns(test_pd_df):
    exp_msg = (
        "Duplicate times (numbers show how many times the given " "time is repeated)"
    )
    inp = pd.concat([test_pd_df, test_pd_df[2015]], axis=1)
    with pytest.raises(DuplicateTimesError) as exc_info:
        ScmRun(inp)

    error_msg = exc_info.value.args[0]
    assert error_msg.startswith(exp_msg)
    pd.testing.assert_index_equal(
        pd.Index([2005, 2010, 2015, 2015], dtype="object", name="time"),
        exc_info.value.time_index,
    )


def test_init_empty(scm_run):
    empty_run = ScmRun()
    assert empty_run.empty
    assert empty_run.filter(model="*").empty

    empty_run.append(scm_run, inplace=True)
    assert not empty_run.empty


def test_repr_empty():
    empty_run = ScmRun()
    assert str(empty_run) == empty_run.__repr__()

    repr = str(empty_run)
    assert "Start: N/A" in repr
    assert "End: N/A" in repr

    assert "timeseries: 0, timepoints: 0" in repr


def test_as_iam(test_iam_df, test_pd_df, iamdf_type):
    df = ScmRun(test_pd_df).to_iamdataframe()

    # test is skipped by test_iam_df fixture if pyam isn't installed
    assert isinstance(df, iamdf_type)

    pd.testing.assert_frame_equal(test_iam_df.meta, df.meta)
    # we switch to time so ensure sensible comparison of columns
    tdf = df.data.copy()
    tdf["year"] = tdf["time"].apply(lambda x: x.year)
    tdf.drop("time", axis="columns", inplace=True)
    pd.testing.assert_frame_equal(test_iam_df.data, tdf, check_like=True)


def test_get_item(scm_run):
    assert scm_run["model"].unique() == ["a_iam"]


@pytest.mark.parametrize(
    "value,output",
    (
        (1, [np.nan, np.nan, 1.0]),
        (1.0, (np.nan, np.nan, 1.0)),
        ("test", ["nan", "nan", "test"]),
    ),
)
def test_get_item_with_nans(scm_run, value, output):
    expected_values = [np.nan, np.nan, value]
    scm_run["extra"] = expected_values
    exp = pd.Series(output, name="extra")

    pd.testing.assert_series_equal(scm_run["extra"], exp, check_exact=value != "test")


def test_get_item_not_in_meta(scm_run):
    dud_key = 0
    error_msg = re.escape("[{}] is not in metadata".format(dud_key))
    with pytest.raises(KeyError, match=error_msg):
        scm_run[dud_key]


def test_set_item(scm_run):
    scm_run["model"] = ["a_iam", "b_iam", "c_iam"]
    assert all(scm_run["model"] == ["a_iam", "b_iam", "c_iam"])


def test_set_item_not_in_meta(scm_run):
    with pytest.raises(ValueError):
        scm_run["junk"] = ["hi", "bye"]

    scm_run["junk"] = ["hi", "bye", "ciao"]
    assert all(scm_run["junk"] == ["hi", "bye", "ciao"])


def test_len(scm_run):
    assert len(scm_run) == len(scm_run.timeseries())


def test_shape(scm_run):
    assert scm_run.shape == scm_run.timeseries().shape


def test_head(scm_run):
    pd.testing.assert_frame_equal(scm_run.head(2), scm_run.timeseries().head(2))


def test_tail(scm_run):
    pd.testing.assert_frame_equal(scm_run.tail(1), scm_run.timeseries().tail(1))


def test_values(scm_run):
    # implicitly checks that `.values` returns the data with each row being a
    # timeseries and each column being a timepoint
    npt.assert_array_equal(scm_run.values, scm_run.timeseries().values)


def test_variable_depth_0(scm_run):
    obs = list(scm_run.filter(level=0)["variable"].unique())
    exp = ["Primary Energy"]
    assert obs == exp


def test_variable_depth_0_with_base():
    tdf = ScmRun(
        data=np.array([[1, 6.0, 7], [0.5, 3, 2], [2, 7, 0], [-1, -2, 3]]).T,
        columns={
            "model": ["a_iam"],
            "climate_model": ["a_model"],
            "scenario": ["a_scenario"],
            "region": ["World"],
            "variable": [
                "Primary Energy",
                "Primary Energy|Coal",
                "Primary Energy|Coal|Electricity",
                "Primary Energy|Gas|Heating",
            ],
            "unit": ["EJ/yr"],
        },
        index=[
            dt.datetime(2005, 1, 1),
            dt.datetime(2010, 1, 1),
            dt.datetime(2015, 6, 12),
        ],
    )

    obs = list(tdf.filter(variable="Primary Energy|*", level=1)["variable"].unique())
    exp = ["Primary Energy|Coal|Electricity", "Primary Energy|Gas|Heating"]
    assert all([e in obs for e in exp]) and len(obs) == len(exp)


def test_variable_depth_0_keep_false(scm_run):
    obs = list(scm_run.filter(level=0, keep=False)["variable"].unique())
    exp = ["Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_0_minus(scm_run):
    obs = list(scm_run.filter(level="0-")["variable"].unique())
    exp = ["Primary Energy"]
    assert obs == exp


def test_variable_depth_0_plus(scm_run):
    obs = list(scm_run.filter(level="0+")["variable"].unique())
    exp = ["Primary Energy", "Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_1(scm_run):
    obs = list(scm_run.filter(level=1)["variable"].unique())
    exp = ["Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_1_minus(scm_run):
    obs = list(scm_run.filter(level="1-")["variable"].unique())
    exp = ["Primary Energy", "Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_1_plus(scm_run):
    obs = list(scm_run.filter(level="1+")["variable"].unique())
    exp = ["Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_raises(scm_run):
    pytest.raises(ValueError, scm_run.filter, level="1/")


def test_filter_error(scm_run):
    pytest.raises(ValueError, scm_run.filter, foo="foo")


def test_filter_year(test_scm_run_datetimes):
    obs = test_scm_run_datetimes.filter(year=2005)
    expected = dt.datetime(2005, 6, 17, 12)

    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


def test_filter_year_error(test_scm_run_datetimes):
    error_msg = re.escape("`year` can only be filtered with ints or lists of ints")
    with pytest.raises(TypeError, match=error_msg):
        test_scm_run_datetimes.filter(year=2005.0)


def test_filter_year_with_own_year(test_scm_run_datetimes):
    res = test_scm_run_datetimes.filter(year=test_scm_run_datetimes["year"].values)
    assert (res["year"].unique() == test_scm_run_datetimes["year"].unique()).all()


@pytest.mark.parametrize(
    "year_list",
    (
        [2005, 2010],
        (2005, 2010),
        np.array([2005, 2010]).astype(int),
    ),
)
def test_filter_year_list(year_list, test_scm_run_datetimes):
    res = test_scm_run_datetimes.filter(year=year_list)
    expected = [2005, 2010]

    assert (res["year"].unique() == expected).all()


def test_filter_inplace(test_scm_run_datetimes):
    test_scm_run_datetimes.filter(year=2005, inplace=True)
    expected = dt.datetime(2005, 6, 17, 12)

    unique_time = test_scm_run_datetimes["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("test_month", [6, "June", "Jun", "jun", ["Jun", "jun"]])
def test_filter_month(test_scm_run_datetimes, test_month):
    obs = test_scm_run_datetimes.filter(month=test_month)
    expected = dt.datetime(2005, 6, 17, 12)
    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("test_month", [6, "Jun", "jun", ["Jun", "jun"]])
def test_filter_year_month(test_scm_run_datetimes, test_month):
    obs = test_scm_run_datetimes.filter(year=2005, month=test_month)
    expected = dt.datetime(2005, 6, 17, 12)
    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("test_day", [17, "Fri", "Friday", "friday", ["Fri", "fri"]])
def test_filter_day(test_scm_run_datetimes, test_day):
    obs = test_scm_run_datetimes.filter(day=test_day)
    expected = dt.datetime(2005, 6, 17, 12)
    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("test_hour", [12, [12, 13]])
def test_filter_hour(test_scm_run_datetimes, test_hour):
    obs = test_scm_run_datetimes.filter(hour=test_hour)
    test_hour = [test_hour] if isinstance(test_hour, int) else test_hour
    expected_rows = (
        test_scm_run_datetimes["time"].apply(lambda x: x.hour).isin(test_hour)
    )
    expected = test_scm_run_datetimes["time"].loc[expected_rows].unique()

    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected[0]


def test_filter_hour_multiple(test_scm_run_datetimes):
    obs = test_scm_run_datetimes.filter(hour=0)
    expected_rows = test_scm_run_datetimes["time"].apply(lambda x: x.hour).isin([0])
    expected = test_scm_run_datetimes["time"].loc[expected_rows].unique()

    unique_time = obs["time"].unique()
    assert len(unique_time) == 2
    assert all([dt in unique_time for dt in expected])


def test_filter_time_exact_match(test_scm_run_datetimes):
    obs = test_scm_run_datetimes.filter(time=dt.datetime(2005, 6, 17, 12))
    expected = dt.datetime(2005, 6, 17, 12)
    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


def test_filter_time_range(test_scm_run_datetimes):
    error_msg = r".*datetime.datetime.*"
    with pytest.raises(TypeError, match=error_msg):
        test_scm_run_datetimes.filter(
            year=range(dt.datetime(2000, 6, 17), dt.datetime(2009, 6, 17))
        )


def test_filter_time_range_year(test_scm_run_datetimes):
    obs = test_scm_run_datetimes.filter(year=range(2000, 2008))

    unique_time = obs["time"].unique()
    expected = dt.datetime(2005, 6, 17, 12)

    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("month_range", [range(3, 7), "Mar-Jun"])
def test_filter_time_range_month(test_scm_run_datetimes, month_range):
    obs = test_scm_run_datetimes.filter(month=month_range)
    expected = dt.datetime(2005, 6, 17, 12)

    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


def test_filter_time_range_month_unrecognised_error(test_scm_run_datetimes):
    fail_filter = "Marb-Jun"
    error_msg = re.escape(
        "Could not convert month '{}' to integer".format(
            [m for m in fail_filter.split("-")]
        )
    )
    with pytest.raises(ValueError, match=error_msg):
        test_scm_run_datetimes.filter(month=fail_filter)


@pytest.mark.parametrize("month_range", [["Mar-Jun", "Nov-Feb"]])
def test_filter_time_range_round_the_clock_error(test_scm_run_datetimes, month_range):
    error_msg = re.escape(
        "string ranges must lead to increasing integer ranges, "
        "Nov-Feb becomes [11, 2]"
    )
    with pytest.raises(ValueError, match=error_msg):
        test_scm_run_datetimes.filter(month=month_range)


@pytest.mark.parametrize("day_range", [range(14, 20), "Thu-Sat"])
def test_filter_time_range_day(test_scm_run_datetimes, day_range):
    obs = test_scm_run_datetimes.filter(day=day_range)
    expected = dt.datetime(2005, 6, 17, 12)
    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


def test_filter_time_range_day_unrecognised_error(test_scm_run_datetimes):
    fail_filter = "Thud-Sat"
    error_msg = re.escape(
        "Could not convert day '{}' to integer".format(
            [m for m in fail_filter.split("-")]
        )
    )
    with pytest.raises(ValueError, match=error_msg):
        test_scm_run_datetimes.filter(day=fail_filter)


@pytest.mark.parametrize("hour_range", [range(10, 14)])
def test_filter_time_range_hour(test_scm_run_datetimes, hour_range):
    obs = test_scm_run_datetimes.filter(hour=hour_range)

    expected_rows = (
        test_scm_run_datetimes["time"].apply(lambda x: x.hour).isin(hour_range)
    )
    expected = test_scm_run_datetimes["time"][expected_rows].unique()

    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected[0]


def test_filter_time_no_match(test_scm_datetime_run):
    obs = test_scm_datetime_run.filter(time=dt.datetime(2004, 6, 18))
    assert len(obs.time_points) == 0
    assert obs.shape[1] == 0
    assert obs.values.shape[1] == 0


def test_filter_time_not_datetime_error(test_scm_run_datetimes):
    error_msg = re.escape("`time` can only be filtered with datetimes")
    with pytest.raises(TypeError, match=error_msg):
        test_scm_run_datetimes.filter(time=2005)


def test_filter_time_not_datetime_range_error(test_scm_run_datetimes):
    error_msg = re.escape("`time` can only be filtered with datetimes")
    with pytest.raises(TypeError, match=error_msg):
        test_scm_run_datetimes.filter(time=range(2000, 2008))


def test_filter_as_kwarg(scm_run):
    obs = list(scm_run.filter(variable="Primary Energy|Coal")["scenario"].unique())
    assert obs == ["a_scenario"]


def test_filter_keep_false_time(scm_run):
    df = scm_run.filter(year=2005, keep=False)
    assert 2005 not in df.time_points.years()
    assert 2010 in df.time_points.years()

    obs = df.filter(scenario="a_scenario").timeseries().values.ravel()
    npt.assert_array_equal(obs, [6, 6, 3, 3])


def test_filter_keep_false_metadata(scm_run):
    df = scm_run.filter(variable="Primary Energy|Coal", keep=False)
    assert "Primary Energy|Coal" not in df["variable"].tolist()
    assert "Primary Energy" in df["variable"].tolist()

    obs = df.filter(scenario="a_scenario").timeseries().values.ravel()
    npt.assert_array_equal(obs, [1, 6, 6])


def test_filter_keep_false_time_and_metadata(scm_run):
    error_msg = (
        "If keep==False, filtering cannot be performed on the temporal axis "
        "and with metadata at the same time"
    )
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        scm_run.filter(variable="Primary Energy|Coal", year=2005, keep=False)


def test_filter_keep_false_successive(scm_run):
    df = scm_run.filter(variable="Primary Energy|Coal", keep=False).filter(
        year=2005, keep=False
    )
    obs = df.filter(scenario="a_scenario").timeseries().values.ravel()
    npt.assert_array_equal(obs, [6, 6])


def test_filter_by_regexp(scm_run):
    obs = scm_run.filter(scenario="a_scenari.$", regexp=True)
    assert obs["scenario"].unique() == "a_scenario"


@pytest.mark.parametrize(
    "regexp,exp_units",
    (
        (True, []),
        (False, ["W/m^2"]),
    ),
)
def test_filter_by_regexp_caret(scm_run, regexp, exp_units):
    tunits = ["W/m2"] * scm_run.shape[1]
    tunits[-1] = "W/m^2"
    scm_run["unit"] = tunits
    obs = scm_run.filter(unit="W/m^2", regexp=regexp)

    if not exp_units:
        assert obs.empty
    else:
        assert obs.get_unique_meta("unit") == exp_units


def test_filter_asterisk_edgecase(scm_run):
    scm_run["extra"] = ["*", "*", "other"]
    obs = scm_run.filter(scenario="*")
    assert len(obs) == len(scm_run)

    obs = scm_run.filter(scenario="*", level=0)
    assert len(obs) == 2

    obs = scm_run.filter(scenario="a_scenario", level=0)
    assert len(obs) == 1

    # Weird case where "*" matches everything instead of "*" in
    obs = scm_run.filter(extra="*", regexp=False)
    assert len(obs) == len(scm_run)
    assert (obs["extra"] == ["*", "*", "other"]).all()

    # Not valid regex
    pytest.raises(re.error, scm_run.filter, extra="*", regexp=True)


def test_filter_timeseries_different_length():
    # This is different to how `ScmDataFrame` deals with nans
    # Nan and empty timeseries remain in the Run
    df = ScmRun(
        pd.DataFrame(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, np.nan]]).T, index=[2000, 2001, 2002]
        ),
        columns={
            "model": ["a_iam"],
            "climate_model": ["a_model"],
            "scenario": ["a_scenario", "a_scenario2"],
            "region": ["World"],
            "variable": ["Primary Energy"],
            "unit": ["EJ/yr"],
        },
    )

    npt.assert_array_equal(
        df.filter(scenario="a_scenario2").timeseries().squeeze(), [4.0, 5.0, np.nan]
    )
    npt.assert_array_equal(df.filter(year=2002).timeseries().squeeze(), [3.0, np.nan])

    exp = pd.Series(["a_scenario", "a_scenario2"], name="scenario")
    obs = df.filter(year=2002)["scenario"]
    pd.testing.assert_series_equal(exp, obs)
    assert not df.filter(scenario="a_scenario2", year=2002).timeseries().empty


def test_filter_timeseries_nan_meta():
    df = ScmRun(
        pd.DataFrame(
            np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]).T, index=[2000, 2001]
        ),
        columns={
            "model": ["a_iam"],
            "climate_model": ["a_model"],
            "scenario": ["a_scenario", "a_scenario2", np.nan],
            "region": ["World"],
            "variable": ["Primary Energy"],
            "unit": ["EJ/yr"],
        },
    )

    def with_nan_assertion(a, b):
        assert len(a) == len(b)
        assert all(
            [(v == b[i]) or (np.isnan(v) and np.isnan(b[i])) for i, v in enumerate(a)]
        )

    res = df.filter(scenario="*")["scenario"].unique()
    exp = ["a_scenario", "a_scenario2", np.nan]
    with_nan_assertion(res, exp)

    res = df.filter(scenario="")["scenario"].unique()
    exp = [np.nan]
    with_nan_assertion(res, exp)

    res = df.filter(scenario=np.nan)["scenario"].unique()
    exp = [np.nan]
    with_nan_assertion(res, exp)


def test_filter_index(scm_run):
    pd.testing.assert_index_equal(scm_run.meta.index, pd.Int64Index([0, 1, 2]))

    run = scm_run.filter(variable="Primary Energy")
    exp_index = pd.Int64Index([0, 2])
    pd.testing.assert_index_equal(run["variable"].index, exp_index)
    pd.testing.assert_index_equal(run.meta.index, exp_index)
    pd.testing.assert_index_equal(run._df.columns, exp_index)

    run = scm_run.filter(variable="Primary Energy", keep=False)
    exp_index = pd.Int64Index([1])
    pd.testing.assert_index_equal(run["variable"].index, exp_index)
    pd.testing.assert_index_equal(run.meta.index, exp_index)
    pd.testing.assert_index_equal(run._df.columns, exp_index)


def test_append_index(scm_run):
    def _check(res, reversed):
        exp_index = pd.Int64Index([0, 1, 2])
        pd.testing.assert_index_equal(res.meta.index, exp_index)

        exp_order = ["Primary Energy", "Primary Energy", "Primary Energy|Coal"]
        if reversed:
            exp_order = exp_order[::-1]
        pd.testing.assert_series_equal(
            res["variable"],
            pd.Series(
                exp_order,
                index=exp_index,
                name="variable",
            ),
        )

    res = run_append(
        [
            scm_run.filter(variable="Primary Energy"),
            scm_run.filter(variable="Primary Energy", keep=False),
        ]
    )
    _check(res, False)

    res = run_append(
        [
            scm_run.filter(variable="Primary Energy", keep=False),
            scm_run.filter(variable="Primary Energy"),
        ]
    )
    _check(res, True)


def test_append_index_extra(scm_run):
    runs = []
    for i in range(3):
        r = scm_run.filter(variable="Primary Energy")
        r["run_id"] = i + 1

        pd.testing.assert_index_equal(r.meta.index, pd.Int64Index([0, 2]))
        runs.append(r)

    res = run_append(runs)

    # note that the indexes are reset for subsequent appends and then increment
    exp_index = pd.Int64Index([0, 1, 2, 3, 4, 5])
    pd.testing.assert_index_equal(res.meta.index, exp_index)
    pd.testing.assert_series_equal(
        res["run_id"],
        pd.Series(
            [1, 1, 2, 2, 3, 3],
            index=exp_index,
            name="run_id",
        ),
    )


@pytest.mark.parametrize("value", [1, 1.0, "test"])
def test_append_nans(scm_run, value):
    run_1 = scm_run.copy()
    run_2 = scm_run.copy()
    run_2["extra"] = value

    res = run_append([run_1, run_2])

    # note that the indexes are reset for subsequent appends and then increment
    pd.testing.assert_series_equal(
        res["extra"],
        pd.Series(
            [np.nan, np.nan, np.nan, value, value, value],
            name="extra",
        ),
    )


times_to_check = (
    np.arange(1, 1000),
    np.arange(1750, 2100),
    np.arange(1850, 2100),
    np.arange(1850, 2300),
    np.arange(3000, 4000),
)


@pytest.mark.parametrize("time_1", times_to_check)
@pytest.mark.parametrize("time_2", times_to_check)
@pytest.mark.parametrize("try_start_1_from_df_with_datetime_index", (True, False))
@pytest.mark.parametrize("try_start_2_from_df_with_datetime_index", (True, False))
def test_append_long_times(
    time_1,
    time_2,
    try_start_1_from_df_with_datetime_index,
    try_start_2_from_df_with_datetime_index,
):
    scmrun_1 = ScmRun(
        data=np.arange(len(time_1)),
        index=time_1,
        columns={
            "model": "model_1",
            "scenario": "scenario_1",
            "variable": "variable_1",
            "region": "region_1",
            "unit": "unit_1",
        },
    )
    if try_start_1_from_df_with_datetime_index:
        scmrun_1_ts = scmrun_1.timeseries()
        try:
            scmrun_1_ts.columns = pd.DatetimeIndex(scmrun_1_ts.columns.values)
        except pd.errors.OutOfBoundsDatetime:
            pytest.skip("pandas datetime error")

    scmrun_2 = ScmRun(
        data=np.arange(len(time_2)),
        index=time_2,
        columns={
            "model": "model_2",
            "scenario": "scenario_2",
            "variable": "variable_2",
            "region": "region_2",
            "unit": "unit_2",
        },
    )
    if try_start_2_from_df_with_datetime_index:
        scmrun_2_ts = scmrun_2.timeseries()
        try:
            scmrun_2_ts.columns = pd.DatetimeIndex(scmrun_2_ts.columns.values)
            scmrun_2 = ScmRun(scmrun_2_ts)

        except pd.errors.OutOfBoundsDatetime:
            pytest.skip("pandas datetime error")

    res = scmrun_1.append(scmrun_2)

    assert not isinstance(res._df.index, pd.DatetimeIndex)
    exp_years = set(time_1).union(set(time_2))
    assert set(res["year"]) == exp_years


def test_timeseries(scm_run):
    dct = {
        "model": ["a_model"] * 3,
        "scenario": ["a_scenario"] * 3,
        "years": [2005, 2010, 2015],
        "value": [1, 6, 6],
    }
    exp = pd.DataFrame(dct).pivot_table(
        index=["model", "scenario"], columns=["years"], values="value"
    )
    obs = scm_run.filter(variable="Primary Energy", scenario="a_scenario").timeseries()
    npt.assert_array_equal(obs, exp)


def test_timeseries_meta(scm_run):
    obs = scm_run.filter(variable="Primary Energy").timeseries(
        meta=["scenario", "model"]
    )
    npt.assert_array_equal(obs.index.names, ["scenario", "model"])


def test_timeseries_duplicated(scm_run):
    pytest.raises(ValueError, scm_run.timeseries, meta=["scenario"])


@pytest.mark.parametrize("time_axis", (None, "year", "year-month"))
@pytest.mark.parametrize("drop_all_nan_times", (True, False))
def test_timeseries_drop_all_nan_times(drop_all_nan_times, time_axis):
    dat = np.arange(12).reshape(4, 3).astype(float)
    dat[3, :] = np.nan
    dat[1, 1] = np.nan
    time = [2010, 2020, 2030, 2040]
    start = ScmRun(
        dat,
        index=time,
        columns={
            "variable": ["v1", "v2", "v3"],
            **{k: k for k in ["model", "scenario", "region", "unit"]},
        },
    )

    res = start.timeseries(drop_all_nan_times=drop_all_nan_times, time_axis=time_axis)
    if drop_all_nan_times:
        # leave the solo nan, drop all others
        assert res.isnull().sum().sum() == 1
        assert len(res.columns) == 3
    else:
        assert res.isnull().sum().sum() == 4
        assert len(res.columns) == 4


def test_timeseries_index_ordered(scm_run):
    def _is_sorted(arr):
        arr = np.asarray(list(arr))
        return np.all(arr[:-1] <= arr[1:])

    def _check_order(r):
        assert _is_sorted(r.timeseries().index.names)
        assert not _is_sorted(r.timeseries(["variable", "scenario"]).index.names)
        assert _is_sorted(r.timeseries(["scenario", "variable"]).index.names)

        assert _is_sorted(r.meta.columns.names)

    _check_order(scm_run)

    new_ts = scm_run.timeseries()
    new_order = list(new_ts.index.names)[::-1]
    assert not _is_sorted(new_order)
    new_ts.index = new_ts.index.reorder_levels(new_order)

    _check_order(ScmRun(new_ts))


def test_quantile_over_lower(test_processing_scm_df):
    exp = pd.DataFrame(
        [
            ["a_model", "a_iam", "World", "Primary Energy", "EJ/yr", -1.0, -2.0, 0.0],
            [
                "a_model",
                "a_iam",
                "World",
                "Primary Energy|Coal",
                "EJ/yr",
                0.5,
                3.0,
                2.0,
            ],
        ],
        columns=[
            "climate_model",
            "model",
            "region",
            "variable",
            "unit",
            dt.datetime(2005, 1, 1),
            dt.datetime(2010, 1, 1),
            dt.datetime(2015, 6, 12),
        ],
    )
    obs = test_processing_scm_df.process_over("scenario", "quantile", q=0)
    pd.testing.assert_frame_equal(
        exp.set_index(obs.index.names), obs, check_like=True, check_column_type=False
    )


def test_quantile_over_upper(test_processing_scm_df):
    exp = pd.DataFrame(
        [
            ["a_model", "World", "Primary Energy", "EJ/yr", 2.0, 7.0, 7.0],
            ["a_model", "World", "Primary Energy|Coal", "EJ/yr", 0.5, 3.0, 2.0],
        ],
        columns=[
            "climate_model",
            "region",
            "variable",
            "unit",
            dt.datetime(2005, 1, 1),
            dt.datetime(2010, 1, 1),
            dt.datetime(2015, 6, 12),
        ],
    )
    obs = test_processing_scm_df.process_over(["model", "scenario"], "quantile", q=1)
    pd.testing.assert_frame_equal(
        exp.set_index(obs.index.names), obs, check_like=True, check_column_type=False
    )


def test_mean_over(test_processing_scm_df):
    exp = pd.DataFrame(
        [
            [
                "a_model",
                "a_iam",
                "World",
                "Primary Energy",
                "EJ/yr",
                2 / 3,
                11 / 3,
                10 / 3,
            ],
            [
                "a_model",
                "a_iam",
                "World",
                "Primary Energy|Coal",
                "EJ/yr",
                0.5,
                3.0,
                2.0,
            ],
        ],
        columns=[
            "climate_model",
            "model",
            "region",
            "variable",
            "unit",
            dt.datetime(2005, 1, 1),
            dt.datetime(2010, 1, 1),
            dt.datetime(2015, 6, 12),
        ],
    )
    obs = test_processing_scm_df.process_over("scenario", "mean")
    pd.testing.assert_frame_equal(
        exp.set_index(obs.index.names), obs, check_like=True, check_column_type=False
    )


def test_median_over(test_processing_scm_df):
    exp = pd.DataFrame(
        [
            ["a_model", "a_iam", "World", "Primary Energy", "EJ/yr", 1.0, 6.0, 3.0],
            [
                "a_model",
                "a_iam",
                "World",
                "Primary Energy|Coal",
                "EJ/yr",
                0.5,
                3.0,
                2.0,
            ],
        ],
        columns=[
            "climate_model",
            "model",
            "region",
            "variable",
            "unit",
            dt.datetime(2005, 1, 1),
            dt.datetime(2010, 1, 1),
            dt.datetime(2015, 6, 12),
        ],
    )
    obs = test_processing_scm_df.process_over("scenario", "median")
    pd.testing.assert_frame_equal(
        exp.set_index(obs.index.names), obs, check_like=True, check_column_type=False
    )


def test_arb_function_over(test_processing_scm_df):
    def same(df):
        return df

    obs = test_processing_scm_df.process_over("scenario", same)

    pd.testing.assert_frame_equal(
        test_processing_scm_df.timeseries(), obs, check_like=True
    )

    def add_2(df):
        return df + 2

    obs = test_processing_scm_df.process_over("scenario", add_2)
    pd.testing.assert_frame_equal(
        (test_processing_scm_df + 2).timeseries(), obs, check_like=True
    )


def test_arb_function_returns_none(test_processing_scm_df):
    def add_2_only_coal(df):
        variable = df.index.get_level_values("variable").unique()[0]
        if variable == "Primary Energy|Coal":
            return df + 2
        # implicit return None for any other variables

    obs = test_processing_scm_df.process_over("scenario", add_2_only_coal)
    exp = (
        test_processing_scm_df.filter(variable="Primary Energy|Coal") + 2
    ).timeseries()
    pd.testing.assert_frame_equal(
        exp,
        obs,
        check_like=True,
    )


@pytest.mark.parametrize(
    "operation",
    [
        "count",
        "cumcount",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
        "first",
        "last",
        "max",
        "mean",
        "median",
        "min",
        "prod",
        "rank",
        "std",
        "sum",
        "var",
    ],
)
def test_process_over_works(operation, test_processing_scm_df):
    obs = test_processing_scm_df.process_over("scenario", "median")
    assert isinstance(obs, pd.DataFrame)


def test_process_over_unrecognised_operation_error(scm_run):
    error_msg = re.escape("invalid process_over operation")
    with pytest.raises(ValueError, match=error_msg):
        scm_run.process_over("scenario", "junk")


def test_process_over_kwargs_error(scm_run):
    v = parse(pd.__version__)

    if v.major == 1 and v.minor < 1:
        exp_exc = UnsupportedFunctionCall
    else:
        exp_exc = TypeError
    with pytest.raises(exp_exc):
        scm_run.process_over("scenario", "mean", junk=4)


@pytest.mark.parametrize("na_override", [10000, -9999, 1.0e6])
def test_process_over_with_nans(scm_run, na_override):
    scm_run["nan_meta"] = np.nan
    res = scm_run.process_over(("variable",), "median", na_override=na_override)
    assert len(res) == 2

    npt.assert_array_equal(res.values[0, :], [0.75, 4.5, 4.5])
    npt.assert_array_equal(
        res.values[1, :],
        scm_run.filter(scenario="a_scenario2"),
    )


@pytest.mark.parametrize("na_override", [10000, -9999, 1.0e6])
def test_process_over_with_nans_raises(scm_run, na_override):
    scm_run["nan_meta"] = na_override

    with pytest.raises(
        ValueError,
        match="na_override clashes with existing meta: {}".format(na_override),
    ):
        scm_run.process_over(("variable",), "median", na_override=na_override)


def test_process_over_without_na_override(scm_run):
    res = scm_run.process_over(("variable",), "median", na_override=None)
    assert len(res) == 2

    npt.assert_array_equal(res.values[0, :], [0.75, 4.5, 4.5])
    npt.assert_array_equal(
        res.values[1, :],
        scm_run.filter(scenario="a_scenario2"),
    )

    scm_run["nan_meta"] = np.nan
    res = scm_run.process_over(("variable",), "median", na_override=None)
    assert len(res) == 0  # This result is incorrect due to the disabling na_override


def test_process_over_as_run(scm_run):
    with pytest.raises(MissingRequiredColumnError):
        res = scm_run.process_over(("variable",), "median", as_run=True)

    res = scm_run.process_over(
        ("variable",),
        "median",
        op_cols={"variable": "New Variable", "extra": "other"},
        as_run=True,
    )

    assert res.get_unique_meta("variable", True) == "New Variable"
    assert res.get_unique_meta("extra", True) == "other"

    # Ops cols are also processed if as_run=False
    res = scm_run.process_over(
        ("variable",),
        "median",
        op_cols={"variable": "New Variable", "extra": "other"},
    )
    assert res.index.get_level_values("variable").unique() == ["New Variable"]
    assert res.index.get_level_values("extra").unique() == ["other"]


def test_process_over_as_run_returns_series(scm_run):
    def total(g):
        return g.sum().sum()

    res = scm_run.process_over(
        ("variable",),
        total,
        op_cols={"variable": "Variable"},
        as_run=False,
    )
    assert isinstance(res, pd.Series)

    with pytest.raises(ValueError, match="Cannot convert pd.Series to ScmRun"):
        res = scm_run.process_over(
            ("variable",),
            total,
            op_cols={"variable": "Variable"},
            as_run=True,
        )


def test_process_over_as_run_with_class(scm_run):
    class CustomRun(BaseScmRun):
        required_cols = ("variable", "unit", "extra")

    with pytest.raises(MissingRequiredColumnError):
        res = scm_run.process_over(("variable",), "median", as_run=CustomRun)

    with pytest.raises(MissingRequiredColumnError):
        # Should still complain about missing columns
        res = scm_run.process_over(
            ("variable",), "median", as_run=CustomRun, op_cols={"variable": "Variable"}
        )

    res = scm_run.process_over(
        ("variable",),
        "median",
        op_cols={"variable": "New Variable", "extra": "other"},
        as_run=CustomRun,
    )
    assert isinstance(res, BaseScmRun)
    assert isinstance(res, CustomRun)

    assert res.get_unique_meta("variable", True) == "New Variable"
    assert res.get_unique_meta("extra", True) == "other"


def test_process_over_as_run_with_invalid_class(scm_run):
    with pytest.raises(
        ValueError,
        match="Invalid value for as_run. Expected True, False or class based on scmdata.run.BaseScmRun",
    ):
        scm_run.process_over(("variable",), "median", as_run=pd.DataFrame)


def test_process_over_as_run_with_metadata(scm_run):
    scm_run.metadata = {"test": "example"}

    res = scm_run.process_over(
        ("variable",),
        "median",
        op_cols={"variable": "New Variable", "extra": "other"},
        as_run=True,
    )

    assert res.metadata == scm_run.metadata


def test_quantiles_over(test_processing_scm_df):
    exp = pd.DataFrame(
        [
            ["a_model", "World", "Primary Energy", "EJ/yr", 0, -1.0, -2.0, 0.0],
            ["a_model", "World", "Primary Energy|Coal", "EJ/yr", 0, 0.5, 3.0, 2.0],
            ["a_model", "World", "Primary Energy", "EJ/yr", 0.5, 1.0, 6.0, 3.0],
            ["a_model", "World", "Primary Energy|Coal", "EJ/yr", 0.5, 0.5, 3.0, 2.0],
            ["a_model", "World", "Primary Energy", "EJ/yr", "median", 1.0, 6.0, 3.0],
            [
                "a_model",
                "World",
                "Primary Energy|Coal",
                "EJ/yr",
                "median",
                0.5,
                3.0,
                2.0,
            ],
            ["a_model", "World", "Primary Energy", "EJ/yr", 1, 2.0, 7.0, 7.0],
            ["a_model", "World", "Primary Energy|Coal", "EJ/yr", 1, 0.5, 3.0, 2.0],
            [
                "a_model",
                "World",
                "Primary Energy",
                "EJ/yr",
                "mean",
                2 / 3,
                11 / 3,
                10 / 3,
            ],
            ["a_model", "World", "Primary Energy|Coal", "EJ/yr", "mean", 0.5, 3.0, 2.0],
        ],
        columns=[
            "climate_model",
            "region",
            "variable",
            "unit",
            "quantile",
            dt.datetime(2005, 1, 1),
            dt.datetime(2010, 1, 1),
            dt.datetime(2015, 6, 12),
        ],
    )

    obs = test_processing_scm_df.quantiles_over(
        cols=["model", "scenario"],
        quantiles=[0, 0.5, 1, "mean", "median"],
    )
    pd.testing.assert_frame_equal(exp.set_index(obs.index.names), obs, check_like=True)


def test_quantiles_over_operation_in_kwargs(test_processing_scm_df):
    error_msg = re.escape(
        "quantiles_over() does not take the keyword argument 'operation', the "
        "operations are inferred from the 'quantiles' argument"
    )
    with pytest.raises(TypeError, match=error_msg):
        test_processing_scm_df.quantiles_over(
            cols=["model", "scenario"], quantiles=[0, 0.5, 1], operation="quantile"
        )


@pytest.mark.parametrize("test_quantile", [0.5, "mean", "median"])
def test_quantiles_over_filter(test_processing_scm_df, test_quantile):
    quantiles = test_processing_scm_df.quantiles_over(
        cols=["model", "scenario"],
        quantiles=[0, 0.5, 1, "mean", "median"],
    )
    quantiles["model"] = "model"
    quantiles["scenario"] = "scenario"
    quantiles = ScmRun(quantiles)

    res = quantiles.filter(quantile=test_quantile)

    assert res.get_unique_meta("quantile", True) == test_quantile


@pytest.mark.parametrize(
    "tfilter",
    [
        ({"time": [dt.datetime(y, 1, 1, 0, 0, 0) for y in range(2005, 2011)]}),
        ({"year": range(2005, 2011)}),
        # # dropped month and day support for now...
        # ({"month": [1, 2, 3]}),
        # ({"day": [1, 2, 3]}),
    ],
)
def test_relative_to_ref_period_mean(test_processing_scm_df, tfilter):
    if "year" in tfilter:
        start_year = tfilter["year"][0]
        end_year = tfilter["year"][-1]

    elif "time" in tfilter:
        start_year = tfilter["time"][0].year
        end_year = tfilter["time"][-1].year

    else:
        raise NotImplementedError(tfilter)

    exp = ScmRun(
        pd.DataFrame(
            [
                [
                    "a_model",
                    "a_iam",
                    "a_scenario",
                    "World",
                    "Primary Energy",
                    "EJ/yr",
                    start_year,
                    end_year,
                    -2.5,
                    2.5,
                    3.5,
                ],
                [
                    "a_model",
                    "a_iam",
                    "a_scenario",
                    "World",
                    "Primary Energy|Coal",
                    "EJ/yr",
                    start_year,
                    end_year,
                    -1.25,
                    1.25,
                    0.25,
                ],
                [
                    "a_model",
                    "a_iam",
                    "a_scenario2",
                    "World",
                    "Primary Energy",
                    "EJ/yr",
                    start_year,
                    end_year,
                    -2.5,
                    2.5,
                    -4.5,
                ],
                [
                    "a_model",
                    "a_iam",
                    "a_scenario3",
                    "World",
                    "Primary Energy",
                    "EJ/yr",
                    start_year,
                    end_year,
                    0.5,
                    -0.5,
                    4.5,
                ],
            ],
            columns=[
                "climate_model",
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "reference_period_start_year",
                "reference_period_end_year",
                dt.datetime(2005, 1, 1),
                dt.datetime(2010, 1, 1),
                dt.datetime(2015, 6, 12),
            ],
        )
    )

    obs = test_processing_scm_df.relative_to_ref_period_mean(**tfilter)

    assert isinstance(obs, ScmRun)

    obs_ts = obs.timeseries()
    exp_ts = exp.timeseries()

    pd.testing.assert_frame_equal(
        exp_ts.reorder_levels(obs_ts.index.names), obs_ts, check_like=True
    )


def test_append(scm_run):
    scm_run["col1"] = [5, 6, 7]
    other = scm_run.filter(scenario="a_scenario2").copy()
    other["variable"] = "Primary Energy clone"
    other["col1"] = 2
    other["col2"] = "b"

    df = scm_run.append(other)
    assert isinstance(df, ScmRun)

    # check that the new meta.index is updated, but not the original one
    assert "col1" in scm_run.meta_attributes

    # assert that merging of meta works as expected
    npt.assert_array_equal(
        df.meta.sort_values(["scenario", "variable"])["col1"].values, [5, 6, 7, 2]
    )
    pd.testing.assert_series_equal(
        df.meta.sort_values(["scenario", "variable"])["col2"].reset_index(drop=True),
        pd.Series([np.nan, np.nan, np.nan, "b"]),
        check_names=False,
    )

    # assert that appending data works as expected
    ts = df.timeseries().sort_index()
    npt.assert_array_equal(ts.iloc[0], ts.iloc[3])
    pd.testing.assert_index_equal(
        df.meta.columns,
        pd.Index(
            [
                "climate_model",
                "col1",
                "col2",
                "model",
                "region",
                "scenario",
                "unit",
                "variable",
            ]
        ),
    )


def test_append_exact_duplicates(scm_run):
    other = copy.deepcopy(scm_run)
    with pytest.warns(UserWarning):
        scm_run.append(other, duplicate_msg="warn").timeseries()

    assert_scmdf_almost_equal(scm_run, other, check_ts_names=False)


def test_append_duplicates(scm_run):
    other = copy.deepcopy(scm_run)
    other["time"] = [2020, 2030, 2040]

    res = scm_run.append(other, duplicate_msg="warn")

    obs = res.filter(scenario="a_scenario2").timeseries().squeeze()
    exp = [2.0, 7.0, 7.0, 2.0, 7.0, 7.0]
    npt.assert_array_equal(res["year"], [2005, 2010, 2015, 2020, 2030, 2040])
    npt.assert_almost_equal(obs, exp)


def test_append_duplicates_order_doesnt_matter(scm_run):
    other = copy.deepcopy(scm_run)
    other["time"] = [2020, 2030, 2040]
    other._df.iloc[2, 2] = 5.0

    res = other.append(scm_run, duplicate_msg="warn")

    obs = res.filter(scenario="a_scenario2").timeseries().squeeze()
    exp = [2.0, 7.0, 7.0, 2.0, 7.0, 5.0]
    npt.assert_array_equal(
        res._time_points.years(), [2005, 2010, 2015, 2020, 2030, 2040]
    )
    npt.assert_almost_equal(obs, exp)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("duplicate_msg", ("warn", True, False))
def test_append_duplicate_times(test_append_scm_runs, duplicate_msg):
    base = test_append_scm_runs["base"]
    other = test_append_scm_runs["other"]
    expected = test_append_scm_runs["expected"]

    if duplicate_msg and not isinstance(duplicate_msg, str):
        exp_msg = (
            "Duplicate metadata (numbers show how many times the given "
            "metadata is repeated)."
        )
        with pytest.raises(NonUniqueMetadataError) as exc_info:
            base.append(other, duplicate_msg=duplicate_msg)

        error_msg = exc_info.value.args[0]
        assert error_msg.startswith(exp_msg)
        pd.testing.assert_frame_equal(base.meta.append(other.meta), exc_info.value.meta)

        return

    with warnings.catch_warnings(record=True) as mock_warn_taking_average:
        res = base.append(other, duplicate_msg=duplicate_msg)

    if duplicate_msg == "warn":
        warn_msg = (
            "Duplicate time points detected, the output will be the average of "
            "the duplicates.  Set `duplicate_msg=False` to silence this message."
        )
        assert len(mock_warn_taking_average) == 1
        assert str(mock_warn_taking_average[0].message) == warn_msg
    else:
        assert not mock_warn_taking_average

    pd.testing.assert_frame_equal(
        res.timeseries(), expected.timeseries(), check_like=True
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_append_doesnt_warn_if_continuous_times(test_append_scm_runs):
    join_year = 2011
    base = test_append_scm_runs["base"].filter(year=range(1, join_year))
    other = test_append_scm_runs["other"].filter(year=range(join_year, 30000))

    with warnings.catch_warnings(record=True) as mock_warn_taking_average:
        base.append(other)

    assert len(mock_warn_taking_average) == 0


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_append_doesnt_warn_if_different(test_append_scm_runs):
    base = test_append_scm_runs["base"].filter(scenario="a_scenario")
    other = test_append_scm_runs["base"].filter(scenario="a_scenario2")

    with warnings.catch_warnings(record=True) as mock_warn_taking_average:
        base.append(other)

    assert len(mock_warn_taking_average) == 0


def test_append_duplicate_times_error_msg(scm_run):
    other = scm_run * 2

    error_msg = re.escape("Unrecognised value for duplicate_msg")
    with pytest.raises(ValueError, match=error_msg):
        scm_run.append(other, duplicate_msg="junk")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_append_inplace(scm_run):
    other = scm_run * 2

    obs = scm_run.filter(scenario="a_scenario2").timeseries().squeeze()
    exp = [2, 7, 7]
    npt.assert_almost_equal(obs, exp)
    with pytest.warns(UserWarning):  # test message elsewhere
        scm_run.append(other, inplace=True, duplicate_msg="warn")

    obs = scm_run.filter(scenario="a_scenario2").timeseries().squeeze()
    exp = [(2.0 + 4.0) / 2, (7.0 + 14.0) / 2, (7.0 + 14.0) / 2]
    npt.assert_almost_equal(obs, exp)


def get_append_col_order_time_dfs(base):
    other_2 = base.filter(variable="Primary Energy|Coal").copy()
    base["runmodus"] = "co2_only"
    other = base.copy()

    other_variable = other["variable"]
    other_variable.iloc[1] = "Primary Energy|Gas"

    other["variable"] = other_variable
    other["time"] = [
        dt.datetime(2002, 1, 1, 0, 0),
        dt.datetime(2008, 1, 1, 0, 0),
        dt.datetime(2009, 1, 1, 0, 0),
    ]

    other_2["ecs"] = 3.0
    other_2["climate_model"] = "a_model2"

    exp = ScmRun(
        pd.DataFrame(
            np.array(
                [
                    [1.0, 1.0, 6.0, 6.0, 6.0, 6.0],
                    [np.nan, 0.5, np.nan, np.nan, 3.0, 3.0],
                    [np.nan, 0.5, np.nan, np.nan, 3.0, 3.0],
                    [0.5, np.nan, 3.0, 3.0, np.nan, np.nan],
                    [2.0, 2.0, 7.0, 7.0, 7.0, 7.0],
                ]
            ).T,
            index=[2002, 2005, 2008, 2009, 2010, 2015],
        ),
        columns={
            "model": ["a_iam"],
            "climate_model": ["a_model", "a_model", "a_model2", "a_model", "a_model"],
            "scenario": [
                "a_scenario",
                "a_scenario",
                "a_scenario",
                "a_scenario",
                "a_scenario2",
            ],
            "region": ["World"],
            "variable": [
                "Primary Energy",
                "Primary Energy|Coal",
                "Primary Energy|Coal",
                "Primary Energy|Gas",
                "Primary Energy",
            ],
            "unit": ["EJ/yr"],
            "runmodus": ["co2_only", "co2_only", np.nan, "co2_only", "co2_only"],
            "ecs": [np.nan, np.nan, 3.0, np.nan, np.nan],
        },
    )

    return base, other, other_2, exp


def test_append_column_order_time_interpolation(scm_run):
    _, other, other_2, exp = get_append_col_order_time_dfs(scm_run)

    res = run_append([scm_run, other, other_2], duplicate_msg=False)

    pd.testing.assert_frame_equal(
        res.timeseries().sort_index(),
        exp.timeseries().reorder_levels(res.timeseries().index.names).sort_index(),
        check_like=True,
    )


def test_run_append_inplace_wrong_base(scm_run):
    error_msg = "Can only append inplace to an ScmRun"
    with pytest.raises(TypeError, match=error_msg):
        with warnings.catch_warnings(record=True):  # ignore warnings in this test
            run_append([scm_run.timeseries(), scm_run], inplace=True)


def test_run_append_empty(scm_run):
    assert run_append([ScmRun()]).empty
    assert run_append([ScmRun(), ScmRun()]).empty

    assert_scmdf_almost_equal(run_append([ScmRun(), scm_run]), scm_run)


def test_append_chain_column_order_time_interpolation(scm_run):
    base, other, other_2, exp = get_append_col_order_time_dfs(scm_run)

    res = scm_run.append(other, duplicate_msg="warn").append(
        other_2, duplicate_msg="warn"
    )

    pd.testing.assert_frame_equal(
        res.timeseries().sort_index(),
        exp.timeseries().reorder_levels(res.timeseries().index.names).sort_index(),
        check_like=True,
    )


def test_append_inplace_column_order_time_interpolation(scm_run):
    base, other, other_2, exp = get_append_col_order_time_dfs(scm_run)

    scm_run.append(other, duplicate_msg="warn", inplace=True)
    scm_run.append(other_2, duplicate_msg="warn", inplace=True)

    pd.testing.assert_frame_equal(
        scm_run.timeseries().sort_index(),
        exp.timeseries().reorder_levels(scm_run.timeseries().index.names).sort_index(),
        check_like=True,
    )


def test_append_inplace_preexisting_nan(scm_run):
    other = scm_run * 2
    other["climate_model"] = "a_model2"
    other["junk"] = np.nan

    original_ts = scm_run.timeseries().copy()
    res = scm_run.append(other)

    # make sure underlying hasn't changed when not appending inplace
    pd.testing.assert_frame_equal(original_ts, scm_run.timeseries())

    exp = pd.concat(
        [scm_run.timeseries().reset_index(), other.timeseries().reset_index()]
    )
    exp["junk"] = np.nan
    exp = exp.set_index(res.meta_attributes)

    pd.testing.assert_frame_equal(
        res.timeseries().reset_index(),
        exp.sort_index().reset_index(),
        check_like=True,
        check_dtype=False,
    )


@pytest.mark.parametrize("join_year", (2010, 2012))
@pytest.mark.parametrize("join_past", (True, False))
def test_append_timewise(join_year, join_past, scm_run_interpolated):
    start = scm_run_interpolated.filter(scenario="a_scenario")

    if join_past:
        base = start.filter(year=range(join_year, 2100))
        other = start.filter(year=range(1, join_year))
    else:
        other = start.filter(year=range(join_year, 2100))
        base = start.filter(year=range(1, join_year))

    other["scenario"] = "other"
    other["model"] = "test"

    res = base.append_timewise(other, align_columns=["variable", "unit"])

    assert_scmdf_almost_equal(res, start)
    assert (res.timeseries().columns == start.timeseries().columns).all()
    assert (res.timeseries().columns == res.timeseries().columns.sort_values()).all()
    assert "Start: 2005" in res.__repr__()


def test_append_timewise_future_and_past(scm_run_interpolated):
    start = scm_run_interpolated.filter(scenario="a_scenario")

    base = start.filter(year=range(2008, 2011))
    other = start.filter(year=base["year"].tolist(), keep=False)
    other["scenario"] = "other"
    other["model"] = "test"

    res = base.append_timewise(other, align_columns=["variable", "unit"])

    assert_scmdf_almost_equal(res, start)
    assert (res.timeseries().columns == start.timeseries().columns).all()
    assert (res.timeseries().columns == res.timeseries().columns.sort_values()).all()
    assert "Start: 2005" in res.__repr__()


def test_append_timewise_extra_col_in_hist(scm_run_interpolated):
    start = scm_run_interpolated.filter(scenario="a_scenario")

    join_year = 2010

    base = start.filter(year=range(join_year, 2100))
    history = start.filter(year=range(1, join_year))
    history["scenario"] = "history"
    history["model"] = "test"
    history["extra_col"] = "tester"

    res = base.append_timewise(history, align_columns=["variable", "unit", "extra_col"])

    exp = start.copy()
    exp["extra_col"] = "tester"
    assert_scmdf_almost_equal(res, exp)


@pytest.mark.xfail(
    _check_pandas_less_120(),
    reason="pandas<1.2.0 can't align properly",
)
def test_append_timewise_align_columns_one_to_many(scm_run_interpolated):
    start = scm_run_interpolated.copy()

    join_year = 2010

    base = start.filter(year=range(join_year, 2100))
    history = start.filter(scenario="a_scenario2", year=range(1, join_year))

    res = base.append_timewise(history, align_columns=["unit"])

    assert_scmdf_almost_equal(
        res.filter(year=range(join_year, 3000)),
        start.filter(year=range(join_year, 3000)),
    )

    for _, row in res.filter(year=range(1, join_year)).timeseries().iterrows():
        # check that history has been written into all timeseries
        npt.assert_allclose(
            row.values.squeeze(),
            history.values.squeeze(),
        )


@pytest.mark.xfail(
    _check_pandas_less_120(),
    reason="pandas<1.2.0 can't align properly",
)
def test_append_timewise_align_columns_many_to_many(scm_run_interpolated):
    start = scm_run_interpolated.copy()

    join_year = 2010

    base = start.filter(year=range(join_year, 2100))
    history = start.filter(variable="Primary Energy", year=range(1, join_year))

    res = base.append_timewise(history, align_columns=["scenario"])

    # unchanged after join year
    assert_scmdf_almost_equal(
        res.filter(year=range(join_year, 3000)),
        start.filter(year=range(join_year, 3000)),
    )

    for scenario, df in (
        res.filter(year=range(1, join_year)).timeseries().groupby("scenario")
    ):
        # check that correct history has been written into all timeseries
        exp_vals = history.filter(scenario=scenario).values.squeeze()
        res_vals = df.values.squeeze()
        npt.assert_allclose(res_vals, np.broadcast_to(exp_vals, res_vals.shape))


def test_append_timewise_ambiguous_history(scm_run_interpolated):
    error_msg = re.escape(
        "Calling ``other.timeseries(meta=align_columns)`` must "
        "result in umabiguous timeseries"
    )
    with pytest.raises(ValueError, match=error_msg):
        scm_run_interpolated.append_timewise(
            scm_run_interpolated,
            align_columns=["variable"],
        )


def test_append_timewise_overlapping_times(scm_run_interpolated):
    start = scm_run_interpolated.filter(scenario="a_scenario")

    base = start.filter(year=range(1, 2011))
    other = start.filter(year=range(2008, 3000))
    other["scenario"] = "other"
    other["model"] = "test"

    error_msg = re.escape("``self`` and ``other`` have overlapping times")
    with pytest.raises(ValueError, match=error_msg):
        base.append_timewise(other, align_columns=["variable", "unit"])


def test_append_timewise_no_match(scm_run_interpolated):
    start = scm_run_interpolated.copy()

    join_year = 2010
    base = start.filter(year=range(join_year, 3000))
    other = start.filter(
        year=range(1, join_year), variable="Primary Energy", scenario="a_scenario"
    )
    other["scenario"] = "other"
    other["model"] = "test"

    res = base.append_timewise(other, align_columns=["variable", "unit"])

    # unchanged after join year
    assert_scmdf_almost_equal(
        res.filter(year=range(join_year, 3000)),
        start.filter(year=range(join_year, 3000)),
        check_ts_names=False,
        allow_unordered=True,
    )

    for variable, df in (
        res.filter(year=range(1, join_year)).timeseries().groupby("variable")
    ):
        # check that correct other has been written into all timeseries
        exp_vals = other.filter(variable=variable)
        if exp_vals.empty:
            # no other provided hence get nans in output
            # question for Jared: should we raise a warning when this happens?
            assert df.isnull().all().all()

        else:
            exp_vals = exp_vals.values.squeeze()
            res_vals = df.values.squeeze()
            npt.assert_allclose(res_vals, np.broadcast_to(exp_vals, res_vals.shape))


def test_interpolate(combo_df):
    combo, df = combo_df
    target_time_points = combo.target

    res = df.interpolate(
        target_time_points,
        interpolation_type=combo.interpolation_type,
        extrapolation_type=combo.extrapolation_type,
    )

    npt.assert_array_almost_equal(res.values.squeeze(), combo.target_values)


@pytest.mark.parametrize(
    "source",
    [[1.0, 2.0, 3.0, np.nan], [1.0, 2.0, np.nan, 4.0], [np.nan, 2.0, 3.0, 4.0]],
)
def test_interpolate_nan(source):
    df = ScmRun(
        source,
        columns={
            "scenario": ["a_scenario"],
            "model": ["a_model"],
            "region": ["World"],
            "variable": ["Emissions|BC"],
            "unit": ["Mg /yr"],
        },
        index=[2000, 2100, 2200, 2300],
    )
    res = df.interpolate(
        [datetime(y, 1, 1) for y in [2000, 2100, 2200, 2300, 2400]],
        interpolation_type="linear",
        extrapolation_type="linear",
    )

    npt.assert_array_almost_equal(
        res.values.squeeze(), [1.0, 2.0, 3.0, 4.0, 5.0], decimal=4
    )


def test_interpolate_nan_constant():
    df = ScmRun(
        [1.0, 2.0, 3.0, np.nan],
        columns={
            "scenario": ["a_scenario"],
            "model": ["a_model"],
            "region": ["World"],
            "variable": ["Emissions|BC"],
            "unit": ["Mg /yr"],
        },
        index=[2000, 2100, 2200, 2300],
    )
    res = df.interpolate(
        [datetime(y, 1, 1) for y in [2000, 2100, 2200, 2300, 2400]],
        interpolation_type="linear",
        extrapolation_type="constant",
    )

    npt.assert_array_almost_equal(
        res.values.squeeze(), [1.0, 2.0, 3.0, 3.0, 3.0], decimal=4
    )


def test_interpolate_single_constant():
    value = 2.0
    df = ScmRun(
        [value],
        columns={
            "scenario": ["a_scenario"],
            "model": ["a_model"],
            "region": ["World"],
            "variable": ["Emissions|BC"],
            "unit": ["Mg /yr"],
        },
        index=[2000],
    )
    target = [datetime(y, 1, 1) for y in [2000, 2100, 2200, 2300, 2400]]
    res = df.interpolate(
        target,
        extrapolation_type="constant",
    )

    npt.assert_array_almost_equal(res.values.squeeze(), [value] * 5, decimal=4)

    # Non-constant extrapolation (default) should fail
    with pytest.raises(InsufficientDataError):
        df.interpolate(target)


def test_time_mean_year_beginning_of_year(test_scm_df_monthly):
    # should be annual mean centred on January 1st of each year
    res = test_scm_df_monthly.time_mean("AS")

    assert isinstance(res, ScmRun)

    # test by hand
    npt.assert_allclose(
        res.filter(variable="Radiative Forcing", year=1992, month=1, day=1).values,
        np.average(np.arange(6)),
    )
    npt.assert_allclose(
        res.filter(variable="Radiative Forcing", year=1996, month=1, day=1).values,
        np.average([42, 43, 44]),
    )

    # automate rest of tests
    def group_annual_mean_beginning_of_year(x):
        if x.month <= 6:
            return x.year
        return x.year + 1

    ts_resampled = (
        test_scm_df_monthly.timeseries()
        .T.groupby(group_annual_mean_beginning_of_year)
        .mean()
        .T
    )
    ts_resampled.columns = ts_resampled.columns.map(lambda x: dt.datetime(x, 1, 1))

    pd.testing.assert_frame_equal(res.timeseries(), ts_resampled, check_like=True)


def test_time_mean_year(test_scm_df_monthly):
    # should be annual mean (using all values in that year)
    res = test_scm_df_monthly.time_mean("AC")

    assert isinstance(res, ScmRun)

    # test by hand
    npt.assert_allclose(
        res.filter(variable="Radiative Forcing", year=1992, month=7, day=1).values,
        np.average(np.arange(12)),
    )
    npt.assert_allclose(
        res.filter(variable="Radiative Forcing", year=1995, month=7, day=1).values,
        np.average(np.arange(36, 45)),
    )

    # automate rest of tests
    def group_annual_mean(x):
        return x.year

    ts_resampled = (
        test_scm_df_monthly.timeseries().T.groupby(group_annual_mean).mean().T
    )
    ts_resampled.columns = ts_resampled.columns.map(lambda x: dt.datetime(x, 7, 1))

    pd.testing.assert_frame_equal(res.timeseries(), ts_resampled, check_like=True)


def test_time_mean_year_end_of_year(test_scm_df_monthly):
    # should be annual mean centred on December 31st of each year
    res = test_scm_df_monthly.time_mean("A")

    assert isinstance(res, ScmRun)

    # test by hand
    npt.assert_allclose(
        res.filter(variable="Radiative Forcing", year=1991, month=12, day=31).values,
        np.average(np.arange(6)),
    )
    npt.assert_allclose(
        res.filter(variable="Radiative Forcing", year=1995, month=12, day=31).values,
        np.average(np.arange(42, 45)),
    )

    # automate rest of tests
    def group_annual_mean_end_of_year(x):
        if x.month >= 7:
            return x.year
        return x.year - 1

    ts_resampled = (
        test_scm_df_monthly.timeseries()
        .T.groupby(group_annual_mean_end_of_year)
        .mean()
        .T
    )
    ts_resampled.columns = ts_resampled.columns.map(lambda x: dt.datetime(x, 12, 31))

    pd.testing.assert_frame_equal(res.timeseries(), ts_resampled, check_like=True)


def test_time_mean_unsupported_style(test_scm_df_monthly):
    error_msg = re.escape("`rule` = `junk` is not supported")
    with pytest.raises(ValueError, match=error_msg):
        test_scm_df_monthly.time_mean("junk")


def test_set_meta_wrong_length(scm_run):
    s = [0.3, 0.4]
    with pytest.raises(ValueError, match="Invalid length for metadata"):
        scm_run["meta_series"] = s


def test_set_meta_as_float(scm_run):
    scm_run["meta_int"] = 3.2

    exp = pd.Series(data=[3.2, 3.2, 3.2], index=scm_run.meta.index, name="meta_int")

    obs = scm_run["meta_int"]
    pd.testing.assert_series_equal(obs, exp)
    pd.testing.assert_index_equal(
        scm_run.meta.columns,
        pd.Index(
            [
                "climate_model",
                "meta_int",
                "model",
                "region",
                "scenario",
                "unit",
                "variable",
            ]
        ),
    )


def test_set_meta_as_str(scm_run):
    scm_run["meta_str"] = "testing"

    exp = pd.Series(
        data=["testing", "testing", "testing"],
        index=scm_run.meta.index,
        name="meta_str",
    )

    obs = scm_run["meta_str"]
    pd.testing.assert_series_equal(obs, exp)

    pd.testing.assert_index_equal(
        scm_run.meta.columns,
        pd.Index(
            [
                "climate_model",
                "meta_str",
                "model",
                "region",
                "scenario",
                "unit",
                "variable",
            ]
        ),
    )


def test_set_meta_as_str_list(scm_run):
    scm_run["category"] = ["testing", "testing2", "testing2"]
    obs = scm_run.filter(category="testing")
    assert obs["scenario"].unique() == "a_scenario"


def test_filter_by_bool(scm_run):
    scm_run["exclude"] = [True, False, False]
    obs = scm_run.filter(exclude=True)
    assert obs["scenario"].unique() == "a_scenario"


def test_filter_by_int(scm_run):
    scm_run["test"] = [1, 2, 3]
    obs = scm_run.filter(test=1)
    assert obs["scenario"].unique() == "a_scenario"


def test_filter_empty(scm_run, caplog):
    with caplog.at_level(logging.WARNING, logger="scmdata.run"):
        empty_run = scm_run.filter(variable="not a variable")

    assert caplog.records[0].stack_info is not None
    assert empty_run.shape == (0, 3)

    # Filtering an empty run should result in an empty run
    res = empty_run.filter(variable="anything")
    assert res.shape == (0, 3)

    # Filtering for an empty run is a noop (the times aren't filtered)
    res = empty_run.filter(year=range(2000, 2010))
    assert res.shape == (0, 3)
    assert id(res) != id(empty_run)


@pytest.mark.xfail(
    _check_pandas_less_110(), reason="pandas<=1.1.0 does not have rtol argument"
)
@pytest.mark.parametrize(
    ("target_unit", "input_units", "filter_kwargs", "expected", "expected_units"),
    [
        ("EJ/yr", "EJ/yr", {}, [1.0, 0.5, 2.0], ["EJ/yr", "EJ/yr", "EJ/yr"]),
        (
            "EJ/yr",
            "EJ/yr",
            {"variable": "Primary Energy"},
            [0.5, 1.0, 2.0],
            ["EJ/yr", "EJ/yr", "EJ/yr"],
        ),
        ("PJ/yr", "EJ/yr", {}, [1000.0, 500.0, 2000.0], ["PJ/yr", "PJ/yr", "PJ/yr"]),
        (
            "PJ/yr",
            "EJ/yr",
            {"scenario": "a_scenario2"},
            [2000.0, 1.0, 0.5],
            ["PJ/yr", "EJ/yr", "EJ/yr"],
        ),
        (
            "PJ/yr",
            ["EJ/yr", "TJ/yr", "Gt C / yr"],
            {"variable": "Primary Energy|Coal"},
            [0.5 * 1e-3, 1.0, 2.0],
            ["PJ/yr", "EJ/yr", "Gt C / yr"],
        ),
        ("W/m^2", "W/m^2", {}, [1.0, 0.5, 2.0], ["W/m^2", "W/m^2", "W/m^2"]),
        (
            "W/km^2",
            "W/m^2",
            {},
            [1.0 * 1e6, 0.5 * 1e6, 2.0 * 1e6],
            ["W/km^2", "W/km^2", "W/km^2"],
        ),
    ],
)
def test_convert_unit(
    scm_run, target_unit, input_units, filter_kwargs, expected, expected_units
):
    scm_run["unit"] = input_units
    obs = scm_run.convert_unit(target_unit, **filter_kwargs)

    exp_units = pd.Series(expected_units, name="unit")

    pd.testing.assert_series_equal(obs["unit"], exp_units, rtol=1e-3)
    npt.assert_array_almost_equal(obs.filter(year=2005).values.squeeze(), expected)
    assert (scm_run["unit"] == input_units).all()


def test_convert_unit_unknown_unit(scm_run):
    unknown_unit = "Unknown"
    scm_run["unit"] = unknown_unit

    error_msg = re.escape(
        "'{}' is not defined in the unit registry".format(unknown_unit)
    )
    with pytest.raises(UndefinedUnitError, match=error_msg):
        scm_run.convert_unit("EJ/yr")


def test_convert_unit_dimensionality(scm_run):
    error_msg = "Cannot convert from 'exajoule / a' .* to 'kelvin'"
    with pytest.raises(DimensionalityError, match=error_msg):
        scm_run.convert_unit("kelvin")


@pytest.mark.xfail(reason="inplace not working")
def test_convert_unit_inplace(scm_run):
    units = scm_run["unit"].copy()

    ret = scm_run.convert_unit("PJ/yr", inplace=True)
    assert ret is None

    assert (scm_run["unit"] != units).all()
    npt.assert_array_almost_equal(
        scm_run.filter(year=2005).values.squeeze(), [1000.0, 500.0, 2000.0]
    )


def test_convert_unit_target_unit_in_input():
    inp = pd.DataFrame(
        [
            {
                "climate_model": "MAGICC6",
                "model": "unspecified",
                "scenario": "historical",
                "region": "World",
                "variable": "Atmospheric Lifetime|CH4",
                "unit": "yr",
                "ensemble_member": 0,
            },
            {
                "climate_model": "MAGICC6",
                "model": "unspecified",
                "scenario": "historical",
                "region": "World",
                "variable": "Atmospheric Lifetime|CH4",
                "unit": "yr",
                "ensemble_member": 1,
            },
            {
                "climate_model": "FaIR1.6",
                "model": "unspecified",
                "scenario": "historical",
                "region": "World",
                "variable": "Atmospheric Lifetime|CH4",
                "unit": "month",
                "ensemble_member": 0,
            },
            {
                "climate_model": "FaIR1.6",
                "model": "unspecified",
                "scenario": "historical",
                "region": "World",
                "variable": "Atmospheric Lifetime|CH4",
                "unit": "month",
                "ensemble_member": 1,
            },
            {
                "climate_model": "FaIR1.6",
                "model": "unspecified",
                "scenario": "historical",
                "region": "World",
                "variable": "Atmospheric Lifetime|CH4",
                "unit": "month",
                "ensemble_member": 10,
            },
            {
                "climate_model": "FaIR1.6",
                "model": "unspecified",
                "scenario": "historical",
                "region": "World",
                "variable": "Atmospheric Lifetime|CH4",
                "unit": "month",
                "ensemble_member": 11,
            },
            {
                "climate_model": "FaIR1.6",
                "model": "unspecified",
                "scenario": "historical",
                "region": "World",
                "variable": "Atmospheric Lifetime|CH4",
                "unit": "month",
                "ensemble_member": 12,
            },
        ]
    )
    inp[2010] = 12
    inp[2015] = 10

    inp = ScmRun(inp)

    res = inp.convert_unit("yr")

    assert res.get_unique_meta("unit", no_duplicates=True) == "yr"


def test_convert_unit_context(scm_run):
    scm_run = scm_run.filter(
        variable="Primary Energy"
    )  # Duplicated meta if set all 3 ts to the same variable name
    scm_run["unit"] = "kg SF5CF3 / yr"
    scm_run["variable"] = "SF5CF3"

    obs = scm_run.convert_unit("kg CO2 / yr", context="AR4GWP100")
    factor = 17700
    expected = [1.0 * factor, 2.0 * factor]
    npt.assert_array_almost_equal(obs.filter(year=2005).values.squeeze(), expected)
    assert all(obs["unit_context"] == "AR4GWP100")

    error_msg = (
        "Cannot convert from 'SF5CF3 * kilogram / a' "
        "([SF5CF3] * [mass] / [time]) to "
        "'CO2 * kilogram / a' ([carbon] * [mass] / [time])"
    )
    with pytest.raises(DimensionalityError, match=re.escape(error_msg)):
        scm_run.convert_unit("kg CO2 / yr")


def test_convert_existing_unit_context(scm_run):
    scm_run = scm_run.filter(
        variable="Primary Energy"
    )  # Duplicated meta if set all 3 ts to the same variable name
    scm_run["unit"] = "kg SF5CF3 / yr"
    scm_run["variable"] = "SF5CF3"
    scm_run["unit_context"] = "AR4GWP100"

    obs = scm_run.convert_unit("kg CO2 / yr", context="AR4GWP100")
    factor = 17700
    expected = [1.0 * factor, 2.0 * factor]
    npt.assert_array_almost_equal(obs.filter(year=2005).values.squeeze(), expected)
    assert all(obs["unit_context"] == "AR4GWP100")


def test_unit_context_not_added_if_context_is_none(scm_run):
    start = scm_run.filter(variable="Primary Energy")
    start["unit"] = "EJ/yr"

    res = start.convert_unit("MJ/yr")

    assert "unit_context" not in res.meta_attributes


def test_unit_context_added_if_context_is_not_none(scm_run):
    start = scm_run.filter(variable="Primary Energy")
    start["unit"] = "EJ/yr"

    res = start.convert_unit("MJ/yr", context="AR4GWP100")

    assert "unit_context" in res.meta_attributes


@pytest.mark.parametrize("context", (None, "AR4GWP100"))
def test_unit_context_no_existing_contexts(scm_run, context):
    to_convert = "*Coal"
    res = scm_run.convert_unit("MJ/yr", variable=to_convert, context=context)

    if context is None:
        assert "unit_context" not in res.meta_attributes
    else:
        assert "unit_context" in res.meta_attributes
        assert "AR4GWP100" == res.filter(variable=to_convert).get_unique_meta(
            "unit_context", no_duplicates=True
        )
        assert np.isnan(
            res.filter(variable=to_convert, keep=False).get_unique_meta(
                "unit_context", no_duplicates=True
            )
        )


def _check_context_or_nan(run, variable, exp, keep=True):
    if exp is None:
        assert np.isnan(
            run.filter(variable=variable, keep=keep).get_unique_meta(
                "unit_context", no_duplicates=True
            )
        )
    else:
        assert (
            run.filter(variable=variable, keep=keep).get_unique_meta(
                "unit_context", no_duplicates=True
            )
            == exp
        )


@pytest.mark.parametrize("to_not_convert_matches", (True, False))
@pytest.mark.parametrize("context", (None, "AR4GWP100"))
def test_unit_context_both_have_existing_context(
    scm_run, context, to_not_convert_matches
):
    to_convert = "*Coal"

    scm_run["unit_context"] = context
    if to_not_convert_matches:
        to_not_convert_context = context
    else:
        to_not_convert_context = "junk"

    def _set_not_convert_context(v):
        if not re.search(r".*Coal", v):
            return to_not_convert_context

    scm_run["unit_context"] = scm_run["variable"].apply(_set_not_convert_context)

    res = scm_run.convert_unit("MJ/yr", variable=to_convert, context=context)

    _check_context_or_nan(res, to_convert, context)
    _check_context_or_nan(res, to_convert, to_not_convert_context, keep=False)


@pytest.mark.parametrize("to_not_convert_matches", (True, False))
@pytest.mark.parametrize("context", (None, "AR4GWP100"))
def test_unit_context_both_have_existing_context_error(
    scm_run, context, to_not_convert_matches
):
    to_convert = "*Coal"

    scm_run["unit_context"] = "junk"
    if to_not_convert_matches:
        scm_run.filter(variable=to_convert, keep=False)["unit_context"] = context

    error_msg = re.escape(
        "Existing unit conversion context(s), `['junk']`, doesn't match input context, "
        "`{}`, drop `unit_context` metadata before doing conversion".format(context)
    )
    with pytest.raises(ValueError, match=error_msg):
        scm_run.convert_unit("MJ/yr", variable=to_convert, context=context)


@pytest.mark.parametrize("context", ("AR5GWP100", "AR4GWP100"))
def test_unit_context_to_convert_has_existing_context(scm_run, context):
    to_convert = "*Coal"
    start = scm_run.convert_unit("MJ/yr", variable=to_convert, context=context)

    _check_context_or_nan(start, to_convert, context)
    _check_context_or_nan(start, to_convert, None, keep=False)

    res = start.convert_unit("GJ/yr", variable=to_convert, context=context)

    _check_context_or_nan(start, to_convert, context)
    _check_context_or_nan(start, to_convert, None, keep=False)
    assert (
        res.filter(variable=to_convert).get_unique_meta("unit", no_duplicates=True)
        == "GJ/yr"
    )
    assert (
        res.filter(variable=to_convert, keep=False).get_unique_meta(
            "unit", no_duplicates=True
        )
        == "EJ/yr"
    )


@pytest.mark.parametrize("context", ("AR5GWP100", "AR4GWP100"))
def test_unit_context_to_convert_has_existing_context_error(scm_run, context):
    to_convert = "*Coal"
    start = scm_run.convert_unit("MJ/yr", variable=to_convert, context=context)

    _check_context_or_nan(start, to_convert, context)
    _check_context_or_nan(start, to_convert, None, keep=False)

    error_msg = re.escape(
        "Existing unit conversion context(s), `['{}']`, doesn't match input context, `junk`, drop "
        "`unit_context` metadata before doing conversion".format(context)
    )
    with pytest.raises(ValueError, match=error_msg):
        start.convert_unit("GJ/yr", variable=to_convert, context="junk")

    with pytest.raises(ValueError, match=error_msg):
        start.convert_unit("MJ/yr", variable=to_convert, context="junk")


@pytest.mark.parametrize("context", ("AR5GWP100", "AR4GWP100", None))
@pytest.mark.parametrize("to_not_convert_context", ("AR5GWP100", "AR4GWP100"))
def test_unit_context_to_not_convert_has_existing_context(
    scm_run, context, to_not_convert_context
):
    to_convert = "*Coal"
    to_not_convert = scm_run.filter(variable=to_convert, keep=False).get_unique_meta(
        "variable"
    )
    start = scm_run.convert_unit(
        "MJ/yr", variable=to_not_convert, context=to_not_convert_context
    )

    _check_context_or_nan(start, to_convert, None)
    _check_context_or_nan(start, to_convert, to_not_convert_context, keep=False)

    # no error, irrespective of context because to_convert context is nan
    res = start.convert_unit("GJ/yr", variable=to_convert, context=context)

    _check_context_or_nan(res, to_convert, context)
    _check_context_or_nan(res, to_not_convert, to_not_convert_context)

    assert (
        res.filter(variable=to_convert).get_unique_meta("unit", no_duplicates=True)
        == "GJ/yr"
    )
    assert (
        res.filter(variable=to_convert, keep=False).get_unique_meta(
            "unit", no_duplicates=True
        )
        == "MJ/yr"
    )


@pytest.mark.parametrize(
    "start_units,target_unit",
    (
        (["W/m^2", "W / m^2", "W/m ^ 2", "W / m ** 2"], "W/m^2"),
        (["W/m^2", "W/m ^ 2", "W / m^2", "W / m ** 2"], "W/m^2"),
        (["W / m^2", "W /m^2", "W / m ** 2", "W/m^2"], "W/m^2"),
        (
            ["GtC/yr/m^2", "GtC / yr / m^2", "GtC/ yr /m ^  2", "GtC/yr/m ^ 2"],
            "GtC / yr / m^2",
        ),
        (["MtCH4 / yr", "Mt CH4/yr", "Mt CH4 / yr", "Mt CH4/ yr"], "MtCH4/yr"),
    ),
)
@pytest.mark.parametrize("duplicated_meta_once_converted", [True, False])
def test_convert_unit_multiple_units(
    start_units, target_unit, duplicated_meta_once_converted
):
    if duplicated_meta_once_converted:
        climate_model = "climate_model"
    else:
        climate_model = ["cma", "cmb", "cmc", "cmd"]

    tdf = ScmRun(
        data=np.arange(12).reshape(3, 4),
        index=range(2010, 2031, 10),
        columns={
            "variable": "Effective Radiative Forcing",
            "unit": start_units,
            "region": "World",
            "scenario": "idealised",
            "model": "idealised",
            "climate_model": climate_model,
        },
    )

    if duplicated_meta_once_converted:
        with pytest.raises(NonUniqueMetadataError):
            tdf.convert_unit(target_unit)

    else:
        res = tdf.convert_unit(target_unit)
        assert res.get_unique_meta("unit", no_duplicates=True) == target_unit
        npt.assert_allclose(
            res.timeseries().sort_index().values,
            tdf.timeseries().sort_index().values,
        )


def test_convert_unit_does_not_warn(scm_run, caplog):
    scm_run["unit"] = "GtC"

    res = scm_run.convert_unit("MtC")

    npt.assert_equal(len(caplog.records), 0)
    npt.assert_array_equal(scm_run.values, res.values / 10**3)


def test_resample():
    df_dts = [
        dt.datetime(2000, 1, 1),
        dt.datetime(2000, 6, 1),
        dt.datetime(2001, 1, 1),
        dt.datetime(2001, 6, 1),
        dt.datetime(2002, 1, 1),
        dt.datetime(2002, 6, 1),
        dt.datetime(2003, 1, 1),
    ]
    df = ScmRun(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        columns={
            "scenario": ["a_scenario"],
            "model": ["a_model"],
            "region": ["World"],
            "variable": ["Emissions|BC"],
            "unit": ["Mg /yr"],
        },
        index=df_dts,
    )
    res = df.resample("AS")

    obs = res.values.squeeze()
    exp = [1.0, 3.0, 5.0, 7.0]
    npt.assert_almost_equal(obs, exp, decimal=1)


def test_resample_long_datetimes():
    df_dts = [dt.datetime(year, 1, 1) for year in np.arange(1700, 2500 + 1, 100)]
    df = ScmRun(
        np.arange(1700, 2500 + 1, 100),
        columns={
            "scenario": ["a_scenario"],
            "model": ["a_model"],
            "region": ["World"],
            "variable": ["Emissions|BC"],
            "unit": ["Mg /yr"],
        },
        index=df_dts,
    )
    res = df.resample("AS")

    obs = res.values.squeeze()
    exp = np.arange(1700, 2500 + 1)
    npt.assert_almost_equal(obs, exp, decimal=1)


def test_init_no_file():
    fname = "/path/to/nowhere"
    error_msg = re.escape("no data file `{}` found!".format(fname))
    with pytest.raises(OSError, match=error_msg):
        ScmRun(fname)


@pytest.mark.xfail(
    _check_pandas_less_120(),
    reason="pandas<1.2.0 gets confused about how to read xlsx files",
)
@pytest.mark.parametrize(
    ("test_file", "test_kwargs"),
    [
        (
            "rcp26_emissions.csv",
            {},
        ),
        (
            "rcp26_emissions.csv.gz",
            {"lowercase_cols": True},
        ),
        (
            "rcp26_emissions_capitalised.csv",
            {"lowercase_cols": True},
        ),
        (
            "rcp26_emissions_int.csv",
            {"lowercase_cols": True},
        ),
        (
            "rcp26_emissions.xls",
            {},
        ),
        (
            "rcp26_emissions_multi_sheet.xlsx",
            {"sheet_name": "rcp26_emissions"},
        ),
        (
            "rcp26_emissions_multi_sheet_capitalised.xlsx",
            {"sheet_name": "rcp26_emissions", "lowercase_cols": True},
        ),
        (
            "rcp26_emissions_multi_sheet_capitalised_int.xlsx",
            {"sheet_name": "rcp26_emissions", "lowercase_cols": True},
        ),
        (
            "rcp26_emissions_multi_sheet_data.xlsx",
            {},
        ),
    ],
)
def test_read_from_disk(test_file, test_kwargs, test_data_path):
    loaded = ScmRun(os.path.join(test_data_path, test_file), **test_kwargs)
    assert (
        loaded.filter(variable="Emissions|N2O", year=1767).timeseries().values.squeeze()
        == 0.010116813
    )


def test_read_from_disk_different_number_of_digits_years(test_data_path):
    loaded = ScmRun(
        os.path.join(test_data_path, "different_number_of_digits_years.csv")
    )

    # make sure data sorts correctly
    assert loaded["year"].iloc[-1] == loaded["year"].max()


def test_read_from_disk_incorrect_labels():
    fname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "test_data",
        "rcp26_emissions_capitalised.csv",
    )

    exp_msg = "Missing required columns"

    with pytest.raises(MissingRequiredColumnError) as exc_info:
        ScmRun(fname)

    error_msg = exc_info.value.args[0]
    assert error_msg.startswith(exp_msg)
    assert "scenario" in exc_info.value.columns
    assert "variable" in exc_info.value.columns
    assert "unit" not in exc_info.value.columns


@pytest.mark.parametrize("separator", ["|", "__", "/", "~", "_", "-"])
def test_separator_changes(scm_run, separator):
    variable = scm_run["variable"]
    scm_run["variable"] = [v.replace("|", separator) for v in variable]

    scm_run.data_hierarchy_separator = separator

    pd.testing.assert_series_equal(
        scm_run.filter(level=0)["variable"],
        pd.Series(["Primary Energy", "Primary Energy"], index=[0, 2], name="variable"),
    )

    pd.testing.assert_series_equal(
        scm_run.filter(level=1)["variable"],
        pd.Series(
            ["Primary Energy{}Coal".format(separator)], index=[1], name="variable"
        ),
    )


def test_get_meta(scm_run):
    assert scm_run.get_unique_meta("climate_model") == ["a_model"]
    assert scm_run.get_unique_meta("variable") == [
        "Primary Energy",
        "Primary Energy|Coal",
    ]


@pytest.mark.parametrize("no_duplicates", [True, False])
def test_get_meta_no_duplicates(scm_run, no_duplicates):
    if no_duplicates:
        assert (
            scm_run.get_unique_meta("climate_model", no_duplicates=no_duplicates)
            == "a_model"
        )

        error_msg = re.escape(
            "`variable` column is not unique (found values: {})".format(
                scm_run["variable"].unique().tolist()
            )
        )
        with pytest.raises(ValueError, match=error_msg):
            scm_run.get_unique_meta("variable", no_duplicates=no_duplicates)
    else:
        assert scm_run.get_unique_meta(
            "climate_model", no_duplicates=no_duplicates
        ) == ["a_model"]
        assert scm_run.get_unique_meta("variable", no_duplicates=no_duplicates) == [
            "Primary Energy",
            "Primary Energy|Coal",
        ]


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("label", ["extra_meta", ["extra", "other"]])
def test_drop_meta(scm_run, label, inplace):
    if type(label) == str:
        scm_run[label] = 1.0
        assert label in scm_run.meta.columns
    else:
        for lbl in label:
            scm_run[lbl] = 1.0
            assert lbl in scm_run.meta.columns

    if inplace:
        scm_run.drop_meta(label, inplace=True)
        res = scm_run
    else:
        res = scm_run.drop_meta(label, inplace=False)
        assert id(res) != id(scm_run)

    if type(label) == str:
        assert label not in res.meta.columns
        assert label not in res.meta_attributes
    else:
        for lbl in label:
            assert lbl not in res.meta.columns
            assert lbl not in res.meta_attributes

    assert "variable" in res.meta.columns


@pytest.mark.parametrize("label", ["extra_meta", ["extra", "other"]])
def test_drop_meta_missing(scm_run, label):
    with pytest.raises(KeyError):
        scm_run.drop_meta(label)

    assert "variable" in scm_run.meta.columns


@pytest.mark.parametrize(
    "label", [["other", "climate_model"], ["climate_model", "other"]]
)
def test_drop_meta_missing_one(scm_run, label):
    with pytest.raises(KeyError):
        scm_run.drop_meta(label)

    assert "variable" in scm_run.meta.columns


def test_drop_meta_not_inplace(scm_run):
    label = "extra"

    scm_run[label] = "test"

    res = scm_run.drop_meta(label, inplace=False)

    assert label in scm_run.meta_attributes
    assert label not in res.meta_attributes

    res = res * 2
    np.testing.assert_almost_equal(res.values, scm_run.values * 2)


def test_drop_meta_inplace_default(scm_run):
    label = "extra"
    scm_run[label] = "test"

    res = scm_run.drop_meta(label)

    assert res is not None
    assert label in scm_run.meta
    assert label not in res.meta


def test_drop_meta_required(scm_run):
    with pytest.raises(MissingRequiredColumnError, match=re.escape("['model']")):
        scm_run.drop_meta(["climate_model", "model"])


time_axis_checks = pytest.mark.parametrize(
    "time_axis,mod_func",
    (
        (None, lambda x: x),
        ("year", lambda x: x.year),
        ("year-month", lambda x: x.year + (x.month - 0.5) / 12),
        ("days since 1970-01-01", lambda x: (x - dt.datetime(1970, 1, 1)).days),
        (
            "seconds since 1970-01-01",
            lambda x: (x - dt.datetime(1970, 1, 1)).total_seconds(),
        ),
    ),
)


@time_axis_checks
def test_timeseries_time_axis(scm_run, time_axis, mod_func):
    res = scm_run.timeseries(time_axis=time_axis)
    assert (res.columns == (scm_run["time"].apply(mod_func))).all()


@time_axis_checks
def test_long_data_time_axis(scm_run, time_axis, mod_func):
    res = scm_run.long_data(time_axis=time_axis)

    assert (res["time"] == (scm_run.long_data()["time"].apply(mod_func))).all()


@time_axis_checks
def test_lineplot_time_axis(scm_run, time_axis, mod_func):
    pytest.importorskip("seaborn")
    mock_return = 4

    with patch("scmdata.plotting.sns.lineplot") as mock_sns_lineplot:
        with patch.object(ScmRun, "long_data") as mock_long_data:
            mock_long_data.return_value = mock_return

            scm_run.lineplot(time_axis=time_axis, other_kwarg="value")

    mock_long_data.assert_called_once()
    mock_long_data.assert_called_with(time_axis=time_axis)

    mock_sns_lineplot.assert_called_once()
    mock_sns_lineplot.assert_called_with(
        x="time",
        y="value",
        estimator=np.median,
        ci="sd",
        hue="scenario",
        other_kwarg="value",
        data=mock_return,
    )


@pytest.mark.parametrize("method_to_call", ("timeseries", "long_data"))
@pytest.mark.parametrize(
    "time_axis,non_unique_vals,exp_raise",
    (
        (
            "year",
            [dt.datetime(2015, 1, 1), dt.datetime(2015, 2, 1), dt.datetime(2016, 1, 1)],
            True,
        ),
        (
            "year",
            [dt.datetime(2015, 1, 1), dt.datetime(2016, 2, 1), dt.datetime(2017, 1, 1)],
            False,
        ),
        (
            "year-month",
            [
                dt.datetime(2015, 1, 1),
                dt.datetime(2015, 1, 10),
                dt.datetime(2015, 2, 1),
            ],
            True,
        ),
        (
            "year-month",
            [
                dt.datetime(2015, 1, 1),
                dt.datetime(2015, 2, 10),
                dt.datetime(2015, 3, 1),
            ],
            False,
        ),
        (
            "days since 1970-01-01",
            [
                dt.datetime(2015, 1, 1, 1),
                dt.datetime(2015, 1, 1, 12),
                dt.datetime(2015, 1, 2, 1),
            ],
            True,
        ),
        (
            "days since 1970-01-01",
            [
                dt.datetime(2015, 1, 1, 1),
                dt.datetime(2015, 1, 2, 1),
                dt.datetime(2015, 1, 3, 1),
            ],
            False,
        ),
    ),
)
def test_timeseries_time_axis_non_unique_raises(
    method_to_call, time_axis, non_unique_vals, exp_raise
):
    start = ScmRun(
        data=np.arange(len(non_unique_vals)),
        index=non_unique_vals,
        columns={
            "scenario": "junk",
            "model": "junk",
            "variable": "Emissions|CO2",
            "unit": "GtC/yr",
            "region": "World",
        },
    )
    error_msg = re.escape(
        "Ambiguous time values with time_axis = '{}'".format(time_axis)
    )

    if exp_raise:
        with pytest.raises(ValueError, match=error_msg):
            getattr(start, method_to_call)(time_axis=time_axis)
    else:
        getattr(start, method_to_call)(time_axis=time_axis)


def test_timeseries_time_axis_junk_error(scm_run):
    error_msg = re.escape("time_axis = 'junk")
    with pytest.raises(NotImplementedError, match=error_msg):
        scm_run.timeseries(time_axis="junk")


def test_timeseries_check_duplicated(scm_run):
    with pytest.raises(NonUniqueMetadataError):
        scm_run.timeseries(meta=["region", "unit"], check_duplicated=True)

    # Default behaviour
    with pytest.raises(NonUniqueMetadataError):
        scm_run.timeseries(meta=["region", "unit"])


def test_long_data_time_axis_junk_error(scm_run):
    error_msg = re.escape("time_axis = 'junk")
    with pytest.raises(NotImplementedError, match=error_msg):
        scm_run.long_data(time_axis="junk")


def test_lineplot_time_axis_junk_error(scm_run):
    pytest.importorskip("seaborn")

    error_msg = re.escape("time_axis = 'junk")

    with patch("scmdata.plotting.sns.lineplot") as mock_sns_lineplot:
        with pytest.raises(NotImplementedError, match=error_msg):
            scm_run.lineplot(time_axis="junk")

    assert not mock_sns_lineplot.called  # doesn't get to trying to plot


@pytest.mark.parametrize(
    "tax1,tax2",
    (
        (
            [dt.datetime(y, 1, 1) for y in range(2000, 2020, 10)],
            [dt.datetime(y, 1, 1) for y in range(2000, 2020, 10)],
        ),
        (
            [dt.datetime(y, 1, 1) for y in range(2000, 2020, 10)],
            [dt.datetime(y, 1, 1) for y in range(2020, 2040, 10)],
        ),
        (
            [dt.datetime(y, 1, 1) for y in range(1000, 1020, 10)],
            [dt.datetime(y, 1, 1) for y in range(2000, 2020, 10)],
        ),
        (
            [dt.datetime(y, 1, 1) for y in range(2000, 2020, 10)],
            [dt.datetime(y, 1, 1) for y in range(3000, 3020, 10)],
        ),
        (
            [dt.datetime(y, 1, 1) for y in range(1000, 2020, 100)],
            [dt.datetime(y, 1, 1) for y in range(2000, 2500, 10)],
        ),
    ),
)
def test_append_long_run(tax1, tax2):
    mdata = {
        "model": "junk",
        "variable": "Emissions|CO2",
        "unit": "GtC",
        "region": "World",
    }
    run1 = ScmRun(
        data=np.arange(len(tax1)), index=tax1, columns={"scenario": "run1", **mdata}
    )
    run2 = ScmRun(
        data=np.arange(len(tax2)), index=tax2, columns={"scenario": "run2", **mdata}
    )

    res = run_append([run1, run2])

    expected = sorted(set(tax1 + tax2))
    assert len(res["time"]) == len(expected)
    for i in range(len(expected)):
        assert res["time"][i] == expected[i]

    assert res.get_unique_meta("scenario") == ["run1", "run2"]


@pytest.mark.parametrize(
    "metadata_1,metadata_2,metadata,expected",
    (
        (
            {"first": "example"},
            {"second": "other_example"},
            None,
            {"first": "example", "second": "other_example"},
        ),
        (
            {"first": "example", "second": "other_example"},
            {"first": "other"},
            None,
            {"first": "example", "second": "other_example"},
        ),
        (
            {"first": "example", "second": "other_example"},
            {"first": "other"},
            {"first": "", "third": "other_example"},
            {"first": "", "third": "other_example"},
        ),
    ),
)
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("use_cls_method", [True, False])
def test_append_metadata(
    scm_run, metadata_1, metadata_2, metadata, expected, inplace, use_cls_method
):
    run1 = scm_run.copy()
    run1["ensemble_member"] = 1
    run1.metadata = metadata_1
    run2 = scm_run.copy()
    run2["ensemble_member"] = 2
    run2.metadata = metadata_2

    if use_cls_method:
        res = run1.append(run2, metadata=metadata, inplace=inplace)
    else:
        res = run_append([run1, run2], metadata=metadata, inplace=inplace)

    if inplace:
        res = run1

    assert res.metadata == expected


def test_append_invalid(scm_run):
    for runs in [None, scm_run]:
        with pytest.raises(TypeError, match="runs is not a list"):
            run_append(runs)

    with pytest.raises(ValueError, match="No runs to append"):
        run_append([])


def test_empty(scm_run):
    assert not scm_run.empty
    assert scm_run.filter(variable="junk nonsense").empty


def test_init_duplicate_metadata_issue_76():
    with pytest.raises(NonUniqueMetadataError):
        ScmRun(
            data=np.arange(6).reshape(2, 3),
            index=[10, 20],
            columns={
                "variable": "Emissions",
                "unit": "Gt",
                "model": "idealised",
                "scenario": "idealised",
                "region": "World",
            },
        )


def test_set_item_duplicate_meta_issue_76(scm_run):
    run = ScmRun(
        data=np.arange(4).reshape(2, 2),
        index=[10, 20],
        columns={
            "variable": ["Emissions", "Emissions removed"],
            "unit": "Gt",
            "model": "idealised",
            "scenario": "idealised",
            "region": "World",
        },
    )

    # check that altering metadata in such a way that it becomes non-unique fails
    with pytest.raises(NonUniqueMetadataError):
        run["variable"] = "Emissions"


def test_non_unique_metadata_error_formatting():
    sdf = pd.DataFrame(
        np.arange(9).reshape(3, 3),
        columns=[dt.datetime(y, 1, 1) for y in [2010, 2020, 2030]],
    )
    sdf["variable"] = ["Emissions", "Emissions", "Temperature"]
    sdf["unit"] = ["Gt", "Gt", "K"]
    sdf["model"] = "idealised"
    sdf["scenario"] = "idealised"
    sdf["region"] = "World"
    sdf = sdf.set_index(["variable", "unit", "model", "scenario", "region"])

    meta = sdf.index.to_frame().reset_index(drop=True)

    exp = meta.groupby(meta.columns.tolist(), as_index=True).size()
    exp = exp[exp > 1]
    exp.name = "repeats"
    exp = exp.to_frame().reset_index()
    error_msg = (
        "Duplicate metadata (numbers show how many times the given "
        "metadata is repeated).\n{}".format(exp)
    )

    with pytest.raises(NonUniqueMetadataError, match=re.escape(error_msg)):
        raise NonUniqueMetadataError(meta)


def test_copy(scm_run):
    orig_run = scm_run
    copy_run = scm_run.copy()

    assert id(orig_run) != id(copy_run)

    assert "test" not in orig_run.metadata
    assert id(orig_run.metadata) != id(copy_run.metadata)
    assert id(orig_run._df) != id(copy_run._df)
    assert id(orig_run._meta) != id(copy_run._meta)

    orig_run["example"] = 1
    assert "example" in orig_run.meta_attributes
    assert "example" not in copy_run.meta_attributes


@pytest.mark.parametrize("model", ["model_a", "model_b"])
def test_metadata_consistency(model):
    start = ScmRun(
        np.arange(6).reshape(3, 2),
        [2010, 2020, 2030],
        columns={
            "model": ["model_a", "model_b"],
            "scenario": "scenario",
            "variable": "variable",
            "region": "region",
            "unit": "unit",
        },
    )
    modified = start.copy()
    modified["new_meta"] = ["hi" for f in modified["model"] if f == model]

    modified_dropped = modified.drop_meta("new_meta", inplace=False)

    assert_scmdf_almost_equal(start, modified_dropped)

    modified.drop_meta("new_meta", inplace=True)
    assert_scmdf_almost_equal(start, modified)


def test_drop_meta_nonunique():
    start = ScmRun(
        np.arange(6).reshape(3, 2),
        [2010, 2020, 2030],
        columns={
            "model": "model",
            "scenario": "scenario",
            "variable": "variable",
            "region": "region",
            "unit": "unit",
            "new_meta": ["a", "b"],
        },
    )

    with pytest.raises(NonUniqueMetadataError):
        start.drop_meta("new_meta")


@pytest.mark.parametrize(
    "output_cls",
    (
        None,
        cftime.DatetimeGregorian,
        cftime.Datetime360Day,
    ),
)
def test_time_as_cftime(scm_run, output_cls):
    if output_cls is None:
        res = scm_run.time_points.as_cftime()
    else:
        res = scm_run.time_points.as_cftime(date_cls=output_cls)

    if output_cls is None:
        assert all([isinstance(v, cftime.DatetimeGregorian) for v in res])
    else:
        assert all([isinstance(v, output_cls) for v in res])


def test_round():
    # There are some quirks with the rounding due to the algo used

    def compare(inp, exp, decimals=1):
        cols = {
            "model": "model",
            "scenario": "scenario",
            "variable": "variable",
            "region": "region",
            "unit": "unit",
        }
        index = [2000, 2025, 2030, 2035, 2040]
        run = ScmRun(
            data=np.array([inp]).T,
            index=index,
            columns=cols,
        )

        res = run.round(decimals)

        exp = ScmRun(
            data=np.array([exp]).T,
            index=index,
            columns=cols,
        )
        assert_scmdf_almost_equal(res, exp)

    compare([3.6565, 5.51, 5.55, 5.45, 1], [3.7, 5.5, 5.6, 5.4, 1.0])
    compare([-3.6565, -5.51, -5.55, -5.45, -1], [-3.7, -5.5, -5.6, -5.4, -1.0])
    compare([-3.6565, -5.51, -5.55, -5.45, -1], [-4, -6, -6, -5.0, -1.0], decimals=0)
    compare([3.6565, 5.51, 5.55, 5.45, 1], [3.6565, 5.51, 5.55, 5.45, 1.0], decimals=5)


def test_round_warns_small():
    run = ScmRun(
        data=np.array([[3.6565, 5.51, 5.55, 1, 2.34e-2]]).T,
        index=[2000, 2025, 2030, 2035, 2040],
        columns={
            "model": "model",
            "scenario": "scenario",
            "variable": "variable",
            "region": "region",
            "unit": "unit",
        },
    )

    match = "There are small values which may be truncated during rounding"

    with pytest.warns(UserWarning, match=match):
        res = run.round(1)

    exp = ScmRun(
        data=np.array([[3.7, 5.5, 5.6, 1.0, 0]]).T,
        index=[2000, 2025, 2030, 2035, 2040],
        columns={
            "model": "model",
            "scenario": "scenario",
            "variable": "variable",
            "region": "region",
            "unit": "unit",
        },
    )
    assert_scmdf_almost_equal(res, exp)
