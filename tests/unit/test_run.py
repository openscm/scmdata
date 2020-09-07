import copy
import datetime as dt
import os
import re
import warnings
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt
from packaging.version import parse
from pandas.errors import UnsupportedFunctionCall
from pint.errors import DimensionalityError, UndefinedUnitError

from scmdata.errors import NonUniqueMetadataError
from scmdata.run import ScmRun, TimeSeries, run_append
from scmdata.testing import assert_scmdf_almost_equal


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

    res_df = res.timeseries()
    res_df.columns = res_df.columns.map(lambda x: x.year)
    res_df = res_df.reset_index()

    pd.testing.assert_frame_equal(
        res_df[test_pd_run_df.columns.tolist()], test_pd_run_df, check_like=True,
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
        "invalid column format, must contain some time (int, float or datetime) columns!"
    )
    with pytest.raises(ValueError, match=error_msg):
        ScmRun(test_init)


def test_init_df_missing_col_error(test_pd_df):
    test_pd_df = test_pd_df.drop("model", axis="columns")
    error_msg = re.escape("missing required columns `['model']`!")
    with pytest.raises(ValueError, match=error_msg):
        ScmRun(test_pd_df)


def test_init_ts_missing_col_error(test_ts):
    error_msg = re.escape("missing required columns `['model']`!")
    with pytest.raises(ValueError, match=error_msg):
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
    df = ScmRun(test_scm_run_datetimes,)

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
    assert len(scm_run) == len(scm_run._ts)


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
    "regexp,exp_units", ((True, []), (False, ["W/m^2"]),),
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


@pytest.mark.parametrize("has_nan", [True, False])
def test_filter_timeseries_nan_meta(has_nan):
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

    # not sure how we want to setup NaN filtering, empty string seems as good as any?
    if not has_nan:
        error_msg = re.escape(
            "String filtering cannot be performed on column 'scenario', which "
            "contains NaN's, unless `has_nan` is True"
        )
        with pytest.raises(TypeError, match=error_msg):
            df.filter(scenario="*", has_nan=has_nan)
        with pytest.raises(TypeError, match=error_msg):
            df.filter(scenario="", has_nan=has_nan)

    else:

        def with_nan_assertion(a, b):
            assert all(
                [
                    (v == b[i]) or (np.isnan(v) and np.isnan(b[i]))
                    for i, v in enumerate(a)
                ]
            )

        res = df.filter(scenario="*", has_nan=has_nan)["scenario"].unique()
        exp = ["a_scenario", "a_scenario2", np.nan]
        with_nan_assertion(res, exp)

        res = df.filter(scenario="", has_nan=has_nan)["scenario"].unique()
        exp = [np.nan]
        with_nan_assertion(res, exp)

        res = df.filter(scenario="nan", has_nan=has_nan)["scenario"].unique()
        exp = [np.nan]
        with_nan_assertion(res, exp)


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
    pd.testing.assert_frame_equal(exp.set_index(obs.index.names), obs, check_like=True)


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
    pd.testing.assert_frame_equal(exp.set_index(obs.index.names), obs, check_like=True)


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
    pd.testing.assert_frame_equal(exp.set_index(obs.index.names), obs, check_like=True)


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
    pd.testing.assert_frame_equal(exp.set_index(obs.index.names), obs, check_like=True)


def test_process_over_unrecognised_operation_error(scm_run):
    error_msg = re.escape("operation must be one of ['median', 'mean', 'quantile']")
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

    exp = type(test_processing_scm_df)(
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
    npt.assert_array_equal(ts.iloc[2], ts.iloc[3])
    pd.testing.assert_index_equal(
        df.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "col1",
                "col2",
            ]
        ),
    )


def test_append_exact_duplicates(scm_run):
    other = copy.deepcopy(scm_run)
    with warnings.catch_warnings(record=True) as mock_warn_taking_average:
        scm_run.append(other, duplicate_msg="warn").timeseries()

    assert len(mock_warn_taking_average) == 1  # test message elsewhere

    assert_scmdf_almost_equal(scm_run, other)


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
    other._ts[2][2] = 5.0

    res = other.append(scm_run, duplicate_msg="warn")

    obs = res.filter(scenario="a_scenario2").timeseries().squeeze()
    exp = [2.0, 7.0, 7.0, 2.0, 7.0, 5.0]
    npt.assert_array_equal(
        res._time_points.years(), [2005, 2010, 2015, 2020, 2030, 2040]
    )
    npt.assert_almost_equal(obs, exp)


@pytest.mark.parametrize("duplicate_msg", ("warn", True, False))
def test_append_duplicate_times(test_append_scm_runs, duplicate_msg):
    base = test_append_scm_runs["base"]
    other = test_append_scm_runs["other"]
    expected = test_append_scm_runs["expected"]

    if duplicate_msg and not isinstance(duplicate_msg, str):
        with pytest.raises(NonUniqueMetadataError):
            base.append(other, duplicate_msg=duplicate_msg)

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


def test_append_inplace(scm_run):
    other = scm_run * 2

    obs = scm_run.filter(scenario="a_scenario2").timeseries().squeeze()
    exp = [2, 7, 7]
    npt.assert_almost_equal(obs, exp)
    with warnings.catch_warnings(record=True) as mock_warn_taking_average:
        scm_run.append(other, inplace=True, duplicate_msg="warn")

    assert len(mock_warn_taking_average) == 1  # test message elsewhere

    obs = scm_run.filter(scenario="a_scenario2").timeseries().squeeze()
    exp = [(2.0 + 4.0) / 2, (7.0 + 14.0) / 2, (7.0 + 14.0) / 2]
    npt.assert_almost_equal(obs, exp)


def get_append_col_order_time_dfs(base):
    other_2 = base.filter(variable="Primary Energy|Coal").copy()
    base["runmodus"] = "co2_only"
    other = base.copy()

    other._ts[1].meta["variable"] = "Primary Energy|Gas"
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
    base, other, other_2, exp = get_append_col_order_time_dfs(scm_run)

    res = run_append([scm_run, other, other_2], duplicate_msg="warn")

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

    exp = pd.concat([scm_run.timeseries(), other.timeseries()])
    exp["junk"] = np.nan
    exp.set_index("junk", append=True, inplace=True)

    pd.testing.assert_frame_equal(
        res.timeseries().reorder_levels(exp.index.names).sort_index().reset_index(),
        exp.sort_index().reset_index(),
        check_like=True,
        check_dtype=False,
    )


@pytest.mark.parametrize("same_times", [True, False])
def test_append_reindexing(scm_run, same_times):
    other = copy.deepcopy(scm_run)
    other["climate_model"] = "other"
    if not same_times:
        other["time"] = [2002, 2010, 2020]

    with patch.object(
        TimeSeries, "reindex", wraps=other._ts[0].reindex
    ) as mock_reindex:
        res = scm_run.append(other, duplicate_msg="warn")

        expected_times = set(
            np.concatenate([other.time_points.values, scm_run.time_points.values])
        )
        if same_times:
            mock_reindex.assert_not_called()
        else:
            mock_reindex.assert_called()

        npt.assert_array_equal(res.time_points.values, sorted(expected_times))
        for t in res._ts:
            npt.assert_array_equal(t.time_points.values, sorted(expected_times))


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


def test_time_mean_year_beginning_of_year(test_scm_df_monthly):
    # should be annual mean centred on January 1st of each year
    res = test_scm_df_monthly.time_mean("AS")

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
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "meta_int",
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
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "meta_str",
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


@pytest.mark.parametrize(
    ("target_unit", "input_units", "filter_kwargs", "expected", "expected_units"),
    [
        ("EJ/yr", "EJ/yr", {}, [1.0, 0.5, 2.0], ["EJ/yr", "EJ/yr", "EJ/yr"]),
        (
            "EJ/yr",
            "EJ/yr",
            {"variable": "Primary Energy"},
            [1.0, 0.5, 2.0],
            ["EJ/yr", "EJ/yr", "EJ/yr"],
        ),
        ("PJ/yr", "EJ/yr", {}, [1000.0, 500.0, 2000.0], ["PJ/yr", "PJ/yr", "PJ/yr"]),
        (
            "PJ/yr",
            "EJ/yr",
            {"scenario": "a_scenario2"},
            [1.0, 0.5, 2000.0],
            ["EJ/yr", "EJ/yr", "PJ/yr"],
        ),
        (
            "PJ/yr",
            ["EJ/yr", "TJ/yr", "Gt C / yr"],
            {"variable": "Primary Energy|Coal"},
            [1.0, 0.5 * 1e-3, 2.0],
            ["EJ/yr", "PJ/yr", "Gt C / yr"],
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

    pd.testing.assert_series_equal(obs["unit"], exp_units, check_less_precise=True)
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


def test_convert_unit_inplace(scm_run):
    units = scm_run["unit"].copy()

    ret = scm_run.convert_unit("PJ/yr", inplace=True)
    assert ret is None

    assert (scm_run["unit"] != units).all()
    npt.assert_array_almost_equal(
        scm_run.filter(year=2005).values.squeeze(), [1000.0, 500.0, 2000.0]
    )


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

    error_msg = "Cannot convert from 'SF5CF3 * kilogram / a' ([SF5CF3] * [mass] / [time]) to 'CO2 * kilogram / a' ([carbon] * [mass] / [time])"
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

    scm_run.filter(variable=to_convert, keep=False)[
        "unit_context"
    ] = to_not_convert_context

    res = scm_run.convert_unit("MJ/yr", variable=to_convert, context=context)

    assert (
        res.filter(variable=to_convert).get_unique_meta(
            "unit_context", no_duplicates=True
        )
        == context
    )
    assert (
        res.filter(variable=to_convert, keep=False).get_unique_meta(
            "unit_context", no_duplicates=True
        )
        == to_not_convert_context
    )


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
        "Existing unit conversion context(s), `['junk']`, doesn't match input context, `{}`, drop "
        "`unit_context` metadata before doing conversion".format(context)
    )
    with pytest.raises(ValueError, match=error_msg):
        scm_run.convert_unit("MJ/yr", variable=to_convert, context=context)


@pytest.mark.parametrize("context", ("AR5GWP100", "AR4GWP100"))
def test_unit_context_to_convert_has_existing_context(scm_run, context):
    to_convert = "*Coal"
    start = scm_run.convert_unit("MJ/yr", variable=to_convert, context=context)

    assert (
        start.filter(variable=to_convert).get_unique_meta(
            "unit_context", no_duplicates=True
        )
        == context
    )
    assert np.isnan(
        start.filter(variable=to_convert, keep=False).get_unique_meta(
            "unit_context", no_duplicates=True
        )
    )

    res = start.convert_unit("GJ/yr", variable=to_convert, context=context)

    assert (
        res.filter(variable=to_convert).get_unique_meta(
            "unit_context", no_duplicates=True
        )
        == context
    )
    assert np.isnan(
        res.filter(variable=to_convert, keep=False).get_unique_meta(
            "unit_context", no_duplicates=True
        )
    )
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

    assert (
        start.filter(variable=to_convert).get_unique_meta(
            "unit_context", no_duplicates=True
        )
        == context
    )
    assert np.isnan(
        start.filter(variable=to_convert, keep=False).get_unique_meta(
            "unit_context", no_duplicates=True
        )
    )

    error_msg = re.escape(
        "Existing unit conversion context(s), `['{}']`, doesn't match input context, `junk`, drop "
        "`unit_context` metadata before doing conversion".format(context)
    )
    with pytest.raises(ValueError, match=error_msg):
        start.convert_unit("GJ/yr", variable=to_convert, context="junk")


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
    assert np.isnan(
        start.filter(variable=to_convert).get_unique_meta(
            "unit_context", no_duplicates=True
        )
    )
    assert (
        start.filter(variable=to_convert, keep=False).get_unique_meta(
            "unit_context", no_duplicates=True
        )
        == to_not_convert_context
    )

    # no error, irrespective of context because to_convert context is nan
    res = start.convert_unit("GJ/yr", variable=to_convert, context=context)

    assert (
        res.filter(variable=to_convert).get_unique_meta(
            "unit_context", no_duplicates=True
        )
        == context
    )
    assert (
        res.filter(variable=to_convert, keep=False).get_unique_meta(
            "unit_context", no_duplicates=True
        )
        == to_not_convert_context
    )
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


def test_convert_unit_does_not_warn(scm_run, caplog):
    scm_run["unit"] = "GtC"

    res = scm_run.convert_unit("MtC")

    npt.assert_equal(len(caplog.records), 0)
    npt.assert_array_equal(scm_run.values, res.values / 10 ** 3)


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


@pytest.mark.parametrize(
    ("test_file", "test_kwargs"),
    [
        ("rcp26_emissions.csv", {},),
        ("rcp26_emissions.csv.gz", {"lowercase_cols": True},),
        ("rcp26_emissions_capitalised.csv", {"lowercase_cols": True},),
        ("rcp26_emissions_int.csv", {"lowercase_cols": True},),
        ("rcp26_emissions.xls", {},),
        ("rcp26_emissions_multi_sheet.xlsx", {"sheet_name": "rcp26_emissions"},),
        (
            "rcp26_emissions_multi_sheet_capitalised.xlsx",
            {"sheet_name": "rcp26_emissions", "lowercase_cols": True},
        ),
        (
            "rcp26_emissions_multi_sheet_capitalised_int.xlsx",
            {"sheet_name": "rcp26_emissions", "lowercase_cols": True},
        ),
        ("rcp26_emissions_multi_sheet_data.xlsx", {},),
    ],
)
def test_read_from_disk(test_file, test_kwargs, test_data_path):
    loaded = ScmRun(os.path.join(test_data_path, test_file), **test_kwargs)
    assert (
        loaded.filter(variable="Emissions|N2O", year=1767).timeseries().values.squeeze()
        == 0.010116813
    )


def test_read_from_disk_incorrect_labels():
    fname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "test_data",
        "rcp26_emissions_capitalised.csv",
    )

    exp_msg = "missing required columns"

    with pytest.raises(ValueError) as exc_info:
        ScmRun(fname)

    error_msg = exc_info.value.args[0]
    assert error_msg.startswith(exp_msg)
    assert "scenario" in error_msg
    assert "variable" in error_msg
    assert "unit" not in error_msg


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


def test_meta_filtered(scm_run):
    scm_run.filter(scenario="a_scenario")["test"] = 1.0
    pd.testing.assert_series_equal(
        pd.Series([1.0, 1.0, np.nan], name="test"), scm_run["test"]
    )


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

    # TODO: remove warning check in v0.7.0
    # Check that the deprecation warning isn't raised
    with warnings.catch_warnings(record=True) as warn:
        if inplace:
            scm_run.drop_meta(label, inplace=True)
            res = scm_run
        else:
            res = scm_run.drop_meta(label, inplace=False)
            assert id(res) != id(scm_run)

        assert len(warn) == 0

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


def test_drop_meta_missing_one(scm_run):
    label = ["variable", "other"]
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

    msg = (
        "drop_meta default behaviour will change to not performing operation inplace in v0.7.0. "
        "Explicitly set inplace=True to retain current behaviour"
    )
    with pytest.warns(DeprecationWarning, match=msg):
        res = scm_run.drop_meta(label)

    # Should default to inplace
    # To change in v0.7.0
    assert res is None
    assert label not in scm_run.meta


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
@patch("scmdata.plotting.sns.lineplot")
@patch.object(ScmRun, "long_data")
def test_lineplot_time_axis(
    mock_long_data, mock_sns_lineplot, scm_run, time_axis, mod_func
):
    mock_return = 4
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


@patch("scmdata.plotting.sns.lineplot")
def test_lineplot_time_axis_junk_error(mock_sns_lineplot, scm_run):
    error_msg = re.escape("time_axis = 'junk")
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
            {"first": "other",},
            None,
            {"first": "example", "second": "other_example"},
        ),
        (
            {"first": "example", "second": "other_example"},
            {"first": "other",},
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


@pytest.mark.parametrize("copy_ts", [True, False])
def test_copy(scm_run, copy_ts):
    orig_run = scm_run
    copy_run = scm_run.copy(copy_ts)

    assert id(orig_run) != id(copy_run)

    assert "test" not in orig_run.metadata
    assert id(orig_run.metadata) != id(copy_run.metadata)

    for o, c in zip(orig_run._ts, copy_run._ts):
        if copy_ts:
            assert id(o) != id(c)
        else:
            assert id(o) == id(c)


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
    modified.filter(model=model)["new_meta"] = "hi"

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
