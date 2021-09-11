import datetime as dt

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import scmdata.processing
from scmdata import ScmRun


@pytest.fixture(scope="function")
def test_processing_scm_df():
    data = np.array(
        [
            [1, 1.1, 1.2, 1.1],
            [1.1, 1.2, 1.3, 1.41],
            [1.3, 1.4, 1.5, 1.6],
            [1.3, 1.5, 1.6, 1.2],
            [1.48, 1.51, 1.72, 1.56],
        ]
    ).T
    yield ScmRun(
        data=data,
        columns={
            "model": ["a_iam"],
            "climate_model": ["a_model"],
            "scenario": ["a_scenario"],
            "region": ["World"],
            "variable": ["Surface Air Temperature Change"],
            "unit": ["K"],
            "ensemble_member": range(data.shape[1]),
        },
        index=[2005, 2006, 2007, 2100],
    )


@pytest.fixture()
def test_processing_scm_df_multi_climate_model(test_processing_scm_df):
    other = test_processing_scm_df + 0.1
    other["climate_model"] = "z_model"

    return test_processing_scm_df.append(other)


crossing_times_year_conversions = pytest.mark.parametrize(
    "return_year,conv_to_year", ((None, True), (True, True), (False, False),)
)


def _get_calculate_crossing_times_call_kwargs(return_year):
    call_kwargs = {}
    if return_year is not None:
        call_kwargs["return_year"] = return_year

    return call_kwargs


def _get_expected_crossing_times(exp_vals, conv_to_year):
    if conv_to_year:
        exp_vals = [v if pd.isnull(v) else v.year for v in exp_vals]
    else:
        exp_vals = [pd.NaT if pd.isnull(v) else v for v in exp_vals]

    return exp_vals


@pytest.mark.parametrize(
    "threshold,exp_vals",
    (
        (
            1.0,
            [
                dt.datetime(2006, 1, 1),  # doesn't cross 1.0 until 2006
                dt.datetime(2005, 1, 1),
                dt.datetime(2005, 1, 1),
                dt.datetime(2005, 1, 1),
                dt.datetime(2005, 1, 1),
            ],
        ),
        (
            1.5,
            [
                np.nan,  # never crosses
                np.nan,  # never crosses
                dt.datetime(2100, 1, 1),  # doesn't cross 1.5 until 2100
                dt.datetime(2007, 1, 1),  # 2007 is first year to actually exceed 1.5
                dt.datetime(2006, 1, 1),
            ],
        ),
        (2.0, [np.nan, np.nan, np.nan, np.nan, np.nan]),
    ),
)
@crossing_times_year_conversions
def test_crossing_times(
    threshold, exp_vals, return_year, conv_to_year, test_processing_scm_df
):
    call_kwargs = _get_calculate_crossing_times_call_kwargs(return_year)
    res = scmdata.processing.calculate_crossing_times(
        test_processing_scm_df, threshold=threshold, **call_kwargs,
    )

    exp_vals = _get_expected_crossing_times(exp_vals, conv_to_year)

    exp = pd.Series(exp_vals, pd.MultiIndex.from_frame(test_processing_scm_df.meta))

    pdt.assert_series_equal(res, exp)


@crossing_times_year_conversions
def test_crossing_times_multi_climate_model(
    return_year, conv_to_year, test_processing_scm_df_multi_climate_model
):
    call_kwargs = _get_calculate_crossing_times_call_kwargs(return_year)

    threshold = 1.5
    exp_vals = [
        # a_model
        np.nan,
        np.nan,
        dt.datetime(2100, 1, 1),
        dt.datetime(2007, 1, 1),
        dt.datetime(2006, 1, 1),
        # z_model
        np.nan,
        dt.datetime(2100, 1, 1),
        dt.datetime(2007, 1, 1),
        dt.datetime(2006, 1, 1),
        dt.datetime(2005, 1, 1),
    ]

    res = scmdata.processing.calculate_crossing_times(
        test_processing_scm_df_multi_climate_model, threshold=threshold, **call_kwargs,
    )

    exp_vals = _get_expected_crossing_times(exp_vals, conv_to_year)

    exp = pd.Series(
        exp_vals,
        pd.MultiIndex.from_frame(test_processing_scm_df_multi_climate_model.meta),
    )

    pdt.assert_series_equal(res, exp)


output_name_options = pytest.mark.parametrize(
    "output_name", (None, "test", "test other")
)


def _get_calculate_exceedance_probs_call_kwargs(output_name):
    call_kwargs = {}
    if output_name is not None:
        call_kwargs["output_name"] = output_name

    return call_kwargs


def _get_calculate_exeedance_probs_expected_name(output_name, threshold):
    if output_name is not None:
        return output_name

    return "{} exceedance probability".format(threshold)


@pytest.mark.parametrize(
    "threshold,exp_vals",
    (
        (1.0, [0.8, 1.0, 1.0, 1.0]),
        (1.5, [0.0, 0.2, 0.4, 0.4]),
        (2.0, [0.0, 0.0, 0.0, 0.0]),
    ),
)
@output_name_options
def test_exceedance_probabilities_over_time(
    output_name, threshold, exp_vals, test_processing_scm_df
):
    call_kwargs = _get_calculate_exceedance_probs_call_kwargs(output_name)
    res = scmdata.processing.calculate_exceedance_probabilities_over_time(
        test_processing_scm_df,
        cols="ensemble_member",
        threshold=threshold,
        **call_kwargs,
    )

    exp_idx = pd.MultiIndex.from_frame(
        test_processing_scm_df.meta.drop(
            "ensemble_member", axis="columns"
        ).drop_duplicates()
    )

    exp = pd.DataFrame(
        np.array(exp_vals)[np.newaxis, :],
        index=exp_idx,
        columns=test_processing_scm_df.time_points.to_index(),
    )
    exp.index = exp.index.set_levels(
        [_get_calculate_exeedance_probs_expected_name(output_name, threshold)],
        level="variable",
    ).set_levels(["dimensionless"], level="unit",)

    pdt.assert_frame_equal(res, exp, check_like=True, check_column_type=False)


def test_exceedance_probabilities_over_time_multiple_res(
    test_processing_scm_df_multi_climate_model,
):
    start = test_processing_scm_df_multi_climate_model.copy()
    threshold = 1.5
    exp_vals = np.array([[0, 1, 2, 2], [1, 2, 3, 3]]) / 5

    res = scmdata.processing.calculate_exceedance_probabilities_over_time(
        start, cols=["ensemble_member"], threshold=threshold,
    )

    exp_idx = pd.MultiIndex.from_frame(
        start.meta.drop(["ensemble_member"], axis="columns").drop_duplicates()
    )

    exp = pd.DataFrame(exp_vals, index=exp_idx, columns=start.time_points.to_index())
    exp.index = exp.index.set_levels(
        [_get_calculate_exeedance_probs_expected_name(None, threshold)],
        level="variable",
    ).set_levels(["dimensionless"], level="unit",)

    pdt.assert_frame_equal(res, exp, check_like=True, check_column_type=False)


def test_exceedance_probabilities_over_time_multiple_grouping(
    test_processing_scm_df_multi_climate_model,
):
    start = test_processing_scm_df_multi_climate_model.copy()
    threshold = 1.5
    exp_vals = np.array([1, 3, 5, 5]) / 10

    res = scmdata.processing.calculate_exceedance_probabilities_over_time(
        start, cols=["climate_model", "ensemble_member"], threshold=threshold,
    )

    exp_idx = pd.MultiIndex.from_frame(
        start.meta.drop(
            ["climate_model", "ensemble_member"], axis="columns"
        ).drop_duplicates()
    )

    exp = pd.DataFrame(
        exp_vals[np.newaxis, :], index=exp_idx, columns=start.time_points.to_index(),
    )
    exp.index = exp.index.set_levels(
        [_get_calculate_exeedance_probs_expected_name(None, threshold)],
        level="variable",
    ).set_levels(["dimensionless"], level="unit",)

    pdt.assert_frame_equal(res, exp, check_like=True, check_column_type=False)


def test_exceedance_probabilities_over_time_multiple_variables(test_processing_scm_df):
    test_processing_scm_df["variable"] = [
        str(i) for i in range(test_processing_scm_df.shape[0])
    ]

    with pytest.raises(ValueError):
        scmdata.processing.calculate_exceedance_probabilities_over_time(
            test_processing_scm_df, cols=["ensemble_member", "variable"], threshold=1.5,
        )


@pytest.mark.parametrize(
    "threshold,exp_val", ((1.0, 1.0), (1.5, 0.6), (2.0, 0.0),),
)
@output_name_options
def test_exceedance_probabilities(
    output_name, threshold, exp_val, test_processing_scm_df
):
    call_kwargs = _get_calculate_exceedance_probs_call_kwargs(output_name)
    res = scmdata.processing.calculate_exceedance_probabilities(
        test_processing_scm_df,
        cols="ensemble_member",
        threshold=threshold,
        **call_kwargs,
    )

    exp_idx = pd.MultiIndex.from_frame(
        test_processing_scm_df.meta.drop(
            "ensemble_member", axis="columns"
        ).drop_duplicates()
    )

    exp = pd.Series(exp_val, index=exp_idx)
    exp.name = _get_calculate_exeedance_probs_expected_name(output_name, threshold)
    exp.index = exp.index.set_levels(["dimensionless"], level="unit",)

    pdt.assert_series_equal(res, exp)


def test_exceedance_probabilities_multiple_res(
    test_processing_scm_df_multi_climate_model,
):
    start = test_processing_scm_df_multi_climate_model.copy()
    threshold = 1.5
    exp_vals = [0.6, 0.8]

    res = scmdata.processing.calculate_exceedance_probabilities(
        start, cols=["ensemble_member"], threshold=threshold,
    )

    exp_idx = pd.MultiIndex.from_frame(
        start.meta.drop("ensemble_member", axis="columns").drop_duplicates()
    )

    exp = pd.Series(exp_vals, index=exp_idx)
    exp.name = _get_calculate_exeedance_probs_expected_name(None, threshold)
    exp.index = exp.index.set_levels(["dimensionless"], level="unit",)

    pdt.assert_series_equal(res, exp)


def test_exceedance_probabilities_multiple_grouping(
    test_processing_scm_df_multi_climate_model,
):
    start = test_processing_scm_df_multi_climate_model.copy()
    threshold = 1.5
    exp_vals = [0.7]

    res = scmdata.processing.calculate_exceedance_probabilities(
        start, cols=["ensemble_member", "climate_model"], threshold=threshold,
    )

    exp_idx = pd.MultiIndex.from_frame(
        start.meta.drop(
            ["ensemble_member", "climate_model"], axis="columns"
        ).drop_duplicates()
    )

    exp = pd.Series(exp_vals, index=exp_idx)
    exp.name = _get_calculate_exeedance_probs_expected_name(None, threshold)
    exp.index = exp.index.set_levels(["dimensionless"], level="unit",)

    pdt.assert_series_equal(res, exp)


def test_exceedance_probabilities_multiple_variables(test_processing_scm_df):
    test_processing_scm_df["variable"] = [
        str(i) for i in range(test_processing_scm_df.shape[0])
    ]

    with pytest.raises(ValueError):
        scmdata.processing.calculate_exceedance_probabilities(
            test_processing_scm_df, cols=["ensemble_member", "variable"], threshold=1.5,
        )
