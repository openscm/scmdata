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
            [1.1, 1.2, 1.3, 1.3],
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
@pytest.mark.parametrize(
    "return_year,conv_to_year", ((None, True), (True, True), (False, False),)
)
def test_crossing_times(
    threshold, exp_vals, return_year, conv_to_year, test_processing_scm_df
):
    call_kwargs = {}
    if return_year is not None:
        call_kwargs["return_year"] = return_year

    res = test_processing_scm_df.process_over(
        [],
        scmdata.processing.calculate_crossing_times,
        threshold=threshold,
        **call_kwargs,
    )

    if conv_to_year:
        exp_vals = [v if pd.isnull(v) else v.year for v in exp_vals]

    exp = pd.Series(exp_vals, pd.MultiIndex.from_frame(test_processing_scm_df.meta),)

    pdt.assert_series_equal(res, exp)


def test_crossing_times_ensemble_group(test_processing_scm_df):
    error_msg = (
        "Only one timeseries should be provided at a time. Received 5:\n"
        "\\s*climate_model"
    )
    with pytest.raises(ValueError, match=error_msg):
        test_processing_scm_df.process_over(
            "ensemble_member",
            scmdata.processing.calculate_crossing_times,
            threshold=1.5,
        )


@pytest.mark.parametrize(
    "threshold,exp_vals",
    (
        (1.0, [0.8, 1.0, 1.0, 1.0]),
        (1.5, [0.0, 0.2, 0.4, 0.4]),
        (2.0, [0.0, 0.0, 0.0, 0.0]),
    ),
)
def test_exceedance_probabilities_over_time(
    threshold, exp_vals, test_processing_scm_df
):
    res = test_processing_scm_df.process_over(
        "ensemble_member",
        scmdata.processing.calculate_exceedance_probabilities_over_time,
        threshold=threshold,
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

    pdt.assert_frame_equal(res, exp, check_like=True, check_column_type=False)


def test_exceedance_probabilities_over_time_multiple_res(test_processing_scm_df):
    other = test_processing_scm_df + 0.1
    other["climate_model"] = "z_model"
    start = test_processing_scm_df.append(other)
    threshold = 1.5
    exp_vals = np.array([[0, 1, 2, 2], [1, 2, 3, 2]]) / 5

    res = start.process_over(
        ["ensemble_member"],
        scmdata.processing.calculate_exceedance_probabilities_over_time,
        threshold=threshold,
    )

    exp_idx = pd.MultiIndex.from_frame(
        start.meta.drop(["ensemble_member"], axis="columns").drop_duplicates()
    )

    exp = pd.DataFrame(
        exp_vals, index=exp_idx, columns=test_processing_scm_df.time_points.to_index(),
    )

    pdt.assert_frame_equal(res, exp, check_like=True, check_column_type=False)


def test_exceedance_probabilities_over_time_multiple_grouping(test_processing_scm_df):
    other = test_processing_scm_df + 0.1
    other["climate_model"] = "z_model"
    start = test_processing_scm_df.append(other)
    threshold = 1.5
    exp_vals = np.array([1, 3, 5, 4]) / 10

    res = start.process_over(
        ["climate_model", "ensemble_member"],
        scmdata.processing.calculate_exceedance_probabilities_over_time,
        threshold=threshold,
    )

    exp_idx = pd.MultiIndex.from_frame(
        start.meta.drop(
            ["climate_model", "ensemble_member"], axis="columns"
        ).drop_duplicates()
    )

    exp = pd.DataFrame(
        exp_vals[np.newaxis, :],
        index=exp_idx,
        columns=test_processing_scm_df.time_points.to_index(),
    )

    pdt.assert_frame_equal(res, exp, check_like=True, check_column_type=False)


@pytest.mark.parametrize(
    "threshold,exp_val",
    (
        (1.0, 1.0),
        (1.5, 0.6),
        (2.0, 0.0),
    ),
)
def test_exceedance_probabilities(
    threshold, exp_val, test_processing_scm_df
):
    res = test_processing_scm_df.process_over(
        "ensemble_member",
        scmdata.processing.calculate_exceedance_probabilities,
        threshold=threshold,
    )

    exp_idx = pd.MultiIndex.from_frame(
        test_processing_scm_df.meta.drop(
            "ensemble_member", axis="columns"
        ).drop_duplicates()
    )

    exp = pd.Series(exp_val, index=exp_idx)

    pdt.assert_series_equal(res, exp)

# TODO:
# - test group by climate model
# - test group by climate model and ensemble member
