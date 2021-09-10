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

    if conv_to_year:
        exp_vals = [v if pd.isnull(v) else v.year for v in exp_vals]
    else:
        exp_vals = [pd.NaT if pd.isnull(v) else v for v in exp_vals]

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
        test_processing_scm_df_multi_climate_model,
        threshold=threshold,
        **call_kwargs,
    )

    if conv_to_year:
        exp_vals = [v if pd.isnull(v) else v.year for v in exp_vals]
    else:
        exp_vals = [pd.NaT if pd.isnull(v) else v for v in exp_vals]

    exp = pd.Series(exp_vals, pd.MultiIndex.from_frame(test_processing_scm_df_multi_climate_model.meta))

    pdt.assert_series_equal(res, exp)

