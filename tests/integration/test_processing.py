import datetime as dt
import os.path
import re

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import scmdata.processing
from scmdata import ScmRun
from scmdata.errors import NonUniqueMetadataError
from scmdata.testing import _check_pandas_less_120


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
        process_over_cols="ensemble_member",
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
        start, process_over_cols=["ensemble_member"], threshold=threshold,
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
        start,
        process_over_cols=["climate_model", "ensemble_member"],
        threshold=threshold,
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
        process_over_cols="ensemble_member",
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
        start, process_over_cols=["ensemble_member"], threshold=threshold,
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
        start,
        process_over_cols=["ensemble_member", "climate_model"],
        threshold=threshold,
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


@pytest.mark.parametrize("col", ["unit", "variable"])
@pytest.mark.parametrize(
    "func,kwargs",
    (
        (scmdata.processing.calculate_exceedance_probabilities, {"threshold": 1.5}),
        (
            scmdata.processing.calculate_exceedance_probabilities_over_time,
            {"threshold": 1.5},
        ),
    ),
)
def test_requires_preprocessing(test_processing_scm_df, col, func, kwargs):
    test_processing_scm_df[col] = [
        str(i) for i in range(test_processing_scm_df.shape[0])
    ]

    with pytest.raises(ValueError):
        func(
            test_processing_scm_df,
            process_over_cols=["ensemble_member", col],
            **kwargs,
        )


def _get_calculate_peak_call_kwargs(output_name, variable):
    call_kwargs = {}
    if output_name is not None:
        call_kwargs["output_name"] = output_name

    return call_kwargs


@output_name_options
def test_peak(output_name, test_processing_scm_df):
    call_kwargs = _get_calculate_peak_call_kwargs(
        output_name, test_processing_scm_df.get_unique_meta("variable", True),
    )

    exp_vals = [1.2, 1.41, 1.6, 1.6, 1.72]
    res = scmdata.processing.calculate_peak(test_processing_scm_df, **call_kwargs,)

    exp_idx = pd.MultiIndex.from_frame(test_processing_scm_df.meta)

    exp = pd.Series(exp_vals, index=exp_idx)
    if output_name is not None:
        exp.index = exp.index.set_levels([output_name], level="variable")
    else:
        idx = exp.index.names
        exp = exp.reset_index()
        exp["variable"] = exp["variable"].apply(lambda x: "Peak {}".format(x))
        exp = exp.set_index(idx)[0]

    pdt.assert_series_equal(res, exp)


def test_peak_multi_variable(test_processing_scm_df_multi_climate_model):
    test_processing_scm_df_multi_climate_model["variable"] = [
        str(i) for i in range(test_processing_scm_df_multi_climate_model.shape[0])
    ]

    exp_vals = [1.2, 1.41, 1.6, 1.6, 1.72, 1.3, 1.51, 1.7, 1.7, 1.82]
    res = scmdata.processing.calculate_peak(test_processing_scm_df_multi_climate_model,)

    exp_idx = pd.MultiIndex.from_frame(test_processing_scm_df_multi_climate_model.meta)

    exp = pd.Series(exp_vals, index=exp_idx)
    idx = exp.index.names
    exp = exp.reset_index()
    exp["variable"] = exp["variable"].apply(lambda x: "Peak {}".format(x))
    exp = exp.set_index(idx)[0]

    pdt.assert_series_equal(res, exp)


def _get_calculate_peak_time_call_kwargs(return_year, output_name):
    call_kwargs = {}

    if return_year is not None:
        call_kwargs["return_year"] = return_year

    if output_name is not None:
        call_kwargs["output_name"] = output_name

    return call_kwargs


@output_name_options
@crossing_times_year_conversions
def test_peak_time(output_name, return_year, conv_to_year, test_processing_scm_df):
    call_kwargs = _get_calculate_peak_time_call_kwargs(return_year, output_name)

    exp_vals = [
        dt.datetime(2007, 1, 1),
        dt.datetime(2100, 1, 1),
        dt.datetime(2100, 1, 1),
        dt.datetime(2007, 1, 1),
        dt.datetime(2007, 1, 1),
    ]
    res = scmdata.processing.calculate_peak_time(test_processing_scm_df, **call_kwargs,)

    exp_idx = pd.MultiIndex.from_frame(test_processing_scm_df.meta)

    if conv_to_year:
        exp_vals = [v.year if conv_to_year else v for v in exp_vals]
        time_name = "Year"
    else:
        time_name = "Time"

    exp = pd.Series(exp_vals, index=exp_idx)
    if output_name is not None:
        exp.index = exp.index.set_levels([output_name], level="variable")
    else:
        idx = exp.index.names
        exp = exp.reset_index()
        exp["variable"] = exp["variable"].apply(
            lambda x: "{} of peak {}".format(time_name, x)
        )
        exp = exp.set_index(idx)[0]

    pdt.assert_series_equal(res, exp)


@crossing_times_year_conversions
def test_peak_time_multi_variable(
    return_year, conv_to_year, test_processing_scm_df_multi_climate_model
):
    test_processing_scm_df_multi_climate_model["variable"] = [
        str(i) for i in range(test_processing_scm_df_multi_climate_model.shape[0])
    ]

    call_kwargs = _get_calculate_peak_time_call_kwargs(return_year, None)

    exp_vals = [
        dt.datetime(2007, 1, 1),
        dt.datetime(2100, 1, 1),
        dt.datetime(2100, 1, 1),
        dt.datetime(2007, 1, 1),
        dt.datetime(2007, 1, 1),
    ] * 2

    res = scmdata.processing.calculate_peak_time(
        test_processing_scm_df_multi_climate_model, **call_kwargs
    )

    if conv_to_year:
        exp_vals = [v.year if conv_to_year else v for v in exp_vals]
        time_name = "Year"
    else:
        time_name = "Time"

    exp_idx = pd.MultiIndex.from_frame(test_processing_scm_df_multi_climate_model.meta)

    exp = pd.Series(exp_vals, index=exp_idx)
    idx = exp.index.names
    exp = exp.reset_index()

    exp["variable"] = exp["variable"].apply(
        lambda x: "{} of peak {}".format(time_name, x)
    )
    exp = exp.set_index(idx)[0]

    pdt.assert_series_equal(res, exp)


@pytest.mark.xfail(
    _check_pandas_less_120(),
    reason="pandas<1.2.0 can't handle non-numeric types in pivot",
)
@pytest.mark.parametrize(
    "index",
    (
        ["climate_model", "model", "scenario", "region"],
        ["climate_model", "scenario", "region"],
        ["climate_model", "model", "scenario", "region", "unit"],
    ),
)
@pytest.mark.parametrize(
    ",".join(
        [
            "exceedance_probabilities_thresholds",
            "exp_exceedance_prob_thresholds",
            "exceedance_probabilities_output_name",
            "exp_exceedance_probabilities_output_name",
            "exceedance_probabilities_variable",
            "exp_exceedance_probabilities_variable",
        ]
    ),
    (
        (
            None,
            [1.5, 2.0, 2.5],
            None,
            "{} exceedance probability",
            None,
            "Surface Air Temperature Change",
        ),
        (
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5],
            "Exceedance Probability|{:.2f}C",
            "Exceedance Probability|{:.2f}C",
            "Surface Temperature",
            "Surface Temperature",
        ),
    ),
)
@pytest.mark.parametrize(
    ",".join(
        [
            "peak_variable",
            "exp_peak_variable",
            "peak_quantiles",
            "exp_peak_quantiles",
            "peak_naming_base",
            "exp_peak_naming_base",
            "peak_time_naming_base",
            "exp_peak_time_naming_base",
            "peak_return_year",
            "exp_peak_return_year",
        ]
    ),
    (
        (
            "Surface Temperature",
            "Surface Temperature",
            None,
            [0.05, 0.17, 0.5, 0.83, 0.95],
            None,
            "{} peak",
            None,
            "{} peak year",
            None,
            True,
        ),
        (
            "Surface Temperature",
            "Surface Temperature",
            None,
            [0.05, 0.17, 0.5, 0.83, 0.95],
            None,
            "{} peak",
            "test {}",
            "test {}",
            None,
            True,
        ),
        (
            "Surface Temperature",
            "Surface Temperature",
            None,
            [0.05, 0.17, 0.5, 0.83, 0.95],
            None,
            "{} peak",
            None,
            "{} peak year",
            True,
            True,
        ),
        (
            None,
            "Surface Air Temperature Change",
            [0.05, 0.95],
            [0.05, 0.95],
            "test {}",
            "test {}",
            "test peak {}",
            "test peak {}",
            True,
            True,
        ),
        (
            None,
            "Surface Air Temperature Change",
            [0.05, 0.95],
            [0.05, 0.95],
            "test {}",
            "test {}",
            None,
            "{} peak time",
            False,
            False,
        ),
        (
            None,
            "Surface Air Temperature Change",
            [0.05, 0.95],
            [0.05, 0.95],
            "test {}",
            "test {}",
            "test peak {}",
            "test peak {}",
            False,
            False,
        ),
    ),
)
@pytest.mark.parametrize("progress", (True, False))
def test_calculate_summary_stats(
    exceedance_probabilities_thresholds,
    exp_exceedance_prob_thresholds,
    index,
    exceedance_probabilities_output_name,
    exp_exceedance_probabilities_output_name,
    exceedance_probabilities_variable,
    exp_exceedance_probabilities_variable,
    peak_quantiles,
    exp_peak_quantiles,
    peak_variable,
    exp_peak_variable,
    peak_naming_base,
    exp_peak_naming_base,
    peak_time_naming_base,
    exp_peak_time_naming_base,
    peak_return_year,
    exp_peak_return_year,
    progress,
    test_processing_scm_df_multi_climate_model,
):
    inp = test_processing_scm_df_multi_climate_model.copy()

    if "unit" not in index:
        exp_index = index + ["unit"]
    else:
        exp_index = index

    process_over_cols = inp.get_meta_columns_except(exp_index)

    exp = []
    for threshold in exp_exceedance_prob_thresholds:
        tmp = scmdata.processing.calculate_exceedance_probabilities(
            inp,
            threshold,
            process_over_cols,
            exp_exceedance_probabilities_output_name.format(threshold),
        )
        exp.append(tmp)

    peaks = scmdata.processing.calculate_peak(inp)
    peak_times = scmdata.processing.calculate_peak_time(
        inp, return_year=exp_peak_return_year
    )
    for q in exp_peak_quantiles:
        peak_q = peaks.groupby(exp_index).quantile(q)
        peak_q.name = exp_peak_naming_base.format(q)

        peak_time_q = peak_times.groupby(exp_index).quantile(q)
        peak_time_q.name = exp_peak_time_naming_base.format(q)

        exp.append(peak_q)
        exp.append(peak_time_q)

    dtype = "object" if not exp_peak_return_year else None
    exp = [v.reorder_levels(exp_index).astype(dtype) for v in exp]
    exp = pd.DataFrame(exp).T
    exp.columns.name = "statistic"
    exp = exp.stack("statistic")
    exp.name = "value"

    call_kwargs = {}
    if exceedance_probabilities_thresholds is not None:
        call_kwargs[
            "exceedance_probabilities_thresholds"
        ] = exceedance_probabilities_thresholds

    if exceedance_probabilities_output_name is not None:
        call_kwargs[
            "exceedance_probabilities_naming_base"
        ] = exceedance_probabilities_output_name

    inp_renamed = inp.copy()
    inp_renamed["variable"] = exp_exceedance_probabilities_variable
    if exceedance_probabilities_variable is not None:
        call_kwargs[
            "exceedance_probabilities_variable"
        ] = exceedance_probabilities_variable

    if peak_quantiles is not None:
        call_kwargs["peak_quantiles"] = peak_quantiles

    tmp = inp.copy()
    tmp["variable"] = exp_peak_variable
    try:
        inp_renamed = inp_renamed.append(tmp)
    except NonUniqueMetadataError:
        # variable already included
        pass
    if peak_variable is not None:
        call_kwargs["peak_variable"] = peak_variable

    if peak_naming_base is not None:
        call_kwargs["peak_naming_base"] = peak_naming_base

    if peak_time_naming_base is not None:
        call_kwargs["peak_time_naming_base"] = peak_time_naming_base

    if peak_time_naming_base is not None:
        call_kwargs["peak_time_naming_base"] = peak_time_naming_base

    if peak_return_year is not None:
        call_kwargs["peak_return_year"] = peak_return_year

    res = scmdata.processing.calculate_summary_stats(
        inp_renamed, index, progress=progress, **call_kwargs,
    )

    pdt.assert_series_equal(res.sort_index(), exp.sort_index())

    # then user can stack etc. if they want
    res.unstack(["statistic", "unit"])

    tmp = res.to_frame().reset_index()
    tmp["statistic"] = tmp["statistic"] + " (" + tmp["unit"] + ")"
    tmp = tmp.drop("unit", axis="columns")
    tmp.pivot_table(index=set(index) - {"unit"}, columns=["statistic"], values="value")


def test_calculate_summary_stats_no_exceedance_probability_var(
    test_processing_scm_df_multi_climate_model,
):
    error_msg = re.escape(
        "exceedance_probabilities_variable `junk` is not available. "
        "Available variables:{}".format(
            test_processing_scm_df_multi_climate_model.get_unique_meta("variable")
        )
    )
    with pytest.raises(ValueError, match=error_msg):
        scmdata.processing.calculate_summary_stats(
            test_processing_scm_df_multi_climate_model,
            ["model", "scenario"],
            exceedance_probabilities_variable="junk",
        )


def test_calculate_summary_stats_no_peak_variable(
    test_processing_scm_df_multi_climate_model,
):
    error_msg = re.escape(
        "peak_variable `junk` is not available. "
        "Available variables:{}".format(
            test_processing_scm_df_multi_climate_model.get_unique_meta("variable")
        )
    )
    with pytest.raises(ValueError, match=error_msg):
        scmdata.processing.calculate_summary_stats(
            test_processing_scm_df_multi_climate_model,
            ["model", "scenario"],
            peak_variable="junk",
        )



@pytest.fixture
def sr15_inferred_temperature_quantiles(test_data_path):
    # fake the temperature quantiles in preparation for the categorisation tests
    # we do this as P33 is not included in the SR1.5 output, yet we need it for
    # the categorisation
    sr15_output = scmdata.ScmRun(
        os.path.join(test_data_path, "sr15", "sr15-output.csv"),
    )
    sr15_exceedance_probs = sr15_output.filter(variable="*Exceedance*")

    out = []
    for cm in ["MAGICC", "FAIR"]:
        cm_ep = sr15_exceedance_probs.filter(variable="*{}*".format(cm))
        cm_median = sr15_output.filter(variable="*{}*MED".format(cm)).timeseries()
        for p in [0.67, 0.5, 0.34]:
            quantile = 1 - p
            cm_q = cm_median.reset_index()
            cm_q["variable"] = cm_q["variable"].str.replace("MED", "P{}".format(int(np.round(quantile*100, 0))))
            cm_q = cm_q.set_index(cm_median.index.names).sort_index()
            cm_q.iloc[:, :] = 10
            for t in [2.0, 1.5]:
                cm_ep_t = cm_ep.filter(variable="*{}*".format(t)).timeseries()
                cm_ep_t_lt = cm_ep_t <= p
                cm_ep_t_lt = cm_ep_t_lt.reorder_levels(cm_q.index.names).sort_index()
                cm_ep_t_lt.index = cm_q.index
                cm_q[cm_ep_t_lt] = t

            out.append(scmdata.ScmRun(cm_q))

    out = scmdata.run_append(out)
    return out


@pytest.fixture()
def sr15_temperatures_unmangled_names(sr15_inferred_temperature_quantiles):
    out = sr15_inferred_temperature_quantiles.copy()
    out["quantile"] = out["variable"].apply(lambda x: float(x.split("|")[-1].strip("P")) / 100)
    out["variable"] = out["variable"].apply(lambda x: "|".join(x.split("|")[:-1]))

    return out


def test_categorisation_sr15(sr15_temperatures_unmangled_names):
    index = ["model", "scenario"]
    exp = sr15_temperatures_unmangled_names.meta[index + ["category"]].drop_duplicates().set_index(index)["category"]

    inp = sr15_temperatures_unmangled_names.drop_meta(["category", "version"]).filter(variable="*MAGICC*")

    res = scmdata.processing.categorisation_sr15(
        inp,
        index=index,
    )

    category_counts = res.value_counts()
    assert category_counts["Above 2C"] == 189
    assert category_counts["Lower 2C"] == 74
    assert category_counts["Higher 2C"] == 58
    assert category_counts["1.5C low overshoot"] == 44
    assert category_counts["1.5C high overshoot"] == 37
    assert category_counts["Below 1.5C"] == 9

# test multiple variable failure
# test unit conversion failures
# test summary stats
