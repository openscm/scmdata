import re
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from scmdata import ScmRun
from scmdata.testing import _check_pandas_less_110

maxes = pytest.importorskip("matplotlib.axes")
plt = pytest.importorskip("matplotlib.pyplot")

sample_quantiles_plumes = pytest.mark.parametrize(
    "quantiles_plumes",
    (
        (
            ((0.05, 0.95), 0.5),
            ((0.5,), 1.0),
        ),
        (((0.17, 0.83), 0.7),),
    ),
)


def test_plumeplot_default(plumeplot_scmrun):
    res = plumeplot_scmrun.plumeplot()
    assert isinstance(res, tuple)
    assert isinstance(res[0], maxes.Axes)
    assert isinstance(res[1], list)


@sample_quantiles_plumes
def test_plumeplot(plumeplot_scmrun, quantiles_plumes):
    plumeplot_scmrun.plumeplot(quantiles_plumes=quantiles_plumes)


@sample_quantiles_plumes
def test_plumeplot_pre_calculated(plumeplot_scmrun, quantiles_plumes):
    quantiles = [v for qv in quantiles_plumes for v in qv[0]]
    summary_stats = ScmRun(
        plumeplot_scmrun.quantiles_over("ensemble_member", quantiles=quantiles)
    )
    summary_stats.plumeplot(
        quantiles_plumes=quantiles_plumes,
        pre_calculated=True,
    )


@pytest.mark.xfail(
    _check_pandas_less_110(), reason="pandas<=1.1.0 spews warnings everywhere"
)
def test_plumeplot_pre_calculated_no_plume_for_one(plumeplot_scmrun):
    quantiles_plumes = (((0.05, 0.95), 0.5), ((0.5,), 1.0))
    quantiles = [v for qv in quantiles_plumes for v in qv[0]]
    summary_stats = ScmRun(
        plumeplot_scmrun.quantiles_over("ensemble_member", quantiles=quantiles)
    )
    # drop out the plume for one of the climate models
    summary_stats = summary_stats.filter(
        climate_model="a_model", quantile=[0.05, 0.95], keep=False
    )

    with pytest.warns(UserWarning) as record:
        summary_stats.plumeplot(
            quantiles_plumes=quantiles_plumes,
            pre_calculated=True,
        )

    assert len(record) == 2
    assert (
        record[0].message.args[0]
        == "Quantile 0.05 not available for a_scenario Surface Air Temperature Change"
    )
    assert (
        record[1].message.args[0]
        == "Quantile 0.95 not available for a_scenario Surface Air Temperature Change"
    )


def test_plumeplot_pre_calculated_no_median_for_one(plumeplot_scmrun):
    quantiles_plumes = (((0.05, 0.95), 0.5), ((0.5,), 1.0))
    quantiles = [v for qv in quantiles_plumes for v in qv[0]]
    summary_stats = ScmRun(
        plumeplot_scmrun.quantiles_over("ensemble_member", quantiles=quantiles)
    )
    # drop out the plume for one of the climate models
    summary_stats = summary_stats.filter(
        climate_model="a_model", quantile=0.5, keep=False
    )

    warn_msg = re.escape(
        "Quantile 0.5 not available for a_scenario Surface Air Temperature Change"
    )
    with pytest.warns(UserWarning, match=warn_msg):
        summary_stats.plumeplot(
            quantiles_plumes=quantiles_plumes,
            pre_calculated=True,
        )


@pytest.mark.xfail(
    _check_pandas_less_110(), reason="pandas<=1.1.0 spews warnings everywhere"
)
def test_plumeplot_pre_calculated_no_plume_for_one_no_median_for_other(
    plumeplot_scmrun,
):
    quantiles_plumes = (((0.05, 0.95), 0.5), ((0.5,), 1.0))
    quantiles = [v for qv in quantiles_plumes for v in qv[0]]
    summary_stats = ScmRun(
        plumeplot_scmrun.quantiles_over("ensemble_member", quantiles=quantiles)
    )
    # drop out the plume for one of the climate models
    summary_stats = summary_stats.filter(
        climate_model="a_model", quantile=[0.05, 0.95], keep=False
    ).filter(climate_model="a_model_2", quantile=0.5, keep=False)

    with pytest.warns(UserWarning) as record:
        _, lh = summary_stats.plumeplot(
            quantiles_plumes=quantiles_plumes,
            pre_calculated=True,
        )

    assert len(record) == 3
    assert (
        record[0].message.args[0]
        == "Quantile 0.05 not available for a_scenario Surface Air Temperature Change"
    )
    assert (
        record[1].message.args[0]
        == "Quantile 0.95 not available for a_scenario Surface Air Temperature Change"
    )
    assert (
        record[2].message.args[0]
        == "Quantile 0.5 not available for a_scenario_2 Surface Air Temperature Change"
    )


@pytest.mark.xfail(
    _check_pandas_less_110(), reason="pandas<=1.1.0 spews warnings everywhere"
)
def test_plumeplot_pre_calculated_no_plume_for_one_no_median_for_other_different_styles(
    plumeplot_scmrun,
):
    quantiles_plumes = (((0.05, 0.95), 0.5), ((0.5,), 1.0))
    quantiles = [v for qv in quantiles_plumes for v in qv[0]]
    summary_stats = ScmRun(
        plumeplot_scmrun.quantiles_over("ensemble_member", quantiles=quantiles)
    )
    # drop out the plume for one of the climate models
    summary_stats = summary_stats.filter(
        climate_model="a_model", quantile=[0.05, 0.95], keep=False
    ).filter(climate_model="a_model_2", quantile=0.5, keep=False)

    with pytest.warns(UserWarning) as record:
        summary_stats.plumeplot(
            quantiles_plumes=quantiles_plumes,
            pre_calculated=True,
            style_var="climate_model",
        )

    assert len(record) == 3
    assert (
        record[0].message.args[0]
        == "Quantile 0.05 not available for a_scenario a_model"
    )
    assert (
        record[1].message.args[0]
        == "Quantile 0.95 not available for a_scenario a_model"
    )
    assert (
        record[2].message.args[0]
        == "Quantile 0.5 not available for a_scenario_2 a_model_2"
    )


def test_plumeplot_warns_dashes_without_lines(scm_run):
    warn_msg = re.escape(
        "`dashes` was passed but no lines were plotted, the style "
        "settings will not be used"
    )
    with pytest.warns(UserWarning, match=warn_msg):
        scm_run.plumeplot(
            quantiles_plumes=(((0.17, 0.83), 0.7),),
            quantile_over="ensemble_member",
            dashes={"Surface Air Temperature Change": "--"},
        )


def test_plumeplot_non_unique_lines(plumeplot_scmrun):
    quantile_over = "climate_model"
    quantile = 0.05
    scenario = "a_scenario"
    variable = "Surface Air Temperature Change"

    summary_stats = ScmRun(
        plumeplot_scmrun.quantiles_over(quantile_over, quantiles=(0.05, 0.5, 0.95))
    )

    error_msg = re.escape(
        "More than one timeseries for "
        "quantile: {}, "
        "scenario: {}, "
        "variable: {}.\n"
        "Please process your data to create unique quantile timeseries "
        "before calling :meth:`plumeplot`.\n"
        "Found: {}".format(
            quantile,
            scenario,
            variable,
            summary_stats.filter(
                quantile=quantile, scenario=scenario, variable=variable
            ),
        )
    )
    with pytest.raises(ValueError, match=error_msg):
        summary_stats.plumeplot(pre_calculated=True)


def test_plumeplot_args(plumeplot_scmrun):
    ax = plt.figure().add_subplot(111)
    palette = {"a_model": "tab:blue", "a_model_2": "tab:red"}
    dashes = {"a_scenario": "-", "a_scenario_2": "--"}
    quantiles_plumes = [((0.05, 0.95), 0.5), ((0.17, 0.83), 0.7), ((0.5,), 1.0)]

    ax_out, legend_items = plumeplot_scmrun.plumeplot(
        ax=ax,
        quantiles_plumes=quantiles_plumes,
        hue_var="climate_model",
        palette=palette,
        hue_label="Climate model",
        style_var="scenario",
        dashes=dashes,
        style_label="Scenario",
        linewidth=3,
        time_axis="year",
    )

    assert ax_out == ax

    quantile_header_idx = 0
    palette_header_idx = quantile_header_idx + 1 + len(quantiles_plumes)
    style_header_idx = palette_header_idx + 1 + len(palette)
    for i, value in enumerate(legend_items):
        if i == quantile_header_idx:
            assert value.get_label() == "Quantiles"

        elif i < palette_header_idx:
            has_a_match = False
            for qp in quantiles_plumes:
                q = qp[0]
                if len(q) == 1:
                    if value.get_label() == "{:.0f}th".format(q[0] * 100):
                        has_a_match = True
                else:
                    if value.get_label() == "{:.0f}th - {:.0f}th".format(
                        q[0] * 100, q[1] * 100
                    ):
                        has_a_match = True

            assert has_a_match

        elif i == palette_header_idx:
            assert value.get_label() == "Climate model"

        elif i < style_header_idx:
            assert value.get_linestyle() == "-"
            has_a_match = False
            for k, v in palette.items():
                if (value.get_label() == k) and (value.get_color() == v):
                    has_a_match = True

            assert has_a_match

        elif i == style_header_idx:
            assert value.get_label() == "Scenario"

        else:
            assert value.get_color() == "gray"
            has_a_match = False
            for k, v in dashes.items():
                if (value.get_label() == k) and (value.get_linestyle() == v):
                    has_a_match = True

            assert has_a_match


@pytest.mark.parametrize("linewidth", (2, 2.5))
@pytest.mark.parametrize("time_axis", ("year", None))
@pytest.mark.parametrize(
    "quantiles_plumes",
    (
        (
            ((0.05, 0.95), 0.5),
            ((0.5,), 1.0),
        ),
        (
            ((0.17, 0.83), 0.7),
            ((0.4,), 1.0),
        ),
    ),
)
def test_plumeplot_values(plumeplot_scmrun, quantiles_plumes, time_axis, linewidth):
    mock_ax = MagicMock()

    palette = {"a_model": "tab:blue", "a_model_2": "tab:red"}
    dashes = {"a_scenario": "-", "a_scenario_2": "--"}

    quantiles = [v for qv in quantiles_plumes for v in qv[0]]
    summary_stats = ScmRun(
        plumeplot_scmrun.quantiles_over("ensemble_member", quantiles=quantiles)
    )
    summary_stats.plumeplot(
        ax=mock_ax,
        quantiles_plumes=quantiles_plumes,
        pre_calculated=True,
        hue_var="climate_model",
        palette=palette,
        style_var="scenario",
        dashes=dashes,
        time_axis=time_axis,
        linewidth=linewidth,
    )

    xaxis = summary_stats.timeseries(time_axis=time_axis).columns.tolist()

    def _is_in_calls(call_to_check, call_args_list):
        pargs_to_check = call_to_check[1]
        kargs_to_check = call_to_check[2]
        in_call = False
        for ca in call_args_list:
            pargs = ca[0]
            pargs_match = True
            for i, p in enumerate(pargs):
                if isinstance(p, np.ndarray):
                    if not np.allclose(p, pargs_to_check[i]):
                        pargs_match = False
                else:
                    if not p == pargs_to_check[i]:
                        pargs_match = False

                if not pargs_match:
                    print(p)
                    print(pargs_to_check[i])

            kargs = ca[1]
            if pargs_match and kargs == kargs_to_check:
                in_call = True

        return in_call

    def _get_with_empty_check(idf_filtered):
        if idf_filtered.empty:
            raise ValueError("Empty")

        return idf_filtered.values.squeeze()

    def _make_fill_between_call(idf, cm, scen, quant_alpha):
        quantiles = quant_alpha[0]
        alpha = quant_alpha[1]

        return call(
            xaxis,
            _get_with_empty_check(
                idf.filter(climate_model=cm, scenario=scen, quantile=quantiles[0])
            ),
            _get_with_empty_check(
                idf.filter(climate_model=cm, scenario=scen, quantile=quantiles[1])
            ),
            alpha=alpha,
            color=palette[cm],
            label="{:.0f}th - {:.0f}th".format(quantiles[0] * 100, quantiles[1] * 100),
        )

    def _make_plot_call(idf, cm, scen, quant_alpha):
        quantiles = quant_alpha[0]
        alpha = quant_alpha[1]

        return call(
            xaxis,
            _get_with_empty_check(
                idf.filter(climate_model=cm, scenario=scen, quantile=quantiles[0])
            ),
            color=palette[cm],
            linestyle=dashes[scen],
            linewidth=linewidth,
            label="{:.0f}th".format(quantiles[0] * 100),
            alpha=alpha,
        )

    cm_scen_combos = summary_stats.meta[["climate_model", "scenario"]].drop_duplicates()
    cm_scen_combos = [v[1].values.tolist() for v in cm_scen_combos.iterrows()]

    plume_qa = [q for q in quantiles_plumes if len(q[0]) == 2]
    fill_between_calls = [
        _make_fill_between_call(summary_stats, cm, scen, qa)
        for cm, scen in cm_scen_combos
        for qa in plume_qa
    ]

    # debug by looking at mock_ax.fill_between.call_args_list
    assert all(
        [
            _is_in_calls(c, mock_ax.fill_between.call_args_list)
            for c in fill_between_calls
        ]
    )

    line_qa = [q for q in quantiles_plumes if len(q[0]) == 1]
    plot_calls = [
        _make_plot_call(summary_stats, cm, scen, qa)
        for cm, scen in cm_scen_combos
        for qa in line_qa
    ]

    # debug by looking at mock_ax.plot.call_args_list
    assert all([_is_in_calls(c, mock_ax.plot.call_args_list) for c in plot_calls])


def test_plumeplot_no_dashes_single_dash_style(plumeplot_scmrun):
    # if there's only a single variable and it is used for style, without
    # specifying dashes then all the lines should come out solid
    ax, legend_items = plumeplot_scmrun.filter(
        variable="Surface Air Temperature Change"
    ).plumeplot(style_var="variable")

    assert legend_items[-1].get_linestyle() == "-"

    for line in ax.lines:
        assert line.get_linestyle() == "-"


# sensible error if missing style etc.
def test_error_missing_palette(plumeplot_scmrun):
    # extra definitions are fine
    plumeplot_scmrun.plumeplot(
        palette={"a_scenario": "blue", "a_scenario_2": "red", "b_scenario": "green"}
    )

    # missing definitions raise
    palette_miss = {"a_scenario_2": "red", "b_scenario": "green"}
    error_msg = re.escape("a_scenario not in palette: {}".format(palette_miss))
    with pytest.raises(KeyError, match=error_msg):
        plumeplot_scmrun.plumeplot(palette=palette_miss)


def test_error_missing_style(plumeplot_scmrun):
    # extra definitions are fine
    plumeplot_scmrun.plumeplot(
        dashes={"Surface Air Temperature Change": "-", "GMST": "--"}
    )

    # missing definitions raise
    dashes_miss = {"GMST": "--"}
    error_msg = re.escape(
        "Surface Air Temperature Change not in dashes: {}".format(dashes_miss)
    )
    with pytest.raises(KeyError, match=error_msg):
        plumeplot_scmrun.plumeplot(dashes=dashes_miss)
