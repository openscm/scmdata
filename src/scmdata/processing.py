"""
Miscellaneous functions for processing :class:`scmdata.ScmRun`

These functions are intended to be able to be used directly with
:meth:`scmdata.ScmRun.process_over`.
"""
import numpy as np
import pandas as pd
import tqdm.autonotebook as tqdman

from .errors import MissingRequiredColumnError
from .run import ScmRun


def _get_ts_gt_threshold(scmrun, threshold):
    timeseries = scmrun.timeseries()

    return timeseries > threshold


def calculate_crossing_times(scmrun, threshold, return_year=True):
    """
    Calculate the time at which each timeseries crosses a given threshold

    Parameters
    ----------
    scmrun : :class:`scmdata.ScmRun`
        Data to calculate the crossing time of

    threshold : float
        Value to use as the threshold for crossing

    return_year : bool
        If ``True``, return the year instead of the datetime

    Returns
    -------
    :class:`pd.Series`
        Crossing time for ``scmrun``, using the meta of ``scmrun`` as the
        output's index. If the threshold is not crossed, ``pd.NA`` is returned.

    Notes
    -----
    This function only returns times that are in the columns of ``scmrun``. If
    you want a finer resolution then you should interpolate your data first.
    For example, if you have data on a ten-year timestep but want crossing
    times on an annual resolution, interpolate (or resample) to annual data
    before calling ``calculate_crossing_times``.
    """
    timeseries_gt_threshold = _get_ts_gt_threshold(scmrun, threshold)
    out = timeseries_gt_threshold.idxmax(axis=1)

    if return_year:
        out = out.apply(lambda x: x.year).astype(int)

    # if don't cross, set to nan
    out[~timeseries_gt_threshold.any(axis=1)] = np.nan

    return out


def calculate_crossing_times_quantiles(
    crossing_times,
    groupby,
    quantiles=(0.05, 0.5, 0.95),
    nan_fill_value=10**6,
    out_nan_threshold=10**5,
    interpolation="linear",
):
    """
    Calculate quantiles of crossing times

    This calculation is non-trivial because some timeseries may never cross
    a given threshold. As a result, some care is required to return
    sensible quantiles. In this function, the quantiles are calculated
    as follows:

        #. all nan values in ``crossing_times`` are filled with ``nan_fill_value``

        #. quantiles are calculated using ``pd.groupby.quantile``

        #. quantiles which never crossed are inferred by examining whether the
           output values are greater than ``out_nan_threshold``. If the calculated
           value is greater than ``out_nan_threshold`` then nan is returned for
           this quantile.

    Parameters
    ----------
    crossing_times : :class:`pd.Series`
        Crossing times, can be calculated using
        :func:`scmdata.processing.calculate_crossing_times`

    groupby : list[str]
        Columns to group the output by

    quantiles : float
        Quantiles to calculate

    nan_fill_value : float
        Value to use to fill in nan values before calculating the quantiles

    out_nan_threshold : float
        Threshold to decide whether a calculated quantile should be nan or not

    interpolation : str
        Interpolation to use when calculating the quantiles, see
        :meth:`pandas.Series.quantile`

    Returns
    -------
    :class:`pd.Series`
        Crossing time quantiles

    Raises
    ------
    NotImplementedError
        ``crossing_times`` contains datetime objects, please raise an
        `issue <https://github.com/openscm/scmdata/issues>`_ if this is your
        use case.

    Examples
    --------
    >>> crossing_times = pd.Series(
    ...     [pd.NA, pd.NA, 2100, 2007, 2006, pd.NA, 2100, 2007, 2006, 2006],
    ...     index=pd.MultiIndex.from_product(
    ...         [["a_scenario"], ["z_model", "x_model"], range(5)],
    ...         names=["scenario", "climate_model", "ensemble_member"]
    ...     )
    ... )
    >>> crossing_times
    scenario    climate_model  ensemble_member
    a_scenario  z_model        0                  <NA>
                               1                  <NA>
                               2                  2100
                               3                  2007
                               4                  2006
                x_model        0                  <NA>
                               1                  2100
                               2                  2007
                               3                  2006
                               4                  2006
    dtype: object
    >>> scmdata.processing.calculate_crossing_times_quantiles(
    ...     crossing_times, groupby=["climate_model", "scenario"]
    ... )
    climate_model  scenario    quantile
    x_model        a_scenario  0.05        2006.0
                               0.50        2007.0
                               0.95           NaN
    z_model        a_scenario  0.05        2006.2
                               0.50        2100.0
                               0.95           NaN
    """
    if pd.api.types.is_datetime64_any_dtype(crossing_times):
        # the issue with datetimes is the fill value, pandas timestamps are
        # somewhat limited so this is not so easy to implement
        raise NotImplementedError(
            "Calculating crossing time quantiles with datetimes is not yet "
            "supported, please raise an issue "
            "(https://github.com/openscm/scmdata/issues) to discuss your use case"
        )

    crossing_times_full = crossing_times.fillna(nan_fill_value)
    crossing_times_quantiles = crossing_times_full.groupby(groupby).quantile(
        q=quantiles, interpolation=interpolation
    )
    out = crossing_times_quantiles.where(
        crossing_times_quantiles < out_nan_threshold, other=pd.NA
    )
    out.index = out.index.set_names("quantile", level=-1)

    return out


def _assert_only_one_value(scmrun, col):
    if len(scmrun.get_unique_meta(col)) > 1:
        raise ValueError(
            "More than one value for {}. "
            "This is unlikely to be what you want.".format(col)
        )


def _get_exceedance_fraction(ts, group_cols):
    grouper = ts.groupby(group_cols)
    number_exceeding = grouper.sum()
    number_members = grouper.count()
    out = number_exceeding / number_members

    return out


_DEFAULT_EXCEEDANCE_PROB_OUTPUT_BASE = "{} exceedance probability"


def _get_exceedance_prob_output_name(output_name, threshold):
    if output_name is None:
        return _DEFAULT_EXCEEDANCE_PROB_OUTPUT_BASE.format(threshold)

    return output_name


def _set_index_level_to(inp, col, val):
    inp.index = inp.index.set_levels([val], level=col)

    return inp


def calculate_exceedance_probabilities(
    scmrun, threshold, process_over_cols, output_name=None
):
    """
    Calculate exceedance probability over all time

    Parameters
    ----------
    scmrun : :class:`scmdata.ScmRun`
        Ensemble of which to calculate the exceedance probability

    threshold : float
        Value to use as the threshold for exceedance

    process_over_cols : list[str]
        Columns to not use when grouping the timeseries (typically "run_id" or
        "ensemble_member" or similar)

    output_name : str
        If supplied, the name of the output series. If not supplied,
        "{threshold} exceedance probability" will be used.

    Returns
    -------
    :class:`pd.Series`
        Exceedance probability over all time over all members of each group in
        ``scmrun``

    Raises
    ------
    ValueError
        ``scmrun`` has more than one variable or more than one unit (convert to
        a single unit before calling this function if needed)

    Notes
    -----
    See the notes of
    :func:`scmdata.processing.calculate_exceedance_probabilities_over_time`
    for an explanation of how the two calculations differ. For most purposes,
    this is the correct function to use.
    """
    _assert_only_one_value(scmrun, "variable")
    _assert_only_one_value(scmrun, "unit")
    timeseries_gt_threshold = _get_ts_gt_threshold(scmrun, threshold)
    group_cols = list(scmrun.get_meta_columns_except(process_over_cols))

    out = _get_exceedance_fraction(
        timeseries_gt_threshold.any(axis=1),
        group_cols,
    )

    if not isinstance(out, pd.Series):  # pragma: no cover # emergency valve
        raise AssertionError("How did we end up without a series?")

    output_name = _get_exceedance_prob_output_name(output_name, threshold)
    out.name = output_name
    out = _set_index_level_to(out, "unit", "dimensionless")

    return out


def calculate_exceedance_probabilities_over_time(
    scmrun, threshold, process_over_cols, output_name=None
):
    """
    Calculate exceedance probability at each point in time

    Parameters
    ----------
    scmrun : :class:`scmdata.ScmRun`
        Ensemble of which to calculate the exceedance probability over time

    threshold : float
        Value to use as the threshold for exceedance

    process_over_cols : list[str]
        Columns to not use when grouping the timeseries (typically "run_id" or
        "ensemble_member" or similar)

    output_name : str
        If supplied, the value to put in the "variable" columns of the output
        :class:`pd.DataFrame`. If not supplied, "{threshold} exceedance
        probability" will be used.

    Returns
    -------
    :class:`pd.DataFrame`
        Timeseries of exceedance probability over time

    Raises
    ------
    ValueError
        ``scmrun`` has more than one variable or more than one unit (convert to
        a single unit before calling this function if needed)

    Notes
    -----
    This differs from
    :func:`scmdata.processing.calculate_exceedance_probabilities` because it
    calculates the exceedance probability at each point in time. That is
    different from calculating the exceedance probability by first determining
    the number of ensemble members which cross the threshold **at any point in
    time** and then dividing by the number of ensemble members. In general,
    this function will produce a maximum exceedance probability which is equal
    to or less than the output of
    :func:`scmdata.processing.calculate_exceedance_probabilities`. In our
    opinion, :func:`scmdata.processing.calculate_exceedance_probabilities` is
    the correct function to use if you want to know the exceedance probability
    of a scenario. This function gives a sense of how the exceedance
    probability evolves over time but, as we said, will generally slightly
    underestimate the exceedance probability over all time.
    """
    _assert_only_one_value(scmrun, "variable")
    _assert_only_one_value(scmrun, "unit")
    timeseries_gt_threshold = _get_ts_gt_threshold(scmrun, threshold)
    group_cols = list(scmrun.get_meta_columns_except(process_over_cols))

    out = _get_exceedance_fraction(
        timeseries_gt_threshold,
        group_cols,
    )

    if not isinstance(out, pd.DataFrame):  # pragma: no cover # emergency valve
        raise AssertionError("How did we end up without a dataframe?")

    output_name = _get_exceedance_prob_output_name(output_name, threshold)
    out = _set_index_level_to(out, "variable", output_name)
    out = _set_index_level_to(out, "unit", "dimensionless")

    return out


def _set_peak_output_name(out, output_name, default_lead):
    if output_name is not None:
        out = _set_index_level_to(out, "variable", output_name)
    else:
        idx = out.index.names
        out = out.reset_index()
        out["variable"] = out["variable"].apply(
            lambda x: "{} {}".format(default_lead, x)
        )
        out = out.set_index(idx)[0]

    return out


def calculate_peak(scmrun, output_name=None):
    """
    Calculate peak i.e. maximum of each timeseries

    Parameters
    ----------
    scmrun : :class:`scmdata.ScmRun`
        Ensemble of which to calculate the exceedance probability over time

    output_name : str
        If supplied, the value to put in the "variable" columns of the output
        series. If not supplied, "Peak {variable}" will be used.

    Returns
    -------
    :class:`pd.Series`
        Peak of each timeseries
    """
    out = scmrun.timeseries().max(axis=1)
    out = _set_peak_output_name(out, output_name, "Peak")

    return out


def calculate_peak_time(scmrun, output_name=None, return_year=True):
    """
    Calculate peak time i.e. the time at which each timeseries reaches its maximum

    Parameters
    ----------
    scmrun : :class:`scmdata.ScmRun`
        Ensemble of which to calculate the exceedance probability over time

    output_name : str
        If supplied, the value to put in the "variable" columns of the output
        series. If not supplied, "Peak {variable}" will be used.

    return_year : bool
        If ``True``, return the year instead of the datetime

    Returns
    -------
    :class:`pd.Series`
        Peak of each timeseries
    """
    out = scmrun.timeseries().idxmax(axis=1)
    if return_year:
        out = out.apply(lambda x: x.year)

    out = _set_peak_output_name(
        out, output_name, "Year of peak" if return_year else "Time of peak"
    )

    return out


def categorisation_sr15(scmrun, index):
    """
    Categorise using the algorithm employed in SR1.5

    For more information, see the SR1.5 scenario analysis
    `notebook <data.ene.iiasa.ac.at/sr15_scenario_analysis/assessment/sr15_2.0_categories_indicators.html>`_.

    Parameters
    ----------
    scmrun : :class: `scmdata.ScmRun`
        Data to use for the classification. This should contain global-mean
        surface air temperatures  (GSAT) relative to 1850-1900 (using another
        reference period will not break this function, but is inconsistent with
        the original algorithm). The data must have a "quantile" column and it
        must have the 0.33, 0.5 and 0.66 quantiles calculated. This can be done
        with :meth:`scmdata.ScmRun.quantiles_over`.

    index : list[str]
        Columns in ``scmrun.meta`` to use as the index of the output

    Returns
    -------
    :class: `pd.Series`
        Categorisation of the timeseries

    Raises
    ------
    ValueError
        More than one variable or one unit is in ``scmrun``

    DimensionalityError
        The units cannot be converted to kelvin
    """
    if "quantile" not in scmrun.meta:
        raise MissingRequiredColumnError(
            "No `quantile` column, calculate quantiles using `.quantiles_over` "
            "to calculate the 0.33, 0.5 and 0.66 quantiles before calling "
            "this function"
        )

    required_quantiles = [0.33, 0.5, 0.66]
    available_quantiles = scmrun.get_unique_meta("quantile")
    if not all([q in available_quantiles for q in required_quantiles]):
        msg = (
            "Not all required quantiles are available, we require the "
            "0.33, 0.5 and 0.66 quantiles, available quantiles: `{}`"
        ).format(available_quantiles)
        raise ValueError(msg)

    _assert_only_one_value(scmrun, "variable")
    scmrun = scmrun.convert_unit("K")
    scmrun["unit"] = ""

    categories = pd.Series(
        name="category",
        index=pd.MultiIndex.from_frame(scmrun.meta[index].drop_duplicates()),
        dtype="object",
    )

    def _get_comp_series(res):
        reset_cols = list(set(res.index.names) - set(index))
        out = res.reset_index(reset_cols, drop=True).reorder_levels(index)

        return out

    peak_median = _get_comp_series(calculate_peak(scmrun.filter(quantile=0.5)))
    peak_p33 = _get_comp_series(calculate_peak(scmrun.filter(quantile=0.33)))
    peak_p66 = _get_comp_series(calculate_peak(scmrun.filter(quantile=0.66)))
    end_of_century_median = _get_comp_series(
        calculate_peak(scmrun.filter(quantile=0.5, year=2100))
    )
    categories[peak_median > 2.0] = "Above 2C"
    categories[peak_median <= 1.5] = "Below 1.5C"

    overshoot_15 = (peak_median > 1.5) & (end_of_century_median <= 1.5)
    categories[
        overshoot_15 & (peak_p33 <= 1.5)  # p exceed <= 0.67
    ] = "1.5C low overshoot"
    categories[
        overshoot_15 & (peak_p33 > 1.5)  # p exceed > 0.67
    ] = "1.5C high overshoot"

    still_uncategorised = categories.isnull()
    peak_p66_lte_2 = peak_p66 <= 2.0  # p exceed < 0.34
    categories[
        still_uncategorised & (peak_median <= 2.0) & ~peak_p66_lte_2
    ] = "Higher 2C"
    categories[still_uncategorised & peak_p66_lte_2] = "Lower 2C"

    if categories.isnull().any():  # pragma: no cover # emergency valve
        raise AssertionError("Unclassified results?")

    return categories


def _calculate_quantile_groupby(base, index, quantile):
    return base.groupby(index).quantile(quantile)


def _raise_missing_variable_error(name, requested, scmrun):
    msg = "{} `{}` is not available. " "Available variables:{}".format(
        name, requested, scmrun.get_unique_meta("variable")
    )
    raise ValueError(msg)


def calculate_summary_stats(
    scmrun,
    index,
    exceedance_probabilities_thresholds=(1.5, 2.0, 2.5),
    exceedance_probabilities_variable="Surface Air Temperature Change",
    exceedance_probabilities_naming_base=None,
    peak_quantiles=(0.05, 0.17, 0.5, 0.83, 0.95),
    peak_variable="Surface Air Temperature Change",
    peak_naming_base=None,
    peak_time_naming_base=None,
    peak_return_year=True,
    categorisation_variable="Surface Air Temperature Change",
    categorisation_quantile_cols=("ensemble_member",),
    progress=False,
):
    """
    Calculate common summary statistics

    Parameters
    ----------
    scmrun : :class:`scmdata.ScmRun`
        Data of which to calculate the stats

    index : list[str]
        Columns to use in the index of the output (unit is added if not
        included)

    exceedance_probabilities_threshold : list[float]
        Thresholds to use for exceedance probabilities

    exceedance_probabilities_variable : str
        Variable to use for exceedance probability calculations

    exceedance_probabilities_naming_base : str
        String to use as the base for naming the exceedance probabilities. Each
        exceedance probability output column will have a name given by
        ``exceedance_probabilities_naming_base.format(threshold)`` where
        threshold is the exceedance probability threshold to use. If not
        supplied, the default output of
        :func:`scmdata.processing.calculate_exceedance_probabilities` will be
        used.

    peak_quantiles : list[float]
        Quantiles to report in peak calculations

    peak_variable : str
        Variable of which to calculate the peak

    peak_naming_base : str
        Base to use for naming the peak outputs. This is combined with the
        quantile. If not supplied, ``"{} peak"`` is used so the outputs will be
        named e.g. "0.05 peak", "0.5 peak", "0.95 peak".

    peak_time_naming_base : str
        Base to use for naming the peak time outputs. This is combined with the
        quantile. If not supplied, ``"{} peak year"`` is used (unless
        ``peak_return_year`` is ``False`` in which case ``"{} peak time"`` is
        used) so the outputs will be named e.g. "0.05 peak year", "0.5 peak
        year", "0.95 peak year".

    peak_return_year : bool
        If ``True``, return the year of the peak of ``peak_variable``,
        otherwise return full dates

    categorisation_variable : str
        Variable to use for categorisation. Note that this variable point to
        timeseries that contain global-mean surface air temperatures  (GSAT)
        relative to 1850-1900 (using another reference period will not break
        this function, but is inconsistent with the original algorithm).

    categorisation_quantile_cols : list[str]
        Columns which represent individual ensemble members in the output (e.g.
        ["ensemble_member"]). The quantiles are taking over these columns
        before the data is passed to
        :func:`scmdata.processing.categorisation_sr15`.

    progress : bool
        Should a progress bar be shown whilst the calculations are done?

    Returns
    -------
    :class:`pd.DataFrame`
        Summary statistics, with each column being a statistic and the index
        being given by ``index``
    """
    if "unit" not in index:
        _index = index + ["unit"]
    else:
        _index = index

    process_over_cols = scmrun.get_meta_columns_except(_index)

    if exceedance_probabilities_naming_base is None:
        exceedance_probabilities_naming_base = _DEFAULT_EXCEEDANCE_PROB_OUTPUT_BASE

    scmrun_exceedance_prob = scmrun.filter(
        variable=exceedance_probabilities_variable,
        log_if_empty=False,
    )
    if scmrun_exceedance_prob.empty:
        _raise_missing_variable_error(
            "exceedance_probabilities_variable",
            exceedance_probabilities_variable,
            scmrun,
        )

    exceedance_prob_calls = [
        (
            calculate_exceedance_probabilities,
            [scmrun_exceedance_prob, t, process_over_cols],
            {"output_name": exceedance_probabilities_naming_base.format(t)},
            exceedance_probabilities_naming_base.format(t),
        )
        for t in exceedance_probabilities_thresholds
    ]

    if peak_naming_base is None:
        peak_naming_base = "{} peak"

    if peak_time_naming_base is None:
        if peak_return_year:
            peak_time_naming_base = "{} peak year"
        else:
            peak_time_naming_base = "{} peak time"

    scmrun_peak = scmrun.filter(
        variable=peak_variable,
        log_if_empty=False,
    )
    if scmrun_peak.empty:
        _raise_missing_variable_error("peak_variable", peak_variable, scmrun)

    # pre-calculate to avoid calculating multiple times
    peaks = calculate_peak(scmrun_peak)
    peak_calls = [
        (
            _calculate_quantile_groupby,
            [peaks, _index, q],
            {},
            peak_naming_base.format(q),
        )
        for q in peak_quantiles
    ]

    # pre-calculate to avoid calculating multiple times
    peak_times = calculate_peak_time(scmrun_peak, return_year=peak_return_year)
    peak_time_calls = [
        (
            _calculate_quantile_groupby,
            [peak_times, _index, q],
            {},
            peak_time_naming_base.format(q),
        )
        for q in peak_quantiles
    ]

    scmrun_categorisation = scmrun.filter(variable=categorisation_variable)
    if scmrun_categorisation.empty:
        _raise_missing_variable_error(
            "categorisation_variable", categorisation_variable, scmrun
        )

    _categorisation_quantile_cols = categorisation_quantile_cols
    if isinstance(_categorisation_quantile_cols, str):
        _categorisation_quantile_cols = [_categorisation_quantile_cols]
    if not all(
        [v in scmrun_categorisation.meta for v in _categorisation_quantile_cols]
    ):
        msg = (
            "categorisation_quantile_cols `{}` not in `scmrun`. "
            "Available columns:{}".format(
                categorisation_quantile_cols, scmrun.meta.columns.tolist()
            )
        )
        raise ValueError(msg)

    scmrun_categorisation = ScmRun(
        scmrun_categorisation.quantiles_over(
            cols=categorisation_quantile_cols,
            quantiles=[0.33, 0.5, 0.66],
        )
    )
    categorisation_calls = [
        (
            categorisation_sr15,
            [scmrun_categorisation, _index],
            {},
            "SR1.5 category",
        )
    ]
    func_calls_args_kwargs = (
        exceedance_prob_calls + peak_calls + peak_time_calls + categorisation_calls
    )

    if progress:
        iterator = tqdman.tqdm(func_calls_args_kwargs)
    else:
        iterator = func_calls_args_kwargs

    def get_result(func, args, kwargs, name):
        res = func(*args, **kwargs)
        res.name = name

        return res

    series = [
        get_result(func, args, kwargs, name).reorder_levels(_index)
        for func, args, kwargs, name in iterator
    ]
    if not peak_return_year:
        series = [s.astype("object") for s in series]

    out = pd.DataFrame(series).T

    out.columns.name = "statistic"
    out = out.stack("statistic")
    out.name = "value"

    return out
