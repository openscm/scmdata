"""
Miscellaneous functions for processing :class:`scmdata.ScmRun`

These functions are intended to be able to be used directly with
:meth:`scmdata.ScmRun.process_over`.
"""
import numpy as np
import pandas as pd

# categorisation
# peak warming
# year of peak warming


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


def calculate_exceedance_probabilities(scmrun, threshold, process_over_cols, output_name=None):
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

    out = _get_exceedance_fraction(timeseries_gt_threshold.any(axis=1), group_cols,)

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
        series. If not supplied, "{threshold} exceedance probability" will be used.

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

    out = _get_exceedance_fraction(timeseries_gt_threshold, group_cols,)

    if not isinstance(out, pd.DataFrame):  # pragma: no cover # emergency valve
        raise AssertionError("How did we end up without a dataframe?")

    output_name = _get_exceedance_prob_output_name(output_name, threshold)
    out = _set_index_level_to(out, "variable", output_name)
    out = _set_index_level_to(out, "unit", "dimensionless")

    return out


def calculate_summary_stats(
    scmrun,
    index,
    exceedance_probabilities_thresholds=[1.5, 2.0, 2.5],
    exceedance_probabilities_variable="Surface Air Temperature Change",
    exceedance_probabilities_naming_base=None,
):
    """
    Calculate common summary statistics

    Parameters
    ----------
    scmrun : :class:`scmdata.ScmRun`
        Data of which to calculate the stats

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

    index : list[str]
        Columns to use in the index of the output (unit is added if not
        included)

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

    exceedance_prob_calls = [
        (
            calculate_exceedance_probabilities,
            [scmrun, t, process_over_cols],
            {"output_name": exceedance_probabilities_naming_base.format(t)}
        )
        for t in exceedance_probabilities_thresholds
    ]
    func_calls_args_kwargs = exceedance_prob_calls

    out = pd.DataFrame([
        func(*args, **kwargs)
        for func, args, kwargs in func_calls_args_kwargs
    ]).T
    out.columns.name = "statistic"
    out = out.stack("statistic")
    out.name = "value"

    return out
