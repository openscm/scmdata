"""
Miscellaneous functions for processing :class:`scmdata.ScmRun`

These functions are intended to be able to be used directly with
:meth:`scmdata.ScmRun.process_over`.
"""
import numpy as np
import pandas as pd

# exceedance probabilities
# categorisation


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


def _get_exceedance_prob_output_name(output_name, threshold):
    if output_name is None:
        return "{} exceedance probability".format(threshold)

    return output_name


def _set_index_level_to(inp, col, val):
    inp.index = inp.index.set_levels([val], level=col)

    return inp


def calculate_exceedance_probabilities(scmrun, threshold, cols, output_name=None):
    """
    Calculate exceedance probability over all time

    Parameters
    ----------
    scmrun : :class:`scmdata.ScmRun`
        Ensemble of which to calculate the exceedance probability

    threshold : float
        Value to use as the threshold for exceedance

    cols : list[str]
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
    group_cols = list(scmrun.get_meta_columns_except(cols))

    out = _get_exceedance_fraction(timeseries_gt_threshold.any(axis=1), group_cols,)

    if not isinstance(out, pd.Series):  # pragma: no cover # emergency valve
        raise AssertionError("How did we end up without a series?")

    output_name = _get_exceedance_prob_output_name(output_name, threshold)
    out.name = output_name
    out = _set_index_level_to(out, "unit", "dimensionless")

    return out


def calculate_exceedance_probabilities_over_time(
    scmrun, threshold, cols, output_name=None
):
    """
    Calculate exceedance probability at each point in time

    Parameters
    ----------
    scmrun : :class:`scmdata.ScmRun`
        Ensemble of which to calculate the exceedance probability over time

    threshold : float
        Value to use as the threshold for exceedance

    cols : list[str]
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
    group_cols = list(scmrun.get_meta_columns_except(cols))

    out = _get_exceedance_fraction(timeseries_gt_threshold, group_cols,)

    if not isinstance(out, pd.DataFrame):  # pragma: no cover # emergency valve
        raise AssertionError("How did we end up without a dataframe?")

    output_name = _get_exceedance_prob_output_name(output_name, threshold)
    out = _set_index_level_to(out, "variable", output_name)
    out = _set_index_level_to(out, "unit", "dimensionless")

    return out
