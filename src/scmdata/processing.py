"""
Miscellaneous functions for processing :class:`scmdata.ScmRun`

These functions are intended to be able to be used directly with
:meth:`scmdata.ScmRun.process_over`.
"""
import numpy as np

# exceedance probabilities
# categorisation


def calculate_crossing_times(ts, threshold, return_year=True):
    """
    Calculate the time at which each timeseries crosses a given threshold

    Parameters
    ----------
    ts : :class:`pd.DataFrame`
        Timeseries to calculate the crossing time of

    threshold : float
        Value to use as the threshold for crossing

    return_year : bool
        If ``True``, return the year instead of the datetime

    Returns
    -------
    :class:`pd.Series`
        Crossing time for ``ts``, using the index of ``ts`` as the output's
        index. If the threshold is not crossed, ``pd.NA`` is returned.

    Raises
    ------
    ValueError
        ``ts`` has more than one timeseries i.e. ``ts.shape[0] > 1``

    Notes
    -----
    This function only returns times that are in the columns of ``ts``. If you
    want a finer resolution then you should interpolate your data first. For
    example, if you have data on a ten-year timestep but want crossing times on
    an annual resolution, interpolate (or resample) to annual data before
    calling :func:`scmdata.processing.calculate_crossing_times`.
    """
    if ts.shape[0] > 1:
        raise ValueError(
            "Only one timeseries should be provided at a time. "
            "Received {}:\n{}".format(ts.shape[0], ts.index.to_frame(index=False))
        )

    ts = ts.iloc[0, :]

    ts_gt_threshold = ts[ts > threshold]
    if ts_gt_threshold.empty:
        return np.nan

    crossing_time = ts_gt_threshold.index[0]

    if return_year:
        return crossing_time.year

    return crossing_time


def calculate_exceedance_probabilities_over_time(ts, threshold):
    """
    Calculate exceedance probability at each point in time

    Parameters
    ----------
    ts : :class:`pd.DataFrame`
        Timeseries of which to calculate the exceedance probabilities over time

    threshold : float
        Value to use as the threshold for exceedance

    Returns
    -------
    :class:`pd.DataFrame`
        Timeseries of exceedance probability over time

    Raises
    ------
    ValueError
        ``ts`` has more than one timeseries i.e. ``ts.shape[0] > 1``

    Notes
    -----
    This differs from
    :func:`scmdata.processing.calculate_exceedance_probabilities` because it
    calculates the exceedance probability in each time point. That is different
    from calculating the exceedance probability by first determining the number
    of ensemble members which cross the threshold **at any point in time** and
    then dividing by the number of ensemble members. In general, this function
    will produce a maximum exceedance probability which is equal to or less
    than the output of
    :func:`scmdata.processingcalculate_exceedance_probabilities`. In our
    opinion, :func:`scmdata.processingcalculate_exceedance_probabilities` is
    the correct function to use if you want to know the exceedance probability
    of a scenario. This function gives a sense of how the exceedance
    probability evolves over time but, as we said, will generally slightly
    underestimate the exceedance probability over all time.
    """
    pass
