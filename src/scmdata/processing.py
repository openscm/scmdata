"""
Miscellaneous functions for processing :class:`scmdata.ScmRun`

These functions are intended to be able to be used directly with
:meth:`scmdata.ScmRun.process_over`.
"""
import numpy as np

# exceedance probabilities
# categorisation


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
    timeseries = scmrun.timeseries()
    timeseries_gt_threshold = timeseries > threshold

    out = timeseries_gt_threshold.idxmax(axis=1)

    if return_year:
        out = out.apply(lambda x: x.year).astype(int)

    # if don't cross, set to nan
    out[~timeseries_gt_threshold.any(axis=1)] = np.nan

    return out
