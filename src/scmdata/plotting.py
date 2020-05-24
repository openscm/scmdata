"""
Plotting helpers for DataFrames

See the example notebook 'plotting-with-seaborn.ipynb' for usage examples
"""

import warnings

import numpy as np

try:
    import seaborn as sns

    has_seaborn = True
except ImportError:  # pragma: no cover
    sns = None
    has_seaborn = False


RCMIP_SCENARIO_COLOURS = {
    "historical": "black",
    "ssp119": "#1e9583",
    "ssp126": "#1d3354",
    "ssp245": "#e9dc3d",
    "ssp370": "#f11111",
    "ssp370-lowNTCF-aerchemmip": "tab:pink",
    "ssp370-lowNTCF-gidden": "tab:red",
    "ssp434": "#63bce4",
    "ssp460": "#e78731",
    "ssp534-over": "#996dc8",
    "ssp585": "#830b22",
}


def lineplot(self, time_axis=None, **kwargs):  # pragma: no cover
    """
    Make a line plot via `seaborn's lineplot <https://seaborn.pydata.org/generated/seaborn.lineplot.html>`_

    Parameters
    ----------
    time_axis : {None, "year", "year-month", "days since 1970-01-01", "seconds since 1970-01-01"}
        Time axis to use for the plot. If `None`, :class:`datetime.datetime` objects will be used.
        If `"year"`, the year of each time point  will be used. If `"year-month", the year plus
        (month - 0.5) / 12  will be used. If `"days since 1970-01-01"`, the number of days  since 1st
        Jan 1970 will be used (calculated using the ``datetime``  module). If `"seconds since 1970-01-01"`,
        the number of seconds  since 1st Jan 1970 will be used (calculated using the ``datetime`` module).

    **kwargs
        Keyword arguments to be passed to ``seaborn.lineplot``. If none are passed,
        sensible defaults will be used.

    Returns
    -------
    :obj:`matplotlib.axes._subplots.AxesSubplot`
        Output of call to ``seaborn.lineplot``
    """
    if not has_seaborn:
        raise ImportError("seaborn is not installed. Run 'pip install seaborn'")

    plt_df = self.long_data(time_axis=time_axis)
    kwargs.setdefault("x", "time")
    kwargs.setdefault("y", "value")
    kwargs.setdefault("hue", "scenario")
    kwargs.setdefault("ci", "sd")
    kwargs.setdefault("estimator", np.median)

    ax = sns.lineplot(data=plt_df, **kwargs)

    return ax


def _deprecated_line_plot(self, **kwargs):  # pragma: no cover
    """
    Make a line plot via `seaborn's lineplot <https://seaborn.pydata.org/generated/seaborn.lineplot.html>`_

    Deprecated: use :func`lineplot` instead

    Parameters
    ----------
    **kwargs
        Keyword arguments to be passed to ``seaborn.lineplot``. If none are passed,
        sensible defaults will be used.

    Returns
    -------
    :obj:`matplotlib.axes._subplots.AxesSubplot`
        Output of call to ``seaborn.lineplot``
    """
    warnings.warn("Use lineplot instead", DeprecationWarning)
    self.lineplot(**kwargs)


def inject_plotting_methods(cls):
    """
    Inject the plotting functions

    Parameters
    ----------
    cls
        Target class
    """
    methods = [
        ("lineplot", lineplot),
        ("line_plot", _deprecated_line_plot),  # for compatibility
    ]

    for name, f in methods:
        setattr(cls, name, f)
