"""
Plotting helpers for :obj:`ScmRun`

See the example notebook 'plotting-with-seaborn.ipynb' for usage examples
"""
import warnings
from itertools import cycle

import numpy as np

try:
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    has_matplotlib = True
except ImportError:  # pragma: no cover
    plt = None
    has_matplotlib = False

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

    If only a single unit is present, it will be used as the y-axis label.
    The axis object is returned so this can be changed by the user if desired.

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
    if "scenario" in self.meta_attributes:
        kwargs.setdefault("hue", "scenario")

    kwargs.setdefault("ci", "sd")
    kwargs.setdefault("estimator", np.median)

    ax = sns.lineplot(data=plt_df, **kwargs)

    try:
        unit = self.get_unique_meta("unit", no_duplicates=True)
        ax.set_ylabel(unit)
    except ValueError:
        pass  # don't set ylabel

    return ax


def plumeplot(  # pragma: no cover
    self,
    ax=None,
    quantiles_plumes=[((0.05, 0.95), 0.5), ((0.5,), 1.0),],
    hue_var="scenario",
    hue_label="Scenario",
    palette=None,
    style_var="variable",
    style_label="Variable",
    dashes=None,
    linewidth=2,
    time_axis=None,
    pre_calculated=False,
    quantile_over=("ensemble_member",),
):
    """
    Make a plume plot, showing plumes for custom quantiles

    Parameters
    ----------
    ax : :obj:`matplotlib.axes._subplots.AxesSubplot`
        Axes on which to make the plot

    quantiles_plumes : list[tuple[tuple, float]]
        Configuration to use when plotting quantiles. Each element is a tuple,
        the first element of which is itself a tuple and the second element of
        which is the alpha to use for the quantile. If the first element has
        length two, these two elements are the quantiles to plot and a plume
        will be made between these two quantiles. If the first element has
        length one, then a line will be plotted to represent this quantile.

    hue_var : str
        The column of ``self.meta`` which should be used to distinguish
        different hues.

    hue_label : str
        Label to use in the legend for ``hue_var``.

    palette : dict
        Dictionary defining the colour to use for different values of
        ``hue_var``.

    style_var : str
        The column of ``self.meta`` which should be used to distinguish
        different styles.

    style_label : str
        Label to use in the legend for ``style_var``.

    dashes : dict
        Dictionary defining the style to use for different values of
        ``style_var``.

    linewidth : float
        Width of lines to use (for quantiles which are not to be shown as
        plumes)

    time_axis : str
        Time axis to use for the plot (see :meth:`~ScmRun.timeseries`)

    pre_calculated : bool
        Are the quantiles pre-calculated? If no, the quantiles will be
        calculated within this function. Pre-calculating the quantiles using
        :meth:`ScmRun.quantiles_over` can lead to faster plotting if multiple
        plots are to be made with the same quantiles.

    quantile_over : str, tuple[str]
        Columns of ``self.meta`` over which the quantiles should be calculated.
        Only used if ``pre_calculated`` is ``False``.

    Returns
    -------
    :obj:`matplotlib.axes._subplots.AxesSubplot`, list
        Axes on which the plot was made and the legend items we have made (in
        case the user wants to move the legend to a different position for
        example)

    Examples
    --------
    >>> scmrun = ScmRun(
    ...     data=np.random.random((10, 3)).T,
    ...     columns={
    ...         "model": ["a_iam"],
    ...         "climate_model": ["a_model"] * 5 + ["a_model_2"] * 5,
    ...         "scenario": ["a_scenario"] * 5 + ["a_scenario_2"] * 5,
    ...         "ensemble_member": list(range(5)) + list(range(5)),
    ...         "region": ["World"],
    ...         "variable": ["Surface Air Temperature Change"],
    ...         "unit": ["K"],
    ...     },
    ...     index=[2005, 2010, 2015],
    ... )

    Plot the plumes, calculated over the different ensemble members.

    >>> scmrun.plumeplot(quantile_over="ensemble_member")

    Pre-calculate the quantiles, then plot

    >>> summary_stats = ScmRun(
    ...     scmrun.quantiles_over("ensemble_member", quantiles=quantiles)
    ... )
    >>> summary_stats.plumeplot(pre_calculated=True)

    Note
    ----
    ``scmdata`` is not a plotting library so this function is provided as is,
    with little testing. In some ways, it is more intended as inspiration for
    other users than as a robust plotting tool.
    """
    if not has_matplotlib:
        raise ImportError("matplotlib is not installed. Run 'pip install matplotlib'")

    if not pre_calculated:
        quantiles = [v for qv in quantiles_plumes for v in qv[0]]
        _pdf = type(self)(self.quantiles_over(quantile_over, quantiles=quantiles))
    else:
        _pdf = self

    if ax is None:
        ax = plt.figure().add_subplot(111)

    _palette = {} if palette is None else palette

    if dashes is None:
        _dashes = {}
        lines = ["-", "--", "-.", ":"]
        linestyle_cycler = cycle(lines)
    else:
        _dashes = dashes

    _plotted_lines = False

    quantile_labels = {}
    for q, alpha in quantiles_plumes:
        for hdf in _pdf.groupby(hue_var):
            hue_value = hdf.get_unique_meta(hue_var, no_duplicates=True)
            pkwargs = {"alpha": alpha}

            for hsdf in hdf.groupby(style_var):
                style_value = hsdf.get_unique_meta(style_var, no_duplicates=True)

                xaxis = hsdf.timeseries(time_axis=time_axis).columns.tolist()
                if palette is not None:
                    try:
                        pkwargs["color"] = _palette[hue_value]
                    except KeyError as exc:
                        error_msg = "{} not in palette: {}".format(hue_value, palette)
                        raise KeyError(error_msg) from exc

                elif hue_value in _palette:
                    pkwargs["color"] = _palette[hue_value]

                if len(q) == 2:
                    label = "{:.0f}th - {:.0f}th".format(q[0] * 100, q[1] * 100)
                    p = ax.fill_between(
                        xaxis,
                        _get_1d_or_raise(
                            hsdf.filter(quantile=q[0]), hue_var, style_var
                        ),
                        _get_1d_or_raise(
                            hsdf.filter(quantile=q[1]), hue_var, style_var
                        ),
                        label=label,
                        **pkwargs
                    )

                    if palette is None:
                        _palette[hue_value] = p.get_facecolor()[0]

                elif len(q) == 1:
                    _plotted_lines = True

                    if dashes is not None:
                        try:
                            pkwargs["linestyle"] = _dashes[style_value]
                        except KeyError as exc:
                            error_msg = "{} not in dashes: {}".format(
                                style_value, dashes
                            )
                            raise KeyError(error_msg) from exc
                    else:
                        _dashes[style_value] = next(linestyle_cycler)
                        pkwargs["linestyle"] = _dashes[style_value]

                    if isinstance(q[0], str):
                        label = q[0]
                    else:
                        label = "{:.0f}th".format(q[0] * 100)

                    p = ax.plot(
                        xaxis,
                        _get_1d_or_raise(
                            hsdf.filter(quantile=q[0]), hue_var, style_var
                        ),
                        label=label,
                        linewidth=linewidth,
                        **pkwargs
                    )[0]

                    if dashes is None:
                        _dashes[style_value] = p.get_linestyle()

                else:
                    raise ValueError(
                        "quantiles to plot must be of length one or two, "
                        "received: {}".format(q)
                    )

                if label not in quantile_labels:
                    quantile_labels[label] = p

    # Fake the line handles for the legend
    hue_val_lines = [
        mlines.Line2D([0], [0], color=_palette[hue_value], label=hue_value)
        for hue_value in self.get_unique_meta(hue_var)
    ]

    legend_items = [
        mpatches.Patch(alpha=0, label="Quantiles"),
        *quantile_labels.values(),
        mpatches.Patch(alpha=0, label=hue_label),
        *hue_val_lines,
    ]

    if _plotted_lines:
        style_val_lines = [
            mlines.Line2D(
                [0],
                [0],
                linestyle=_dashes[style_value],
                label=style_value,
                color="gray",
                linewidth=linewidth,
            )
            for style_value in self.get_unique_meta(style_var)
        ]
        legend_items += [
            mpatches.Patch(alpha=0, label=style_label),
            *style_val_lines,
        ]
    else:
        if dashes is not None:
            warnings.warn(
                "`dashes` was passed but no lines were plotted, the style settings will not be used"
            )

    ax.legend(handles=legend_items, loc="best")

    units = self.get_unique_meta("unit")
    if len(units) == 1:
        ax.set_ylabel(units[0])

    return ax, legend_items


def _get_1d_or_raise(in_scmrun, hue_var, style_var):
    out_arr = in_scmrun.values.squeeze()
    if len(out_arr.shape) > 1:
        quantile = in_scmrun.get_unique_meta("quantile", True)
        hue_var_value = in_scmrun.get_unique_meta(hue_var, True)
        style_var_value = in_scmrun.get_unique_meta(style_var, True)
        error_msg = (
            "More than one timeseries for "
            "quantile: {}, "
            "{}: {}, "
            "{}: {}.\n"
            "Please process your data to create unique quantile timeseries "
            "before calling :meth:`plumeplot`.\n"
            "Found: {}".format(
                quantile, hue_var, hue_var_value, style_var, style_var_value, in_scmrun,
            )
        )
        raise ValueError(error_msg)

    return out_arr


def _deprecated_line_plot(self, **kwargs):  # pragma: no cover
    """
    Make a line plot via `seaborn's lineplot <https://seaborn.pydata.org/generated/seaborn.lineplot.html>`_

    Deprecated: use :func:`lineplot` instead

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
    Inject the plotting methods

    Parameters
    ----------
    cls
        Target class
    """
    methods = [
        ("lineplot", lineplot),
        ("line_plot", _deprecated_line_plot),  # for compatibility
        ("plumeplot", plumeplot),
    ]

    for name, f in methods:
        setattr(cls, name, f)
