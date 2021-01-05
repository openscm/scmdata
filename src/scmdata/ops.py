"""
Operations for :obj:`ScmRun`

These largely rely on
`Pint's Pandas interface <https://pint.readthedocs.io/en/0.13/pint-pandas.html>`_
to handle unit conversions automatically
"""
import warnings

import numpy as np
import pandas as pd
import pint_pandas
from openscm_units import unit_registry

from .time import TimePoints

try:
    import scipy.integrate

    has_scipy = True
except ImportError:  # pragma: no cover
    scipy = None
    has_scipy = False


def prep_for_op(inp, op_cols, meta, ur=unit_registry):
    """
    Prepare dataframe for operation

    Parameters
    ----------
    inp : :obj:`ScmRun`
        :obj:`ScmRun` containing data to prepare

    op_cols : dict of str: str
        Dictionary containing the columns to drop in order to prepare
        for the operation as the keys (the values are not used). For
        example, if we have
        ``op_cols={"variable": "Emissions|CO2 - Emissions|CO2|Fossil"}``
        then we will drop the "variable" column from the index.

    ur : :obj:`pint.UnitRegistry`
        Pint unit registry to use for the operation

    Returns
    -------
    :obj:`pd.DataFrame`
        Timeseries to use for the operation. They are the transpose of
        the normal :meth:`ScmRun.timeseries` output with the columns
        being Pint arrays (unless "unit" is in op_cols in which case no
        units are available to be used so the columns are standard numpy
        arrays). We do this so that we can use
        `Pint's Pandas interface <https://pint.readthedocs.io/en/0.13/pint-pandas.html>`_
        to handle unit conversions automatically.
    """
    pint_pandas.PintType.ureg = ur

    key_cols = list(op_cols.keys())

    out = inp.timeseries(meta=meta).reset_index(key_cols, drop=True)

    out = out.T

    if "unit" not in op_cols:
        unit_level = out.columns.names.index("unit")
        out = out.pint.quantify(level=unit_level)

    return out


def set_op_values(output, op_cols):
    """
    Set operation values in output

    Parameters
    ----------
    output : :obj:`pd.Dataframe`
        Dataframe of which to update the values

    op_cols : dict of str: str
        Dictionary containing the columns to update as the keys and the value
        those columns should hold in the output as the values. For example,
        if we have
        ``op_cols={"variable": "Emissions|CO2 - Emissions|CO2|Fossil"}`` then
        the output will have a "variable" column with the value
        "Emissions|CO2 - Emissions|CO2|Fossil".

    Returns
    -------
    :obj:`pd.Dataframe`
        ``output`` with the relevant columns being set according to ``op_cols``.
    """
    for k, v in op_cols.items():
        output[k] = v

    return output


def _perform_op(base, other, op, use_pint_units=True):
    # pint handling means we have to do series by series
    out = []
    col_names = base.columns.names
    for col in base:
        if op in ["add", "subtract"]:
            try:
                if op == "add":
                    out.append(base[col] + other[col])

                elif op == "subtract":
                    out.append(base[col] - other[col])

            except KeyError:
                raise KeyError(
                    "No equivalent in `other` for {}".format(list(zip(col_names, col)))
                )

        elif op == "multiply":
            out.append(base[col] * other[col])

        elif op == "divide":
            out.append(base[col] / other[col])

        else:  # pragma: no cover
            raise NotImplementedError(op)

    if len(out) == 1:
        out = out[0].to_frame()
    else:
        out = pd.concat(out, axis="columns")

    out.columns.names = base.columns.names

    if use_pint_units:
        out = out.pint.dequantify()

    out = out.T

    return out


def subtract(self, other, op_cols, **kwargs):
    """
    Subtract values

    Parameters
    ----------
    other : :obj:`ScmRun`
        :obj:`ScmRun` containing data to subtract

    op_cols : dict of str: str
        Dictionary containing the columns to drop before subtracting as
        the keys and the value those columns should hold in the output
        as the values. For example, if we have
        ``op_cols={"variable": "Emissions|CO2 - Emissions|CO2|Fossil"}``
        then the subtraction will be performed with an index that uses
        all columns except the "variable" column and the output will
        have a "variable" column with the value
        "Emissions|CO2 - Emissions|CO2|Fossil".

    **kwargs : any
        Passed to :func:`prep_for_op`

    Returns
    -------
    :obj:`ScmRun`
        Difference between ``self`` and ``other``, using ``op_cols`` to
        define the columns which should be dropped before the data is
        aligned and to define the value of these columns in the output.

    Examples
    --------
        >>> import numpy as np
    >>> from scmdata import ScmRun
    >>>
    >>> IDX = [2010, 2020, 2030]
    >>>
    >>>
    >>> start = ScmRun(
    ...     data=np.arange(18).reshape(3, 6),
    ...     index=IDX,
    ...     columns={
    ...         "variable": [
    ...             "Emissions|CO2|Fossil",
    ...             "Emissions|CO2|AFOLU",
    ...             "Emissions|CO2|Fossil",
    ...             "Emissions|CO2|AFOLU",
    ...             "Cumulative Emissions|CO2",
    ...             "Surface Air Temperature Change",
    ...         ],
    ...         "unit": ["GtC / yr", "GtC / yr", "GtC / yr", "GtC / yr", "GtC", "K"],
    ...         "region": ["World|NH", "World|NH", "World|SH", "World|SH", "World", "World"],
    ...         "model": "idealised",
    ...         "scenario": "idealised",
    ...     },
    ... )
    >>>
    >>> start.head()
    time                                                            2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable                 unit     region   model     scenario
    Emissions|CO2|Fossil     GtC / yr World|NH idealised idealised                  0.0                  6.0                 12.0
    Emissions|CO2|AFOLU      GtC / yr World|NH idealised idealised                  1.0                  7.0                 13.0
    Emissions|CO2|Fossil     GtC / yr World|SH idealised idealised                  2.0                  8.0                 14.0
    Emissions|CO2|AFOLU      GtC / yr World|SH idealised idealised                  3.0                  9.0                 15.0
    Cumulative Emissions|CO2 GtC      World    idealised idealised                  4.0                 10.0                 16.0
    >>> fos = start.filter(variable="*Fossil")
    >>> fos.head()
    time                                                        2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable             unit     region   model     scenario
    Emissions|CO2|Fossil GtC / yr World|NH idealised idealised                  0.0                  6.0                 12.0
                                  World|SH idealised idealised                  2.0                  8.0                 14.0
    >>>
    >>> afolu = start.filter(variable="*AFOLU")
    >>> afolu.head()
    time                                                       2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable            unit     region   model     scenario
    Emissions|CO2|AFOLU GtC / yr World|NH idealised idealised                  1.0                  7.0                 13.0
                                 World|SH idealised idealised                  3.0                  9.0                 15.0
    >>>
    >>> fos_minus_afolu = fos.subtract(
    ...     afolu, op_cols={"variable": "Emissions|CO2|Fossil - AFOLU"}
    ... )
    >>> fos_minus_afolu.head()
    time                                                                  2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    model     scenario  region   variable                     unit
    idealised idealised World|NH Emissions|CO2|Fossil - AFOLU gigatC / a                 -1.0                 -1.0                 -1.0
                        World|SH Emissions|CO2|Fossil - AFOLU gigatC / a                 -1.0                 -1.0                 -1.0
    >>>
    >>> nh_minus_sh = nh.subtract(sh, op_cols={"region": "World|NH - SH"})
    >>> nh_minus_sh.head()
    time                                                               2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    model     scenario  region        variable             unit
    idealised idealised World|NH - SH Emissions|CO2|Fossil gigatC / a                 -2.0                 -2.0                 -2.0
                                      Emissions|CO2|AFOLU  gigatC / a                 -2.0                 -2.0                 -2.0
    """
    out = _perform_op(
        prep_for_op(self, op_cols, self.meta.columns, **kwargs),
        prep_for_op(other, op_cols, self.meta.columns, **kwargs),
        "subtract",
        use_pint_units="unit" not in op_cols,
    )

    out = set_op_values(out, op_cols)

    return type(self)(out)


def add(self, other, op_cols, **kwargs):
    """
    Add values

    Parameters
    ----------
    other : :obj:`ScmRun`
        :obj:`ScmRun` containing data to add

    op_cols : dict of str: str
        Dictionary containing the columns to drop before adding as the
        keys and the value those columns should hold in the output as
        the values. For example, if we have
        ``op_cols={"variable": "Emissions|CO2 - Emissions|CO2|Fossil"}``
        then the addition will be performed with an index that uses
        all columns except the "variable" column and the output will
        have a "variable" column with the value
        "Emissions|CO2 - Emissions|CO2|Fossil".

    **kwargs : any
        Passed to :func:`prep_for_op`

    Returns
    -------
    :obj:`ScmRun`
        Sum of ``self`` and ``other``, using ``op_cols`` to define the
        columns which should be dropped before the data is aligned and
        to define the value of these columns in the output.

    Examples
    --------
    >>> import numpy as np
    >>> from scmdata import ScmRun
    >>>
    >>> IDX = [2010, 2020, 2030]
    >>>
    >>>
    >>> start = ScmRun(
    ...     data=np.arange(18).reshape(3, 6),
    ...     index=IDX,
    ...     columns={
    ...         "variable": [
    ...             "Emissions|CO2|Fossil",
    ...             "Emissions|CO2|AFOLU",
    ...             "Emissions|CO2|Fossil",
    ...             "Emissions|CO2|AFOLU",
    ...             "Cumulative Emissions|CO2",
    ...             "Surface Air Temperature Change",
    ...         ],
    ...         "unit": ["GtC / yr", "GtC / yr", "GtC / yr", "GtC / yr", "GtC", "K"],
    ...         "region": ["World|NH", "World|NH", "World|SH", "World|SH", "World", "World"],
    ...         "model": "idealised",
    ...         "scenario": "idealised",
    ...     },
    ... )
    >>>
    >>> start.head()
    time                                                            2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable                 unit     region   model     scenario
    Emissions|CO2|Fossil     GtC / yr World|NH idealised idealised                  0.0                  6.0                 12.0
    Emissions|CO2|AFOLU      GtC / yr World|NH idealised idealised                  1.0                  7.0                 13.0
    Emissions|CO2|Fossil     GtC / yr World|SH idealised idealised                  2.0                  8.0                 14.0
    Emissions|CO2|AFOLU      GtC / yr World|SH idealised idealised                  3.0                  9.0                 15.0
    Cumulative Emissions|CO2 GtC      World    idealised idealised                  4.0                 10.0                 16.0
    >>> fos = start.filter(variable="*Fossil")
    >>> fos.head()
    time                                                        2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable             unit     region   model     scenario
    Emissions|CO2|Fossil GtC / yr World|NH idealised idealised                  0.0                  6.0                 12.0
                                  World|SH idealised idealised                  2.0                  8.0                 14.0
    >>>
    >>> afolu = start.filter(variable="*AFOLU")
    >>> afolu.head()
    time                                                       2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable            unit     region   model     scenario
    Emissions|CO2|AFOLU GtC / yr World|NH idealised idealised                  1.0                  7.0                 13.0
                                 World|SH idealised idealised                  3.0                  9.0                 15.0
    >>>
    >>> total = fos.add(afolu, op_cols={"variable": "Emissions|CO2"})
    >>> total.head()
    time                                                   2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    model     scenario  region   variable      unit
    idealised idealised World|NH Emissions|CO2 gigatC / a                  1.0                 13.0                 25.0
                        World|SH Emissions|CO2 gigatC / a                  5.0                 17.0                 29.0
    >>>
    >>> nh = start.filter(region="*NH")
    >>> nh.head()
    time                                                        2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable             unit     region   model     scenario
    Emissions|CO2|Fossil GtC / yr World|NH idealised idealised                  0.0                  6.0                 12.0
    Emissions|CO2|AFOLU  GtC / yr World|NH idealised idealised                  1.0                  7.0                 13.0
    >>>
    >>> sh = start.filter(region="*SH")
    >>> sh.head()
    time                                                        2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable             unit     region   model     scenario
    Emissions|CO2|Fossil GtC / yr World|SH idealised idealised                  2.0                  8.0                 14.0
    Emissions|CO2|AFOLU  GtC / yr World|SH idealised idealised                  3.0                  9.0                 15.0
    >>>
    >>> world = nh.add(sh, op_cols={"region": "World"})
    >>> world.head()
    time                                                        2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    model     scenario  region variable             unit
    idealised idealised World  Emissions|CO2|Fossil gigatC / a                  2.0                 14.0                 26.0
                               Emissions|CO2|AFOLU  gigatC / a                  4.0                 16.0                 28.0
    """
    out = _perform_op(
        prep_for_op(self, op_cols, self.meta.columns, **kwargs),
        prep_for_op(other, op_cols, self.meta.columns, **kwargs),
        "add",
        use_pint_units="unit" not in op_cols,
    )

    out = set_op_values(out, op_cols)

    return type(self)(out)


def multiply(self, other, op_cols, **kwargs):
    """
    Multiply values

    Parameters
    ----------
    other : :obj:`ScmRun`
        :obj:`ScmRun` containing data to multiply

    op_cols : dict of str: str
        Dictionary containing the columns to drop before multiplying as the keys and the value those columns should hold in the output as the values. For example, if we have ``op_cols={"variable": "Emissions|CO2 - Emissions|CO2|Fossil"}`` then the multiplication will be performed with an index that uses all columns except the "variable" column and the output will have a "variable" column with the value "Emissions|CO2 - Emissions|CO2|Fossil".

    **kwargs : any
        Passed to :func:`prep_for_op`

    Returns
    -------
    :obj:`ScmRun`
        Product of ``self`` and ``other``, using ``op_cols`` to define the columns which should be dropped before the data is aligned and to define the value of these columns in the output.

    Examples
    --------
    >>> import numpy as np
    >>> from scmdata import ScmRun
    >>>
    >>> IDX = [2010, 2020, 2030]
    >>>
    >>>
    >>> start = ScmRun(
    ...     data=np.arange(18).reshape(3, 6),
    ...     index=IDX,
    ...     columns={
    ...         "variable": [
    ...             "Emissions|CO2|Fossil",
    ...             "Emissions|CO2|AFOLU",
    ...             "Emissions|CO2|Fossil",
    ...             "Emissions|CO2|AFOLU",
    ...             "Cumulative Emissions|CO2",
    ...             "Surface Air Temperature Change",
    ...         ],
    ...         "unit": ["GtC / yr", "GtC / yr", "GtC / yr", "GtC / yr", "GtC", "K"],
    ...         "region": ["World|NH", "World|NH", "World|SH", "World|SH", "World", "World"],
    ...         "model": "idealised",
    ...         "scenario": "idealised",
    ...     },
    ... )
    >>>
    >>> start.head()
    time                                                            2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable                 unit     region   model     scenario
    Emissions|CO2|Fossil     GtC / yr World|NH idealised idealised                  0.0                  6.0                 12.0
    Emissions|CO2|AFOLU      GtC / yr World|NH idealised idealised                  1.0                  7.0                 13.0
    Emissions|CO2|Fossil     GtC / yr World|SH idealised idealised                  2.0                  8.0                 14.0
    Emissions|CO2|AFOLU      GtC / yr World|SH idealised idealised                  3.0                  9.0                 15.0
    Cumulative Emissions|CO2 GtC      World    idealised idealised                  4.0                 10.0                 16.0
    >>> fos = start.filter(variable="*Fossil")
    >>> fos.head()
    time                                                        2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable             unit     region   model     scenario
    Emissions|CO2|Fossil GtC / yr World|NH idealised idealised                  0.0                  6.0                 12.0
                                  World|SH idealised idealised                  2.0                  8.0                 14.0
    >>>
    >>> afolu = start.filter(variable="*AFOLU")
    >>> afolu.head()
    time                                                       2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable            unit     region   model     scenario
    Emissions|CO2|AFOLU GtC / yr World|NH idealised idealised                  1.0                  7.0                 13.0
                                 World|SH idealised idealised                  3.0                  9.0                 15.0
    >>>
    >>> fos_times_afolu = fos.multiply(
    ...     afolu, op_cols={"variable": "Emissions|CO2|Fossil : AFOLU"}
    ... )
    >>> fos_times_afolu.head()
    time                                                                            2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    model     scenario  region   variable                     unit
    idealised idealised World|NH Emissions|CO2|Fossil : AFOLU gigatC ** 2 / a ** 2                  0.0                 42.0                156.0
                        World|SH Emissions|CO2|Fossil : AFOLU gigatC ** 2 / a ** 2                  6.0                 72.0                210.0
    """
    out = _perform_op(
        prep_for_op(self, op_cols, self.meta.columns, **kwargs),
        prep_for_op(other, op_cols, self.meta.columns, **kwargs),
        "multiply",
        use_pint_units="unit" not in op_cols,
    )

    out = set_op_values(out, op_cols)

    return type(self)(out)


def divide(self, other, op_cols, **kwargs):
    """
    Divide values (self / other)

    Parameters
    ----------
    other : :obj:`ScmRun`
        :obj:`ScmRun` containing data to divide

    op_cols : dict of str: str
        Dictionary containing the columns to drop before dividing as the keys and the value those columns should hold in the output as the values. For example, if we have ``op_cols={"variable": "Emissions|CO2 - Emissions|CO2|Fossil"}`` then the division will be performed with an index that uses all columns except the "variable" column and the output will have a "variable" column with the value "Emissions|CO2 - Emissions|CO2|Fossil".

    **kwargs : any
        Passed to :func:`prep_for_op`

    Returns
    -------
    :obj:`ScmRun`
        Quotient of ``self`` and ``other``, using ``op_cols`` to define the columns which should be dropped before the data is aligned and to define the value of these columns in the output.

    Examples
    --------
    >>> import numpy as np
    >>> from scmdata import ScmRun
    >>>
    >>> IDX = [2010, 2020, 2030]
    >>>
    >>>
    >>> start = ScmRun(
    ...     data=np.arange(18).reshape(3, 6),
    ...     index=IDX,
    ...     columns={
    ...         "variable": [
    ...             "Emissions|CO2|Fossil",
    ...             "Emissions|CO2|AFOLU",
    ...             "Emissions|CO2|Fossil",
    ...             "Emissions|CO2|AFOLU",
    ...             "Cumulative Emissions|CO2",
    ...             "Surface Air Temperature Change",
    ...         ],
    ...         "unit": ["GtC / yr", "GtC / yr", "GtC / yr", "GtC / yr", "GtC", "K"],
    ...         "region": ["World|NH", "World|NH", "World|SH", "World|SH", "World", "World"],
    ...         "model": "idealised",
    ...         "scenario": "idealised",
    ...     },
    ... )
    >>>
    >>> start.head()
    time                                                            2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable                 unit     region   model     scenario
    Emissions|CO2|Fossil     GtC / yr World|NH idealised idealised                  0.0                  6.0                 12.0
    Emissions|CO2|AFOLU      GtC / yr World|NH idealised idealised                  1.0                  7.0                 13.0
    Emissions|CO2|Fossil     GtC / yr World|SH idealised idealised                  2.0                  8.0                 14.0
    Emissions|CO2|AFOLU      GtC / yr World|SH idealised idealised                  3.0                  9.0                 15.0
    Cumulative Emissions|CO2 GtC      World    idealised idealised                  4.0                 10.0                 16.0
    >>> fos = start.filter(variable="*Fossil")
    >>> fos.head()
    time                                                        2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable             unit     region   model     scenario
    Emissions|CO2|Fossil GtC / yr World|NH idealised idealised                  0.0                  6.0                 12.0
                                  World|SH idealised idealised                  2.0                  8.0                 14.0
    >>>
    >>> afolu = start.filter(variable="*AFOLU")
    >>> afolu.head()
    time                                                       2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable            unit     region   model     scenario
    Emissions|CO2|AFOLU GtC / yr World|NH idealised idealised                  1.0                  7.0                 13.0
                                 World|SH idealised idealised                  3.0                  9.0                 15.0
    >>>
    >>> fos_afolu_ratio = fos.divide(
    ...     afolu, op_cols={"variable": "Emissions|CO2|Fossil : AFOLU"}
    ... )
    >>> fos_afolu_ratio.head()
    time                                                                     2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    model     scenario  region   variable                     unit
    idealised idealised World|NH Emissions|CO2|Fossil : AFOLU dimensionless             0.000000             0.857143             0.923077
                        World|SH Emissions|CO2|Fossil : AFOLU dimensionless             0.666667             0.888889             0.933333
    """
    out = _perform_op(
        prep_for_op(self, op_cols, self.meta.columns, **kwargs),
        prep_for_op(other, op_cols, self.meta.columns, **kwargs),
        "divide",
        use_pint_units="unit" not in op_cols,
    )

    out = set_op_values(out, op_cols)

    return type(self)(out)


def integrate(self, out_var=None):
    """
    Integrate with respect to time

    Parameters
    ----------
    out_var : str
        If provided, the variable column of the output is set equal to
        ``out_var``. Otherwise, the output variables are equal to the input
        variables, prefixed with "Cumulative " .

    Returns
    -------
    :obj:`scmdata.ScmRun`
        :obj:`scmdata.ScmRun` containing the integral of ``self`` with respect
        to time

    Warns
    -----
    UserWarning
        The data being integrated contains nans. If this happens, the output
        data will also contain nans.
    """
    if not has_scipy:
        raise ImportError("scipy is not installed. Run 'pip install scipy'")

    time_unit = "s"
    times_in_s = self.time_points.values.astype(
        "datetime64[{}]".format(time_unit)
    ).astype("int")

    ts = self.timeseries()
    if ts.isnull().sum().sum() > 0:
        warnings.warn(
            "You are integrating data which contains nans so your result will "
            "also contain nans. Perhaps you want to remove the nans before "
            "performing the integration using a combination of :meth:`filter` "
            "and :meth:`interpolate`?"
        )
    # If required, we can remove the hard-coding of initial, it just requires
    # some thinking about unit handling
    _initial = 0.0
    out = pd.DataFrame(
        scipy.integrate.cumtrapz(y=ts, x=times_in_s, axis=1, initial=_initial)
    )
    out.index = ts.index
    out.columns = ts.columns

    out = type(self)(out)
    out *= unit_registry(time_unit)

    try:
        out_unit = out.get_unique_meta("unit", no_duplicates=True).replace(" ", "")
        out_unit = str(unit_registry(out_unit).to_reduced_units().units)
        out = out.convert_unit(out_unit)

    except ValueError:
        # more than one unit, don't try to clean up
        pass

    if out_var is None:
        out["variable"] = "Cumulative " + out["variable"]
    else:
        out["variable"] = out_var

    return out


def delta_per_delta_time(self, out_var=None):
    """
    Calculate change in timeseries values for each timestep, divided by the size of the timestep

    The output is placed on the middle of each timestep and is one timestep
    shorter than the input.

    Parameters
    ----------
    out_var : str
        If provided, the variable column of the output is set equal to
        ``out_var``. Otherwise, the output variables are equal to the input
        variables, prefixed with "Delta " .

    Returns
    -------
    :obj:`scmdata.ScmRun`
        :obj:`scmdata.ScmRun` containing the changes in values of ``self``,
        normalised by the change in time

    Warns
    -----
    UserWarning
        The data contains nans. If this happens, the output data will also
        contain nans.
    """
    time_unit = "s"
    times_numpy = self.time_points.values.astype("datetime64[{}]".format(time_unit))
    times_deltas_numpy = times_numpy[1:] - times_numpy[:-1]
    times_in_s = times_numpy.astype("int")
    time_deltas_in_s = times_in_s[1:] - times_in_s[:-1]

    ts = self.timeseries()
    if ts.isnull().sum().sum() > 0:
        warnings.warn(
            "You are calculating deltas of data which contains nans so your "
            "result will also contain nans. Perhaps you want to remove the "
            "nans before calculating the deltas using a combination of "
            ":meth:`filter` and :meth:`interpolate`?"
        )

    out = ts.diff(periods=1, axis="columns")
    if not out.iloc[:, 0].isnull().all():  # pragma: no cover
        raise AssertionError(
            "Did pandas change their API? The first timestep is not all nan."
        )

    out = out.iloc[:, 1:] / time_deltas_in_s

    new_times = times_numpy[:-1] + times_deltas_numpy / 2
    out.columns = TimePoints(new_times).to_index()

    out = type(self)(out)
    out /= unit_registry(time_unit)

    try:
        out_unit = out.get_unique_meta("unit", no_duplicates=True).replace(" ", "")
        out_unit = str(unit_registry(out_unit).to_reduced_units().units)
        out = out.convert_unit(out_unit)

    except ValueError:
        # more than one unit, don't try to clean up
        pass

    if out_var is None:
        out["variable"] = "Delta " + out["variable"]
    else:
        out["variable"] = out_var

    return out


def linear_regression(self):
    """
    Calculate linear regression of each timeseries

    Note
    ----
    Times in seconds since 1970-01-01 are used as the x-axis for the
    regressions. Such values can be accessed with
    ``self.time_points.values.astype("datetime64[s]").astype("int")``. This
    decision does not matter for the gradients, but is important for the
    intercept values.

    Returns
    -------
    list of dict[str : Any]
        List of dictionaries. Each dictionary contains the metadata for the
        timeseries plus the gradient (with key ``"gradient"``) and intercept (
        with key ``"intercept"``). The gradient and intercept are stored as
        :obj:`pint.Quantity`.
    """
    _, _, time_unit, gradients, intercepts, meta = _calculate_linear_regression(self)

    out = []
    for row_meta, gradient, intercept in zip(
        meta.to_dict("records"), gradients, intercepts,
    ):
        unit = row_meta.pop("unit")

        row_meta["gradient"] = gradient * unit_registry(
            "{} / {}".format(unit, time_unit)
        )
        row_meta["intercept"] = intercept * unit_registry(unit)

        out.append(row_meta)

    return out


def _convert_linear_regression_raw_to_pdf(raw, key_to_keep, unit):
    pdf_dicts = []
    for r in raw:
        transformed = {
            k: v for k, v in r.items() if k not in ["gradient", "intercept", "unit"]
        }
        if unit is None:
            transformed[key_to_keep] = r[key_to_keep].magnitude
            transformed["unit"] = str(r["gradient"].units)
        else:
            transformed[key_to_keep] = r[key_to_keep].to(unit).magnitude
            transformed["unit"] = unit

        pdf_dicts.append(transformed)

    return pd.DataFrame(pdf_dicts)


def linear_regression_gradient(self, unit=None):
    """
    Calculate gradients of a linear regression of each timeseries

    Parameters
    ----------
    unit : str
        Output unit for gradients. If not supplied, the gradients' units will
        not be converted to a common unit.

    Returns
    -------
    :obj:`pd.DataFrame`
        ``self.meta`` plus a column with the value of the gradient for each
        timeseries. The ``"unit"`` column is updated to show the unit of the
        gradient.
    """
    raw = self.linear_regression()

    return _convert_linear_regression_raw_to_pdf(raw, "gradient", unit)


def linear_regression_intercept(self, unit=None):
    """
    Calculate intercepts of a linear regression of each timeseries

    Note
    ----
    Times in seconds since 1970-01-01 are used as the x-axis for the
    regressions. Such values can be accessed with
    ``self.time_points.values.astype("datetime64[s]").astype("int")``. This
    decision does not matter for the gradients, but is important for the
    intercept values.

    Parameters
    ----------
    unit : str
        Output unit for gradients. If not supplied, the gradients' units will
        not be converted to a common unit.

    Returns
    -------
    :obj:`pd.DataFrame`
        ``self.meta`` plus a column with the value of the gradient for each
        timeseries. The ``"unit"`` column is updated to show the unit of the
        gradient.
    """
    raw = self.linear_regression()

    return _convert_linear_regression_raw_to_pdf(raw, "intercept", unit)


def linear_regression_scmrun(self):
    """
    Re-calculate the timeseries based on a linear regression

    Returns
    -------
    :obj:`scmdata.ScmRun`
        The timeseries, re-calculated based on a linear regression
    """
    (
        times_numpy,
        times_in_s,
        time_unit,
        gradients,
        intercepts,
        meta,
    ) = _calculate_linear_regression(self)

    out_shape = (meta.shape[0], len(times_in_s))
    regression_timeseries = (
        np.broadcast_to(gradients, out_shape[::-1]).T * times_in_s
        + intercepts[:, np.newaxis]
    )

    out = type(self)(
        data=regression_timeseries.T,
        index=times_numpy,
        columns=meta.to_dict(orient="list"),
    )

    return out


def _calculate_linear_regression(in_scmrun):
    time_unit = "s"
    times_numpy = in_scmrun.time_points.values.astype(
        "datetime64[{}]".format(time_unit)
    )
    times_in_s = times_numpy.astype("int")

    ts = in_scmrun.timeseries()
    if ts.isnull().sum().sum() > 0:
        warnings.warn(
            "You are calculating a linear regression of data which contains "
            "nans so your result will also contain nans. Perhaps you want to "
            "remove the nans before calculating the regression using a "
            "combination of :meth:`filter` and :meth:`interpolate`?"
        )

    res = np.polyfit(times_in_s, ts.T, 1)
    gradients = res[0, :]
    intercepts = res[1, :]

    meta = ts.index.to_frame().reset_index(drop=True)

    return times_numpy, times_in_s, time_unit, gradients, intercepts, meta


def inject_ops_methods(cls):
    """
    Inject the operation methods

    Parameters
    ----------
    cls
        Target class
    """
    methods = [
        ("subtract", subtract),
        ("add", add),
        ("multiply", multiply),
        ("divide", divide),
        ("integrate", integrate),
        ("delta_per_delta_time", delta_per_delta_time),
        ("linear_regression", linear_regression),
        ("linear_regression_gradient", linear_regression_gradient),
        ("linear_regression_intercept", linear_regression_intercept),
        ("linear_regression_scmrun", linear_regression_scmrun),
    ]

    for name, f in methods:
        setattr(cls, name, f)
