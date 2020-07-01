"""
Operations for :obj:`ScmRun`
"""
import pandas as pd


def prep_for_op(inp, op_cols):
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
        the normal :meth:`ScmRun.timeseries` output. We do this so that
        we can transition to using
        `Pint's Pandas interface <https://pint.readthedocs.io/en/0.13/pint-pandas.html>`_
        to handle unit conversions automatically in future.
    """
    key_cols = list(op_cols.keys())

    out = inp.timeseries().reset_index(key_cols, drop=True)

    out = out.T
    # pint pandas interface to come
    # unit_level = out.columns.names.index("unit")
    # out = out.pint.quantify(level=unit_level)

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


def _perform_op(base, other, op):
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

        # elif op == "multiply":
        #     out.append(base[col] * other[col])

        # elif op == "divide":
        #     out.append(base[col] / other[col])

        else:  # pragma: no cover
            raise NotImplementedError(op)

    if len(out) == 1:
        out = out[0].to_frame()
    else:
        out = pd.concat(out, axis="columns")

    out.columns.names = base.columns.names

    # when we add pint back index
    # out = out.pint.dequantify()
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
    >>> from scmdata import ScmRun
    >>>
    >>> IDX = [2010, 2020, 2030]
    >>>
    >>>
    >>> start = ScmRun(
    ...     data=np.arange(12).reshape(3, 4),
    ...     index=IDX,
    ...     columns={
    ...         "variable": [
    ...             "Emissions|CO2|Fossil", "Emissions|CO2|AFOLU",
    ...             "Emissions|CO2|Fossil", "Emissions|CO2|AFOLU",
    ...         ],
    ...         "unit": "GtC / yr",
    ...         "region": [
    ...             "World|NH", "World|NH",
    ...             "World|SH", "World|SH",
    ...         ],
    ...         "model": "idealised",
    ...         "scenario": "idealised",
    ...     }
    ... )
    >>>
    >>> start.head()
    time                                                        2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable             unit     region   model     scenario
    Emissions|CO2|Fossil GtC / yr World|NH idealised idealised                  0.0                  4.0                  8.0
    Emissions|CO2|AFOLU  GtC / yr World|NH idealised idealised                  1.0                  5.0                  9.0
    Emissions|CO2|Fossil GtC / yr World|SH idealised idealised                  2.0                  6.0                 10.0
    Emissions|CO2|AFOLU  GtC / yr World|SH idealised idealised                  3.0                  7.0                 11.0
    >>>
    >>> fos = start.filter(variable="*Fossil")
    >>> fos.head()
    time                                                        2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable             unit     region   model     scenario
    Emissions|CO2|Fossil GtC / yr World|NH idealised idealised                  0.0                  4.0                  8.0
                                  World|SH idealised idealised                  2.0                  6.0                 10.0
    >>>
    >>> afolu = start.filter(variable="*AFOLU")
    >>> afolu.head()
    time                                                       2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable            unit     region   model     scenario
    Emissions|CO2|AFOLU GtC / yr World|NH idealised idealised                  1.0                  5.0                  9.0
                                 World|SH idealised idealised                  3.0                  7.0                 11.0
    >>>
    >>> fos_minus_afolu = fos.subtract(afolu, op_cols={"variable": "Emissions|CO2|Fossil - AFOLU"})
    >>> fos_minus_afolu.head()
    time                                                                2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    model     scenario  region   variable                     unit
    idealised idealised World|NH Emissions|CO2|Fossil - AFOLU GtC / yr                 -1.0                 -1.0                 -1.0
                        World|SH Emissions|CO2|Fossil - AFOLU GtC / yr                 -1.0                 -1.0                 -1.0
    >>>
    >>> nh = start.filter(region="*NH")
    >>> nh.head()
    time                                                        2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable             unit     region   model     scenario
    Emissions|CO2|Fossil GtC / yr World|NH idealised idealised                  0.0                  4.0                  8.0
    Emissions|CO2|AFOLU  GtC / yr World|NH idealised idealised                  1.0                  5.0                  9.0
    >>>
    >>> sh = start.filter(region="*SH")
    >>> sh.head()
    time                                                        2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable             unit     region   model     scenario
    Emissions|CO2|Fossil GtC / yr World|SH idealised idealised                  2.0                  6.0                 10.0
    Emissions|CO2|AFOLU  GtC / yr World|SH idealised idealised                  3.0                  7.0                 11.0
    >>>
    >>> nh_minus_sh = nh.subtract(sh, op_cols={"region": "World|NH - SH"})
    >>> nh_minus_sh.head()
    time                                                             2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    model     scenario  region        variable             unit
    idealised idealised World|NH - SH Emissions|CO2|Fossil GtC / yr                 -2.0                 -2.0                 -2.0
                                      Emissions|CO2|AFOLU  GtC / yr                 -2.0                 -2.0                 -2.0
    """
    out = _perform_op(
        prep_for_op(self, op_cols, **kwargs),
        prep_for_op(other, op_cols, **kwargs),
        "subtract",
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
    >>> from scmdata import ScmRun
    >>>
    >>> IDX = [2010, 2020, 2030]
    >>>
    >>>
    >>> start = ScmRun(
    ...     data=np.arange(12).reshape(3, 4),
    ...     index=IDX,
    ...     columns={
    ...         "variable": [
    ...             "Emissions|CO2|Fossil", "Emissions|CO2|AFOLU",
    ...             "Emissions|CO2|Fossil", "Emissions|CO2|AFOLU",
    ...         ],
    ...         "unit": "GtC / yr",
    ...         "region": [
    ...             "World|NH", "World|NH",
    ...             "World|SH", "World|SH",
    ...         ],
    ...         "model": "idealised",
    ...         "scenario": "idealised",
    ...     }
    ... )
    >>>
    >>> start.head()
    time                                                        2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable             unit     region   model     scenario
    Emissions|CO2|Fossil GtC / yr World|NH idealised idealised                  0.0                  4.0                  8.0
    Emissions|CO2|AFOLU  GtC / yr World|NH idealised idealised                  1.0                  5.0                  9.0
    Emissions|CO2|Fossil GtC / yr World|SH idealised idealised                  2.0                  6.0                 10.0
    Emissions|CO2|AFOLU  GtC / yr World|SH idealised idealised                  3.0                  7.0                 11.0
    >>>
    >>> fos = start.filter(variable="*Fossil")
    >>> fos.head()
    time                                                        2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable             unit     region   model     scenario
    Emissions|CO2|Fossil GtC / yr World|NH idealised idealised                  0.0                  4.0                  8.0
                                  World|SH idealised idealised                  2.0                  6.0                 10.0
    >>>
    >>> afolu = start.filter(variable="*AFOLU")
    >>> afolu.head()
    time                                                       2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable            unit     region   model     scenario
    Emissions|CO2|AFOLU GtC / yr World|NH idealised idealised                  1.0                  5.0                  9.0
                                 World|SH idealised idealised                  3.0                  7.0                 11.0
    >>>
    >>> total = fos.add(afolu, op_cols={"variable": "Emissions|CO2"})
    >>> total.head()
    time                                                 2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    model     scenario  region   variable      unit
    idealised idealised World|NH Emissions|CO2 GtC / yr                  1.0                  9.0                 17.0
                        World|SH Emissions|CO2 GtC / yr                  5.0                 13.0                 21.0
    >>>
    >>> nh = start.filter(region="*NH")
    >>> nh.head()
    time                                                        2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable             unit     region   model     scenario
    Emissions|CO2|Fossil GtC / yr World|NH idealised idealised                  0.0                  4.0                  8.0
    Emissions|CO2|AFOLU  GtC / yr World|NH idealised idealised                  1.0                  5.0                  9.0
    >>>
    >>> sh = start.filter(region="*SH")
    >>> sh.head()
    time                                                        2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    variable             unit     region   model     scenario
    Emissions|CO2|Fossil GtC / yr World|SH idealised idealised                  2.0                  6.0                 10.0
    Emissions|CO2|AFOLU  GtC / yr World|SH idealised idealised                  3.0                  7.0                 11.0
    >>>
    >>> world = nh.add(sh, op_cols={"region": "World"})
    >>> world.head()
    time                                                      2010-01-01 00:00:00  2020-01-01 00:00:00  2030-01-01 00:00:00
    model     scenario  region variable             unit
    idealised idealised World  Emissions|CO2|Fossil GtC / yr                  2.0                 10.0                 18.0
                               Emissions|CO2|AFOLU  GtC / yr                  4.0                 12.0                 20.0
    """
    out = _perform_op(
        prep_for_op(self, op_cols, **kwargs),
        prep_for_op(other, op_cols, **kwargs),
        "add",
    )

    out = set_op_values(out, op_cols)

    return type(self)(out)


# # ops below require unit awareness so reserve until pint is added
# def multiply(self, other, op_cols, **kwargs):
#     """
#     Multiply values

#     Parameters
#     ----------
#     other : :obj:`ScmRun`
#         :obj:`ScmRun` containing data to multiply

#     op_cols : dict of str: str
#         Dictionary containing the columns to drop before multiplying as the keys and the value those columns should hold in the output as the values. For example, if we have ``op_cols={"variable": "Emissions|CO2 - Emissions|CO2|Fossil"}`` then the multiplication will be performed with an index that uses all columns except the "variable" column and the output will have a "variable" column with the value "Emissions|CO2 - Emissions|CO2|Fossil".

#     **kwargs : any
#         Passed to :func:`prep_for_op`

#     Returns
#     -------
#     :obj:`ScmRun`
#         Product of ``self`` and ``other``, using ``op_cols`` to define the columns which should be dropped before the data is aligned and to define the value of these columns in the output.
#     """
#     out = _perform_op(
#         prep_for_op(self, op_cols, **kwargs),
#         prep_for_op(other, op_cols, **kwargs),
#         "multiply",
#     )

#     out = set_op_values(out, op_cols)

#     return type(self)(out)


# def divide(self, other, op_cols, **kwargs):
#     """
#     Divide values (self / other)

#     Parameters
#     ----------
#     other : :obj:`ScmRun`
#         :obj:`ScmRun` containing data to divide

#     op_cols : dict of str: str
#         Dictionary containing the columns to drop before dividing as the keys and the value those columns should hold in the output as the values. For example, if we have ``op_cols={"variable": "Emissions|CO2 - Emissions|CO2|Fossil"}`` then the division will be performed with an index that uses all columns except the "variable" column and the output will have a "variable" column with the value "Emissions|CO2 - Emissions|CO2|Fossil".

#     **kwargs : any
#         Passed to :func:`prep_for_op`

#     Returns
#     -------
#     :obj:`ScmRun`
#         Quotient of ``self`` and ``other``, using ``op_cols`` to define the columns which should be dropped before the data is aligned and to define the value of these columns in the output.
#     """
#     out = _perform_op(
#         prep_for_op(self, op_cols, **kwargs),
#         prep_for_op(other, op_cols, **kwargs),
#         "divide",
#     )

#     out = set_op_values(out, op_cols)

#     return type(self)(out)


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
        # ("multiply", multiply),
        # ("divide", divide),
    ]

    for name, f in methods:
        setattr(cls, name, f)
