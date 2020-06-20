def _prep_for_op(inp, op_cols):
    key_cols = list(op_cols.keys())

    return inp.timeseries().reset_index(key_cols, drop=True)


def _set_op_values(inp, op_cols):
    for k, v in op_cols.items():
        inp[k] = v

    return inp


def subtract(self, other, op_cols):
    """
    Subtract values

    Parameters
    ----------
    other : :obj:`ScmRun`
        :obj:`ScmRun` containing data to subtract

    op_cols : dict of str: str
        Dictionary containing the columns to drop before subtracting as the keys and the value those columns should hold in the output as the values. For example, if we have ``op_cols={"variable": "Emissions|CO2 - Emissions|CO2|Fossil"}`` then the subtraction will be performed with an index that uses all columns except the "variable" column and the output will have a "variable" column with the value "Emissions|CO2 - Emissions|CO2|Fossil".

    Returns
    -------
    :obj:`ScmRun`
        Difference between ``self`` and ``other``, using ``op_cols`` to define the columns which should be dropped before the data is aligned and to define the value of these columns in the output.
    """
    out = _prep_for_op(self, op_cols) - _prep_for_op(other, op_cols)

    out = _set_op_values(out, op_cols)

    return type(self)(out)


def add(self, other, op_cols):
    """
    Add values

    Parameters
    ----------
    other : :obj:`ScmRun`
        :obj:`ScmRun` containing data to add

    op_cols : dict of str: str
        Dictionary containing the columns to drop before adding as the keys and the value those columns should hold in the output as the values. For example, if we have ``op_cols={"variable": "Emissions|CO2 - Emissions|CO2|Fossil"}`` then the addition will be performed with an index that uses all columns except the "variable" column and the output will have a "variable" column with the value "Emissions|CO2 - Emissions|CO2|Fossil".

    Returns
    -------
    :obj:`ScmRun`
        Sum of ``self`` and ``other``, using ``op_cols`` to define the columns which should be dropped before the data is aligned and to define the value of these columns in the output.
    """
    out = _prep_for_op(self, op_cols) + _prep_for_op(other, op_cols)

    out = _set_op_values(out, op_cols)

    return type(self)(out)


def multiply(self, other, op_cols):
    """
    Multiply values

    Parameters
    ----------
    other : :obj:`ScmRun`
        :obj:`ScmRun` containing data to multiply

    op_cols : dict of str: str
        Dictionary containing the columns to drop before multiplying as the keys and the value those columns should hold in the output as the values. For example, if we have ``op_cols={"variable": "Emissions|CO2 - Emissions|CO2|Fossil"}`` then the multiplication will be performed with an index that uses all columns except the "variable" column and the output will have a "variable" column with the value "Emissions|CO2 - Emissions|CO2|Fossil".

    Returns
    -------
    :obj:`ScmRun`
        Product of ``self`` and ``other``, using ``op_cols`` to define the columns which should be dropped before the data is aligned and to define the value of these columns in the output.
    """
    out = _prep_for_op(self, op_cols) * _prep_for_op(other, op_cols)

    out = _set_op_values(out, op_cols)

    return type(self)(out)


def divide(self, other, op_cols):
    """
    Divide values (self / other)

    Parameters
    ----------
    other : :obj:`ScmRun`
        :obj:`ScmRun` containing data to divide

    op_cols : dict of str: str
        Dictionary containing the columns to drop before dividing as the keys and the value those columns should hold in the output as the values. For example, if we have ``op_cols={"variable": "Emissions|CO2 - Emissions|CO2|Fossil"}`` then the division will be performed with an index that uses all columns except the "variable" column and the output will have a "variable" column with the value "Emissions|CO2 - Emissions|CO2|Fossil".

    Returns
    -------
    :obj:`ScmRun`
        Quotient of ``self`` and ``other``, using ``op_cols`` to define the columns which should be dropped before the data is aligned and to define the value of these columns in the output.
    """
    out = _prep_for_op(self, op_cols) / _prep_for_op(other, op_cols)

    out = _set_op_values(out, op_cols)

    return type(self)(out)


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
    ]

    for name, f in methods:
        setattr(cls, name, f)
