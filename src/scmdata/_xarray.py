"""
Interface with `xarray <https://xarray.pydata.org/en/stable/index.html>`_
"""
import numpy as np
import pint.errors
import xarray as xr

from .errors import NonUniqueMetadataError


def to_xarray(self, dimensions=("region",), extras=(), unify_units=True):
    """
    Convert to a :class:`xarray.Dataset`

    Parameters
    ----------
    dimensions : iterable of str
        Dimensions for each variable in the returned dataset. If an "_id" co-ordinate is
         required (see ``extras`` documentation for when "_id" is required) and is not
         included in ``dimensions`` then it will be the last dimension (or second last
         dimension if "time" is also not included in ``dimensions``). If "time" is not
         included in ``dimensions`` it will be the last dimension.

    extras : iterable of str
        Columns in ``self.meta`` from which to create "non-dimension co-ordinates" (see
        `xarray terminology <https://xarray.pydata.org/en/stable/terminology.html>`_
        for more details). These non-dimension co-ordinates store extra information and
        can be mapped to each timeseries found in the data variables of the output
        :class:`xarray.Dataset`. Where possible, these non-dimension co-ordinates will use
        dimension co-ordinates as their own co-ordinates. However, if the metadata in
        ``extras`` is not defined by a single dimension in ``dimensions``, then the
        ``extras`` co-ordinates will have dimensions of "_id". This "_id" co-ordinate
        maps the values in the ``extras`` co-ordinates to each timeseries in the
        serialised dataset. Where "_id" is required, an extra "_id" dimension will
        also be added to ``dimensions``.

    unify_units : bool
        If a given variable has multiple units, should we attempt to unify them?

    Returns
    -------
    :class:`xarray.Dataset`
        Data in self, re-formatted as an :class:`xarray.Dataset`

    Raises
    ------
    ValueError
        If a variable has multiple units and ``unify_units`` is ``False``.

    ValueError
        If a variable has multiple units which are not able to be converted to a common
        unit because they have different base units.
    """
    dimensions = list(dimensions)
    extras = list(extras)

    dimensions_extras_overlap = set(dimensions).intersection(set(extras))
    if dimensions_extras_overlap:
        raise ValueError(
            "dimensions and extras cannot have any overlap. "
            "Current values in both dimensions and extras: {}".format(
                dimensions_extras_overlap
            )
        )

    timeseries_dims = list(set(dimensions) - {"time"} - {"_id"})

    self_unified_units = _unify_scmrun_units(self, unify_units)
    timeseries = _get_timeseries_for_xr_dataset(
        self_unified_units, timeseries_dims, extras
    )
    non_dimension_extra_metadata = _get_other_metdata_for_xr_dataset(
        self_unified_units, dimensions, extras
    )

    if extras:
        ids, ids_dimensions = _get_ids_for_xr_dataset(
            self_unified_units, extras, timeseries_dims
        )
    else:
        ids = None
        ids_dimensions = None

    for_xarray = _get_dataframe_for_xr_dataset(
        timeseries, timeseries_dims, extras, ids, ids_dimensions
    )
    xr_ds = xr.Dataset.from_dataframe(for_xarray)

    if extras:
        xr_ds = _add_extras(xr_ds, ids, ids_dimensions, self_unified_units)

    unit_map = (
        self_unified_units.meta[["variable", "unit"]]
        .drop_duplicates()
        .set_index("variable")["unit"]
    )
    xr_ds = _add_units(xr_ds, unit_map)
    xr_ds = _add_scmdata_metadata(xr_ds, non_dimension_extra_metadata)
    xr_ds = _set_dimensions(xr_ds, dimensions)

    return xr_ds


def _unify_scmrun_units(run, unify_units):
    variable_unit_table = run.meta[["variable", "unit"]].drop_duplicates()
    variable_units = variable_unit_table.set_index("variable")["unit"]

    variable_counts = variable_unit_table["variable"].value_counts()
    more_than_one_unit_variables = variable_counts[variable_counts > 1]
    if not more_than_one_unit_variables.empty:
        if not unify_units:
            error_msg = (
                "The following variables are reported in more than one unit. "
                "Found variable-unit combinations are:\n{}".format(
                    variable_unit_table[
                        variable_unit_table["variable"].isin(
                            more_than_one_unit_variables.index.values
                        )
                    ]
                )
            )

            raise ValueError(error_msg)

        for variable in more_than_one_unit_variables.index:
            out_unit = variable_units[variable].iloc[0]
            try:
                run = run.convert_unit(out_unit, variable=variable)
            except pint.errors.DimensionalityError as exc:
                error_msg = (
                    "Variable `{}` cannot be converted to a common unit. "
                    "Units in the provided dataset: {}.".format(
                        variable, variable_units[variable].values.tolist()
                    )
                )
                raise ValueError(error_msg) from exc

    return run


def _get_timeseries_for_xr_dataset(run, dimensions, extras):
    for d in dimensions:
        vals = sorted(run.meta[d].unique())
        if not all([isinstance(v, str) for v in vals]) and np.isnan(vals).any():
            raise AssertionError("nan in dimension: `{}`".format(d))

    try:
        timeseries = run.timeseries(dimensions + extras + ["variable"])
    except NonUniqueMetadataError as exc:
        error_msg = (
            "dimensions: `{}` and extras: `{}` do not uniquely define the "
            "timeseries, please add extra dimensions and/or extras".format(
                dimensions, extras
            )
        )
        raise ValueError(error_msg) from exc

    timeseries.columns = run.time_points.as_cftime()

    return timeseries


def _get_other_metdata_for_xr_dataset(run, dimensions, extras):
    other_dimensions = list(
        set(run.meta.columns) - set(dimensions) - set(extras) - {"variable", "unit"}
    )
    other_metdata = run.meta[other_dimensions].drop_duplicates()
    if other_metdata.shape[0] > 1 and not other_metdata.empty:
        error_msg = (
            "Other metadata is not unique for dimensions: `{}` and extras: `{}`. "
            "Please add meta columns with more than one value to dimensions or "
            "extras.\nNumber of unique values in each column:\n{}.\n"
            "Existing values in the other metadata:\n{}.".format(
                dimensions,
                extras,
                other_metdata.nunique(),
                other_metdata.drop_duplicates(),
            )
        )
        raise ValueError(error_msg)

    return other_metdata


def _get_ids_for_xr_dataset(run, extras, dimensions):
    # these loops could be very slow with lots of extras and dimensions...
    ids_dimensions = {}
    for extra in extras:
        for col in dimensions:
            if _many_to_one(run.meta, extra, col):
                dim_col = col
                break
        else:
            dim_col = "_id"

        ids_dimensions[extra] = dim_col

    ids = run.meta[extras].drop_duplicates()
    ids["_id"] = range(ids.shape[0])
    ids = ids.set_index(extras)

    return ids, ids_dimensions


def _many_to_one(df, col1, col2):
    """
    Check if there is a many to one mapping between col2 and col1
    """
    # thanks https://stackoverflow.com/a/59091549
    checker = df[[col1, col2]].drop_duplicates()

    max_count = checker.groupby(col2).count().max()[0]
    if max_count < 1:  # pragma: no cover # emergency valve
        raise AssertionError

    return max_count == 1


def _get_dataframe_for_xr_dataset(timeseries, dimensions, extras, ids, ids_dimensions):
    timeseries = timeseries.reset_index()

    add_id_dimension = extras and "_id" in set(ids_dimensions.values())
    if add_id_dimension:
        timeseries = (
            timeseries.set_index(ids.index.names)
            .join(ids)
            .reset_index(drop=True)
            .set_index(dimensions + ["variable", "_id"])
        )
    else:
        timeseries = timeseries.set_index(dimensions + ["variable"])
        if extras:
            timeseries = timeseries.drop(extras, axis="columns")

    timeseries.columns.names = ["time"]

    if (
        len(timeseries.index.unique()) != timeseries.shape[0]
    ):  # pragma: no cover # emergency valve
        # shouldn't be able to get here because any issues should be caught
        # by initial creation of timeseries but just in case
        raise AssertionError("something not unique")

    for_xarray = (
        timeseries.T.stack(dimensions + ["_id"])
        if add_id_dimension
        else timeseries.T.stack(dimensions)
    )

    return for_xarray


def _add_extras(xr_ds, ids, ids_dimensions, run):
    # this loop could also be slow...
    extra_coords = {}
    for extra, id_dimension in ids_dimensions.items():
        if id_dimension in ids:
            ids_extra = ids.reset_index().set_index(id_dimension)
        else:
            ids_extra = (
                run.meta[[extra, id_dimension]]
                .drop_duplicates()
                .set_index(id_dimension)
            )

        extra_coords[extra] = (
            id_dimension,
            ids_extra[extra].loc[xr_ds[id_dimension].values],
        )

    xr_ds = xr_ds.assign_coords(extra_coords)

    return xr_ds


def _add_units(xr_ds, unit_map):
    for data_var in xr_ds.data_vars:
        unit = unit_map[data_var]
        if (
            not isinstance(unit, str) and len(unit) > 1
        ):  # pragma: no cover # emergency valve
            # should have already been caught...
            raise AssertionError(
                "Found multiple units ({}) for {}".format(unit, data_var)
            )

        xr_ds[data_var].attrs["units"] = unit

    return xr_ds


def _add_scmdata_metadata(xr_ds, others):
    for col in others:
        vals = others[col].unique()
        if len(vals) > 1:  # pragma: no cover # emergency valve
            # should have already been caught...
            raise AssertionError("More than one value for meta: {}".format(col))

        xr_ds.attrs["scmdata_metadata_{}".format(col)] = vals[0]

    return xr_ds


def _set_dimensions(xr_ds, dimensions):
    out_dimensions = dimensions
    if "time" not in dimensions:
        out_dimensions += ["time"]

    if "_id" in xr_ds.dims and "_id" not in dimensions:
        out_dimensions += ["_id"]

    return xr_ds.transpose(*out_dimensions)


def inject_xarray_methods(cls):
    """
    Inject the xarray methods

    Parameters
    ----------
    cls
        Target class
    """
    methods = [
        ("to_xarray", to_xarray),
    ]

    for name, f in methods:
        setattr(cls, name, f)
