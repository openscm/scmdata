"""
NetCDF4 file operations

Reading and writing :obj:`ScmRun` to disk as binary
"""
try:
    import netCDF4 as nc

    has_netcdf = True
except ImportError:  # pragma: no cover
    nc = None
    has_netcdf = False

from datetime import datetime
from logging import getLogger

import numpy as np
import xarray as xr

from . import __version__
from .errors import NonUniqueMetadataError

logger = getLogger(__name__)


"""
Default to writing float data as 8 byte floats
"""
DEFAULT_FLOAT = "f8"


def _var_to_nc(var):
    return var.replace("|", "_pipe_").replace(" ", "_space_")


def _nc_to_var(var):
    return var.replace("_pipe_", "|").replace("_space_", " ")


def _write_nc(fname, run, dimensions, extras):
    """
    Low level function to write the dimensions, variables and metadata to disk
    """
    xr_ds = _get_xr_dataset(run, dimensions, extras)

    xr_ds.attrs["created_at"] = datetime.utcnow().isoformat()
    xr_ds.attrs["_scmdata_version"] = __version__

    if run.metadata:
        xr_ds.attrs.update(run.metadata)

    xr_ds.to_netcdf(fname)


def _get_xr_dataset(run, dimensions, extras):
    timeseries = _get_timeseries_for_xr_dataset(run, dimensions, extras)
    non_dimension_extra_metadata = _get_other_metdata_for_xr_dataset(
        run, dimensions, extras
    )

    if extras:
        ids, ids_dimensions = _get_ids_for_xr_dataset(run, extras, dimensions)
    else:
        ids = None
        ids_dimensions = None

    for_xarray = _get_dataframe_for_xr_dataset(
        timeseries, dimensions, extras, ids, ids_dimensions
    )
    xr_ds = xr.Dataset.from_dataframe(for_xarray)

    if extras:
        xr_ds = _add_extras(xr_ds, ids, ids_dimensions, run)

    unit_map = (
        run.meta[["variable", "unit"]].drop_duplicates().set_index("variable")["unit"]
    )
    xr_ds = _add_units(xr_ds, unit_map)
    xr_ds = _add_scmdata_metadata(xr_ds, non_dimension_extra_metadata)

    return xr_ds


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
        xr_ds[data_var].attrs["units"] = unit

    xr_ds = xr_ds.rename_vars({v: _var_to_nc(v) for v in xr_ds.data_vars})

    return xr_ds


def _add_scmdata_metadata(xr_ds, others):
    for col in others:
        vals = others[col].unique()
        if len(vals) > 1:  # pragma: no cover # emergency valve
            # should have already been caught...
            raise AssertionError("More than one value for meta: {}".format(col))

        xr_ds.attrs["_scmdata_metadata_{}".format(col)] = vals[0]

    return xr_ds


def _read_nc(cls, fname):
    loaded = xr.load_dataset(fname)
    dataframe = loaded.to_dataframe()

    dataframe = _reshape_to_scmrun_dataframe(dataframe, loaded)
    run = _convert_to_cls_and_add_metadata(dataframe, loaded, cls)

    return run


def _reshape_to_scmrun_dataframe(dataframe, loaded):
    index_cols = list(set(dataframe.columns) - set(loaded.data_vars))
    dataframe = dataframe.set_index(index_cols, append=True)
    if "_id" in dataframe.index.names:
        dataframe = dataframe.reset_index("_id", drop=True)

    dataframe.columns.name = "variable"
    dataframe.columns = dataframe.columns.map(_nc_to_var)

    dataframe = dataframe.stack("variable").unstack("time").reset_index()

    unit_map = {
        data_var: loaded[data_var].attrs["units"] for data_var in loaded.data_vars
    }
    dataframe["unit"] = dataframe["variable"].map(_var_to_nc).map(unit_map).values

    return dataframe


def _convert_to_cls_and_add_metadata(dataframe, loaded, cls):
    for k in list(loaded.attrs.keys()):
        if k.startswith("_scmdata_metadata_"):
            dataframe[k.replace("_scmdata_metadata_", "")] = loaded.attrs.pop(k)

    run = cls(dataframe)
    run.metadata.update(loaded.attrs)

    return run


def run_to_nc(run, fname, dimensions=("region",), extras=()):
    """
    Write timeseries to disk as a netCDF4 file

    Each unique variable will be written as a variable within the netCDF file.
    Choosing the dimensions and extras such that there are as few empty (or
    nan) values as possible will lead to the best compression on disk.

    Parameters
    ----------
    fname: str
        Path to write the file into

    dimensions: iterable of str
        Dimensions to include in the netCDF file. The time dimension is always included, even if not provided. An additional co-ordinate, "_id", will be included if ``extras`` is provided and any of the metadata in ``extras`` is not uniquely defined by ``dimensions``. "_id" maps the timeseries in each variable to their relevant metadata.

    extras : iterable of str
        Metadata columns to write as co-ordinates in the netCDF file. Where possible, the metadata in ``dimensions`` will be the co-ordinates of these co-ordinates. However, if the metadata in ``extras`` is not defined by a single dimension in ``dimensions``, then the ``extras`` co-ordinates will have co-ordinates of "_id", which maps the metadata to each timeseries in the serialised dataset.

    See Also
    --------
    :meth:`scmdata.run.ScmRun.to_nc`
    """
    if not has_netcdf:
        raise ImportError("netcdf4 is not installed. Run 'pip install netcdf4'")

    dimensions = list(dimensions)
    extras = list(extras)

    if "time" in dimensions:
        dimensions.remove("time")

    if "variable" in dimensions:
        dimensions.remove("variable")

    _write_nc(fname, run, dimensions, extras)


def nc_to_run(cls, fname):
    """
    Read a netCDF4 file from disk

    Parameters
    ----------
    fname: str
        Filename to read

    See Also
    --------
    :meth:`scmdata.run.ScmRun.from_nc`
    """
    if not has_netcdf:
        raise ImportError("netcdf4 is not installed. Run 'pip install netcdf4'")

    try:
        return _read_nc(cls, fname)
    except Exception:
        logger.exception("Failed reading netcdf file: {}".format(fname))
        raise


def inject_nc_methods(cls):
    """
    Add the to/from nc methods to a class

    Parameters
    ----------
    cls
        Class to add methods to
    """
    name = "to_nc"
    func = run_to_nc
    func.__name__ = name
    func.__doc__ = func.__doc__
    setattr(cls, name, func)

    name = "from_nc"
    func = classmethod(nc_to_run)
    func.__name__ = name
    func.__doc__ = func.__doc__
    setattr(cls, name, func)
