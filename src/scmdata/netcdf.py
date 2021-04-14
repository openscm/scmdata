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

from collections import defaultdict
from datetime import datetime
from logging import getLogger

import numpy as np
import xarray as xr
from xarray.coding.times import decode_cf_datetime, encode_cf_datetime

from . import __version__

logger = getLogger(__name__)

_TO_NC_DOCSTRING = """\
Write data to disk as a netCDF4 file

Parameters
----------
path: str
    Path to write the file into

dimensions: iterable of str
    Dimensions to include in the netCDF file. The order of the dimensions in the netCDF file will be the same
    as the order provided.
    The time dimension is always included as the last dimension, even if not provided.
"""


_FROM_NC_DOCSTRING = """\
Read netCDF4 file from disk

Parameters
----------
path: str
    Path to write the file into

"""

"""
Default to writing float data as 8 byte floats
"""
DEFAULT_FLOAT = "f8"


def _var_to_nc(var):
    return var.replace("|", "__").replace(" ", "_")


def _nc_to_var(var):
    return var.replace("__", "|").replace("_", " ")


def _get_idx(vals, v):
    if v not in vals:
        raise AssertionError("{} is not in {}".format(v, vals))

    return np.where(vals == v)[0][0]


def _get_nc_type(np_type):
    if np_type == int:
        return {
            "datatype": "i8",
        }
    elif np_type == float:
        return {"datatype": DEFAULT_FLOAT, "fill_value": np.nan}

    return {"datatype": str, "fill_value": None}


def _create_time_variable(ds, run):
    """
    Create a CF-compliant time variable
    Note that the CF dictates the use of units, rather than unit which we use else where
    """
    ds.createDimension("time", run.shape[1])
    ds.createVariable(
        "time", "i8", "time",
    )

    num, units, calendar = encode_cf_datetime(run.time_points.as_cftime())
    ds.variables["time"][:] = num
    ds.variables["time"].setncatts({"calendar": calendar, "units": units})


def _read_time_variable(time_var):
    # If times use the f8 datatype, convert to datetime64[s]
    if time_var.dtype == np.dtype("f8"):
        return time_var[:].astype("datetime64[s]")
    else:
        # Use CF-compliant time handling
        attrs = time_var.ncattrs()
        units = time_var.units if "units" in attrs else None
        calendar = time_var.calendar if "calendar" in attrs else None

        return decode_cf_datetime(time_var[:], units, calendar)


def _write_nc(fname, run, dimensions, extras):
    """
    Low level function to write the dimensions, variables and metadata to disk
    """
    unit_name = "unit"

    id_dimensions = list(
        set(run.meta.columns) - set(dimensions) - {"variable", "unit"}
    )

    tmp = run.timeseries(dimensions + id_dimensions + ["variable"])
    tmp.columns = run.time_points.as_cftime()

    ids = run.meta[id_dimensions].drop_duplicates()
    ids["_id"] = range(ids.shape[0])
    ids = ids.set_index(id_dimensions)

    unit_table = (
        run.meta[["variable", unit_name]]
        .drop_duplicates()
        .set_index("variable")["unit"]
    )

    joint = tmp.reset_index().set_index(id_dimensions).join(ids)
    joint = joint.reset_index(drop=True).set_index(dimensions + ["variable", "_id"])
    joint.columns.names = ["time"]
    assert (
        len(joint.index.unique()) == joint.shape[0]
    ), "something not unique (also caught by initial call to timeseries so this is just another check)..."

    for_xarray = joint.T.stack(dimensions + ["_id"])

    xr_tmp = xr.Dataset.from_dataframe(for_xarray)
    ids_tmp = ids.reset_index().set_index("_id")

    extras = {}
    for c in ids_tmp:
        extras[c] = ("_id", ids_tmp[c].loc[xr_tmp["_id"].values])

    xr_tmp = xr_tmp.assign_coords(extras)

    for data_var in xr_tmp.data_vars:
        unit = unit_table[data_var]
        xr_tmp[data_var].attrs["units"] = unit

    xr_tmp.attrs["created_at"] = datetime.utcnow().isoformat()
    xr_tmp.attrs["_scmdata_version"] = __version__
    if hasattr(run, "metadata"):
        xr_tmp.attrs.update(run.metadata)

    xr_tmp.to_netcdf(fname)


def _read_nc(cls, fname):
    loaded = xr.load_dataset(fname)

    df = loaded.to_dataframe()  # .unstack("region")
    index_cols = list(set(df.columns) - set(loaded.data_vars))
    df = df.set_index(index_cols, append=True).reset_index("_id", drop=True)
    df.columns.name = "variable"
    df = df.unstack("time").stack("variable")
    df.columns = df.columns.astype(object)
    df = df.reset_index()
    unit_map = {data_var: loaded[data_var].attrs["units"] for data_var in loaded.data_vars}
    df["unit"] = df["variable"].map(unit_map).values

    run = cls(df)

    run.metadata.update(loaded.attrs)

    return run


def run_to_nc(run, fname, dimensions=("region",), extras=()):
    """
    Write timeseries to disk as a netCDF4 file

    Each unique variable will be written as a netCDF file.

    Parameters
    ----------
    fname: str
        Path to write the file into

    dimensions: iterable of str
        Dimensions to include in the netCDF file. The order of the dimensions
        in the netCDF file will be the same as the order provided. The time
        dimension is always included as the last dimension, even if not provided.

    extras : iterable of tuples or str
        Metadata attributes to write as variables in the netCDF file.

    See Also
    --------
    :meth:`scmdata.run.ScmRun.to_nc`
    """
    if not has_netcdf:
        raise ImportError("netcdf4 is not installed. Run 'pip install netcdf4'")

    dimensions = list(dimensions)
    if "time" in dimensions:
        dimensions.remove("time")
    if "variable" in dimensions:
        dimensions.remove("variable")

    _write_nc(fname, run, dimensions, extras)
    # with nc.Dataset(fname, "w", diskless=True, persist=True) as ds:
    #     ds.created_at = datetime.utcnow().isoformat()
    #     ds._scmdata_version = __version__

    #     if hasattr(run, "metadata"):
    #         ds.setncatts(run.metadata)

    #     _write_nc(ds, run, dimensions, extras)


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

    # with nc.Dataset(fname) as ds:
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
