"""
NetCDF4 file operations

Reading and writing Run's to disk as binary
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


def _write_nc(ds, df, dimensions, extras):
    """
    Low level function to write the dimensions, variables and metadata to disk
    """
    all_dims = list(dimensions) + ["time"]

    # Create the dimensions
    ds.createDimension("time", len(df.time_points))
    ds.createVariable(
        "time", "f8", "time",
    )
    ds.variables["time"][:] = df.time_points.values

    dims = {}
    for d in dimensions:
        vals = sorted(df.meta[d].unique())
        if not all([isinstance(v, str) for v in vals]) and np.isnan(vals).any():
            raise AssertionError("nan in dimension: `{}`".format(d))

        ds.createDimension(d, len(vals))
        dtype = type(vals[0])
        ds.createVariable(d, dtype, d)
        for i, v in enumerate(vals):  # Iteration needed for str types
            ds.variables[d][i] = v
        dims[d] = np.asarray(vals)

    var_shape = [len(dims[d]) for d in dimensions] + [len(df.time_points)]

    # Write any extra variables
    for e in extras:
        metadata = df.meta[[e, *dimensions]].drop_duplicates()

        if metadata[dimensions].duplicated().any():
            raise ValueError(
                "metadata for {} is not unique for requested dimensions".format(e)
            )

        type_info = _get_nc_type(metadata[e].dtype)
        ds.createVariable(e, dimensions=dimensions, zlib=True, **type_info)
        ds.variables[e]._is_metadata = 1

        data_to_write = np.zeros(
            [len(dims[d]) for d in dimensions], dtype=metadata[e].dtype
        )
        if "fill_value" in type_info:
            data_to_write.fill(type_info["fill_value"])

        df_values = metadata[e].values
        for i, (_, m) in enumerate(metadata.iterrows()):
            idx = [_get_idx(dims[d], m[d]) for d in dimensions]
            data_to_write[tuple(idx)] = df_values[i]

        ds.variables[e][:] = data_to_write

    for var_df in df.groupby("variable"):
        v = var_df.get_unique_meta("variable", True)
        meta = var_df.meta.copy().drop("variable", axis=1)

        # Check that the varying dimensions are all unique
        if meta[dimensions].duplicated().any():
            raise ValueError(
                "{} dimensions are not unique for variable {}".format(dimensions, v)
            )

        # Check that the other meta are consistent
        var_attrs = {"_is_metadata": 0}
        for d in set(meta.columns) - set(dimensions) - set(extras):
            if len(meta[d].unique()) != 1:
                raise ValueError(
                    "metadata for {} is not unique for variable {}".format(d, v)
                )
            var_attrs[d] = meta[d].unique()[0]

        var_name = _var_to_nc(v)
        ds.createVariable(
            var_name, DEFAULT_FLOAT, all_dims, zlib=True, fill_value=np.nan
        )

        # We need to write in one dimension at a time
        data_to_write = np.zeros(var_shape)
        data_to_write.fill(np.nan)
        df_values = var_df.values
        for i, (_, m) in enumerate(meta.iterrows()):
            idx = [_get_idx(dims[d], m[d]) for d in dimensions]
            idx.append(slice(None))  # time dim
            data_to_write[tuple(idx)] = df_values[i]
        # Write in one call to the nc library
        ds.variables[var_name][:] = data_to_write

        # Set variable metadata
        ds.variables[var_name].setncatts(var_attrs)


def _read_nc(cls, ds):
    dims = {d: ds.variables[d][:] for d in ds.dimensions}
    dims["time"] = dims["time"].astype("datetime64[s]")

    data = []
    columns = defaultdict(list)

    def _read_var(name, var):
        var_data = var[:]
        valid_mask = ~np.isnan(var_data).all(axis=-1)

        var_meta = {"variable": name}
        for v in var.ncattrs():
            if not v.startswith("_"):
                var_meta[v] = var.getncattr(v)

        # Iterate over all combinations of dimensions
        meta_at_coord = np.asarray(
            np.meshgrid(*[dims[d] for d in var.dimensions[:-1]], indexing="ij")
        )
        meta_at_coord = meta_at_coord[0]

        with np.nditer(meta_at_coord, ["refs_ok", "multi_index"], order="F") as it:
            for _ in it:
                if not valid_mask[it.multi_index]:
                    continue
                data.append(var_data[it.multi_index])
                for i, v in enumerate(it.multi_index):
                    dim_name = var.dimensions[i]
                    columns[dim_name].append(dims[dim_name][v])
                for v in var_meta:
                    columns[v].append(var_meta[v])

    extra_cols = []
    for var_name in ds.variables:
        var = ds.variables[var_name]
        if var_name in dims:
            continue

        # Check if metadata column
        if var.getncattr("_is_metadata"):
            extra_cols.append(var_name)
            continue

        name = _nc_to_var(var_name)
        _read_var(name, var)

    df = cls(np.asarray(data).T, columns=columns, index=dims["time"])

    # Parse any extra metadata columns
    # Requires 1 filter per item

    # stupid dataframes and their returning of copies
    from .dataframe import ScmDataFrame

    is_scmdf = isinstance(df, ScmDataFrame)
    if is_scmdf:
        df = df.timeseries().reset_index()

    for col in extra_cols:
        var = ds.variables[col]

        values = var[:]
        meta_at_coord = np.asarray(
            np.meshgrid(*[dims[d] for d in var.dimensions], indexing="ij")
        )
        with np.nditer(meta_at_coord[0], ["refs_ok", "multi_index"], order="F") as it:
            for _ in it:
                meta_vals = {
                    k: v
                    for k, v in zip(
                        var.dimensions, meta_at_coord[(slice(None),) + it.multi_index]
                    )
                }
                if is_scmdf:
                    for c, v in meta_vals.items():
                        df.loc[df[c] == v, col] = values[it.multi_index]

                else:
                    df.filter(**meta_vals)[col] = values[it.multi_index]

        if is_scmdf:
            df.loc[:, col] = df.loc[:, col].astype(values[0].dtype)

    if is_scmdf:
        df = ScmDataFrame(df)

    return df


def run_to_nc(df, fname, dimensions=("region",), extras=()):
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
    """
    if not has_netcdf:
        raise ImportError("netcdf4 is not installed. Run 'pip install netcdf4'")

    dimensions = list(dimensions)
    if "time" in dimensions:
        dimensions.remove("time")

    with nc.Dataset(fname, "w", diskless=True, persist=True) as ds:
        ds.created_at = datetime.utcnow().isoformat()
        ds._scmdata_version = __version__
        _write_nc(ds, df, dimensions, extras)


def nc_to_run(cls, fname):
    """
    Read a netCDF4 file from disk

    Parameters
    ----------
    fname: str
        Filename to read
    """
    if not has_netcdf:
        raise ImportError("netcdf4 is not installed. Run 'pip install netcdf4'")

    with nc.Dataset(fname) as ds:
        try:
            return _read_nc(cls, ds)
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
