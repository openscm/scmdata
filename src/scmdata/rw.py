try:
    import netCDF4 as nc

    has_netcdf = True
except ImportError:
    nc = None
    has_netcdf = False

from datetime import datetime

import numpy as np
from scmdata import __version__


def _var_to_nc(var):
    return (
        var.replace("|", "__")
            .replace(" ", "_")
            .lower()
    )


def _nc_to_var(var):
    return (
        var.replace("__", "|")
            .replace("_", " ")
            .capitalise()
    )


def _get_idx(vals, v):
    assert v in vals
    return np.where(vals == v)[0][0]


def _write_nc(ds, df, dimensions):
    """
    Low level function to write the dimensions, variables and metadata to disk

    Parameters
    ----------
    ds : `nc.Dataset`
    df:  Dataframe
    dimensions : list of str
        Excluding time diment
    """

    all_dims = dimensions + ["time"]

    # Create the dimensions
    ds.createDimension("time", len(df.time_points))
    ds.createVariable("time", "f8", "time", )
    ds.variables["time"][:] = df.time_points

    for d in dimensions:
        vals = sorted(df.meta[d].unique())
        ds.createDimension(d, len(vals))
        dtype = type(vals[0])
        ds.createVariable(d, dtype, d)
        for i, v in enumerate(vals):  # Iteration needed for str types
            ds.variables[d][i] = v

    for v in df.meta["variable"].unique():
        var_df = df.filter(variable=v)

        meta = var_df.meta.copy().drop("variable", axis=1)

        # Check that the varying dimensions are all unique
        for d in dimensions:
            if meta[d].duplicated().any():
                raise ValueError("{} dimension is not unique for variable {}".format(d, v))

        # Check that the other meta are consistent
        var_attrs = {}
        for d in set(meta.columns) - set(dimensions):
            if len(meta[d].unique()) != 1:
                raise ValueError("metadata for {} is not unique for variable {}".format(d, v))
            var_attrs[d] = meta[d].unique()[0]

        var_name = _var_to_nc(v)
        ds.createVariable(var_name, "f8", all_dims, zlib=True, fill_value=np.nan)

        # We need to write in dimension at a time
        for row_id, m in meta.iterrows():
            idx = [_get_idx(ds.variables[d][:], m[d]) for d in dimensions]
            idx.append(slice(None))  # time dim
            ds.variables[var_name][idx] = var_df._data[row_id].values

        # Set variable metadata
        ds.variables[var_name].setncatts(var_attrs)


def df_to_nc(df, fname, dimensions=("region",)):
    """
    Writes a ScmDataFrame to disk as a netCDF4 file

    Each unique variable will be written as a netCDF file.

    Parameters
    ----------
    path: str
        Path to write the file into
    dimensions: iterable of str
        Dimensions to include in the netCDF file. The order of the dimensions in the netCDF file will be the same
        as the order provided.
        The time dimension is always included as the last dimension, even if not provided.
    """
    if not has_netcdf:
        raise ImportError("netcdf4 is not installed. Run 'pip install netcdf4'")

    dimensions = list(dimensions)
    if "time" in dimensions:
        dimensions.remove("time")

    with nc.Dataset(fname, "w") as ds:
        ds.created_at = datetime.utcnow().isoformat()
        ds._scmdata_version = __version__
        _write_nc(ds, df, dimensions)
