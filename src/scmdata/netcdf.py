"""
NetCDF4 file operations

Reading and writing :class:`ScmRun <scmdata.run.ScmRun>` to disk as binary
"""
try:
    import netCDF4 as nc

    has_netcdf = True
except ImportError:  # pragma: no cover
    nc = None
    has_netcdf = False

from datetime import datetime
from logging import getLogger

import xarray as xr

from . import __version__

logger = getLogger(__name__)


def _var_to_nc(var):
    return var.replace("|", "__").replace(" ", "_")


def _rename_variables(xr_ds):
    name_mapping = {}
    for data_var in xr_ds.data_vars:
        serialised_name = _var_to_nc(data_var)
        name_mapping[data_var] = serialised_name
        xr_ds[data_var].attrs["long_name"] = data_var

    xr_ds = xr_ds.rename_vars(name_mapping)

    return xr_ds


def _get_xr_dataset_to_write(run, dimensions, extras):
    xr_ds = run.to_xarray(dimensions, extras)
    xr_ds = _rename_variables(xr_ds)

    return xr_ds


def _write_nc(fname, run, dimensions, extras, **kwargs):
    """
    Low level function to write the dimensions, variables and metadata to disk
    """
    xr_ds = _get_xr_dataset_to_write(run, dimensions, extras)

    xr_ds.attrs["created_at"] = datetime.utcnow().isoformat()
    xr_ds.attrs["_scmdata_version"] = __version__

    if run.metadata:
        xr_ds.attrs.update(run.metadata)

    write_kwargs = _update_kwargs_to_match_serialised_variable_names(xr_ds, kwargs)
    xr_ds.to_netcdf(fname, **write_kwargs)


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

    variable_name_map = {k: v.attrs["long_name"] for k, v in loaded.data_vars.items()}
    dataframe.columns.name = "variable"
    dataframe.columns = dataframe.columns.map(variable_name_map)

    dataframe = dataframe.stack("variable").unstack("time").reset_index()

    unit_map = {
        data_var: loaded[data_var].attrs["units"] for data_var in loaded.data_vars
    }
    dataframe["unit"] = dataframe["variable"].map(_var_to_nc).map(unit_map).values

    return dataframe


def _convert_to_cls_and_add_metadata(dataframe, loaded, cls):
    for k in list(loaded.attrs.keys()):
        if k.startswith("scmdata_metadata_"):
            dataframe[k.replace("scmdata_metadata_", "")] = loaded.attrs.pop(k)

    run = cls(dataframe)
    run.metadata.update(loaded.attrs)

    return run


def _update_kwargs_to_match_serialised_variable_names(xr_ds, in_kwargs):
    variable_name_map = {v.attrs["long_name"]: k for k, v in xr_ds.data_vars.items()}

    def _update_kwargs(dict_in):
        dict_out = {}
        for key, value in dict_in.items():

            if isinstance(value, dict):
                new_val = _update_kwargs(value)
            elif value in variable_name_map:
                new_val = variable_name_map[value]
            else:
                new_val = value

            if key in variable_name_map:
                dict_out[variable_name_map[key]] = new_val
            else:
                dict_out[key] = new_val

        return dict_out

    return _update_kwargs(in_kwargs)


def run_to_nc(run, fname, dimensions=("region",), extras=(), **kwargs):
    """
    Write timeseries to disk as a netCDF4 file

    Each unique variable will be written as a variable within the netCDF file.
    Choosing the dimensions and extras such that there are as few empty (or
    nan) values as possible will lead to the best compression on disk.

    Parameters
    ----------
    fname : str
        Path to write the file into

    dimensions : iterable of str
        Dimensions to include in the netCDF file. The time dimension is always included
        (if not provided it will be the last dimension). An additional dimension
        (specifically a co-ordinate in xarray terms), "_id", will be included if
        ``extras`` is provided and any of the metadata in ``extras`` is not uniquely
        defined by ``dimensions``. "_id" maps the timeseries in each variable to
        their relevant metadata.

    extras : iterable of str
        Metadata columns to write as variables in the netCDF file (specifically as
        "non-dimension co-ordinates" in xarray terms, see `xarray terminology
        <https://xarray.pydata.org/en/stable/terminology.html>`_ for more details).
        Where possible, these non-dimension co-ordinates will use dimension co-ordinates
        as their own co-ordinates. However, if the metadata in ``extras`` is not defined
        by a single dimension in ``dimensions``, then the ``extras`` co-ordinates will
        have dimensions of "_id". This "_id" co-ordinate maps the values in the
        ``extras`` co-ordinates to each timeseries in the serialised dataset. Where
        "_id" is required, an extra "_id" dimension will also be added to
        ``dimensions``.

    kwargs
        Passed through to :meth:`xarray.Dataset.to_netcdf`

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

    _write_nc(fname, run, dimensions, extras, **kwargs)


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
