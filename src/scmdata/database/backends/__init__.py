from .base import DatabaseBackend
from .netcdf import NetCDFBackend

"""
Loaded backends for ScmDatabase

Additional backends should be based upon :class:`DatabaseBackend`
"""
backend_classes = {"netcdf": NetCDFBackend}