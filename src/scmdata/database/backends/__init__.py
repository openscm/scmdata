"""
Database backends are responsible for the fetching and storage of ScmRun objects. All
backends should be based upon :class:`BaseDatabaseBackend`.
"""

from .base import BaseDatabaseBackend  # noqa: F401, E402
from .netcdf import NetCDFDatabaseBackend

"""
Loaded backends for ScmDatabase

Additional backends should be based upon :class:`BaseDatabaseBackend`
"""
backend_classes = {"netcdf": NetCDFDatabaseBackend}
