"""
scmdata, simple data handling for simple climate model data
"""
from ._version import get_versions  # isort:skip

__version__ = get_versions()["version"]
del get_versions

from scmdata.dataframe import ScmDataFrame, df_append  # noqa: F401, E402
from scmdata.run import ScmRun, run_append  # noqa: F401, E402
