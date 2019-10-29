from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from scmdata.dataframe import ScmDataFrame, df_append
