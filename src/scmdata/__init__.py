"""
scmdata, simple data handling for simple climate model data
"""
from ._version import get_versions  # isort:skip

__version__ = get_versions()["version"]
del get_versions

UNITS_COL = "unit"
"""
Column which contains units

This defaults to "unit".
"""

REQUIRED_COLS = ("model", "scenario", "region", "variable", UNITS_COL)
"""
Minimum metadata columns required by an ScmRun.

If an application requires a different set of required metadata, this
can be specified by overriding :attr:`required_cols` on a custom class
inheriting :class:`scmdata.run.BaseScmRun`. Note that at a minimum,
("variable", UNITS_COL) columns are required.
"""


from scmdata.run import ScmRun, run_append  # noqa: F401, E402
