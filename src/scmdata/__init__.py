"""
scmdata, simple data handling for simple climate model data.

See README and docs for more info.
"""
try:
    from importlib.metadata import version as _version
except ImportError:
    # no recourse if the fallback isn't there either...
    from importlib_metadata import version as _version

try:
    __version__ = _version("openscm_runner")
except Exception:  # pylint: disable=broad-except  # pragma: no cover
    # Local copy, not installed with setuptools
    __version__ = "unknown"

from scmdata.run import ScmRun, run_append  # noqa: F401, E402
