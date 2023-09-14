# scmdata

<!---
Can use start-after and end-before directives in docs, see
https://myst-parser.readthedocs.io/en/latest/syntax/organising_content.html#inserting-other-documents-directly-into-the-current-document
-->

<!--- sec-begin-description -->

**scmdata** provides some useful data handling routines for dealing with data related to simple climate models
(SCMs aka reduced complexity climate models, RCMs). In particular, it provides a high-performance way of
handling and serialising (including to netCDF) timeseries data along with attached metadata.

**scmdata** was inspired by [pyam](https://github.com/IAMconsortium/pyam) and was originally part of the
[openscm](https://github.com/openscm/openscm>) package.

[![CI](https://github.com/openscm/scmdata/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/openscm/scmdata/actions/workflows/ci.yaml)
[![Coverage](https://codecov.io/gh/climate-resource/scmdata/branch/main/graph/badge.svg)](https://codecov.io/gh/openscm/scmdata)
[![Docs](https://readthedocs.org/projects/scmdata/badge/?version=latest)](https://scmdata.readthedocs.io)

**PyPI :**
[![PyPI](https://img.shields.io/pypi/v/scmdata.svg)](https://pypi.org/project/scmdata/)
[![PyPI: Supported Python versions](https://img.shields.io/pypi/pyversions/scmdata.svg)](https://pypi.org/project/scmdata/)
[![PyPI install](https://github.com/openscm/scmdata/actions/workflows/install.yaml/badge.svg?branch=main)](https://github.com/openscm/scmdata/actions/workflows/install.yaml)

**Other info :**
[![License](https://img.shields.io/github/license/openscm/scmdata.svg)](https://github.com/openscm/scmdata/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/openscm/scmdata.svg)](https://github.com/openscm/scmdata/commits/main)
[![Contributors](https://img.shields.io/github/contributors/openscm/scmdata.svg)](https://github.com/openscm/scmdata/graphs/contributors)


<!--- sec-end-description -->

Full documentation can be found at:
[scmdata.readthedocs.io](https://scmdata.readthedocs.io/en/latest/).
We recommend reading the docs there because the internal documentation links
don't render correctly on GitHub's viewer.

## Installation

<!--- sec-begin-installation -->

scmdata can be installed with conda or pip:

```bash
pip install scmdata
conda install -c conda-forge scmdata
```

Additional dependencies can be installed using

```bash
# To add plotting dependencies
pip install scmdata[plots]
# To add notebook dependencies
pip install scmdata[notebooks]

# If you are installing with conda, we recommend
# installing the extras by hand because there is no stable
# solution yet (issue here: https://github.com/conda/conda/issues/7502)
```

<!--- sec-end-installation -->

### For developers

<!--- sec-begin-installation-dev -->

For development, we rely on [poetry](https://python-poetry.org) for all our
dependency management. To get started, you will need to make sure that poetry
is installed
([instructions here](https://python-poetry.org/docs/#installing-with-the-official-installer),
we found that pipx and pip worked better to install on a Mac).

For all of work, we use our `Makefile`.
You can read the instructions out and run the commands by hand if you wish,
but we generally discourage this because it can be error prone.
In order to create your environment, run `make virtual-envir˚onment`.

If there are any issues, the messages from the `Makefile` should guide you
through. If not, please raise an issue in the [issue tracker](https://github.com/openscm/scmdata/issues)˚.

For the rest of our developer docs, please see [](development-reference).

[issue_tracker]: https://github.com/openscm/scmdata/issues

<!--- sec-end-installation-dev -->
