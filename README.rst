scmdata
=======

+-------------------+----------------+--------------+--------+
| Repository health |    |CI CD|     |  |Coverage|  | |Docs| |
+-------------------+----------------+--------------+--------+

+------+------------------+----------------+------------------+
| Pypi |  |PyPI Install|  |     |PyPI|     |  |PyPI Version|  |
+------+------------------+----------------+------------------+

+-------+-----------------+-------------------+-----------------+
| Conda | |conda install| | |conda platforms| | |conda version| |
+-------+-----------------+-------------------+-----------------+

+-----------------+----------------+---------------+-----------+
|   Other info    | |Contributors| | |Last Commit| | |License| |
+-----------------+----------------+---------------+-----------+

.. sec-begin-links

.. |CI CD| image:: https://github.com/openscm/scmdata/workflows/scmdata%20CI-CD/badge.svg
    :target: https://github.com/openscm/scmdata/actions?query=workflow%3A%22scmdata+CI-CD%22
.. |Coverage| image:: https://codecov.io/gh/openscm/scmdata/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/openscm/scmdata
.. |Docs| image:: https://readthedocs.org/projects/scmdata/badge/?version=latest
    :target: https://scmdata.readthedocs.io/en/latest/?badge=latest
.. |PyPI Install| image:: https://github.com/openscm/scmdata/workflows/Test%20PyPI%20install/badge.svg
    :target: https://github.com/openscm/scmdata/actions?query=workflow%3A%22Test+PyPI+install%22
.. |PyPI| image:: https://img.shields.io/pypi/pyversions/scmdata.svg
    :target: https://pypi.org/project/scmdata/
.. |PyPI Version| image:: https://img.shields.io/pypi/v/scmdata.svg
    :target: https://pypi.org/project/scmdata/
.. |conda install| image:: https://github.com/openscm/scmdata/workflows/Test%20conda%20install/badge.svg
    :target: https://github.com/openscm/scmdata/actions?query=workflow%3A%22Test+conda+install%22
.. |conda platforms| image:: https://img.shields.io/conda/pn/conda-forge/scmdata.svg
    :target: https://anaconda.org/conda-forge/scmdata
.. |conda version| image:: https://img.shields.io/conda/vn/conda-forge/scmdata.svg
    :target: https://anaconda.org/conda-forge/scmdata
.. |Contributors| image:: https://img.shields.io/github/contributors/openscm/scmdata.svg
    :target: https://github.com/openscm/scmdata/graphs/contributors
.. |Last Commit| image:: https://img.shields.io/github/last-commit/openscm/scmdata.svg
    :target: https://github.com/openscm/scmdata/commits/master
.. |License| image:: https://img.shields.io/github/license/openscm/scmdata.svg
    :target: https://github.com/openscm/scmdata/blob/master/LICENSE

.. sec-end-links

Brief summary
+++++++++++++

.. sec-begin-long-description
.. sec-begin-index

**scmdata** provides some useful data handling routines for dealing with data related to simple climate models (SCMs aka reduced complexity climate models, RCMs).
In particular, it provides a high-performance way of handling and serialising (including to netCDF) timeseries data along with attached metadata.
**scmdata** was inspired by `pyam <https://github.com/IAMconsortium/pyam>`_ and was originally part of the `openscm <https://github.com/openscm/openscm>`_ package.

.. sec-end-index

.. sec-begin-license

License
-------

**scmdata** is free software under a BSD 3-Clause License, see `LICENSE <https://github.com/openscm/scmdata/blob/master/LICENSE>`_.

.. sec-end-license
.. sec-end-long-description

.. sec-begin-installation

Installation
------------

**scmdata** can be installed with pip

.. code:: bash

    pip install scmdata

If you also want to run the example notebooks install additional
dependencies using

.. code:: bash

    pip install scmdata[notebooks]

OpenSCM-Units can also be installed with conda

.. code:: bash

    conda install -c conda-forge scmdata

.. sec-end-installation

Documentation
-------------

Documentation can be found at our `documentation pages <https://scmdata.readthedocs.io/en/latest/>`_
(we are thankful to `Read the Docs <https://readthedocs.org/>`_ for hosting us).

Contributing
------------

If you'd like to contribute, please make a pull request!
The pull request templates should ensure that you provide all the necessary information.

