SCMData
=======

.. sec-begin-links

+-------------------+----------------+--------------+
| Repository health |    |CI CD|     |  |Coverage|  |
+-------------------+----------------+--------------+

+------+------------------+----------------+------------------+
| Pypi |  |PyPI Install|  |     |PyPI|     |  |PyPI Version|  |
+------+------------------+----------------+------------------+

+-------+-----------------+-------------------+-----------------+
| Conda | |conda install| | |conda platforms| | |conda version| |
+-------+-----------------+-------------------+-----------------+

+-----------------+----------------+---------------+-----------+
|   Other info    | |Contributors| | |Last Commit| | |License| |
+-----------------+----------------+---------------+-----------+

.. |CI CD| image:: https://github.com/openscm/scmdata/workflows/scmdata%20CI-CD/badge.svg
    :target: https://github.com/openscm/scmdata/actions?query=workflow%3A%22scmdata+CI-CD%22
.. |Coverage| image:: https://codecov.io/gh/openscm/scmdata/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/openscm/scmdata
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
.. |Contributors| image:: https://img.shields.io/github/contributors/openscm/scmdata.svg
    :target: https://github.com/openscm/scmdata/graphs/contributors
.. |Last Commit| image:: https://img.shields.io/github/last-commit/openscm/scmdata.svg
    :target: https://github.com/openscm/scmdata/commits/master
.. |License| image:: https://img.shields.io/github/license/openscm/scmdata.svg
    :target: https://github.com/openscm/scmdata/blob/master/LICENSE

.. sec-end-links

.. sec-begin-index

SCMData provides some useful data handling routines for dealing with data pertaining to Simple Climate Models (SCMs).

An ``ScmDataFrame`` provides a subset of the functionality provided by `pyam <https://github.com/IAMconsortium/pyam>`_'s IamDataFrame,
but is adapted to provide better performance for timeseries data. This package was originally part of `openscm <https://github.com/openclimatedata/openscm>`_.

.. sec-end-index

Contributing
------------

If you'd like to contribute, please make a pull request!
The pull request templates should ensure that you provide all the necessary information.

.. sec-begin-license

License
-------

ScmData is free software under a BSD 3-Clause License, see `LICENSE <https://github.com/openscm/license/blob/master/LICENSE>`_.

.. sec-end-license

