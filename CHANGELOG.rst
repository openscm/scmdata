Changelog
=========

master
------

v0.4.3
------

- (`#46 <https://github.com/openscm/scmdata/pull/46>`_) Add test of conda installation

v0.4.2
------

- (`#45 <https://github.com/openscm/scmdata/pull/45>`_) Make installing seaborn optional

v0.4.1
------

- (`#44 <https://github.com/openscm/scmdata/pull/44>`_) Add multi-dimensional handling to ``scmdata.netcdf``
- (`#43 <https://github.com/openscm/scmdata/pull/43>`_) Fix minor bugs in netCDF handling and address minor code coverage issues
- (`#41 <https://github.com/openscm/scmdata/pull/41>`_) Update documentation of the data model. Additionally:

    - makes ``.time_points`` atttributes consistently return ``scmdata.time.TimePoints`` instances
    - ensures ``.meta`` is used consistently throughout the code base (removing ``.metadata``)

- (`#33 <https://github.com/openscm/scmdata/pull/33>`_) Remove dependency on `pyam <https://github.com/IAMconsortium/pyam>`_. Plotting is done with `seaborn <https://github.com/mwaskom/seaborn>`_ instead.
- (`#34 <https://github.com/openscm/scmdata/pull/34>`_) Allow the serialization/deserialization of ``scmdata.run.ScmRun`` and ``scmdata.ScmDataFrame`` as netCDF4 files.
- (`#30 <https://github.com/lewisjared/scmdata/pull/30>`_) Swap to using `openscm-units <https://github.com/openscm/openscm-units>`_ for unit handling (hence remove much of the ``scmdata.units`` module)
- (`#21 <https://github.com/openscm/scmdata/pull/21>`_) Added ``scmdata.run.ScmRun`` as a proposed replacement for ``scmdata.dataframe.ScmDataFrame``. This new class provides an identical interface as a ``ScmDataFrame``, but uses a different underlying data structure to the ``ScmDataFrame``. The purpose of ``ScmRun`` is to provide performance improvements when handling large sets of time-series data. Removed support for Python 3.5 until `pyam` dependency is optional
- (`#31 <https://github.com/openscm/scmdata/pull/31>`_) Tidy up repository after changing location

v0.4.0
------

- (`#28 <https://github.com/openscm/scmdata/pull/28>`_) Expose ``scmdata.units.unit_registry``

v0.3.1
------

- (`#25 <https://github.com/openscm/scmdata/pull/25>`_) Make scipy an optional dependency
- (`#24 <https://github.com/openscm/scmdata/pull/24>`_) Fix missing "N2O" unit (see `#14 <https://github.com/openscm/scmdata/pull/14>`_). Also updates test of year to day conversion, it is 365.25 to within 0.01% (but depends on the Pint release).

v0.3.0
------

- (`#20 <https://github.com/openscm/scmdata/pull/20>`_) Add support for python=3.5
- (`#19 <https://github.com/openscm/scmdata/pull/19>`_) Add support for python=3.6

v0.2.2
------

- (`#16 <https://github.com/openscm/scmdata/pull/16>`_) Only rename columns when initialising data if needed

v0.2.1
------

- (`#13 <https://github.com/openscm/scmdata/pull/13>`_) Ensure ``LICENSE`` is included in package
- (`#11 <https://github.com/openscm/scmdata/pull/11>`_) Add SO2F2 unit and update to Pyam v0.3.0
- (`#12 <https://github.com/openscm/scmdata/pull/12>`_) Add ``get_unique_meta`` convenience method
- (`#10 <https://github.com/openscm/scmdata/pull/10>`_) Fix extrapolation bug which prevented any extrapolation from occuring

v0.2.0
------

- (`#9 <https://github.com/openscm/scmdata/pull/9>`_) Add ``time_mean`` method
- (`#8 <https://github.com/openscm/scmdata/pull/8>`_) Add ``make docs`` target

v0.1.2
------

- (`#7 <https://github.com/openscm/scmdata/pull/7>`_) Add notebook tests
- (`#4 <https://github.com/openscm/scmdata/pull/4>`_) Unit conversions for CH4 and N2O contexts now work for compound units (e.g. 'Mt CH4 / yr' to 'Gt C / day')
- (`#6 <https://github.com/openscm/scmdata/pull/6>`_) Add auto-formatting

v0.1.1
------

- (`#5 <https://github.com/openscm/scmdata/pull/5>`_) Add ``scmdata.dataframe.df_append`` to ``__init__.py``

v0.1.0
------

- (`#3 <https://github.com/openscm/scmdata/pull/3>`_) Added documentation for the api and Makefile targets for releasing
- (`#2 <https://github.com/openscm/scmdata/pull/2>`_) Refactored scmdataframe from openclimatedata/openscm@077f9b5 into a standalone package
- (`#1 <https://github.com/openscm/scmdata/pull/1>`_) Add docs folder
