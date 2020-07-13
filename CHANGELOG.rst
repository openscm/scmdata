Changelog
=========

master
------

v0.6.1
------

- (`#74 <https://github.com/openscm/scmdata/pull/74>`_) Update handling of unit conversion context during unit conversions
- (`#73 <https://github.com/openscm/scmdata/pull/73>`_) Only reindex timeseries when dealing with different time points

v0.5.2
------

- (`#65 <https://github.com/openscm/scmdata/pull/65>`_) Use pint for ops, making them automatically unit aware
- (`#71 <https://github.com/openscm/scmdata/pull/71>`_) Start adding arithmetic support via ``scmdata.ops``. So far only add and subtract are supported.
- (`#70 <https://github.com/openscm/scmdata/pull/70>`_) Automatically set y-axis label to units if it makes sense in :obj:`ScmRun`'s :meth:`lineplot` method

v0.5.1
------

- (`#68 <https://github.com/openscm/scmdata/pull/68>`_) Rename :func`scmdata.run.df_append` to :func`scmdata.run.run_append`. :func`scmdata.run.df_append` deprecated and will be removed in v0.6.0
- (`#67 <https://github.com/openscm/scmdata/pull/67>`_) Update the documentation for :meth`ScmRun.append`
- (`#66 <https://github.com/openscm/scmdata/pull/66>`_) Raise ValueError if index/columns arguments are not provided when instantiating a :class`ScmRun` object with a numpy array. Add ``lowercase_cols`` argument to coerce the column names in CSV files to lowercase

v0.5.0
------

- (`#64 <https://github.com/openscm/scmdata/pull/64>`_) Remove spurious warning from :obj:`ScmRun`'s :meth:`filter` method
- (`#63 <https://github.com/openscm/scmdata/pull/63>`_) Remove :meth:`set_meta` from :class:`ScmRun` in preference for using the :meth:`__setitem__` method
- (`#62 <https://github.com/openscm/scmdata/pull/62>`_) Fix interpolation when the data contains nan values
- (`#61 <https://github.com/openscm/scmdata/pull/61>`_) Hotfix filters to also include caret ("^") in pseudo-regexp syntax. Also adds :meth:`empty` property to :obj:`ScmRun`
- (`#59 <https://github.com/openscm/scmdata/pull/59>`_) Deprecate :class:`ScmDataFrame`. To be removed in v0.6.0
- (`#58 <https://github.com/openscm/scmdata/pull/58>`_) Use ``cftime`` datetimes when appending :class:`ScmRun` objects to avoid OutOfBounds errors when datetimes span many centuries
- (`#55 <https://github.com/openscm/scmdata/pull/55>`_) Add ``time_axis`` keyword argument to ``ScmRun.timeseries``, ``ScmRun.long_data`` and ``ScmRun.lineplot`` to give greater control of the time axis when retrieving data
- (`#54 <https://github.com/openscm/scmdata/pull/54>`_) Add :meth:`drop_meta` to :class:`ScmRun` for dropping metadata columns
- (`#53 <https://github.com/openscm/scmdata/pull/53>`_) Don't convert case of variable names written to file. No longer convert case of serialized dataframes
- (`#51 <https://github.com/openscm/scmdata/pull/51>`_) Refactor :meth:`relative_to_ref_period_mean` so that it returns an instance of the input data type (rather than a :obj:`pd.DataFrame`) and puts the reference period in separate meta columns rather than mangling the variable name.
- (`#47 <https://github.com/openscm/scmdata/pull/47>`_) Update README and ``setup.py`` to make it easier for new users

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
