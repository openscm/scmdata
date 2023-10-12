# Changelog

Versions follow [Semantic Versioning](https://semver.org/) (`<major>.<minor>.<patch>`).

Backward incompatible (breaking) changes will only be introduced in major versions
with advance notice in the **Deprecations** section of releases.


<!--
You should *NOT* be adding new changelog entries to this file, this
file is managed by towncrier. See changelog/README.md.

You *may* edit previous changelogs to fix problems like typo corrections or such.
To add a new changelog entry, please see
https://pip.pypa.io/en/latest/development/contributing/#news-entries,
noting that we use the `changelog` directory instead of news, md instead
of rst and use slightly different categories.
-->

<!-- towncrier release notes start -->

## v0.15.2

- ([#257](https://github.com/openscm/scmdata/pull/257)) Updated to support the latest version of {mod}`notebook`
- ([#252](https://github.com/openscm/scmdata/pull/252)) Add ``py.typed`` file do downstream packages can use
    the provided type-hints. Improved the coverage of the type-hints in ``run.py``
- ([#255](https://github.com/openscm/scmdata/pull/255)) Unpin upper limit of the version of numpy
- ([#248](https://github.com/openscm/scmdata/pull/248)) Correctly filter the time index of empty ScmRuns. Resolves #245
- ([#247](https://github.com/openscm/scmdata/pull/247)) Better performance for ScmRun.__setitem__

## v0.15.1

-   ([#239](https://github.com/openscm/scmdata/pull/239)) Move notebooks into the documentation and an update
    of the documentation configuration
-   ([#238](https://github.com/openscm/scmdata/pull/238)) Support {func}`scmdata.ScmRun` reading and writing files using `pathlib.Path` objects.
-   ([#232](https://github.com/openscm/scmdata/pull/232)) Update inplace
    operations to always return a result (closes
    [#230](https://github.com/openscm/scmdata/issues/230)). Removes
    support for pandas==1.0.5

## v0.15.0

-   ([#223](https://github.com/openscm/scmdata/pull/223)) Loosen the
    pandas requirement to cover pandas\>=1.4.3. Also support officially
    support Python 3.10 and 3.11
-   ([#222](https://github.com/openscm/scmdata/pull/222)) Decrease the
    minimum number of time points for interpolation to 2
-   ([#221](https://github.com/openscm/scmdata/pull/221)) Add option to
    {func}`scmdata.ScmRun.interpolate` to allow
    for interpolation which ignores leap-years. This also fixes a bug
    where {func}`scmdata.ScmRun.interpolate`
    converts integer values into unix time. This functionality isn't
    consistent with the behaviour of the TimePoints class where integers
    are converted into years.
-   ([#218](https://github.com/openscm/scmdata/pull/218)) Replaced internal calls to
    {func}`scmdata.groupby.RunGroupby.map` with {func}`scmdata.groupby.RunGroupby.apply`
-   ([#210](https://github.com/openscm/scmdata/pull/210)) Update github
    actions to avoid the use of a deprecated workflow command

## v0.14.2

-   ([#209](https://github.com/openscm/scmdata/pull/209)) Lazy import plotting modules to speed up startup time
-   ([#208](https://github.com/openscm/scmdata/pull/208)) Ensure that all unit operations in {mod}`scmdata` use
    {attr}`scmdata.units.UNIT_REGISTRY`. This now defaults to {attr}`openscm_units.unit_registry` instead of
    unique unit registry for {mod}`scmdata`.
-   ([#202](https://github.com/openscm/scmdata/pull/202)) Add {func}`scmdata.run.ScmRun.set_meta` to
    enable setting of metadata for a subset of timeseries
-   ([#194](https://github.com/openscm/scmdata/pull/194)) Deprecate {func}`scmdata.groupby.RunGroupBy.map` in
    preference to {meth}`scmdata.groupby.RunGroupBy.apply` which is identical in functionality. Add
    {class}`scmdata.ScmRun.apply` for applying a function to each timeseries
-   ([#195](https://github.com/openscm/scmdata/pull/195)) Refactor {mod}`scmdata.database` to a package. The
    database backends have been moved to {mod}`scmdata.database.backends`.
-   ([#197](https://github.com/openscm/scmdata/pull/197)) Workaround
    [regression in Panda\'s](https://github.com/pandas-dev/pandas/issues/47071) handling of xarray's {class}`xr.CFTimeIndex`
-   ([#193](https://github.com/openscm/scmdata/pull/193)) Pin the version of black used for code formatting to
    ensure consistency

## v0.14.1

-   ([#192](https://github.com/openscm/scmdata/pull/192)) Bugfix for the versioning of the package
-   ([#191](https://github.com/openscm/scmdata/pull/191)) Add check of PyPI distribution to CI

## v0.14.0

-   ([#190](https://github.com/openscm/scmdata/pull/190)) Add special case for extrapolating timeseries
    containing a single timestep using`constant` extrapolation. Moved
    {class}`scmdata.errors.InsufficientDataError` from {mod}`scmdata.time` to {mod}`scmdata.errors`
-   ([#186](https://github.com/openscm/scmdata/pull/186) and
    [#187](https://github.com/openscm/scmdata/pull/187)) Fix the handling of non-alphanumeric characters in
    filenames on Windows for {class}`scmdata.database.ScmDatabase`. `*` values are no longer included in
    {class}`scmdata.database.ScmDatabase` filenames
-   ([#186](https://github.com/openscm/scmdata/pull/186) Move to `pyproject.toml` for setup etc.

## v0.13.2

-   ([#185](https://github.com/openscm/scmdata/pull/185)) Allow {class}`scmdata.run.ScmRun` to read remote
    files by providing a URL to the constructor
-   ([#183](https://github.com/openscm/scmdata/pull/183)) Deprecate {func}`scmdata.ops.integrate`, replacing
    with to {func}`scmdata.ops.cumsum` and {func}`scmdata.ops.cumtrapz`
-   ([#184](https://github.com/openscm/scmdata/pull/184)) Add {func}`scmdata.run.ScmRun.round`
-   ([#182](https://github.com/openscm/scmdata/pull/182)) Updated incorrect `conda` install instructions

## v0.13.1

-   ([#181](https://github.com/openscm/scmdata/pull/181)) Allow the initialisation of empty
    {class}`scmdata.ScmRun` objects
-   ([#180](https://github.com/openscm/scmdata/pull/180)) Add
    {func}`scmdata.processing.calculate_crossing_times_quantiles` to handle quantile calculations with nan
    values involved
-   ([#176](https://github.com/openscm/scmdata/pull/176)) Add `as_run`
    argument to {meth}`scmdata.ScmRun.process_over` (closes
    [#160](https://github.com/openscm/scmdata/issues/160))

## v0.13.0

-   ([#174](https://github.com/openscm/scmdata/pull/174)) Add
    {func}`scmdata.processing.categorisation_sr15` (also added functionality for this to
    {func}`scmdata.processing.calculate_summary_stats`)
-   ([#173](https://github.com/openscm/scmdata/pull/173)) Add
    {func}`scmdata.processing.calculate_peak`and {func}`scmdata.processing.calculate_peak_time`
    (also added functionality for these to {func}`scmdata.processing.calculate_summary_stats`)
-   ([#175](https://github.com/openscm/scmdata/pull/175)) Remove unused
    {obj}`scmdata.REQUIRED_COLS` (closes [#166](https://github.com/openscm/scmdata/issues/166))
-   ([#172](https://github.com/openscm/scmdata/pull/172)) Add
    {func}`scmdata.processing.calculate_summary_stats`
-   ([#171](https://github.com/openscm/scmdata/pull/171)) Add
    {func}`scmdata.processing.calculate_exceedance_probabilities`,
    {func}`scmdata.processing.calculate_exceedance_probabilities_over_time`and
    {meth}`scmdata.ScmRun.get_meta_columns_except`
-   ([#170](https://github.com/openscm/scmdata/pull/170)) Added {meth}`scmdata.ScmRun.groupby_all_except`
    to allow greater use of the concept of grouping by columns except a given set
-   ([#169](https://github.com/openscm/scmdata/pull/169)) Make
    {func}`scmdata.processing.calculate_crossing_times` able to be used as a standalone function rather than
    being intended to be called via {meth}`scmdata.ScmRun.process_over`
-   ([#168](https://github.com/openscm/scmdata/pull/168)) Improve the
    error messages when checking that {class}`scmdata.ScmRun` objects are identical
-   ([#165](https://github.com/openscm/scmdata/pull/165)) Add
    {func}`scmdata.processing.calculate_crossing_times`
-   ([#164](https://github.com/openscm/scmdata/pull/164)) Added {meth}`scmdata.ScmRun.append_timewise` to
    allow appending of data along the time axis with broadcasting along multiple meta dimensions
-   ([#164](https://github.com/openscm/scmdata/pull/164)) Sort time axis
    internally (ensures that {meth}`scmdata.ScmRun.__repr__` renders properly)
-   ([#164](https://github.com/openscm/scmdata/pull/164)) Added
    {class}`scmdata.errors.DuplicateTimesError`, raised when duplicate times are passed to
    {class}`scmdata.ScmRun`
-   ([#164](https://github.com/openscm/scmdata/pull/164)) Unified
    capitalisation of error messages in `scmdata.errors` and added the
    `meta` table to `exc_info` of
    {class}`NonUniqueMetadataError`
-   ([#163](https://github.com/openscm/scmdata/pull/163)) Added
    {meth}`scmdata.ScmRun.adjust_median_to_target` to allow for the median of an ensemble of timeseries to
    be adjusted to a given value
-   ([#163](https://github.com/openscm/scmdata/pull/163)) Update
    `scmdata.plotting.RCMIP_SCENARIO_COLOURS` to new AR6 colours

## v0.12.1

-   ([#162](https://github.com/openscm/scmdata/pull/162)) Fix bug which
    led to a bad read in when the saved data spanned from before year
    1000
-   ([#162](https://github.com/openscm/scmdata/pull/162)) Allowed
    {meth}`scmdata.ScmRun.plumeplot` to handle
    the case where not all data will make complete plumes or have a
    best-estimate line if `pre_calculated` is `True`. This allows a
    dataset with one source that has a best-estimate only to be plotted
    at the same time as a dataset which has a range too with only a
    single call to {meth}`scmdata.ScmRun.plumeplot`.

## v0.12.0

-   ([#161](https://github.com/openscm/scmdata/pull/161)) Loosen
    requirements and drop Python3.6 support

## v0.11.0

-   ([#159](https://github.com/openscm/scmdata/pull/159)) Allow access
    to more functions in
    {class}`scmdata.run.BaseScmRun.process_over`, including arbitrary functions
-   ([#158](https://github.com/openscm/scmdata/pull/158)) Return {class}`cftime.DatetimeGregorian` rather
    than {class}`cftime.datetime` from {meth}`scmdata.time.TimePoints.as_cftime`
    and {func}`scmdata.offsets.generate_range` to ensure better interoperability with other libraries (e.g.
    xarray\'s plotting functionality). Add `date_cls` argument to {meth}`scmdata.time.TimePoints.as_cftime`
    and {func}`scmdata.offsets.generate_range` so that the output date type can be user specified.
-   ([#148](https://github.com/openscm/scmdata/pull/148)) Refactor
    {class}`scmdata.database.ScmDatabase` to be able to use custom backends
-   ([#157](https://github.com/openscm/scmdata/pull/157)) Add `disable_tqdm` parameter to
    {meth}`scmdata.database.ScmDatabase.load`
    and {meth}`scmdata.database.ScmDatabase.save` to disable displaying progress bars
-   ([#156](https://github.com/openscm/scmdata/pull/156)) Fix
    {mod}`pandas` and {mod}`xarray` documentation links
-   ([#155](https://github.com/openscm/scmdata/pull/155)) Simplify flake8 configuration

## v0.10.1

-   ([#154](https://github.com/openscm/scmdata/pull/154)) Refactor common binary operators for
    {class}`scmdata.run.BaseScmRun` and {class}`scmdata.timeseries.Timeseries` into
    a mixin following the removal of {func}`xarray.core.ops.inject_binary_ops` in `xarray==1.18.0`

## v0.10.0

-   ([#151](https://github.com/openscm/scmdata/pull/151)) Add {meth}`ScmRun.to_xarray` (improves conversion
    to xarray and ability of user to control dimensions etc. when writing netCDF files)
-   ([#149](https://github.com/openscm/scmdata/pull/149)) Fix bug in testcase for `xarray<=0.16.1`
-   ([#147](https://github.com/openscm/scmdata/pull/147)) Re-do netCDF reading and writing to make use of
    xarray and provide better handling of extras (results in speedups of 10-100x)
-   ([#146](https://github.com/openscm/scmdata/pull/146)) Update CI-CD
    workflow to include more sensible dependencies and also test Python3.9
-   ([#145](https://github.com/openscm/scmdata/pull/145)) Allow {meth}`ScmDatabase.load` to handle lists as
    filter values

## v0.9.1

-   ([#144](https://github.com/openscm/scmdata/pull/144)) Fix {meth}`ScmRun.plumeplot` style handling
    (previously, if `dashes` was not supplied each line would be a different style even if all the lines had
    the same value for `style_var`)

## v0.9.0

-   ([#143](https://github.com/openscm/scmdata/pull/143)) Alter time axis when serialising to netCDF so that
    time axis is easily read by other tools (e.g. xarray)

## v0.8.0

-   ([#139](https://github.com/openscm/scmdata/pull/139)) Update filter to handle metadata columns which
    contain a mix of data types
-   ([#139](https://github.com/openscm/scmdata/pull/139)) Add {meth}`ScmRun.plumeplot`
-   ([#140](https://github.com/openscm/scmdata/pull/140)) Add workaround
    for installing scmdata with Python 3.6 on windows to handle lack of
    cftime 1.3.1 wheel
-   ([#138](https://github.com/openscm/scmdata/pull/138)) Add {meth}`ScmRun.quantiles_over`
-   ([#137](https://github.com/openscm/scmdata/pull/137)) Fix {meth}`scmdata.ScmRun.to_csv` so that
    writing and reading is circular (i.e. you end up where you started
    if you write a file and then read it straight back into a new
    {obj}`scmdata.ScmRun <scmdata.run.ScmRun>` instance)

## v0.7.6

-   ([#136](https://github.com/openscm/scmdata/pull/136)) Make filtering by year able to handle a
    {obj}`np.ndarray`of integers (previously this would raise a {class}`TypeError`)
-   ([#135](https://github.com/openscm/scmdata/pull/135)) Make scipy
    lazy loading in `scmdata.time` follow lazy loading seen in other
    modules
-   ([#134](https://github.com/openscm/scmdata/pull/134)) Add CI run in
    which seaborn is not installed to check scipy importing

## v0.7.5

-   ([#133](https://github.com/openscm/scmdata/pull/133)) Pin
    pandas<1.2 to avoid pint-pandas installation failure (see
    [pint-pandas #51](https://github.com/hgrecco/pint-pandas/issues/51))

## v0.7.4

-   ([#132](https://github.com/openscm/scmdata/pull/132)) Update to new `openscm-units 0.2`
-   ([#130](https://github.com/openscm/scmdata/pull/130)) Add stack info
    to warning message when filtering results in an empty
    {obj}`scmdata.run.ScmRun`

## v0.7.3

-   ([#124](https://github.com/openscm/scmdata/pull/124)) Add {class}`scmdata.run.BaseScmRun` and
    {attr}`scmdata.run.BaseScmRun.required_cols` so new sub-classes can be defined which use a different
    set of required columns from {class}`scmdata.run.ScmRun`. Also added
    {class}`scmdata.errors.MissingRequiredColumn` and tidied up the docs.
-   ([#75](https://github.com/openscm/scmdata/pull/75)) Add test to ensure that {meth}`scmdata.ScmRun.groupby`
    cannot pick up the same timeseries twice even if metadata is changed
    by the function being applied
-   ([#125](https://github.com/openscm/scmdata/pull/125)) Fix edge-case
    when filtering an empty {class}`scmdata.ScmRun <scmdata.run.ScmRun>`
-   ([#123](https://github.com/openscm/scmdata/pull/123)) Add
    {class}`scmdata.database.ScmDatabase` to read/write data using multiple files. (closes
    [#103](https://github.com/openscm/scmdata/issues/103))

## v0.7.2

-   ([#121](https://github.com/openscm/scmdata/pull/121)) Faster
    implementation of {func}`scmdata.run.run_append`. The original timeseries indexes and order are no
    longer maintained after an append.
-   ([#120](https://github.com/openscm/scmdata/pull/120)) Check the type
    and length of the runs argument in
    {func}`scmdata.run.run_append` (closes
    [#101](https://github.com/openscm/scmdata/issues/101))

## v0.7.1

-   ([#119](https://github.com/openscm/scmdata/pull/119)) Make groupby
    support grouping by metadata with integer values
-   ([#119](https://github.com/openscm/scmdata/pull/119)) Ensure using {func}`scmdata.run.run_append` does not
    mangle the index to {obj}`pd.DatetimeIndex`

## v0.7.0

-   ([#118](https://github.com/openscm/scmdata/pull/118)) Make scipy an optional dependency
-   ([#117](https://github.com/openscm/scmdata/pull/117)) Sort timeseries index ordering (closes
    [#97](https://github.com/openscm/scmdata/issues/97))
-   ([#116](https://github.com/openscm/scmdata/pull/116)) Update {meth}`scmdata.ScmRun.drop_meta` inplace
    behaviour
-   ([#115](https://github.com/openscm/scmdata/pull/115)) Add `na-override` argument to
    {meth}`scmdata.ScmRun.process_over` for handling nan metadata (closes
    [#113](https://github.com/openscm/scmdata/issues/113))
-   ([#114](https://github.com/openscm/scmdata/pull/114)) Add operations:
    {meth}`scmdata.ScmRun.linear_regression`,
    {meth}`scmdata.ScmRun.linear_regression_gradient`,
    {meth}`scmdata.ScmRun.linear_regression_intercept` and
    {meth}`scmdata.ScmRun.linear_regression_scmrun`
-   ([#111](https://github.com/openscm/scmdata/pull/111)) Add operation:
    {meth}`scmdata.ScmRun.delta_per_delta_time`
-   ([#112](https://github.com/openscm/scmdata/pull/112)) Ensure unit
    conversion doesn't fall over when the target unit is in the input
-   ([#110](https://github.com/openscm/scmdata/pull/110)) Revert to
    using {class}`pd.DataFrame` with {class}`pd.Categorical`
    series as meta indexes.
-   ([#108](https://github.com/openscm/scmdata/pull/108)) Remove
    deprecated {class}`ScmDataFrame` (closes
    [#60](https://github.com/openscm/scmdata/issues/60))
-   ([#105](https://github.com/openscm/scmdata/pull/105)) Add
    performance benchmarks for {obj}`ScmRun`
-   ([#106](https://github.com/openscm/scmdata/pull/106)) Add
    {meth}`ScmRun.integrate` so we can
    integrate timeseries with respect to time
-   ([#104](https://github.com/openscm/scmdata/pull/104)) Fix bug when
    reading csv/excel files which use integer years and
    `lowercase_cols=True` (closes
    [#102](https://github.com/openscm/scmdata/issues/102))

## v0.6.4

-   ([#96](https://github.com/openscm/scmdata/pull/96)) Fix non-unique timeseries metadata checks for
    {meth}`ScmRun.timeseries`
-   ([#100](https://github.com/openscm/scmdata/pull/100)) When initialising {obj}`ScmRun` from file, make
    the default be to read with {func}`pd.read_csv`. This means we now initialising reading from gzipped
    CSV files.
-   ([#99](https://github.com/openscm/scmdata/pull/99)) Hotfix failing notebook test
-   ([#94](https://github.com/openscm/scmdata/pull/94)) Fix edge-case issue with drop_meta (closes
    [#92](https://github.com/openscm/scmdata/issues/92))
-   ([#95](https://github.com/openscm/scmdata/pull/95)) Add `drop_all_nan_times` keyword argument to
    {meth}`ScmRun.timeseries` so time points with no data of interest can easily be removed

## v0.6.3

-   ([#91](https://github.com/openscm/scmdata/pull/91)) Provide support for pandas==1.1

## v0.6.2

-   ([#87](https://github.com/openscm/scmdata/pull/87)) Upgrade workflow to use `isort>=5`
-   ([#82](https://github.com/openscm/scmdata/pull/82)) Add support for
    adding Pint scalars and vectors to {class}`scmdata.Timeseries` and
    {class}`scmdata.ScmRun <scmdata.run.ScmRun>` instances
-   ([#85](https://github.com/openscm/scmdata/pull/85)) Allow required
    columns to be read as `extras` from netCDF files (closes
    [#83](https://github.com/openscm/scmdata/issues/83))
-   ([#84](https://github.com/openscm/scmdata/pull/84)) Raise a DeprecationWarning if no default `inplace`
    argument is provided for {meth}`ScmRun.drop_meta`. inplace default behaviour scheduled to be changed to
    `False` in v0.7.0
-   ([#81](https://github.com/openscm/scmdata/pull/81)) Add {attr}`scmdata.run.ScmRun.metadata` to track
    {class}`ScmRun` instance-specific metadata (closes [#77](https://github.com/openscm/scmdata/issues/77))
-   ([#80](https://github.com/openscm/scmdata/pull/80)) No longer use
    {class}`pandas.tseries.offsets.BusinessMixin` to determine Business-related offsets in
    {meth}`scmdata.offsets.to_offset`. (closes [#78](https://github.com/openscm/scmdata/issues/78))
-   ([#79](https://github.com/openscm/scmdata/pull/79)) Introduce
    `scmdata.errors.NonUniqueMetadataError`. Update handling of
    duplicate metadata so default behaviour of `run_append` is to raise
    a `NonUniqueMetadataError`. (closes [#76](https://github.com/openscm/scmdata/issues/76))

## v0.6.1

-   ([#74](https://github.com/openscm/scmdata/pull/74)) Update handling
    of unit conversion context during unit conversions
-   ([#73](https://github.com/openscm/scmdata/pull/73)) Only reindex
    timeseries when dealing with different time points

## v0.5.2

-   ([#65](https://github.com/openscm/scmdata/pull/65)) Use pint for
    ops, making them automatically unit aware
-   ([#71](https://github.com/openscm/scmdata/pull/71)) Start adding
    arithmetic support via {mod}`scmdata.ops`.
    So far only add and subtract are supported.
-   ([#70](https://github.com/openscm/scmdata/pull/70)) Automatically
    set y-axis label to units if it makes sense in
    {obj}`ScmRun`\'s
    {meth}`lineplot` method

## v0.5.1

-   ([#68](https://github.com/openscm/scmdata/pull/68)) Rename
    {func}`scmdata.run.df_append` to
    :func\`scmdata.run.run_append\`. :func\`scmdata.run.df_append\`
    deprecated and will be removed in v0.6.0
-   ([#67](https://github.com/openscm/scmdata/pull/67)) Update the
    documentation for {meth}`ScmRun.append`
-   ([#66](https://github.com/openscm/scmdata/pull/66)) Raise ValueError
    if index/columns arguments are not provided when instantiating a
    :class\`ScmRun\` object with a numpy array. Add `lowercase_cols`
    argument to coerce the column names in CSV files to lowercase

## v0.5.0

-   ([#64](https://github.com/openscm/scmdata/pull/64)) Remove spurious warning from {obj}`ScmRun`'s
    {meth}`filter` method([#63](https://github.com/openscm/scmdata/pull/63))
    Remove {meth}`set_meta` from {class}`ScmRun` in preference for using the {meth}`__setitem__` method
-   ([#62](https://github.com/openscm/scmdata/pull/62)) Fix interpolation when the data contains nan values
-   ([#61](https://github.com/openscm/scmdata/pull/61)) Hotfix filters
    to also include caret (\"\^\") in pseudo-regexp syntax. Also adds
    {meth}`empty` property to {obj}`ScmRun`
-   ([#59](https://github.com/openscm/scmdata/pull/59)) Deprecate {class}`ScmDataFrame`. To be removed in
    `v0.6.0`
-   ([#58](https://github.com/openscm/scmdata/pull/58)) Use `cftime` datetimes when appending {class}`ScmRun`
    objects to avoid OutOfBounds errors when datetimes span many centuries
-   ([#55](https://github.com/openscm/scmdata/pull/55)) Add `time_axis`keyword argument to
    `ScmRun.timeseries`, `ScmRun.long_data` and `ScmRun.lineplot` to give greater control of the time axis
    when retrieving data
-   ([#54](https://github.com/openscm/scmdata/pull/54)) Add {meth}`drop_meta` to {class}`ScmRun` for dropping
    metadata columns
-   ([#53](https://github.com/openscm/scmdata/pull/53)) Don\'t convert
    case of variable names written to file. No longer convert case of
    serialized dataframes
-   ([#51](https://github.com/openscm/scmdata/pull/51)) Refactor
    {meth}`relative_to_ref_period_mean` so that
    it returns an instance of the input data type (rather than a
    {obj}`pd.DataFrame`) and puts the reference
    period in separate meta columns rather than mangling the variable
    name.
-   ([#47](https://github.com/openscm/scmdata/pull/47)) Update README and `setup.py` to make it easier for new users

## v0.4.3

-   ([#46](https://github.com/openscm/scmdata/pull/46)) Add test of conda installation

## v0.4.2

-   ([#45](https://github.com/openscm/scmdata/pull/45)) Make installing seaborn optional

## v0.4.1

-   ([#44](https://github.com/openscm/scmdata/pull/44)) Add multi-dimensional handling to `scmdata.netcdf`

-   ([#43](https://github.com/openscm/scmdata/pull/43)) Fix minor bugs in netCDF handling and address minor code coverage issues

-   ([#41](https://github.com/openscm/scmdata/pull/41)) Update
    documentation of the data model. Additionally:

    > -   makes `.time_points` atttributes consistently return
    >     `scmdata.time.TimePoints` instances
    > -   ensures `.meta` is used consistently throughout the code base
    >     (removing `.metadata`)

-   ([#33](https://github.com/openscm/scmdata/pull/33)) Remove dependency on [pyam](https://github.com/IAMconsortium/pyam).
    Plotting is done with [seaborn](https://github.com/mwaskom/seaborn)instead.

-   ([#34](https://github.com/openscm/scmdata/pull/34)) Allow the serialization/deserialization of `scmdata.run.ScmRun` and `scmdata.ScmDataFrame`
    as netCDF4 files.

-   ([#30](https://github.com/lewisjared/scmdata/pull/30)) Swap to using [openscm-units](https://github.com/openscm/openscm-units) for unit handling (hence remove much of the
    `scmdata.units` module)

-   ([#21](https://github.com/openscm/scmdata/pull/21)) Added `scmdata.run.ScmRun` as a proposed replacement for `scmdata.dataframe.ScmDataFrame`.
    This new class provides an identical interface as a `ScmDataFrame`, but uses a different underlying data
    structure to the `ScmDataFrame`. The purpose of `ScmRun` is to provide performance improvements when
    handling large sets of time-series data. Removed support for Python 3.5 until`pyam`
   dependency is optional

-   ([#31](https://github.com/openscm/scmdata/pull/31)) Tidy up repository after changing location

## v0.4.0

-   ([#28](https://github.com/openscm/scmdata/pull/28)) Expose `scmdata.units.unit_registry`

## v0.3.1

-   ([#25](https://github.com/openscm/scmdata/pull/25)) Make scipy an optional dependency
-   ([#24](https://github.com/openscm/scmdata/pull/24)) Fix missing "N2O" unit (see [#14](https://github.com/openscm/scmdata/pull/14)). Also updates test of year to day conversion,
    it is 365.25 to within 0.01% (but depends on the Pint release).

## v0.3.0

-   ([#20](https://github.com/openscm/scmdata/pull/20)) Add support for python=3.5
-   ([#19](https://github.com/openscm/scmdata/pull/19)) Add support for python=3.6

## v0.2.2

-   ([#16](https://github.com/openscm/scmdata/pull/16)) Only rename columns when initialising data if needed

## v0.2.1

-   ([#13](https://github.com/openscm/scmdata/pull/13)) Ensure `LICENSE` is included in package
-   ([#11](https://github.com/openscm/scmdata/pull/11)) Add SO2F2 unit and update to Pyam v0.3.0
-   ([#12](https://github.com/openscm/scmdata/pull/12)) Add `get_unique_meta` convenience method
-   ([#10](https://github.com/openscm/scmdata/pull/10)) Fix extrapolation bug which prevented any extrapolation from occurring

## v0.2.0

-   ([#9](https://github.com/openscm/scmdata/pull/9)) Add `time_mean` method
-   ([#8](https://github.com/openscm/scmdata/pull/8)) Add `make docs` target

## v0.1.2

-   ([#7](https://github.com/openscm/scmdata/pull/7)) Add notebook tests
-   ([#4](https://github.com/openscm/scmdata/pull/4)) Unit conversions for CH4 and N2O contexts now work for compound units (e.g. 'Mt CH4 / yr' to
    'Gt C / day')
-   ([#6](https://github.com/openscm/scmdata/pull/6)) Add auto-formatting

## v0.1.1

-   ([#5](https://github.com/openscm/scmdata/pull/5)) Add `scmdata.dataframe.df_append` to `__init__.py`

## v0.1.0

-   ([#3](https://github.com/openscm/scmdata/pull/3)) Added documentation for the api and Makefile targets for releasing
-   ([#2](https://github.com/openscm/scmdata/pull/2)) Refactored scmdataframe from <openclimatedata/openscm@077f9b5> into a standalone package
-   ([#1](https://github.com/openscm/scmdata/pull/1)) Add docs folder
