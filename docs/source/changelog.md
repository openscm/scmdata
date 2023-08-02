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


## v0.15.1

-   ([#239](https://github.com/openscm/scmdata/pull/239)) Move notebooks
    into the documentation and an update of the documentation
    configuration
-   ([#238](https://github.com/openscm/scmdata/pull/238)) Support
    `scmdata.ScmRun`{.interpreted-text role="func"} reading and writing
    files using [pathlib.Path]{.title-ref} objects.
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
    `scmdata.ScmRun.interpolate`{.interpreted-text role="func"} to allow
    for interpolation which ignores leap-years. This also fixes a bug
    where `scmdata.ScmRun.interpolate`{.interpreted-text role="func"}
    converts integer values into unix time. This functionality isn\'t
    consistent with the behaviour of the TimePoints class where integers
    are converted into years.
-   ([#218](https://github.com/openscm/scmdata/pull/218)) Replaced
    internal calls to `scmdata.groupby.RunGroupby.map`{.interpreted-text
    role="func"} with
    `scmdata.groupby.RunGroupby.apply`{.interpreted-text role="func"}
-   ([#210](https://github.com/openscm/scmdata/pull/210)) Update github
    actions to avoid the use of a deprecated workflow command

## v0.14.2

-   ([#209](https://github.com/openscm/scmdata/pull/209)) Lazy import
    plotting modules to speed up startup time
-   ([#208](https://github.com/openscm/scmdata/pull/208)) Ensure that
    all unit operations in `scmdata`{.interpreted-text role="mod"} use
    `scmdata.units.UNIT_REGISTRY`{.interpreted-text role="attr"}. This
    now defaults to `openscm_units.unit_registry`{.interpreted-text
    role="attr"} instead of unique unit registry for
    `scmdata`{.interpreted-text role="mod"}.
-   ([#202](https://github.com/openscm/scmdata/pull/202)) Add
    `scmdata.run.ScmRun.set_meta`{.interpreted-text role="func"} to
    enable setting of metadata for a subset of timeseries
-   ([#194](https://github.com/openscm/scmdata/pull/194)) Deprecate
    `scmdata.groupby.RunGroupBy.map`{.interpreted-text role="func"} in
    preference to `scmdata.groupby.RunGroupBy.apply`{.interpreted-text
    role="func"} which is identical in functionality. Add
    `scmdata.ScmRun.apply`{.interpreted-text role="class"} for applying
    a function to each timeseries
-   ([#195](https://github.com/openscm/scmdata/pull/195)) Refactor
    `scmdata.database`{.interpreted-text role="mod"} to a package. The
    database backends have been moved to
    `scmdata.database.backends`{.interpreted-text role="mod"}.
-   ([#197](https://github.com/openscm/scmdata/pull/197)) Workaround
    [regression in
    Panda\'s](https://github.com/pandas-dev/pandas/issues/47071)
    handling of xarray\'s `CFTimeIndex`{.interpreted-text role="class"}
-   ([#193](https://github.com/openscm/scmdata/pull/193)) Pin the
    version of black used for code formatting to ensure consistency

## v0.14.1

-   ([#192](https://github.com/openscm/scmdata/pull/192)) Bugfix for the
    versioning of the package
-   ([#191](https://github.com/openscm/scmdata/pull/191)) Add check of
    PyPI distribution to CI

## v0.14.0

-   ([#190](https://github.com/openscm/scmdata/pull/190)) Add special
    case for extrapolating timeseries containing a single timestep using
    `constant` extrapolation. Moved
    `scmdata.errors.InsufficientDataError`{.interpreted-text
    role="attr"} from `scmdata.time`{.interpreted-text role="mod"} to
    `scmdata.errors`{.interpreted-text role="mod"}
-   ([#186](https://github.com/openscm/scmdata/pull/186) and
    [#187](https://github.com/openscm/scmdata/pull/187)) Fix the
    handling of non-alphanumeric characters in filenames on Windows for
    `scmdata.database.ScmDatabase`{.interpreted-text role="class"}. `*`
    values are no longer included in
    `scmdata.database.ScmDatabase`{.interpreted-text role="class"}
    filenames
-   ([#186](https://github.com/openscm/scmdata/pull/186) Move to
    `pyproject.toml` for setup etc.

## v0.13.2

-   ([#185](https://github.com/openscm/scmdata/pull/185)) Allow
    `scmdata.run.ScmRun`{.interpreted-text role="class"} to read remote
    files by providing a URL to the constructor
-   ([#183](https://github.com/openscm/scmdata/pull/183)) Deprecate
    `scmdata.ops.integrate`{.interpreted-text role="func"}, replacing
    with to `scmdata.ops.cumsum`{.interpreted-text role="func"} and
    `scmdata.ops.cumtrapz`{.interpreted-text role="func"}
-   ([#184](https://github.com/openscm/scmdata/pull/184)) Add
    `scmdata.run.ScmRun.round`{.interpreted-text role="func"}
-   ([#182](https://github.com/openscm/scmdata/pull/182)) Updated
    incorrect [conda]{.title-ref} install instructions

## v0.13.1

-   ([#181](https://github.com/openscm/scmdata/pull/181)) Allow the
    initialisation of empty `scmdata.ScmRun`{.interpreted-text
    role="class"} objects
-   ([#180](https://github.com/openscm/scmdata/pull/180)) Add
    `scmdata.processing.calculate_crossing_times_quantiles`{.interpreted-text
    role="func"} to handle quantile calculations with nan values
    involved
-   ([#176](https://github.com/openscm/scmdata/pull/176)) Add `as_run`
    argument to `scmdata.ScmRun.process_over`{.interpreted-text
    role="func"} (closes
    [#160](https://github.com/openscm/scmdata/issues/160))

## v0.13.0

-   ([#174](https://github.com/openscm/scmdata/pull/174)) Add
    `scmdata.processing.categorisation_sr15`{.interpreted-text
    role="func"} (also added functionality for this to
    `scmdata.processing.calculate_summary_stats`{.interpreted-text
    role="func"})
-   ([#173](https://github.com/openscm/scmdata/pull/173)) Add
    `scmdata.processing.calculate_peak`{.interpreted-text role="func"}
    and `scmdata.processing.calculate_peak_time`{.interpreted-text
    role="func"} (also added functionality for these to
    `scmdata.processing.calculate_summary_stats`{.interpreted-text
    role="func"})
-   ([#175](https://github.com/openscm/scmdata/pull/175)) Remove unused
    `scmdata.REQUIRED_COLS`{.interpreted-text role="obj"} (closes
    [#166](https://github.com/openscm/scmdata/issues/166))
-   ([#172](https://github.com/openscm/scmdata/pull/172)) Add
    `scmdata.processing.calculate_summary_stats`{.interpreted-text
    role="func"}
-   ([#171](https://github.com/openscm/scmdata/pull/171)) Add
    `scmdata.processing.calculate_exceedance_probabilities`{.interpreted-text
    role="func"},
    `scmdata.processing.calculate_exceedance_probabilities_over_time`{.interpreted-text
    role="func"} and
    `scmdata.ScmRun.get_meta_columns_except`{.interpreted-text
    role="meth"}
-   ([#170](https://github.com/openscm/scmdata/pull/170)) Added
    `scmdata.ScmRun.groupby_all_except`{.interpreted-text role="meth"}
    to allow greather use of the concept of grouping by columns except a
    given set
-   ([#169](https://github.com/openscm/scmdata/pull/169)) Make
    `scmdata.processing.calculate_crossing_times`{.interpreted-text
    role="func"} able to be used as a standalone function rather than
    being intended to be called via
    `scmdata.ScmRun.process_over`{.interpreted-text role="meth"}
-   ([#168](https://github.com/openscm/scmdata/pull/168)) Improve the
    error messages when checking that `scmdata.ScmRun`{.interpreted-text
    role="class"} objects are identical
-   ([#165](https://github.com/openscm/scmdata/pull/165)) Add
    `scmdata.processing.calculate_crossing_times`{.interpreted-text
    role="func"}
-   ([#164](https://github.com/openscm/scmdata/pull/164)) Added
    `scmdata.ScmRun.append_timewise`{.interpreted-text role="meth"} to
    allow appending of data along the time axis with broadcasting along
    multiple meta dimensions
-   ([#164](https://github.com/openscm/scmdata/pull/164)) Sort time axis
    internally (ensures that `scmdata.ScmRun.__repr__`{.interpreted-text
    role="meth"} renders properly)
-   ([#164](https://github.com/openscm/scmdata/pull/164)) Added
    `scmdata.errors.DuplicateTimesError`{.interpreted-text
    role="class"}, raised when duplicate times are passed to
    `scmdata.ScmRun`{.interpreted-text role="class"}
-   ([#164](https://github.com/openscm/scmdata/pull/164)) Unified
    capitalisation of error messages in `scmdata.errors` and added the
    `meta` table to `exc_info` of
    `NonUniqueMetadataError`{.interpreted-text role="class"}
-   ([#163](https://github.com/openscm/scmdata/pull/163)) Added
    `scmdata.ScmRun.adjust_median_to_target`{.interpreted-text
    role="meth"} to allow for the median of an ensemble of timeseries to
    be adjusted to a given value
-   ([#163](https://github.com/openscm/scmdata/pull/163)) Update
    `scmdata.plotting.RCMIP_SCENARIO_COLOURS` to new AR6 colours

## v0.12.1

-   ([#162](https://github.com/openscm/scmdata/pull/162)) Fix bug which
    led to a bad read in when the saved data spanned from before year
    1000
-   ([#162](https://github.com/openscm/scmdata/pull/162)) Allowed
    `scmdata.ScmRun.plumeplot`{.interpreted-text role="meth"} to handle
    the case where not all data will make complete plumes or have a
    best-estimate line if `pre_calculated` is `True`. This allows a
    dataset with one source that has a best-estimate only to be plotted
    at the same time as a dataset which has a range too with only a
    single call to `scmdata.ScmRun.plumeplot`{.interpreted-text
    role="meth"}.

## v0.12.0

-   ([#161](https://github.com/openscm/scmdata/pull/161)) Loosen
    requirements and drop Python3.6 support

## v0.11.0

-   ([#159](https://github.com/openscm/scmdata/pull/159)) Allow access
    to more functions in
    `scmdata.run.BaseScmRun.process_over`{.interpreted-text
    role="class"}, including arbitrary functions
-   ([#158](https://github.com/openscm/scmdata/pull/158)) Return
    `cftime.DatetimeGregorian`{.interpreted-text role="class"} rather
    than `cftime.datetime`{.interpreted-text role="class"} from
    `scmdata.time.TimePoints.as_cftime`{.interpreted-text role="meth"}
    and `scmdata.offsets.generate_range`{.interpreted-text role="func"}
    to ensure better interoperability with other libraries (e.g.
    xarray\'s plotting functionality). Add `date_cls` argument to
    `scmdata.time.TimePoints.as_cftime`{.interpreted-text role="meth"}
    and `scmdata.offsets.generate_range`{.interpreted-text role="func"}
    so that the output date type can be user specified.
-   ([#148](https://github.com/openscm/scmdata/pull/148)) Refactor
    `scmdata.database.ScmDatabase`{.interpreted-text role="class"} to be
    able to use custom backends
-   ([#157](https://github.com/openscm/scmdata/pull/157)) Add
    `disable_tqdm` parameter to
    `scmdata.database.ScmDatabase.load`{.interpreted-text role="meth"}
    and `scmdata.database.ScmDatabase.save`{.interpreted-text
    role="meth"} to disable displaying progress bars
-   ([#156](https://github.com/openscm/scmdata/pull/156)) Fix
    `pandas`{.interpreted-text role="mod"} and
    `xarray`{.interpreted-text role="mod"} documentation links
-   ([#155](https://github.com/openscm/scmdata/pull/155)) Simplify
    flake8 configuration

## v0.10.1

-   ([#154](https://github.com/openscm/scmdata/pull/154)) Refactor
    common binary operators for
    `scmdata.run.BaseScmRun`{.interpreted-text role="class"} and
    `scmdata.timeseries.Timeseries`{.interpreted-text role="class"} into
    a mixin following the removal of
    `xarray.core.ops.inject_binary_ops`{.interpreted-text role="func"}
    in [xarray==1.18.0]{.title-ref}

## v0.10.0

-   ([#151](https://github.com/openscm/scmdata/pull/151)) Add
    `ScmRun.to_xarray`{.interpreted-text role="meth"} (improves
    conversion to xarray and ability of user to control dimensions etc.
    when writing netCDF files)
-   ([#149](https://github.com/openscm/scmdata/pull/149)) Fix bug in
    testcase for xarray\<=0.16.1
-   ([#147](https://github.com/openscm/scmdata/pull/147)) Re-do netCDF
    reading and writing to make use of xarray and provide better
    handling of extras (results in speedups of 10-100x)
-   ([#146](https://github.com/openscm/scmdata/pull/146)) Update CI-CD
    workflow to include more sensible dependencies and also test
    Python3.9
-   ([#145](https://github.com/openscm/scmdata/pull/145)) Allow
    `ScmDatabase.load`{.interpreted-text role="meth"} to handle lists as
    filter values

## v0.9.1

-   ([#144](https://github.com/openscm/scmdata/pull/144)) Fix
    `ScmRun.plumeplot`{.interpreted-text role="meth"} style handling
    (previously, if `dashes` was not supplied each line would be a
    different style even if all the lines had the same value for
    `style_var`)

## v0.9.0

-   ([#143](https://github.com/openscm/scmdata/pull/143)) Alter time
    axis when serialising to netCDF so that time axis is easily read by
    other tools (e.g. xarray)

## v0.8.0

-   ([#139](https://github.com/openscm/scmdata/pull/139)) Update filter
    to handle metadata columns which contain a mix of data types
-   ([#139](https://github.com/openscm/scmdata/pull/139)) Add
    `ScmRun.plumeplot`{.interpreted-text role="meth"}
-   ([#140](https://github.com/openscm/scmdata/pull/140)) Add workaround
    for installing scmdata with Python 3.6 on windows to handle lack of
    cftime 1.3.1 wheel
-   ([#138](https://github.com/openscm/scmdata/pull/138)) Add
    `ScmRun.quantiles_over`{.interpreted-text role="meth"}
-   ([#137](https://github.com/openscm/scmdata/pull/137)) Fix
    `scmdata.ScmRun.to_csv`{.interpreted-text role="meth"} so that
    writing and reading is circular (i.e. you end up where you started
    if you write a file and then read it straight back into a new
    `scmdata.ScmRun <scmdata.run.ScmRun>`{.interpreted-text role="obj"}
    instance)

## v0.7.6

-   ([#136](https://github.com/openscm/scmdata/pull/136)) Make filtering
    by year able to handle a `np.ndarray`{.interpreted-text role="obj"}
    of integers (previously this would raise a
    `TypeError`{.interpreted-text role="class"})
-   ([#135](https://github.com/openscm/scmdata/pull/135)) Make scipy
    lazy loading in `scmdata.time` follow lazy loading seen in other
    modules
-   ([#134](https://github.com/openscm/scmdata/pull/134)) Add CI run in
    which seaborn is not installed to check scipy importing

## v0.7.5

-   ([#133](https://github.com/openscm/scmdata/pull/133)) Pin
    pandas\<1.2 to avoid pint-pandas installation failure (see
    [pint-pandas #51](https://github.com/hgrecco/pint-pandas/issues/51))

## v0.7.4

-   ([#132](https://github.com/openscm/scmdata/pull/132)) Update to new
    openscm-units 0.2
-   ([#130](https://github.com/openscm/scmdata/pull/130)) Add stack info
    to warning message when filtering results in an empty
    `scmdata.run.ScmRun`{.interpreted-text role="obj"}

## v0.7.3

-   ([#124](https://github.com/openscm/scmdata/pull/124)) Add
    `scmdata.run.BaseScmRun`{.interpreted-text role="class"} and
    `scmdata.run.BaseScmRun.required_cols`{.interpreted-text
    role="attr"} so new sub-classes can be defined which use a different
    set of required columns from `scmdata.run.ScmRun`{.interpreted-text
    role="class"}. Also added
    `scmdata.errors.MissingRequiredColumn`{.interpreted-text
    role="class"} and tidied up the docs.
-   ([#75](https://github.com/openscm/scmdata/pull/75)) Add test to
    ensure that `scmdata.ScmRun.groupby`{.interpreted-text role="meth"}
    cannot pick up the same timeseries twice even if metadata is changed
    by the function being applied
-   ([#125](https://github.com/openscm/scmdata/pull/125)) Fix edge-case
    when filtering an empty
    `scmdata.ScmRun <scmdata.run.ScmRun>`{.interpreted-text
    role="class"}
-   ([#123](https://github.com/openscm/scmdata/pull/123)) Add
    `scmdata.database.ScmDatabase`{.interpreted-text role="class"} to
    read/write data using multiple files. (closes
    [#103](https://github.com/openscm/scmdata/issues/103))

## v0.7.2

-   ([#121](https://github.com/openscm/scmdata/pull/121)) Faster
    implementation of `scmdata.run.run_append`{.interpreted-text
    role="func"}. The original timeseries indexes and order are no
    longer maintained after an append.
-   ([#120](https://github.com/openscm/scmdata/pull/120)) Check the type
    and length of the runs argument in
    `scmdata.run.run_append`{.interpreted-text role="func"} (closes
    [#101](https://github.com/openscm/scmdata/issues/101))

## v0.7.1

-   ([#119](https://github.com/openscm/scmdata/pull/119)) Make groupby
    support grouping by metadata with integer values
-   ([#119](https://github.com/openscm/scmdata/pull/119)) Ensure using
    `scmdata.run.run_append`{.interpreted-text role="func"} does not
    mangle the index to `pd.DatetimeIndex`{.interpreted-text role="obj"}

## v0.7.0

-   ([#118](https://github.com/openscm/scmdata/pull/118)) Make scipy an
    optional dependency
-   ([#117](https://github.com/openscm/scmdata/pull/117)) Sort
    timeseries index ordering (closes
    [#97](https://github.com/openscm/scmdata/issues/97))
-   ([#116](https://github.com/openscm/scmdata/pull/116)) Update
    `scmdata.ScmRun.drop_meta`{.interpreted-text role="meth"} inplace
    behaviour
-   ([#115](https://github.com/openscm/scmdata/pull/115)) Add
    [na_override]{.title-ref} argument to
    `scmdata.ScmRun.process_over`{.interpreted-text role="meth"} for
    handling nan metadata (closes
    [#113](https://github.com/openscm/scmdata/issues/113))
-   ([#114](https://github.com/openscm/scmdata/pull/114)) Add
    operations: `scmdata.ScmRun.linear_regression`{.interpreted-text
    role="meth"},
    `scmdata.ScmRun.linear_regression_gradient`{.interpreted-text
    role="meth"},
    `scmdata.ScmRun.linear_regression_intercept`{.interpreted-text
    role="meth"} and
    `scmdata.ScmRun.linear_regression_scmrun`{.interpreted-text
    role="meth"}
-   ([#111](https://github.com/openscm/scmdata/pull/111)) Add operation:
    `scmdata.ScmRun.delta_per_delta_time`{.interpreted-text role="meth"}
-   ([#112](https://github.com/openscm/scmdata/pull/112)) Ensure unit
    conversion doesn\'t fall over when the target unit is in the input
-   ([#110](https://github.com/openscm/scmdata/pull/110)) Revert to
    using [pd.DataFrame]{.title-ref} with [pd.Categorical]{.title-ref}
    series as meta indexes.
-   ([#108](https://github.com/openscm/scmdata/pull/108)) Remove
    deprecated `ScmDataFrame`{.interpreted-text role="class"} (closes
    [#60](https://github.com/openscm/scmdata/issues/60))
-   ([#105](https://github.com/openscm/scmdata/pull/105)) Add
    performance benchmarks for `ScmRun`{.interpreted-text role="obj"}
-   ([#106](https://github.com/openscm/scmdata/pull/106)) Add
    `ScmRun.integrate`{.interpreted-text role="meth"} so we can
    integrate timeseries with respect to time
-   ([#104](https://github.com/openscm/scmdata/pull/104)) Fix bug when
    reading csv/excel files which use integer years and
    `lowercase_cols=True` (closes
    [#102](https://github.com/openscm/scmdata/issues/102))

## v0.6.4

-   ([#96](https://github.com/openscm/scmdata/pull/96)) Fix non-unique
    timeseries metadata checks for `ScmRun.timeseries`{.interpreted-text
    role="meth"}
-   ([#100](https://github.com/openscm/scmdata/pull/100)) When
    initialising `ScmRun`{.interpreted-text role="obj"} from file, make
    the default be to read with `pd.read_csv`{.interpreted-text
    role="func"}. This means we now initialising reading from gzipped
    CSV files.
-   ([#99](https://github.com/openscm/scmdata/pull/99)) Hotfix failing
    notebook test
-   ([#94](https://github.com/openscm/scmdata/pull/94)) Fix edge-case
    issue with drop_meta (closes
    [#92](https://github.com/openscm/scmdata/issues/92))
-   ([#95](https://github.com/openscm/scmdata/pull/95)) Add
    `drop_all_nan_times` keyword argument to
    `ScmRun.timeseries`{.interpreted-text role="meth"} so time points
    with no data of interest can easily be removed

## v0.6.3

-   ([#91](https://github.com/openscm/scmdata/pull/91)) Provide support
    for pandas==1.1

## v0.6.2

-   ([#87](https://github.com/openscm/scmdata/pull/87)) Upgrade workflow
    to use `isort>=5`
-   ([#82](https://github.com/openscm/scmdata/pull/82)) Add support for
    adding Pint scalars and vectors to
    `scmdata.Timeseries`{.interpreted-text role="class"} and
    `scmdata.ScmRun <scmdata.run.ScmRun>`{.interpreted-text
    role="class"} instances
-   ([#85](https://github.com/openscm/scmdata/pull/85)) Allow required
    columns to be read as `extras` from netCDF files (closes
    [#83](https://github.com/openscm/scmdata/issues/83))
-   ([#84](https://github.com/openscm/scmdata/pull/84)) Raise a
    DeprecationWarning if no default `inplace` argument is provided for
    `ScmRun.drop_meta`{.interpreted-text role="meth"}. inplace default
    behaviour scheduled to be changed to `False` in v0.7.0
-   ([#81](https://github.com/openscm/scmdata/pull/81)) Add
    `scmdata.run.ScmRun.metadata`{.interpreted-text role="attr"} to
    track `ScmRun`{.interpreted-text role="class"} instance-specific
    metadata (closes
    [#77](https://github.com/openscm/scmdata/issues/77))
-   ([#80](https://github.com/openscm/scmdata/pull/80)) No longer use
    `pandas.tseries.offsets.BusinessMixin`{.interpreted-text
    role="class"} to determine Business-related offsets in
    `scmdata.offsets.to_offset`{.interpreted-text role="meth"}. (closes
    [#78](https://github.com/openscm/scmdata/issues/78))
-   ([#79](https://github.com/openscm/scmdata/pull/79)) Introduce
    `scmdata.errors.NonUniqueMetadataError`. Update handling of
    duplicate metadata so default behaviour of `run_append` is to raise
    a `NonUniqueMetadataError`. (closes
    [#76](https://github.com/openscm/scmdata/issues/76))

## v0.6.1

-   ([#74](https://github.com/openscm/scmdata/pull/74)) Update handling
    of unit conversion context during unit conversions
-   ([#73](https://github.com/openscm/scmdata/pull/73)) Only reindex
    timeseries when dealing with different time points

## v0.5.2

-   ([#65](https://github.com/openscm/scmdata/pull/65)) Use pint for
    ops, making them automatically unit aware
-   ([#71](https://github.com/openscm/scmdata/pull/71)) Start adding
    arithmetic support via `scmdata.ops`{.interpreted-text role="mod"}.
    So far only add and subtract are supported.
-   ([#70](https://github.com/openscm/scmdata/pull/70)) Automatically
    set y-axis label to units if it makes sense in
    `ScmRun`{.interpreted-text role="obj"}\'s
    `lineplot`{.interpreted-text role="meth"} method

## v0.5.1

-   ([#68](https://github.com/openscm/scmdata/pull/68)) Rename
    `scmdata.run.df_append`{.interpreted-text role="func"} to
    :func\`scmdata.run.run_append\`. :func\`scmdata.run.df_append\`
    deprecated and will be removed in v0.6.0
-   ([#67](https://github.com/openscm/scmdata/pull/67)) Update the
    documentation for `ScmRun.append`{.interpreted-text role="meth"}
-   ([#66](https://github.com/openscm/scmdata/pull/66)) Raise ValueError
    if index/columns arguments are not provided when instantiating a
    :class\`ScmRun\` object with a numpy array. Add `lowercase_cols`
    argument to coerce the column names in CSV files to lowercase

## v0.5.0

-   ([#64](https://github.com/openscm/scmdata/pull/64)) Remove spurious
    warning from `ScmRun`{.interpreted-text role="obj"}\'s
    `filter`{.interpreted-text role="meth"} method
-   ([#63](https://github.com/openscm/scmdata/pull/63)) Remove
    `set_meta`{.interpreted-text role="meth"} from
    `ScmRun`{.interpreted-text role="class"} in preference for using the
    `__setitem__`{.interpreted-text role="meth"} method
-   ([#62](https://github.com/openscm/scmdata/pull/62)) Fix
    interpolation when the data contains nan values
-   ([#61](https://github.com/openscm/scmdata/pull/61)) Hotfix filters
    to also include caret (\"\^\") in pseudo-regexp syntax. Also adds
    `empty`{.interpreted-text role="meth"} property to
    `ScmRun`{.interpreted-text role="obj"}
-   ([#59](https://github.com/openscm/scmdata/pull/59)) Deprecate
    `ScmDataFrame`{.interpreted-text role="class"}. To be removed in
    v0.6.0
-   ([#58](https://github.com/openscm/scmdata/pull/58)) Use `cftime`
    datetimes when appending `ScmRun`{.interpreted-text role="class"}
    objects to avoid OutOfBounds errors when datetimes span many
    centuries
-   ([#55](https://github.com/openscm/scmdata/pull/55)) Add `time_axis`
    keyword argument to `ScmRun.timeseries`, `ScmRun.long_data` and
    `ScmRun.lineplot` to give greater control of the time axis when
    retrieving data
-   ([#54](https://github.com/openscm/scmdata/pull/54)) Add
    `drop_meta`{.interpreted-text role="meth"} to
    `ScmRun`{.interpreted-text role="class"} for dropping metadata
    columns
-   ([#53](https://github.com/openscm/scmdata/pull/53)) Don\'t convert
    case of variable names written to file. No longer convert case of
    serialized dataframes
-   ([#51](https://github.com/openscm/scmdata/pull/51)) Refactor
    `relative_to_ref_period_mean`{.interpreted-text role="meth"} so that
    it returns an instance of the input data type (rather than a
    `pd.DataFrame`{.interpreted-text role="obj"}) and puts the reference
    period in separate meta columns rather than mangling the variable
    name.
-   ([#47](https://github.com/openscm/scmdata/pull/47)) Update README
    and `setup.py` to make it easier for new users

## v0.4.3

-   ([#46](https://github.com/openscm/scmdata/pull/46)) Add test of
    conda installation

## v0.4.2

-   ([#45](https://github.com/openscm/scmdata/pull/45)) Make installing
    seaborn optional

## v0.4.1

-   ([#44](https://github.com/openscm/scmdata/pull/44)) Add
    multi-dimensional handling to `scmdata.netcdf`

-   ([#43](https://github.com/openscm/scmdata/pull/43)) Fix minor bugs
    in netCDF handling and address minor code coverage issues

-   ([#41](https://github.com/openscm/scmdata/pull/41)) Update
    documentation of the data model. Additionally:

    > -   makes `.time_points` atttributes consistently return
    >     `scmdata.time.TimePoints` instances
    > -   ensures `.meta` is used consistently throughout the code base
    >     (removing `.metadata`)

-   ([#33](https://github.com/openscm/scmdata/pull/33)) Remove
    dependency on [pyam](https://github.com/IAMconsortium/pyam).
    Plotting is done with [seaborn](https://github.com/mwaskom/seaborn)
    instead.

-   ([#34](https://github.com/openscm/scmdata/pull/34)) Allow the
    serialization/deserialization of `scmdata.run.ScmRun` and
    `scmdata.ScmDataFrame` as netCDF4 files.

-   ([#30](https://github.com/lewisjared/scmdata/pull/30)) Swap to using
    [openscm-units](https://github.com/openscm/openscm-units) for unit
    handling (hence remove much of the `scmdata.units` module)

-   ([#21](https://github.com/openscm/scmdata/pull/21)) Added
    `scmdata.run.ScmRun` as a proposed replacement for
    `scmdata.dataframe.ScmDataFrame`. This new class provides an
    identical interface as a `ScmDataFrame`, but uses a different
    underlying data structure to the `ScmDataFrame`. The purpose of
    `ScmRun` is to provide performance improvements when handling large
    sets of time-series data. Removed support for Python 3.5 until
    [pyam]{.title-ref} dependency is optional

-   ([#31](https://github.com/openscm/scmdata/pull/31)) Tidy up
    repository after changing location

## v0.4.0

-   ([#28](https://github.com/openscm/scmdata/pull/28)) Expose
    `scmdata.units.unit_registry`

## v0.3.1

-   ([#25](https://github.com/openscm/scmdata/pull/25)) Make scipy an
    optional dependency
-   ([#24](https://github.com/openscm/scmdata/pull/24)) Fix missing
    \"N2O\" unit (see
    [#14](https://github.com/openscm/scmdata/pull/14)). Also updates
    test of year to day conversion, it is 365.25 to within 0.01% (but
    depends on the Pint release).

## v0.3.0

-   ([#20](https://github.com/openscm/scmdata/pull/20)) Add support for
    python=3.5
-   ([#19](https://github.com/openscm/scmdata/pull/19)) Add support for
    python=3.6

## v0.2.2

-   ([#16](https://github.com/openscm/scmdata/pull/16)) Only rename
    columns when initialising data if needed

## v0.2.1

-   ([#13](https://github.com/openscm/scmdata/pull/13)) Ensure `LICENSE`
    is included in package
-   ([#11](https://github.com/openscm/scmdata/pull/11)) Add SO2F2 unit
    and update to Pyam v0.3.0
-   ([#12](https://github.com/openscm/scmdata/pull/12)) Add
    `get_unique_meta` convenience method
-   ([#10](https://github.com/openscm/scmdata/pull/10)) Fix
    extrapolation bug which prevented any extrapolation from occuring

## v0.2.0

-   ([#9](https://github.com/openscm/scmdata/pull/9)) Add `time_mean`
    method
-   ([#8](https://github.com/openscm/scmdata/pull/8)) Add `make docs`
    target

## v0.1.2

-   ([#7](https://github.com/openscm/scmdata/pull/7)) Add notebook tests
-   ([#4](https://github.com/openscm/scmdata/pull/4)) Unit conversions
    for CH4 and N2O contexts now work for compound units (e.g. \'Mt CH4
    / yr\' to \'Gt C / day\')
-   ([#6](https://github.com/openscm/scmdata/pull/6)) Add
    auto-formatting

## v0.1.1

-   ([#5](https://github.com/openscm/scmdata/pull/5)) Add
    `scmdata.dataframe.df_append` to `__init__.py`

## v0.1.0

-   ([#3](https://github.com/openscm/scmdata/pull/3)) Added
    documentation for the api and Makefile targets for releasing
-   ([#2](https://github.com/openscm/scmdata/pull/2)) Refactored
    scmdataframe from <openclimatedata/openscm@077f9b5> into a
    standalone package
-   ([#1](https://github.com/openscm/scmdata/pull/1)) Add docs folder
