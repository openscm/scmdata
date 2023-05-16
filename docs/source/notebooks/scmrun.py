# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ScmRun
#
# *Suggestions for update:* add examples of handling of timeseries interpolation plus how the guessing works
#
# In this notebook we provide an overview of the capabilities provided by scmdata's `ScmRun` class. `ScmRun` provides a efficient interface to analyse timeseries data.
# ## Imports
# %% tags=["remove-stdout", "remove-stderr"]
import traceback

import numpy as np
from openscm_units import unit_registry as ur
from pint.errors import DimensionalityError

from scmdata import ScmRun
from scmdata.errors import NonUniqueMetadataError

# %% [markdown]
# ## Loading data
#
# `ScmRun`'s can read many different data types and be loaded in many different ways.
# For a full explanation, see the docstring of `ScmRun`'s `__init__` method.

# %%
print(ScmRun.__init__.__doc__)

# %% [markdown]
# Here we load data from a file.
#
# *Note:* here we load RCP26 emissions data. This originally came from http://www.pik-potsdam.de/~mmalte/rcps/ and has since been re-written into a format which can be read by scmdata using the [pymagicc](https://github.com/openclimatedata/pymagicc) library. We are not currently planning on importing Pymagicc's readers into scmdata by default, please raise an issue [here](https://github.com/openscm/scmdata/issues) if you would like us to consider doing so.

# %%
rcp26 = ScmRun("rcp26_emissions.csv", lowercase_cols=True)

# %% [markdown]
# ## Timeseries
#
# `ScmDataFrame` is ideally suited to working with timeseries data.
# The `timeseries` method allows you to easily get the data back in wide format as a *pandas* `DataFrame`.
# Here 'wide' format refers to representing timeseries as a row with metadata being contained in the row labels.

# %%
rcp26.timeseries().head()

# %%
type(rcp26.timeseries())

# %% [markdown]
# ## Operations with scalars
#
# Basic operations with scalars are easily performed.

# %%
rcp26.head()

# %%
(rcp26 + 2).head()

# %%
(rcp26 / 4).head()

# %% [markdown]
# `ScmRun` instances also support operations with [Pint](https://github.com/hgrecco/pint) scalars, permitting automatic unit conversion and error raising. For interested readers, the scmdata package uses the [OpenSCM-Units](https://openscm-units.readthedocs.io/) unit registry.

# %%
to_add = 500 * ur("MtCO2 / yr")

# %% [markdown]
# If we try to add 0.5 GtC / yr to all the timeseries, we'll get a `DimensionalityError`.

# %%
try:
    rcp26 + to_add
except DimensionalityError:
    traceback.print_exc(limit=0, chain=False)

# %% [markdown]
# However, if we filter things correctly, this operation is perfectly valid.

# %%
(rcp26.filter(variable="Emissions|CO2|MAGICC AFOLU") + to_add).head()

# %% [markdown]
# This can be compared to the raw data as shown below.

# %%
rcp26.filter(variable="Emissions|CO2|MAGICC AFOLU").head()

# %% [markdown]
# ## Unit conversion
#
# The scmdata package uses the [OpenSCM-Units](https://openscm-units.readthedocs.io/) unit registry and uses the [Pint](https://github.com/hgrecco/pint) library to handle unit conversion.
#
# Calling the `convert_unit` method of an `ScmRun` returns a new `ScmRun` instance with converted units.

# %%
rcp26.filter(variable="Emissions|BC").timeseries()

# %%
rcp26.filter(variable="Emissions|BC").convert_unit("kg BC / day").timeseries()

# %% [markdown]
# Note that you must filter your data first as the unit conversion is applied to all available variables. If you do not, you will receive `DimensionalityError`'s.

# %%
try:
    rcp26.convert_unit("kg BC / day").timeseries()
except DimensionalityError:
    traceback.print_exc(limit=0, chain=False)

# %% [markdown]
# Having said this, thanks to Pint's idea of contexts, we are able to trivially convert to CO<sub>2</sub> equivalent units (as long as we restrict our conversion to variables which have a CO<sub>2</sub> equivalent).

# %%
rcp26.filter(variable=["*CO2*", "*CH4*", "*N2O*"]).timeseries()

# %%

rcp26.filter(variable=["*CO2*", "*CH4*", "*N2O*"]).convert_unit(
    "Mt CO2 / yr", context="AR4GWP100"
).timeseries()

# %% [markdown]
# Without the context, a `DimensionalityError` is once again raised.

# %%
try:
    rcp26.convert_unit("Mt CO2 / yr").timeseries()
except DimensionalityError:
    traceback.print_exc(limit=0, chain=False)

# %% [markdown]
# In addition, when we do a conversion with contexts, the context information is automatically added to the metadata. This ensures we can't accidentally use a different context for further conversions.

# %%
ar4gwp100_converted = rcp26.filter(variable=["*CO2*", "*CH4*", "*N2O*"]).convert_unit(
    "Mt CO2 / yr", context="AR4GWP100"
)
ar4gwp100_converted.timeseries()

# %% [markdown]
# Trying to convert without a context, or with a different context, raises an error.

# %%
try:
    ar4gwp100_converted.convert_unit("Mt CO2 / yr")
except ValueError:
    traceback.print_exc(limit=0, chain=False)

# %%
try:
    ar4gwp100_converted.convert_unit("Mt CO2 / yr", context="AR5GWP100")
except ValueError:
    traceback.print_exc(limit=0, chain=False)

# %% [markdown]
# ## Metadata handling
#
# Each timeseries within an `ScmRun` object has metadata associated with it. The `meta` attribute provides the `Timeseries` specific metadata of the timeseries as a `pd.DataFrame`. This DataFrame is effectively the `index` of the `ScmRun.timeseries()` function.
#
# This `Timeseries` specific metadata can be modified using the `[]` notation which modify the metadata inplace or alternatively using the `set_meta` function which returns a new `ScmRun` with updated metadata. `set_meta` also makes it easy to update a subset of timeseries.

# %%
ar4gwp100_converted.meta

# %%
# Update inplace
ar4gwp100_converted["unit_context"] = "inplace"
ar4gwp100_converted["unit_context"]

# %%
# set_meta returns a new `ScmRun` with the updated metadata
ar4gwp100_converted.set_meta(
    "unit_context", "updated-in-set_meta", variable="Emissions|CO2|*"
)

# %%
# The original `ScmRun` was not modified by `set_meta`
ar4gwp100_converted

# %% [markdown]
# `ScmRun` instances are strict with respect to metadata handling. If you either try to either a) instantiate an `ScmRun` instance with duplicate metadata or b) change an existing `ScmRun` instance so that it has duplicate metadata then you will receive a `NonUniqueMetadataError`.

# %%
try:
    ScmRun(
        data=np.arange(6).reshape(2, 3),
        index=[10, 20],
        columns={
            "variable": "Emissions",
            "unit": "Gt",
            "model": "idealised",
            "scenario": "idealised",
            "region": "World",
        },
    )
except NonUniqueMetadataError:
    traceback.print_exc(limit=0, chain=False)

# %%

try:
    rcp26["variable"] = "Emissions|CO2|MAGICC AFOLU"
except NonUniqueMetadataError:
    traceback.print_exc(limit=0, chain=False)

# %% [markdown]
# There is also a `metadata` attribute which provides metadata for the `ScmRun` instance.
#
# These metadata can be used to store information about the collection of runs as a whole, such as the file where the data are stored or longer-form information about a particular dataset.

# %%
rcp26.metadata["filename"] = "rcp26_emissions.csv"
rcp26.metadata

# %% [markdown]
# ## Convenience methods
#
# Below we showcase a few convenience methods of `ScmRun`. These will grow over time, please add a pull request adding more where they are useful!

# %% [markdown]
# ### get_unique_meta
#
# This method helps with getting the unique metadata values in an `ScmRun`. Here we show how it can be useful. Check out its docstring for full details.

# %% [markdown]
# By itself, it doesn't do anything special, just returns the unique metadata values as a list.

# %%
rcp26.get_unique_meta("variable")

# %% [markdown]
# However, it can be useful if you expect there to only be one unique metadata value. In such a case, you can use the `no_duplicates` argument to ensure that you only get a single value as its native type (not a list) and that an error will be raised if this isn't the case.

# %%
rcp26.get_unique_meta("model", no_duplicates=True)

# %%
try:
    rcp26.get_unique_meta("unit", no_duplicates=True)
except ValueError:
    traceback.print_exc(limit=0, chain=False)

# %%
