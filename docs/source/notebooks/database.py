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
# # ScmDatabase
#
# In this notebook, we provide an example of the `ScmDatabase` class. `ScmDatabase` helps read and write large bunches of timeseries data by splitting them up into multiple files on disk and allowing users to read/write selections at a time.
#
# This allows handling very large datasets which may exceed the amount of system memory a user has available.

# %%

import tempfile
import traceback

import numpy as np
import pandas as pd

from scmdata import ScmRun, run_append
from scmdata.database import ScmDatabase
from scmdata.errors import NonUniqueMetadataError

pd.set_option("display.width", 160)

# %% [markdown]
# ## Initialisation
#
# There are two main things to think about when creating a `ScmDatabase`. Namely:
#
# * Where the data is going to be stored (`root_dir`)
# * How the data will be split up (`levels`)
#
# When data is to be written to disk it is split into different files, each with a unique combination of metadata values. The `levels` option defines the metadata columns used to split up the data.
#
# Choosing an appropriate value for `levels` could play a large role in determining the performance of reading/writing. For example, if you were storing output from a number of different climate models, you may define `levels` as `["climate_model", "scenario", "variable", "region"]`. This would allow loading a particular variable and region, say `Surface Temperature` for the `World` region, from all climate models and scenarios without needing to load the other variables and regions. Specifying too many groups may result in slow writing if a very large number of database files are written.
#
# If you wish load a subset of a particular metadata dimension then it must be specified in this list.

# %%
print(ScmDatabase.__init__.__doc__)

# %%
temp_out_dir = tempfile.TemporaryDirectory()
database = ScmDatabase(temp_out_dir.name, levels=["climate_model", "scenario"])

# %%

database


# %% [markdown]
# ## Saving data
#
# Data can be added to the database using the `save_to_database` method. Subsequent calls merge new data into the database.


# %%
def create_timeseries(
    n=500,
    count=1,
    b_factor=1 / 1000,
    model="example",
    scenario="ssp119",
    variable="Surface Temperature",
    unit="K",
    region="World",
    **kwargs,
):
    a = np.random.rand(count)
    b = np.random.rand(count) * b_factor
    data = a + np.arange(n)[:, np.newaxis] ** 2 * b
    index = 2000 + np.arange(n)
    return ScmRun(
        data,
        columns={
            "model": model,
            "scenario": scenario,
            "variable": variable,
            "region": region,
            "unit": unit,
            "ensemble_member": range(count),
            **kwargs,
        },
        index=index,
    )


# %%
runs_low = run_append(
    [
        create_timeseries(
            scenario="low",
            climate_model="model_a",
            count=10,
            b_factor=1 / 1000,
        ),
        create_timeseries(
            scenario="low",
            climate_model="model_b",
            count=10,
            b_factor=1 / 1000,
        ),
    ]
)
runs_high = run_append(
    [
        create_timeseries(
            scenario="high",
            climate_model="model_a",
            count=10,
            b_factor=2 / 1000,
        ),
        create_timeseries(
            scenario="high",
            climate_model="model_b",
            count=10,
            b_factor=2 / 1000,
        ),
    ]
)

# %%

run_append([runs_low, runs_high]).line_plot(hue="scenario", style="climate_model")

# %%

database.save(runs_low)

# %%
database.available_data()

# %% [markdown]
# Internally, each row shown in `available_data()` is stored as a netCDF file in a directory structure following ``database.levels``.

# %%

# !pushd {temp_out_dir.name}; tree; popd

# %% [markdown]
# Additional calls to `save` will merge the new data into the database, creating any new files as required.
#
# If existing data is found, it is first loaded and merged with the saved data before writing to prevent losing existing data.

# %%

database.save(runs_high)

# %%
database.available_data()

# %% [markdown]
# These data still need unique metadata otherwise a `NonUniqueMetadataError` is raised.

# %%

try:
    database.save(runs_high)
except NonUniqueMetadataError:
    traceback.print_exc(limit=0, chain=False)

# %%

runs_high_extra = runs_high.copy()
runs_high_extra["ensemble_member"] = runs_high_extra["ensemble_member"] + 10
database.save(runs_high_extra)

# %% [markdown]
# ## Loading data
#
# When loading data we can select a subset of data, similar to `ScmRun.filter` but limited to filtering for the metadata columns as specified in `levels`

# %%

run = database.load(scenario="high")
run.meta

# %%

database.load(climate_model="model_b").meta

# %% [markdown]
# The entire dataset can also be loaded if needed. This may not be possible for very large datasets depending on the amount of system memory available.

# %%

all_data = database.load()
all_data.meta

# %%

all_data.line_plot(hue="scenario", style="climate_model")

# %%
temp_out_dir.cleanup()
