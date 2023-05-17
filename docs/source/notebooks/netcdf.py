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
# # NetCDF handling
#
# NetCDF formatted files are much faster to read and write for large datasets.
# In order to make the most of this, the `ScmRun` objects have the ability to
# read and write netCDF files.

# %%

import traceback
from glob import glob

import numpy as np
import seaborn as sns
import xarray as xr

from scmdata.netcdf import nc_to_run
from scmdata.run import ScmRun, run_append

# %% [markdown]
# ## Helper bits and piecs

# %%
OUT_FNAME = "/tmp/out_runs.nc"


# %%
def new_timeseries(
    n=100,
    count=1,
    model="example",
    scenario="ssp119",
    variable="Surface Temperature",
    unit="K",
    region="World",
    cls=ScmRun,
    **kwargs,
):
    data = np.random.rand(n, count) * np.arange(n)[:, np.newaxis]
    index = 2000 + np.arange(n)
    return cls(
        data,
        columns={
            "model": model,
            "scenario": scenario,
            "variable": variable,
            "region": region,
            "unit": unit,
            **kwargs,
        },
        index=index,
    )


# %% [markdown]
# Let's create an `ScmRun` which contains a few variables and a number of runs.
# Such a dataframe would be used to store the results from an ensemble of
# simple climate model runs.

# %%

runs = run_append(
    [
        new_timeseries(
            count=3,
            variable=[
                "Surface Temperature",
                "Atmospheric Concentrations|CO2",
                "Radiative Forcing",
            ],
            unit=["K", "ppm", "W/m^2"],
            run_id=run_id,
        )
        for run_id in range(10)
    ]
)
runs.metadata["source"] = "fake data"
runs

# %% [markdown]
# ## Reading/Writing to NetCDF4

# %% [markdown]
# ### Basics
#
# Writing the runs to disk is easy. The one trick is that each variable and dimension combination must have unique metadata. If they do not, you will receive an error message like the below.

# %%
try:
    runs.to_nc(OUT_FNAME, dimensions=["region"])
except ValueError:
    traceback.print_exc(limit=0, chain=False)

# %% [markdown]
# In our dataset, there is more than one "run_id" per variable hence we need to use a different dimension, `run_id`, because this will result in each variable's remaining metadata being unique.

# %%
runs.to_nc(OUT_FNAME, dimensions=["run_id"])

# %% [markdown]
# The output netCDF file can be read using the `from_nc` method, `nc_to_run` function or directly using `xarray`.

# %%

runs_netcdf = ScmRun.from_nc(OUT_FNAME)
runs_netcdf

# %%

nc_to_run(ScmRun, OUT_FNAME)

# %%

xr.load_dataset(OUT_FNAME)

# %% [markdown]
# The additional `metadata` in `runs` is also serialized and deserialized in the netCDF files. The `metadata` of the loaded `ScmRun` will also contain some additional fields about the file creation.

# %%

assert "source" in runs_netcdf.metadata
runs_netcdf.metadata

# %% [markdown]
# ### Splitting your data
#
# Sometimes if you have complicated ensemble runs it might be more efficient to split the data into smaller subsets.
#
# In the below example we iterate over scenarios to produce a netCDF file per scenario.

# %%
large_run = []

# 10 runs for each scenario
for sce in ["ssp119", "ssp370", "ssp585"]:
    large_run.extend(
        [
            new_timeseries(
                count=3,
                scenario=sce,
                variable=[
                    "Surface Temperature",
                    "Atmospheric Concentrations|CO2",
                    "Radiative Forcing",
                ],
                unit=["K", "ppm", "W/m^2"],
                paraset_id=paraset_id,
            )
            for paraset_id in range(10)
        ]
    )

large_run = run_append(large_run)

# also set a run_id (often we'd have paraset_id and run_id,
# one which keeps track of the parameter set we've run and
# the other which keeps track of the run in a large ensemble)
large_run["run_id"] = large_run.meta.index.values
large_run

# %% [markdown]
# Data for each scenario can then be loaded independently instead of having to load all the data and then filtering

# %%
for sce_run in large_run.groupby("scenario"):
    sce = sce_run.get_unique_meta("scenario", True)
    sce_run.to_nc(
        "/tmp/out-{}-sparse.nc".format(sce),
        dimensions=["run_id", "paraset_id"],
    )

# %%

ScmRun.from_nc("/tmp/out-ssp585-sparse.nc").filter("Surface Temperature").line_plot()

# %% [markdown]
# For such a data set, since both `run_id` and `paraset_id` vary, both could be added as dimensions in the file.
#
# The one problem with this approach is that you get very sparse arrays because the data is written on a 100 x 30 x 90 (time points x paraset_id x run_id) grid but there's only 90 timeseries so you end up with 180 timeseries worth of nans (although this is a relatively small problem because the netCDF files use compression to minismise the impact of the extra nan values).

# %%

xr.load_dataset("/tmp/out-ssp585-sparse.nc")

# %%

# Load all scenarios
run_append([ScmRun.from_nc(fname) for fname in glob("/tmp/out-ssp*-sparse.nc")])

# %% [markdown]
# An alternative to the sparse arrays is to specify the variables in the `extras` attribute. If possible, this adds the metadata to the netCDF file as an extra co-ordinate, which uses one of the dimensions as it's co-ordinate. If using one of the dimensions as a co-ordinate would not specify the metadata uniquely, we add the extra as an additional co-ordinate, which itself has co-ordinates of `_id`. This `_id` co-ordinate provides a unique mapping between the extra metadata and the timeseries.

# %%
for sce_run in large_run.groupby("scenario"):
    sce = sce_run.get_unique_meta("scenario", True)
    sce_run.to_nc(
        "/tmp/out-{}-extras.nc".format(sce),
        dimensions=["run_id"],
        extras=["paraset_id"],
    )

# %% [markdown]
# `paraset_id` is uniquely defined by `run_id` so we don't end up with an extra `_id` co-ordinate.

# %%

xr.load_dataset("/tmp/out-ssp585-extras.nc")

# %%

ScmRun.from_nc("/tmp/out-ssp585-extras.nc").filter("Surface Temperature").line_plot()

# %% [markdown]
# If we use dimensions and extra such that our extra co-ordinates are not uniquely defined by the regions, an `_id` dimension is automatically added to ensure we don't lose any information.

# %%
large_run.to_nc(
    "/tmp/out-extras-sparse.nc",
    dimensions=["scenario"],
    extras=["paraset_id", "run_id"],
)

# %%

xr.load_dataset("/tmp/out-extras-sparse.nc")

# %% [markdown]
# ### Multi-dimensional data
#
# **scmdata** can also handle having more than one dimension. This can be especially helpful if you have output from a number of models (IAMs), scenarios, regions and runs.

# %%
multi_dimensional_run = []

for model in ["AIM", "GCAM", "MESSAGE", "REMIND"]:
    for sce in ["ssp119", "ssp370", "ssp585"]:
        for region in ["World", "R5LAM", "R5MAF", "R5ASIA", "R5OECD", "R5REF"]:
            multi_dimensional_run.extend(
                [
                    new_timeseries(
                        count=3,
                        model=model,
                        scenario=sce,
                        region=region,
                        variable=[
                            "Surface Temperature",
                            "Atmospheric Concentrations|CO2",
                            "Radiative Forcing",
                        ],
                        unit=["K", "ppm", "W/m^2"],
                        paraset_id=paraset_id,
                    )
                    for paraset_id in range(10)
                ]
            )

multi_dimensional_run = run_append(multi_dimensional_run)

multi_dimensional_run

# %%
multi_dim_outfile = "/tmp/out-multi-dimensional.nc"

# %%
multi_dimensional_run.to_nc(
    multi_dim_outfile,
    dimensions=("region", "model", "scenario", "paraset_id"),
)

# %%

xr.load_dataset(multi_dim_outfile)

# %%

multi_dim_loaded_co2_conc = ScmRun.from_nc(multi_dim_outfile).filter(
    "Atmospheric Concentrations|CO2"
)

seaborn_df = multi_dim_loaded_co2_conc.long_data()
seaborn_df.head()

# %%

sns.relplot(
    data=seaborn_df,
    x="time",
    y="value",
    units="paraset_id",
    estimator=None,
    hue="scenario",
    style="model",
    col="region",
    col_wrap=3,
    kind="line",
)
