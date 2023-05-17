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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Summary statistics
#
# `ScmRun` objects have methods specific to calculating summary statistics. In this notebook we demonstrate them.
#
# At present, the following methods are available:
#
# - `process_over`
# - `quantiles_over`
# - `groupby`
# - `groupby_all_except`

# %%

import numpy as np
import pandas as pd

from scmdata.run import ScmRun, run_append

# %% [markdown]
# ## Helper bits and piecs


# %%
def new_timeseries(
    n=101,
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
# Let's create an `ScmRun` which contains a few variables and a number of runs. Such a dataframe would be used to store the results from an ensemble of simple climate model runs.

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
# ## `process_over`
#
# The `process_over` method allows us to calculate a specific set of statistics on groups of timeseries. A number of pandas functions can be called including "sum", "mean" and "describe".

# %%
print(runs.process_over.__doc__)

# %% [markdown]
# ### Mean

# %%

mean = runs.process_over(cols="run_id", operation="mean")
mean

# %% [markdown]
# ### Median

# %%

median = runs.process_over(cols="run_id", operation="median")
median


# %% [markdown]
# ### Arbitrary functions
#
# You are also able to run arbitrary functions for each group

# %%


def mean_and_invert(df, axis=0):
    # Take a mean across the group and then invert the result
    return -df.mean(axis=axis)


runs.process_over("run_id", operation=mean_and_invert)

# %%

runs.process_over("run_id", operation=mean_and_invert, axis=1)

# %% [markdown]
# ### Other quantiles

# %%

lower_likely_quantile = runs.process_over(cols="run_id", operation="quantile", q=0.17)
lower_likely_quantile

# %% [markdown]
# ## `quantiles_over`
#
# If you want to calculate more than one summary statistic, `quantiles_over` will calculate and label multiple summary statistics before returning them.

# %%
print(runs.quantiles_over.__doc__)

# %%

summary_stats = runs.quantiles_over(
    cols="run_id", quantiles=[0.05, 0.17, 0.5, 0.83, 0.95, "mean", "median"]
)
summary_stats

# %% [markdown]
# ### Plotting

# %% [markdown]
# #### Calculate quantiles within plotting function
#
# We can use `plumeplot` directly to plot quantiles. This will calculate the quantiles as part of making the plot so if you're doing this lots it might be faster to pre-calculate the quantiles, then make the plot instead (see below)

# %% [markdown]
# Note that in this case the default setttings in `plumeplot` don't produce anything that helpful, we show how to modify them in the cell below.

# %%

runs.plumeplot(quantile_over="run_id")

# %%

runs.plumeplot(
    quantile_over="run_id",
    quantiles_plumes=[
        ((0.05, 0.95), 0.2),
        ((0.17, 0.83), 0.5),
        (("median",), 1.0),
    ],
    hue_var="variable",
    hue_label="Variable",
    style_var="scenario",
    style_label="Scenario",
)

# %% [markdown]
# #### Pre-calculated quantiles
#
# Alternately, we can cast the output of `quantiles_over` to an `ScmRun` object for ease of filtering and plotting.

# %%

summary_stats_scmrun = ScmRun(summary_stats)
summary_stats_scmrun

# %% [markdown]
# As discussed above, casting the output of `quantiles_over` to an `ScmRun` object helps avoid repeatedly calculating the quantiles.

# %%

summary_stats_scmrun.plumeplot(
    quantiles_plumes=[
        ((0.05, 0.95), 0.2),
        ((0.17, 0.83), 0.5),
        (("median",), 1.0),
    ],
    hue_var="variable",
    hue_label="Variable",
    style_var="scenario",
    style_label="Scenario",
    pre_calculated=True,
)

# %% [markdown]
# If we don't want a plume plot, we can always our standard lineplot method.

# %%

summary_stats_scmrun.filter(variable="Radiative Forcing").lineplot(hue="quantile")

# %% [markdown]
# ## `groupby`
#
# The `groupby` method allows us to group the data by columns in `scmrun.meta` and then perform operations. An example is given below.

# %%

variable_means = []
for vdf in runs.groupby("variable"):
    vdf_mean = vdf.timeseries().mean(axis=0)
    vdf_mean.name = vdf.get_unique_meta("variable", True)
    variable_means.append(vdf_mean)

pd.DataFrame(variable_means)

# %% [markdown]
# ## `groupby_all_except`
#
# The `groupby_all_except` method allows us to group the data by all columns in `scmrun.meta` except for a certain set. Like with `groupby`, we can then use the groups to perform operations. An example is given below. Note that, in most cases, using `process_over` is likely to be more useful.

# %%

ensemble_means = []
for edf in runs.groupby_all_except("run_id"):
    edf_mean = edf.timeseries().mean(axis=0)
    edf_mean.name = edf.get_unique_meta("variable", True)
    ensemble_means.append(edf_mean)

pd.DataFrame(ensemble_means)

# %% [markdown]
# As we said, in most cases using `process_over` is likely to be more useful. For example the above can be done using `process_over` in one line (and more metadata is retained).

# %%

runs.process_over("run_id", "mean")
