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
# # Processing
#
# scmdata has some support for processing `ScmRun` instances to calculate statistics
# of interest. Here we provide examples of how to use them.
#
# At present, we can calculate:
#
# - crossing times (e.g. 1.5C crossing times)
# - crossing time quantiles
# - exceedance probabilities
# - peak
# - peak year
# - categorisation in line with SR1.5
# - a set of summary variables

# %% [markdown]
# ## Load some data
#
# For this demonstration, we are going to use MAGICC output from [RCMIP Phase 2]
# as available at [https://zenodo.org/record/4624566/files/data-processed-submission-database-hadcrut5-target-MAGICCv7.5.1.tar.gz?download=1](). Here we have just extracted the air temperature output for the SSPs from 1995 to 2100.

# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import scmdata.processing
from scmdata import ScmRun, run_append

# %%

magicc_output = ScmRun("magicc-rcmip-phase-2-gsat-output.csv")
magicc_output

# %% [markdown]
# ## Crossing times
#
# The first thing we do is show how to calculate the crossing times of a given threshold.

# %%

crossing_time_15 = scmdata.processing.calculate_crossing_times(
    magicc_output,
    threshold=1.5,
)
crossing_time_15

# %% [markdown]
# The output is a `pd.Series`, which is useful for many other pieces of work.
#
# For example, we could make a plot with e.g. seaborn.

# %%
label = "1.5C crossing time"
pdf = crossing_time_15.reset_index().rename({0: label}, axis="columns")
sns.histplot(data=pdf, x=label, hue="scenario")

# %%
label = "2.0C crossing time"
crossing_time_20 = scmdata.processing.calculate_crossing_times(
    magicc_output,
    threshold=2.0,
)

pdf = crossing_time_20.reset_index().rename({0: label}, axis="columns")
sns.histplot(data=pdf, x=label, hue="scenario")

# %% [markdown]
# ### Crossing time quantiles
#
# Calculating the quantiles of crossing times is a bit fiddly because some
# ensemble members will not cross the threshold at all. We show how quantiles
# can be calculated sensibly below. The calculation will return nan if that
# quantile in the crossing times corresponds to an ensemble member which never
# crosses the threshold.

# %%
scmdata.processing.calculate_crossing_times_quantiles(
    crossing_time_15,
    groupby=["climate_model", "model", "scenario"],
    quantiles=(0.05, 0.17, 0.5, 0.83, 0.95),
)

# %% [markdown]
# In the above, we can see that the 83rd percentile of crossing times is in fact
# to not cross at all under the SSP1-1.9 scenario.

# %% [markdown]
# ### Datetime output
#
# If desired, data could be interpolated first before calculating the crossing
# times. In such cases, returning the output as datetime rather than year might be helpful.

# %%
scmdata.processing.calculate_crossing_times(
    magicc_output.resample("MS"),
    threshold=2.0,
    return_year=False,
)

# %% [markdown]
# ## Exceedance probabilities
#
# Next we show how to calculate exceedance probabilities.

# %%
exceedance_probability_2C = scmdata.processing.calculate_exceedance_probabilities(
    magicc_output,
    process_over_cols=["ensemble_member", "variable"],
    threshold=2.0,
)
exceedance_probability_2C

# %% [markdown]
#
# We can make a plot to compare exceedance probabilities over multiple scenarios.

# %%
exceedance_probability_15C = scmdata.processing.calculate_exceedance_probabilities(
    magicc_output,
    process_over_cols=["ensemble_member", "variable"],
    threshold=1.5,
)

pdf = (
    pd.DataFrame(
        [
            exceedance_probability_2C,
            exceedance_probability_15C,
        ]
    )
    .T.melt(ignore_index=False, value_name="Exceedance probability")
    .reset_index()
)
display(pdf)

ax = sns.barplot(data=pdf, x="scenario", y="Exceedance probability", hue="variable")
ax.tick_params(labelrotation=30)
ax.set_ylim([0, 1])
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="threshold")
plt.tight_layout()

# %% [markdown]
# ## Exceedance probabilities over time
#
# It is also possible to calculate exceedance probabilities over time.

# %%
res = scmdata.processing.calculate_exceedance_probabilities_over_time(
    magicc_output,
    process_over_cols="ensemble_member",
    threshold=1.5,
)
res = scmdata.ScmRun(res)
res.lineplot(style="variable")

# %% [markdown]
# Note that taking the maximum exceedance probability over all time will be less
# than or equal to the exceedance probability calculated with
# `calculate_exceedance_probabilities` because the order of operations matters:
# calculating whether each ensemble member exceeds the threshold or not then
# seeing how many ensemble members out of the total exceed the threshold is not
# the same as seeing how many ensemble members exceed the threshold at each
# timestep and then taking the maximum over all timesteps. In general, taking
# the maximum value from `calculate_exceedance_probabilities_over_time` will be
# less than or equal to the results of `calculate_exceedance_probabilities`,
# as demonstrated below.

# %%
comparison = (
    pd.DataFrame(
        {
            "calculate_exceedance_probabilities": exceedance_probability_15C,
            "max of calculate_exceedance_probabilities_over_time": (
                res.timeseries(meta=exceedance_probability_15C.index.names).max(axis=1)
            ),
        }
    )
    * 100
)
comparison.round(1)

# %% [markdown]
# ## Peak
#
# We can calculate the peaks in each timeseries.

# %%
peak_warming = scmdata.processing.calculate_peak(magicc_output)
peak_warming

# %% [markdown]
# From this we can then calculate median peak warming by scenario (or other quantiles).

# %%
peak_warming.groupby(["model", "scenario"]).median()

# %% [markdown]
# Or make a plot.

# %%
label = "Peak warming"
pdf = peak_warming.reset_index().rename({0: label}, axis="columns")
sns.histplot(data=pdf, x=label, hue="scenario")

# %% [markdown]
# Similarly to exceedance probabilties, the order of operations matters: calculating the median of the peaks is different to calculating the median then taking the peak of this median timeseries. In general, the max of the median timeseries is less than or equal to the median of the peak in each timeseries.

# %%
comparison = pd.DataFrame(
    {
        "median of peak warming": peak_warming.groupby(["model", "scenario"]).median(),
        "max of median timeseries": (
            magicc_output.process_over(
                list(set(magicc_output.meta.columns) - {"model", "scenario"}),
                "median",
            ).max(axis=1)
        ),
    }
)
comparison.round(3)

# %% [markdown]
# ## Peak time
#
# We can calculate the peak time in each timeseries.

# %%
peak_warming_year = scmdata.processing.calculate_peak_time(magicc_output)
peak_warming_year

# %% [markdown]
# From this we can then calculate median peak warming time by scenario (or other quantiles).

# %%
peak_warming_year.groupby(["model", "scenario"]).median()

# %%
label = "Peak warming year"
pdf = peak_warming_year.reset_index().rename({0: label}, axis="columns")
ax = sns.histplot(data=pdf, x=label, hue="scenario", bins=np.arange(2025, 2100 + 1))

# %% [markdown]
# ## SR1.5 categorisation
#
# It is also possible to categorise the scenarios using the same categorisation as in SR1.5. To do this we have to first calculate the appropriate quantiles.

# %%
magicc_output_categorisation_quantiles = scmdata.ScmRun(
    magicc_output.quantiles_over("ensemble_member", quantiles=[0.33, 0.5, 0.66])
)
magicc_output_categorisation_quantiles

# %%
scmdata.processing.categorisation_sr15(
    magicc_output_categorisation_quantiles, ["climate_model", "scenario"]
)

# %% [markdown]
# ## Set of summary variables
#
# It is also possible to calculate a set of summary variables using the convenience function `calculate_summary_stats`. The documentation is given below.

# %%
print(scmdata.processing.calculate_summary_stats.__doc__)

# %% [markdown]
# It can be used to calculate summary statistics as shown below.

# %%
summary_stats = scmdata.processing.calculate_summary_stats(
    magicc_output,
    ["climate_model", "model", "scenario", "region"],
    exceedance_probabilities_thresholds=np.arange(1, 2.51, 0.1),
    exceedance_probabilities_variable="Surface Air Temperature Change",
    exceedance_probabilities_naming_base="Exceedance Probability|{:.2f}K",
    peak_quantiles=[0.05, 0.5, 0.95],
    peak_variable="Surface Air Temperature Change",
    peak_naming_base="Peak Surface Air Temperature Change|{}th quantile",
    peak_time_naming_base="Year of peak Surface Air Temperature Change|{}th quantile",
    peak_return_year=True,
    progress=True,
)
summary_stats

# %% [markdown]
# We can then use pandas to create summary tables of interest.

# %%
summary_stats.unstack(["climate_model", "statistic", "unit"])

# %%

index = ["climate_model", "scenario"]
pivot_merge_unit = summary_stats.to_frame().reset_index()
pivot_merge_unit["statistic"] = pivot_merge_unit["statistic"] + pivot_merge_unit[
    "unit"
].apply(lambda x: "({})".format(x) if x else "")
pivot_merge_unit = pivot_merge_unit.drop("unit", axis="columns")
pivot_merge_unit = pivot_merge_unit.set_index(
    list(set(pivot_merge_unit.columns) - {"value"})
).unstack("statistic")
pivot_merge_unit
