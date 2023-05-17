# ---
# jupyter:
#   jupytext:
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
# # Time operations
#
# Time operations are notoriously difficult. In this notebook we go through some of scmdata's time operation capabilities.

# %% [markdown]
# ## Imports

# %%

import datetime as dt
import traceback

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

import scmdata.errors
import scmdata.time
from scmdata import ScmRun, run_append

register_matplotlib_converters()


# %% [markdown]
# ## Data
#
# Here we use the RCP26 emissions data. This originally came from http://www.pik-potsdam.de/~mmalte/rcps/ and has since been re-written into a format which can be read by scmdata using the [pymagicc](https://github.com/openclimatedata/pymagicc) library. We are not currently planning on importing Pymagicc's readers into scmdata by default, please raise an issue [here](https://github.com/openscm/scmdata/issues) if you would like us to consider doing so.

# %%
var_to_plot = "Emissions|BC"

rcp26 = ScmRun("rcp26_emissions.csv")
rcp26["time operation"] = "raw"

# %%
rcp26.filter(variable=var_to_plot).lineplot(hue="time operation")

# %% [markdown]
# For illustrative purposes, we shift the time points of the raw data before moving on.

# %%
rcp26["time"] = rcp26["time"].map(lambda x: dt.datetime(x.year, 3, 17))
rcp26 = ScmRun(rcp26)
rcp26.head()

# %% [markdown]
# ## Resampling
#
# The first method to consider is `resample`. This allows us to resample a dataframe onto different timesteps. Below, we resample the data onto monthly timesteps.

# %%
rcp26_monthly = rcp26.resample("MS")
rcp26_monthly["time operation"] = "start of month"
rcp26_monthly.head()

# %% [markdown]
# We can also resample to e.g. start of year or end of year.

# %%
rcp26_end_of_year = rcp26.resample("A")
rcp26_end_of_year["time operation"] = "end of year"
rcp26_end_of_year.head()

# %%
rcp26_start_of_year = rcp26.resample("AS")
rcp26_start_of_year["time operation"] = "start of year"
rcp26_start_of_year.head()

# %% [markdown]
# ## Interpolating
#
# Not all time points are supported by resampling. If we want to use custom time points (e.g. middle of year), we can do that with interpolate.

# %%
rcp26_middle_of_year = rcp26.interpolate(
    target_times=sorted(
        [dt.datetime(v, 7, 1) for v in set([v.year for v in rcp26["time"]])]
    )
)
rcp26_middle_of_year["time operation"] = "middle of year"
rcp26_middle_of_year.head()

# %% [markdown]
# ## Extrapolating
#
# Extrapolating is also supported by scmdata.

# %%
rcp26_extrap = rcp26.interpolate(
    target_times=sorted([dt.datetime(v, 7, 1) for v in range(1700, 2551)])
)
rcp26_extrap["time operation"] = "extrapolated"
rcp26_extrap.head()

# %%
rcp26_extrap_const = rcp26.interpolate(
    target_times=sorted([dt.datetime(v, 7, 1) for v in range(1700, 2551)]),
    extrapolation_type="constant",
)
rcp26_extrap_const["time operation"] = "extrapolated constant"
rcp26_extrap_const.head()

# %%
rcp26.head()

# %%
rcp26_extrap.head()

# %%
rcp26_extrap_const.head()

# %%

pdf = run_append([rcp26, rcp26_extrap, rcp26_extrap_const])

pdf.filter(variable=var_to_plot).lineplot(hue="time operation")

# %% [markdown]
# If we try to extrapolate beyond our source data but set `extrapolation_type=None`, we will receive an `InsufficientDataError`.

# %%
try:
    rcp26.interpolate(
        target_times=sorted([dt.datetime(v, 7, 1) for v in range(1700, 2551)]),
        extrapolation_type=None,
    )
except scmdata.time.InsufficientDataError:
    traceback.print_exc(limit=0, chain=False)

# %% [markdown]
# Generally the `interpolate` requires at minimum 3 times in order to perform any interpolation/extrapolation, otherwise an `InsufficientDataError` is raised. There is a special case where `constant` extrapolation can be used on a single time-step.

# %%
rcp26_yr2000 = rcp26.filter(year=2000)
rcp26_extrap_const_single = rcp26_yr2000.interpolate(
    target_times=sorted([dt.datetime(v, 7, 1) for v in range(1700, 2551)]),
    extrapolation_type="constant",
)
rcp26_extrap_const_single["time operation"] = "extrapolated constant"
rcp26_extrap_const_single.head()

# %% [markdown]
# ## Time means
#
# With monthly data, we can then take time means. Most of the time we just want to take the annual mean. This can be done as shown below.

# %% [markdown]
# ### Annual mean

# %%
rcp26_annual_mean = rcp26_monthly.time_mean("AC")
rcp26_annual_mean["time operation"] = "annual mean"
rcp26_annual_mean.head()

# %% [markdown]
# As the data is an annual mean, we put it in July 1st (which is more or less the centre of the year).

# %% [markdown]
# ### Annual mean centred on January 1st
#
# Sometimes we want to take annual means centred on January 1st, rather than the middle of the year. This can be done as shown.

# %%
rcp26_annual_mean_jan_1 = rcp26_monthly.time_mean("AS")
rcp26_annual_mean_jan_1["time operation"] = "annual mean Jan 1"
rcp26_annual_mean_jan_1.head()

# %% [markdown]
# As the data is centred on January 1st, we put it in January 1st.

# %% [markdown]
# ### Annual mean centred on December 31st
#
# Sometimes we want to take annual means centred on December 31st, rather than the middle of the year. This can be done as shown.

# %%
rcp26_annual_mean_dec_31 = rcp26_monthly.time_mean("A")
rcp26_annual_mean_dec_31["time operation"] = "annual mean Dec 31"
rcp26_annual_mean_dec_31.head()

# %% [markdown]
# As the data is centred on December 31st, we put it in December 31st.

# %% [markdown]
# ## Comparing the results
#
# We can compare the impact of these different methods with a plot as shown below.

# %%

var_to_plot = "Emissions|CF4"
pdf = run_append(
    [
        rcp26,
        rcp26_monthly,
        rcp26_start_of_year,
        rcp26_middle_of_year,
        rcp26_end_of_year,
        rcp26_annual_mean,
        rcp26_annual_mean_jan_1,
        rcp26_annual_mean_dec_31,
    ]
)

fig = plt.figure(figsize=(16, 9))

ax = fig.add_subplot(121)
pdf.filter(variable=var_to_plot).lineplot(ax=ax, hue="time operation")

ax = fig.add_subplot(122)
pdf.filter(variable=var_to_plot, year=range(1998, 2001)).lineplot(
    ax=ax, hue="time operation"
)

plt.tight_layout()

# %% [markdown]
# When the timeseries is particularly noisy, the different operations result in slightly different timeseries. For example, shifting to start of month smooths the data a bit (as you're interpolating and resampling the underlying data) while taking means centred on different points in time changes your mean as you take different windows of your monthly data.

# %%

fig = plt.figure(figsize=(16, 9))

ax = fig.add_subplot(121)
pdf.filter(variable=var_to_plot).lineplot(ax=ax, hue="time operation")

ax = fig.add_subplot(122)
pdf.filter(variable=var_to_plot, year=range(1998, 2001)).lineplot(
    ax=ax, hue="time operation", legend=False
)

plt.tight_layout()

# %% [markdown]
# The lines above don't match the underlying timeseries e.g. the monthly data minimum is in the wrong place.

# %%

rcp26_monthly.filter(
    variable=var_to_plot, year=range(1998, 2001), month=[2, 3, 4, 5]
).timeseries()

# %%

pdf.filter(variable=var_to_plot, year=range(1998, 2001)).timeseries().T.plot(
    figsize=(16, 9)
)

# %%
pdf.filter(variable=var_to_plot, year=range(1998, 2001)).timeseries().T.sort_index()
