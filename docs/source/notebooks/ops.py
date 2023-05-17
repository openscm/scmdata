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

# %%
import os
import traceback
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from scmdata import ScmRun, run_append

# %% [markdown]
# # Operations
#
# scmdata has limited support for operations with `ScmRun` instances. Here we provide examples of how to use them.

# %% [markdown]
# ## Available operations
#
# At present, the following options are available:
#
# - add
# - subtract
# - divide
# - multiply
# - integrate (sum and trapizoidal)
# - change per unit time (numerical differentiation essentially)
# - calculate linear regression
# - shift median of an ensemble
#
# These operations are unit aware so are fairly powerful.

# %% [markdown]
# ## Load some data
#
# We first load some test data.
# %%

db_emms = ScmRun("rcp26_emissions.csv", lowercase_cols=True)
db_emms.tail()

# %%
db_forcing = ScmRun(
    "rcmip-radiative-forcing-annual-means-v4-0-0.csv", lowercase_cols=True
).drop_meta(["mip_era", "activity_id"], inplace=False)

db_forcing.head()

# %% [markdown]
# ## Add

# %% [markdown]
# A very simple example is adding two variables together. For example, below we calculate total CO$_2$ emissions for the RCP2.6 scenario.

# %%
emms_co2 = db_emms.filter(variable="Emissions|CO2|MAGICC Fossil and Industrial").add(
    db_emms.filter(variable="Emissions|CO2|MAGICC AFOLU"),
    op_cols={"variable": "Emissions|CO2"},
)
emms_co2.head()

# %%

ax = plt.figure(figsize=(12, 7)).add_subplot(111)
plt_df = run_append([db_emms, emms_co2])
plt_df.filter(variable="*CO2*").lineplot(hue="variable", ax=ax)

# %% [markdown]
# The `op_cols` argument tells scmdata which columns to ignore when aligning the data
# and what value to give this column in the output. So we could do the same calculation
# but give the output a different name as shown below.

# %%
emms_co2_different_name = db_emms.filter(
    variable="Emissions|CO2|MAGICC Fossil and Industrial"
).add(
    db_emms.filter(variable="Emissions|CO2|MAGICC AFOLU"),
    op_cols={"variable": "Emissions|CO2|Total"},
)
emms_co2_different_name.head()

# %% [markdown]
# ## Subtract
#
# Subtraction works much the same way. Below we calculate the total effective radiative
# forcing and CO$_2$ effective radiative forcing in the [RCMIP](https://rcmip.org) data.

# %%
non_co2_rf = db_forcing.filter(variable="Effective Radiative Forcing").subtract(
    db_forcing.filter(variable="Effective Radiative Forcing|Anthropogenic|CO2"),
    op_cols={"variable": "Effective Radiative Forcing|Non-CO2"},
)
non_co2_rf.head()

# %%

ax = plt.figure(figsize=(12, 7)).add_subplot(111)
plt_df_forcing = run_append([db_forcing, non_co2_rf])
plt_df_forcing.filter(
    variable=["Effective Radiative Forcing", "Effective*CO2*"]
).lineplot(style="variable")

# %% [markdown]
# We could also calculate the difference between some SSP and RCP scenarios. The
# first thing to try would be to simply subtract the SSP126 total effective
# radiative forcing from the RCP26 total radiative forcing.

# %%
try:
    ssp126_minus_rcp26 = db_forcing.filter(
        scenario="ssp126", variable="Effective Radiative Forcing"
    ).subtract(
        db_forcing.filter(scenario="rcp26", variable="Radiative Forcing"),
        op_cols={
            "scenario": "ssp126 - rcp26",
        },
    )
except KeyError:
    traceback.print_exc(limit=0, chain=False)

# %% [markdown]
# Doing this gives us a `KeyError`. The reason is that the SSP126 variable
# is `Effective Radiative Forcing` whilst the RCP26 variable is `Radiative Forcing`
# hence the two datasets don't align. We can work around this using the `op_cols` argument.

# %%

ssp126_minus_rcp26 = db_forcing.filter(
    scenario="ssp126", variable="Effective Radiative Forcing"
).subtract(
    db_forcing.filter(scenario="rcp26", variable="Radiative Forcing"),
    op_cols={
        "scenario": "ssp126 - rcp26",
        "variable": "RF",
    },
)
ssp126_minus_rcp26.lineplot()

# %% [markdown]
# We could create plots of all the differences as shown below.

# %%
ssp_rcp_diffs = []
for target in ["26", "45", "60", "85"]:
    ssp = db_forcing.filter(
        scenario="ssp*{}".format(target),
        variable="Effective Radiative Forcing",
    )
    ssp_scen = ssp.get_unique_meta("scenario", no_duplicates=True)
    ssp_model = ssp.get_unique_meta("model", no_duplicates=True)

    rcp = db_forcing.filter(
        scenario="rcp{}".format(target), variable="Radiative Forcing"
    )
    rcp_scen = rcp.get_unique_meta("scenario", no_duplicates=True)
    rcp_model = rcp.get_unique_meta("model", no_duplicates=True)

    ssp_rcp_diff = ssp.subtract(
        rcp,
        op_cols={
            "scenario": "{} - {}".format(ssp_scen, rcp_scen),
            "model": "{} - {}".format(ssp_model, rcp_model),
            "variable": "RF",
        },
    )
    ssp_rcp_diffs.append(ssp_rcp_diff)

ssp_rcp_diffs = run_append(ssp_rcp_diffs)
ssp_rcp_diffs.head()

# %%

ax = plt.figure(figsize=(12, 7)).add_subplot(111)
ssp_rcp_diffs.lineplot(ax=ax, style="model")

# %% [markdown]
# ## Divide
#
# The divide (and multiply) operations clearly have to also be aware of units.
# Thanks to [Pint's pandas interface](https://pint.readthedocs.io/en/0.13/pint-pandas.html),
# this can happen automatically. For example, in our calculation below the units are
# automatically returned as `dimensionless`.

# %%
ssp126_to_rcp26 = db_forcing.filter(
    scenario="ssp126", variable="Effective Radiative Forcing"
).divide(
    db_forcing.filter(scenario="rcp26", variable="Radiative Forcing"),
    op_cols={
        "scenario": "ssp126 / rcp26",
        "variable": "RF",
    },
)
ssp126_to_rcp26.head()

# %%

ssp126_to_rcp26.lineplot()

# %% [markdown]
# ## Integrate
#
# We can also integrate our data. As previously, thanks to
# [Pint](https://pint.readthedocs.io) and some other work, this operation
# is also unit aware.
#
# There are two methods of integration available, `cumtrapz` and `cumsum`.
# The method that should be used depends on the data you are integrating,
# specifically whether the data are piecewise linear or piecewise constant.
#
# Annual timeseries of emissions are piecewise constant (each value represents
# a total which is constant over an year) so should use the `cumsum` method.
#
# Other output such as effective radiative forcing, concentrations or decadal
# timeseries of emissions, represent a  point estimate or an average over a period.
# These timeseries are therefore piecewise linear and should use the `cumtrapz`
# method.

# %%
with warnings.catch_warnings():
    # Ignore warning about nans in the historical timeseries
    warnings.filterwarnings("ignore", module="scmdata.ops")

    # Radiative Forcings are piecewise-linear so `cumtrapz` should be used
    erf_integral = (
        db_forcing.filter(variable="Effective Radiative Forcing")
        .cumtrapz()
        .convert_unit("TJ / m^2")
    )

erf_integral

# %%

ax = plt.figure(figsize=(12, 7)).add_subplot(111)
erf_integral.lineplot(ax=ax, style="model")

# %%
with warnings.catch_warnings():
    # Ignore warning about nans in the historical timeseries
    warnings.filterwarnings("ignore", module="scmdata.ops")

    # Emissions are piecewise-constant so `cumsum` should be used
    cumulative_emissions = emms_co2.cumsum().convert_unit("Gt C")

cumulative_emissions

# %%

ax = plt.figure(figsize=(12, 7)).add_subplot(111)
cumulative_emissions.lineplot(ax=ax, style="model")

# %% [markdown]
# ## Time deltas
#
# We can also calculate the change per unit time of our data. As previously,
# thanks to [Pint](https://pint.readthedocs.io/en/0.13) and some other work,
# this operation is also unit aware.

# %%
with warnings.catch_warnings():
    # Ignore warning about nans in the historical timeseries
    warnings.filterwarnings("ignore", module="scmdata.ops")

    erf_delta = (
        db_forcing.filter(variable="Effective Radiative Forcing")
        .delta_per_delta_time()
        .convert_unit("W / m^2 / yr")
    )

erf_delta

# %%

fig, axes = plt.subplots(figsize=(16, 9), ncols=2)
erf_delta.lineplot(ax=axes[0], style="model")
erf_delta.filter(year=range(2000, 2500)).lineplot(
    ax=axes[1], style="model", legend=False
)
axes[1].set_ylim([-0.1, 0.2])

# %% [markdown]
# ## Regression
#
# We can also calculate linear regressions of our data. As previously, thanks
# to [Pint](https://pint.readthedocs.io) and some other work, this operation
# is also unit aware.

# %%
erf_total = db_forcing.filter(variable="Effective Radiative Forcing").filter(
    scenario="historical", keep=False
)
erf_total_for_reg = erf_total.filter(year=range(2010, 2050))

# %% [markdown]
# The default return type of `linear_regression` is a list of dictionaries.

# %%
linear_regression_raw = erf_total_for_reg.linear_regression()
type(linear_regression_raw)

# %%
type(linear_regression_raw[0])

# %% [markdown]
# If we want, we can make a `DataFrame` from this list.

# %%

linear_regression_df = pd.DataFrame(linear_regression_raw)
linear_regression_df

# %% [markdown]
# Alternately, we can request only the gradients or only the intercepts
# (noting that intercepts are calculated using a time base which has zero at 1970-01-01 again).

# %%
erf_total_for_reg.linear_regression_gradient("W / m^2 / yr")

# %%
erf_total_for_reg.linear_regression_intercept("W / m^2")

# %% [markdown]
# If we want to plot the regressions, we can request an `ScmRun` instance based
# on the linear regressions is returned using the `linear_regression_scmrun` method.

# %%
linear_regression = erf_total_for_reg.linear_regression_scmrun()
linear_regression["label"] = "linear regression"
linear_regression

# %%
erf_total_for_reg["label"] = "Raw"
pdf = run_append([erf_total_for_reg, linear_regression])

# %%

fig, axes = plt.subplots(figsize=(12, 7))
pdf.filter(scenario=["ssp1*", "ssp5*"]).lineplot(ax=axes, style="label")

# %%

fig, axes = plt.subplots(figsize=(16, 9))
pdf.lineplot(ax=axes, style="label")

# %% [markdown]
# ## Shift median
#
# Sometimes we wish to simply move an ensemble of timeseries so that its median
# matches some value, whilst preserving the spread of the ensemble. This can be
# done with the `adjust_median_to_target` method. For example, let's say that we
# wanted to shift our ensemble of forcing values so that their 2030 median was equal
# to 4.5 (god knows why we would want to do this, but it will serve as an example).

# %%
erf_total = db_forcing.filter(variable="Effective Radiative Forcing").filter(
    scenario="historical", keep=False
)
erf_total_for_shift = erf_total.filter(year=range(2010, 2100))
erf_total_for_shift["label"] = "Raw"

# %%
erf_total_for_shift

# %%
erf_total_shifted = erf_total_for_shift.adjust_median_to_target(
    target=4.5,
    evaluation_period=2030,
    process_over=("scenario", "model"),
)
erf_total_shifted["label"] = "Shifted"

# %%
pdf = run_append([erf_total_for_shift, erf_total_shifted])

# %%

fig, axes = plt.subplots(figsize=(12, 7))
pdf.lineplot(ax=axes, style="label")

# %% [markdown]
# If we wanted, we could adjust the timeseries relative to some reference period
# first before doing the shift.

# %%
ref_period = range(2010, 2040 + 1)
erf_total_for_shift_rel_to_ref_period = erf_total_for_shift.relative_to_ref_period_mean(
    year=ref_period
)
erf_total_for_shift_rel_to_ref_period["label"] = "rel. to {} - {}".format(
    ref_period[0], ref_period[-1]
)

# %%
target = -5
evaluation_period = range(2020, 2030 + 1)
erf_total_for_shift_rel_to_ref_period_shifted = (
    erf_total_for_shift_rel_to_ref_period.adjust_median_to_target(
        target=target,
        evaluation_period=evaluation_period,
        process_over=("scenario", "model"),
    )
)
erf_total_for_shift_rel_to_ref_period_shifted[
    "label"
] = "rel. to {} - {} (median of {} - {} mean adjusted to {})".format(
    ref_period[0],
    ref_period[-1],
    evaluation_period[0],
    evaluation_period[-1],
    target,
)

# %%
pdf = run_append(
    [
        erf_total_for_shift,
        erf_total_shifted,
        erf_total_for_shift_rel_to_ref_period,
        erf_total_for_shift_rel_to_ref_period_shifted,
    ]
)

# %%

fig, axes = plt.subplots(figsize=(12, 12))
pdf.lineplot(ax=axes, style="label")
