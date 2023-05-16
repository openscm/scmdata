# ---
# jupyter:
#   jupytext:
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
# # xarray compatibility
#
# **scmdata** allows datat to be exported to xarray. This makes it easy to use xarray's many helpful features, most of which are not natively provided in scmdata.

# %%

import numpy as np

from scmdata import ScmRun


# %%
def get_data(years, n_ensemble_members, end_val, rand_pct):
    return (np.arange(years.shape[0]) / years.shape[0] * end_val)[:, np.newaxis] * (
        rand_pct * np.random.random((years.shape[0], n_ensemble_members)) + 1
    )


# %%

years = np.arange(1750, 2500 + 1)
variables = ["gsat", "gmst"]
n_variables = len(variables)
n_ensemble_members = 100


start = ScmRun(
    np.hstack(
        [
            get_data(years, n_ensemble_members, 5.5, 0.1),
            get_data(years, n_ensemble_members, 6.0, 0.05),
        ]
    ),
    index=years,
    columns={
        "model": "a_model",
        "scenario": "a_scenario",
        "variable": [v for v in variables for i in range(n_ensemble_members)],
        "region": "World",
        "unit": "K",
        "ensemble_member": [i for v in variables for i in range(n_ensemble_members)],
    },
)
start

# %% [markdown]
# The usual scmdata methods are of course available.

# %%

start.plumeplot(
    quantile_over="ensemble_member", hue_var="variable", hue_label="Variable"
)

# %% [markdown]
# However, we can cast to an xarray DataSet and then all the xarray methods become available too.

# %%

xr_ds = start.to_xarray(dimensions=("ensemble_member",))
xr_ds

# %% [markdown]
# For example, calculating statistics.

# %%

xr_ds.median(dim="ensemble_member")

# %% [markdown]
# Plotting timeseries.

# %%

xr_ds["gsat"].plot.line(hue="ensemble_member", add_legend=False)

# %% [markdown]
# Selecting and plotting timeseries.

# %%

xr_ds["gsat"].sel(ensemble_member=range(10)).plot.line(
    hue="ensemble_member", add_legend=False
)

# %% [markdown]
# Scatter plots.

# %%

xr_ds.plot.scatter(x="gsat", y="gmst", hue="ensemble_member", alpha=0.3)

# %% [markdown]
# Or combinations of calculations and plots.

# %%

xr_ds.median(dim="ensemble_member").plot.scatter(x="gsat", y="gmst")
