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
# # Plotting with seaborn
#
# ScmData provides limited support for plotting. However, we make it as easy as possible to
# return data in a format which can be used with the
# [seaborn](https://seaborn.pydata.org/index.html) plotting library. Given the power of this
# library, we recommend having a look through its documentation if you want to make anything
# more than the most basic plots.

# %%

import matplotlib.pyplot as plt
import seaborn as sns

from scmdata.plotting import RCMIP_SCENARIO_COLOURS
from scmdata.run import ScmRun

# %% [markdown]
# ## Data
#
# For this notebook we use the RCMIP radiative forcings, available at rcmip.org.

# %%

rcmip_db = ScmRun("rcmip-radiative-forcing-annual-means-v4-0-0.csv")
rcmip_db.head()

# %% [markdown]
# ## Plotting with ScmRun
#
# For the most common plotting patterns, we provide a very simple `lineplot` method in `ScmRun`.

# %%

out = rcmip_db.filter(variable="Effective Radiative Forcing").lineplot()
out

# %% [markdown]
# ``kwargs`` passed to this method are given directly to
# [``seaborn.lineplot``](https://seaborn.pydata.org/generated/seaborn.lineplot.html),
# which allows an extra layer of control.
#
# For example, we can plot on slightly bigger axes, make the lines slightly transparent,
# add markers for the different models, specify the colour to use for each scenario and
# specify the order to display the scenarios in.

# %%

ax = plt.figure(figsize=(16, 9)).add_subplot(111)
rcmip_db.filter(variable="Effective Radiative Forcing").lineplot(
    ax=ax,
    hue="scenario",
    palette=RCMIP_SCENARIO_COLOURS,
    hue_order=RCMIP_SCENARIO_COLOURS.keys(),
    style="model",
    alpha=0.7,
)

# %% [markdown]
# ### Specifying the time axis
#
# Plotting with a `datetime.datetime` time axis is not always convenient.
# To address this, we provide the `time_axis` keyword argument. The options are available in the `lineplot` docstring.

# %%
print(rcmip_db.lineplot.__doc__)

# %%

fig, axes = plt.subplots(figsize=(16, 9), nrows=2, ncols=2)

pdb = rcmip_db.filter(variable="Effective Radiative Forcing")
for ax, time_axis in zip(
    axes.flatten(),
    [
        "year",
        "year-month",
        "days since 1970-01-01",
        "seconds since 1970-01-01",
    ],
):
    pdb.lineplot(
        ax=ax,
        hue="scenario",
        palette=RCMIP_SCENARIO_COLOURS,
        hue_order=RCMIP_SCENARIO_COLOURS.keys(),
        style="model",
        alpha=0.7,
        time_axis=time_axis,
        legend=False,
    )
    ax.set_title(time_axis)

plt.tight_layout()

# %% [markdown]
# These same options can also be passed to the `timeseries` and `long_data` methods.

# %%

rcmip_db.timeseries(time_axis="year-month")

# %%

rcmip_db.long_data(time_axis="days since 1970-01-01")

# %% [markdown]
# ## Plotting with seaborn
#
# If you wish to make plots which are more complex than this most basic pattern, a combination of seaborn and pandas reshaping is your best bet.

# %% [markdown]
# ### Plotting on a grid
#
# Often we wish to look at lots of different variables at once. Seaborn allows this sort of 'gridded' plotting, as shown below.

# %%
vars_to_plot = ["Effective Radiative Forcing"] + [
    "Effective Radiative Forcing|{}".format(v)
    for v in [
        "Anthropogenic",
        "Anthropogenic|Aerosols",
        "Anthropogenic|CO2",
        "Anthropogenic|CH4",
        "Anthropogenic|N2O",
    ]
]
vars_to_plot

# %%

seaborn_df = rcmip_db.filter(variable=vars_to_plot).long_data()
seaborn_df.head()

# %% [markdown]
# With the output of `.long_data()` we can directly use [``seaborn.relplot``](https://seaborn.pydata.org/generated/seaborn.relplot.html).

# %%

sns.relplot(
    data=seaborn_df,
    x="time",
    y="value",
    col="variable",
    col_wrap=3,
    hue="scenario",
    palette=RCMIP_SCENARIO_COLOURS,
    hue_order=RCMIP_SCENARIO_COLOURS.keys(),
    alpha=0.7,
    facet_kws={"sharey": False},
    kind="line",
)

# %% [markdown]
# ### Variable scatter plots
#
# Sometimes we don't want to plot against time, rather we want to plot variables against each other. For example, we might want to see how the effective radiative forcings relate to each other in the different scenarios. In such a case we can reshape the data using pandas before using seaborn.

# %%

ts = rcmip_db.filter(variable=vars_to_plot[:4]).timeseries()
ts.head()

# %%

ts_reshaped = ts.unstack("variable").stack("time").reset_index()
ts_reshaped.head()

# %%

sns.pairplot(
    ts_reshaped,
    hue="scenario",
    palette=RCMIP_SCENARIO_COLOURS,
    hue_order=RCMIP_SCENARIO_COLOURS.keys(),
    corner=True,
    height=4,
)
