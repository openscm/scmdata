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
# # Emissions units with Pint
#
# In this notebook we give some examples of how units are handled in SCMData and are built on top of the [Pint](https://github.com/hgrecco/pint) package.

# %%

import traceback
import warnings

import pint
from pint.errors import DimensionalityError

from scmdata.units import UnitConverter

# %% [markdown]
# ## UnitConverter
#
# The `UnitConverter` class handles all unit conversions for us. It is used as shown.

# %%
uc = UnitConverter("GtC/yr", "Mt CO2 / yr")
uc.convert_from(1)

# %%
uc.convert_to(1)

# %% [markdown]
# ## Pint Unit Registry
#
# The `unit_registry` which sits underneath all conversions can be accessed via `scmdata.units.get_unit_registry`or via `UnitConverter`'s `unit_registry` property.

# %%
unit_registry = uc.unit_registry

# %% [markdown]
# Having accessed the `unit_registry`, all the units available in SCMData can be shown like so.

# %%

dir(unit_registry)

# %% [markdown]
# Additional units can be added to the unit registry using `define` as shown below. By default, `scmdata.units.UNIT_REGISTRY` uses the same registry as `openscm_units.unit_registry` so any additional units will be available for any other packages which use `openscm_units.unit_registry`.

# %%
unit_registry.define("population = [population]")

assert "population" in dir(unit_registry)

# %% [markdown]
# ## Using Pint Directly
#
# For completeness, below we show how to use pint directly. Note that all of these operations are used by `UnitConverter` so the user shouldn't ever have to access pint in this way.

# %% [markdown]
# With the `unit_registry`, we can also create Pint variables/arrays which are unit aware.

# %%
one_carbon = 1 * unit_registry("C")
print(one_carbon)

# %%
type(one_carbon)

# %%
one_co2 = 1 * unit_registry.CO2
three_sulfur = 3 * unit_registry.S

# %% [markdown]
# Pint quantities also print in an intuitive way.

# %%
print(one_co2)
print(three_sulfur)

# %% [markdown]
# We can convert them to base units or to each other.

# %%
print(one_carbon.to_base_units())
print(one_co2.to("C"))
print(three_sulfur.to("SO2"))

# %% [markdown]
# Operations are units aware.

# %%
print(one_carbon + one_co2)
print(one_carbon * one_co2)
print((one_carbon * one_co2).to_base_units())
print(one_carbon / one_co2)
print((one_carbon / one_co2).to_base_units())

# %% [markdown]
# If we have compound units (e.g. emissions units which are [mass] * [substance] / [time]), we can convert any bit of the unit we want.

# %%
eg1 = 1 * unit_registry("Mt") * unit_registry("C") / unit_registry("yr")
print(eg1)
eg2 = 5 * unit_registry("t") * unit_registry("CO2") / unit_registry("s")
print(eg2)

# %%
print(eg1.to("Gt CO2 / day"))
print(eg2.to("Gt C / yr"))

# %% [markdown]
# ## Contexts
#
# With a context, we can use metric conversion definitions to do emissions conversions that would otherwise raise a `DimensionalityError`. For example, converting CO2 to N2O using AR4GWP100 (where 298 tCO2 = 1 tN2O).

# %%

ar4gwp100uc = UnitConverter("N2O", "CO2", context="AR4GWP100")
ar4gwp100uc.convert_from(1)

# %%
ar4gwp100uc = UnitConverter("N2O", "CH4", context="AR4GWP100")
ar4gwp100uc.convert_from(1)

# %% [markdown]
# We can see which contexts we have (which we can use for e.g. metric conversions).

# %%
ar4gwp100uc.contexts

# %% [markdown]
# Such context dependent conversions can also be done directly with Pint.

# %%
base = 1 * unit_registry("N2O")
with unit_registry.context("AR4GWP100"):
    print(one_carbon)
    print(one_carbon.to("CO2"))
    print(
        one_carbon.to("N2O") + 3 * unit_registry.N2O
    )  # I am not sure why you need to force the conversion of `a` first...

# %% [markdown]
# Without a context to tell us about metrics, if we try to do an invalid conversion, a `DimensionalityError` will be raised.

# %%
try:
    ar4gwp100uc = UnitConverter("N2O", "CO2")
    ar4gwp100uc.convert_from(1)
except DimensionalityError:
    traceback.print_exc(limit=0, chain=False)

# %%
try:
    base.to("CO2")
except DimensionalityError:
    traceback.print_exc(limit=0, chain=False)


# %% [markdown]
# If the context you use does not have the conversion you request, a warning will be raised. Any subsequent conversions will result in NaN's.


# %%
# modify the way the warning appears to remove the path,
# thank you https://stackoverflow.com/a/26433913
def custom_formatting(message, category, filename, lineno, file=None, line=None):
    return "{}: {}\n".format(category.__name__, message)


warnings.formatwarning = custom_formatting

ucnan = UnitConverter("N2O", "Halon2402", context="SARGWP100")
ucnan.convert_from(1)

# %%
