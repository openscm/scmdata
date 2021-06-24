import pytest


def test_xarray_plot_line(plumeplot_scmrun):
    pytest.importorskip("matplotlib")
    pytest.importorskip("nc_time_axis")

    plumeplot_scmrun["climate_model"] = plumeplot_scmrun.get_unique_meta(
        "climate_model"
    )[0]

    xr_ds = plumeplot_scmrun.to_xarray(dimensions=("scenario", "ensemble_member"))

    xr_ds["Surface Air Temperature Change"].median(dim="ensemble_member").plot.line(
        hue="scenario"
    )
    xr_ds["Surface Air Temperature Change"].sel(scenario="a_scenario").plot.line(
        hue="ensemble_member"
    )


@pytest.mark.parametrize("scatter_kwargs", ({}, dict(hue="scenario")))
def test_xarray_plot_scatter(plumeplot_scmrun, scatter_kwargs):
    pytest.importorskip("matplotlib")

    plumeplot_scmrun["climate_model"] = plumeplot_scmrun.get_unique_meta(
        "climate_model"
    )[0]
    gmst = plumeplot_scmrun.copy()
    gmst["variable"] = "Surface Air Ocean Blended Temperature Change"
    plumeplot_scmrun = plumeplot_scmrun.append(gmst)

    xr_ds = plumeplot_scmrun.to_xarray(dimensions=("scenario", "ensemble_member"))

    xr_ds.plot.scatter(
        x="Surface Air Ocean Blended Temperature Change",
        y="Surface Air Temperature Change",
        **scatter_kwargs
    )
