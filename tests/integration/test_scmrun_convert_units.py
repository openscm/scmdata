import numpy as np

from scmdata import ScmRun


def test_convert_to_scen_units():
    var_start_units_scen_units_conv_factor = (
        ("Emissions|CO2", "Mt CO2/yr", "Gt C/yr", 12 / 44000, None),
        ("Emissions|CH4", "Mt C/yr", "Mt CH4/yr", (12 + 4) / 12, "CH4_conversions"),
        (
            "Emissions|N2O",
            "kt N2O/yr",
            "Mt N2ON/yr",
            (2 * 14) / (1000 * (2 * 14 + 16)),
            None,
        ),
        ("Emissions|SOx", "Mt SO2/yr", "Mt S/yr", 32 / (32 + 2 * 16), None),
        ("Emissions|CO", "Mt CO/yr", "Mt CO/yr", 1, None),
        ("Emissions|NMVOC", "Mt VOC/yr", "Mt NMVOC/yr", 1, None),
        (
            "Emissions|NOx",
            "Mt NOx/yr",
            "Mt N/yr",
            14 / (14 + 2 * 16),
            "NOx_conversions",
        ),
        ("Emissions|BC", "Mt BC/yr", "Mt BC/yr", 1, None),
        ("Emissions|OC", "Mt OC/yr", "Mt OC/yr", 1, None),
        ("Emissions|NH3", "Mt NH3/yr", "Mt N/yr", 14 / (14 + 3), "NH3_conversions"),
        ("Emissions|CF4", "Mt CF4/yr", "kt CF4/yr", 1000, None),
        ("Emissions|HFC4310mee", "kt HFC4310mee/yr", "kt HFC4310/yr", 1, None),
        ("Emissions|SF6", "t SF6/yr", "kt SF6/yr", 1 / 1000, None),
    )

    start = ScmRun(
        data=np.ones((1, len(var_start_units_scen_units_conv_factor))),
        index=np.array([2015]),
        columns={
            "model": ["unspecified"],
            "scenario": ["test"],
            "region": ["World"],
            "variable": [v[0] for v in var_start_units_scen_units_conv_factor],
            "unit": [v[1] for v in var_start_units_scen_units_conv_factor],
        },
    )

    res = start.copy()
    for variable, _, target_unit, _, context in var_start_units_scen_units_conv_factor:
        res = res.convert_unit(target_unit, context=context, variable=variable)

    for (
        variable,
        _,
        target_unit,
        conv_factor,
        _,
    ) in var_start_units_scen_units_conv_factor:
        np.testing.assert_allclose(
            res.filter(variable=variable).timeseries(), conv_factor, rtol=1e-15
        )
