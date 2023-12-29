import cProfile

import numpy as np
import openscm_units
from line_profiler import LineProfiler

from scmdata.run import BaseScmRun

UR = openscm_units.unit_registry

scenarios = ["a", "b", "c"]
variables = ["T|upper", "T|lower", "Heat uptake", "Carbon uptake"]
units = ["K", "K", "ZJ / yr", "GtC / yr"]
n_ts = len(scenarios) * len(variables)

years = np.arange(1850, 5000 + 1)
n_yrs = years.size
n_vals = n_ts * n_yrs
model_res = BaseScmRun(
    data=np.arange(n_vals).reshape(n_ts, n_yrs).T,
    columns={
        "variable": variables * len(scenarios),
        "unit": units * len(scenarios),
        "scenario": scenarios * len(variables),
        "model": "an_esm",
    },
    index=years,
)
normalisation = model_res.copy()
normalisation.values[:] = 1.0
normalisation["model"] = "norm"


def op_scmrun(res, target, norm):
    sse = (
        (
            res.subtract(target, op_cols={"model": "res - target"}).divide(
                norm, op_cols={"model": "m"}
            )
            ** 2
        )
        .convert_unit("dimensionless")
        .values.sum()
        .sum()
    )

    return sse


def op_pandas_align_then_pint(res, target, norm, check_duplicated=True):
    op_meta = ["variable", "scenario", "unit"]
    res_ts = res.timeseries(meta=op_meta, check_duplicated=check_duplicated)
    target_ts = target.timeseries(meta=op_meta, check_duplicated=check_duplicated)
    norm_ts = norm.timeseries(meta=op_meta, check_duplicated=check_duplicated)

    res_ts_unit_groups = {}
    target_ts_unit_groups = {}
    norm_ts_unit_groups = {}
    for unit, udf in res_ts.groupby("unit", sort=False):
        unitless = udf.reset_index("unit", drop=True)
        res_ts_unit_groups[unit] = unitless
        # Doesn't work with different units, but will deal with that in a minute
        target_ts_unit_groups[unit] = target_ts.loc[udf.index, :].reset_index(
            "unit", drop=True
        )
        norm_ts_unit_groups[unit] = norm_ts.loc[udf.index, :].reset_index(
            "unit", drop=True
        )

    sse = 0
    for unit, res_df in res_ts_unit_groups.items():
        res_array = UR.Quantity(res_df.values, unit)
        target_array = UR.Quantity(target_ts_unit_groups[unit].values, unit)
        norm_array = UR.Quantity(norm_ts_unit_groups[unit].values, unit)

        sse += np.sum(
            (((res_array - target_array) / norm_array) ** 2).to("dimensionless").m
        )

    return sse


def op_col_by_col_no_pint_pandas(res, target, norm, check_duplicated=True):
    op_meta = ["variable", "scenario", "unit"]
    res_ts_T = res.timeseries(meta=op_meta, check_duplicated=check_duplicated).T
    target_ts_T = target.timeseries(meta=op_meta, check_duplicated=check_duplicated).T
    norm_ts_T = norm.timeseries(meta=op_meta, check_duplicated=check_duplicated).T

    sse = 0
    unit_index = op_meta.index("unit")
    for c in res_ts_T:
        # Need to think more for different unit ops
        unit = c[unit_index]

        res_c_array = UR.Quantity(res_ts_T[c].values, unit)
        target_c_array = UR.Quantity(target_ts_T[c].values, unit)
        norm_c_array = UR.Quantity(norm_ts_T[c].values, unit)

        sse += np.sum(
            (((res_c_array - target_c_array) / norm_c_array) ** 2).to("dimensionless").m
        )

    return sse


for call, cprofile_out_file, profile_funcs in [
    (
        "op_scmrun(model_res, model_res, normalisation)",
        "op_scmrun.prof",
        [op_scmrun, BaseScmRun.timeseries],
    ),
    (
        "op_pandas_align_then_pint(model_res, model_res, normalisation)",
        "op_pandas_align_then_pint.prof",
        [op_pandas_align_then_pint, BaseScmRun.timeseries],
    ),
    (
        "op_col_by_col_no_pint_pandas(model_res, model_res, normalisation)",
        "op_col_by_col_no_pint_pandas.prof",
        [op_col_by_col_no_pint_pandas, BaseScmRun.timeseries],
    ),
]:
    cProfile.run(call, cprofile_out_file)

    header = f"Profiling: {call}"
    print(header)
    print("-" * len(header))
    profile = LineProfiler(*profile_funcs)
    profile.run(call)
    profile.print_stats(
        output_unit=1e-3,
        rich=True,
        sort=False,
        summarize=True,
        #        details=False,
        details=True,
    )
    print("-" * len(header))
    print()
