import cProfile

import numpy as np
import openscm_units
from line_profiler import LineProfiler

from scmdata.run import BaseScmRun

UR = openscm_units.unit_registry

scenarios = ["a", "b", "c"]
variables = ["T|upper", "T|lower", "Heat uptake", "Carbon uptake"]
units = ["K", "K", "ZJ / yr", "GtCO2 / yr"]
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

model_res_diff_unit = model_res.copy()
model_res_diff_unit = (
    model_res_diff_unit.convert_unit("mK", variable="T|upper")
    .convert_unit("kK", variable="T|lower")
    .convert_unit("J / yr", variable="Heat uptake")
    .convert_unit("PgC / yr", variable="Carbon uptake")
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
    assert op_meta[-1] == "unit", "Stuff below may break..."
    unit_index = len(op_meta) - 1

    res_ts_T = res.timeseries(meta=op_meta, check_duplicated=check_duplicated).T
    target_ts_T = target.timeseries(meta=op_meta, check_duplicated=check_duplicated).T
    norm_ts_T = norm.timeseries(meta=op_meta, check_duplicated=check_duplicated).T

    sse = 0
    for c in res_ts_T:
        res_c = res_ts_T[c]
        res_unit = res_c.name[unit_index]
        res_c_array = UR.Quantity(res_c.values.squeeze(), res_unit)

        # Can use -1 because unit is the last column in the metadata
        target_c = target_ts_T[c[:-1]]
        target_unit = target_c.columns[0]
        target_c_array = UR.Quantity(target_c.values.squeeze(), target_unit)

        # Can use -1 because unit is the last column in the metadata
        norm_c = norm_ts_T[c[:-1]]
        norm_unit = norm_c.columns[0]
        norm_c_array = UR.Quantity(norm_c.values.squeeze(), norm_unit)

        # TODO: Should be checking alignment of time axis before doing this
        sse += np.sum(
            (((res_c_array - target_c_array) / norm_c_array) ** 2).to("dimensionless").m
        )

    return sse


def op_align_iloc_iteration(res, target, norm, iloc=False):
    op_meta = ["variable", "scenario"]

    res_ts = res.timeseries(meta=[*op_meta, "unit"])
    res_ts_units = {tuple(v[:-1]): v[-1] for v in res_ts.index.tolist()}
    target_ts = target.timeseries(meta=[*op_meta, "unit"])
    target_ts_units = {tuple(v[:-1]): v[-1] for v in target_ts.index.tolist()}
    norm_ts = norm.timeseries(meta=[*op_meta, "unit"])
    norm_ts_units = {tuple(v[:-1]): v[-1] for v in norm_ts.index.tolist()}

    res_ts_no_unit = res_ts.reset_index("unit", drop=True)
    res_ts_no_unit, target_ts_no_unit = res_ts_no_unit.align(
        target_ts.reset_index("unit", drop=True), copy=False
    )
    res_ts_no_unit, norm_ts_no_unit = res_ts_no_unit.align(
        norm_ts.reset_index("unit", drop=True), copy=False
    )

    assert not res_ts_no_unit.isna().any().any()
    assert not target_ts_no_unit.isna().any().any()
    assert not norm_ts_no_unit.isna().any().any()

    sse = 0
    for i, (idx, row) in enumerate(res_ts_no_unit.iterrows()):
        res_array = UR.Quantity(row.values, res_ts_units[idx])
        if iloc:
            # Can iloc here with confidence as we have already aligned
            target_array = UR.Quantity(
                target_ts_no_unit.iloc[i].values, target_ts_units[idx]
            )
            norm_array = UR.Quantity(norm_ts_no_unit.iloc[i].values, norm_ts_units[idx])
        else:
            target_array = UR.Quantity(
                target_ts_no_unit.loc[idx].values, target_ts_units[idx]
            )
            norm_array = UR.Quantity(
                norm_ts_no_unit.loc[idx].values, norm_ts_units[idx]
            )

        sse += np.sum(
            (((res_array - target_array) / norm_array) ** 2).to("dimensionless").m
        )

    return sse


for call, cprofile_out_file, profile_funcs in [
    (
        "op_scmrun(model_res, model_res, normalisation)",
        "op_scmrun.prof",
        [op_scmrun],
    ),
    (
        "op_scmrun(model_res, model_res_diff_unit, normalisation)",
        "op_scmrun.prof",
        [op_scmrun],
    ),
    (
        "op_pandas_align_then_pint(model_res, model_res, normalisation)",
        "op_pandas_align_then_pint.prof",
        [op_pandas_align_then_pint],
    ),
    # (
    #     "op_pandas_align_then_pint(model_res, model_res_diff_unit, normalisation)",
    #     "op_pandas_align_then_pint.prof",
    #     [op_pandas_align_then_pint],
    # ),
    (
        "op_col_by_col_no_pint_pandas(model_res, model_res, normalisation)",
        "op_col_by_col_no_pint_pandas.prof",
        [op_col_by_col_no_pint_pandas],
    ),
    (
        "op_col_by_col_no_pint_pandas(model_res, model_res_diff_unit, normalisation)",
        "op_col_by_col_no_pint_pandas.prof",
        [op_col_by_col_no_pint_pandas],
    ),
    (
        "op_align_iloc_iteration(model_res, model_res_diff_unit, normalisation)",
        "op_align_iloc_iteration.prof",
        [op_align_iloc_iteration],
    ),
    (
        "op_align_iloc_iteration(model_res, model_res_diff_unit, normalisation, iloc=True)",
        "op_align_iloc_iteration_iloc.prof",
        [op_align_iloc_iteration],
    ),
]:
    cProfile.run(call, cprofile_out_file)

    header = f"Profiling: {call}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    profile = LineProfiler(*profile_funcs)
    profile.run(call)
    profile.print_stats(
        output_unit=1e-3,
        rich=True,
        sort=False,
        summarize=True,
        # details=False,
        details=True,
    )
    print("=" * len(header))
    print()
