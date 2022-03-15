import re

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import xarray as xr

import scmdata


def do_basic_to_xarray_checks(res, start_run, dimensions, extras):
    assert isinstance(res, xr.Dataset)
    assert set(res.data_vars) == set(start_run.get_unique_meta("variable"))

    for variable_name, data_var in res.data_vars.items():
        assert data_var.dims == dimensions

        unit = start_run.filter(variable=variable_name).get_unique_meta("unit")
        assert data_var.units in unit

    # all other metadata should be in attrs
    for meta_col in (
        set(start_run.meta.columns)
        - set(dimensions)
        - set(extras)
        - {"variable", "unit"}
    ):
        meta_val = start_run.get_unique_meta(meta_col, True)
        assert res.attrs["scmdata_metadata_{}".format(meta_col)] == meta_val


def do_basic_check_of_data_points(res, start_run, dimensions):
    for variable_name, data_var in res.data_vars.items():
        # check a couple of data points to make sure the translation is correct
        for idx in [0, -1]:
            xarray_spot = data_var.isel({v: idx for v in dimensions})
            fkwargs = {k: [v.values.tolist()] for k, v in xarray_spot.coords.items()}
            fkwargs["variable"] = variable_name
            xarray_unit = data_var.units

            start_run_spot = start_run.filter(**fkwargs).convert_unit(xarray_unit)
            if np.isnan(xarray_spot):
                assert start_run_spot.empty
            else:
                start_run_vals = float(start_run_spot.values.squeeze())
                npt.assert_array_equal(xarray_spot.values, start_run_vals)


@pytest.mark.parametrize(
    "dimensions,expected_dimensions",
    (
        (("region", "scenario", "time"), ("region", "scenario", "time")),
        (("time", "region", "scenario"), ("time", "region", "scenario")),
        (("region", "time", "scenario"), ("region", "time", "scenario")),
        (("region", "scenario"), ("region", "scenario", "time")),
        (("scenario", "region"), ("scenario", "region", "time")),
        (("scenario",), ("scenario", "time")),
    ),
)
def test_to_xarray(scm_run, dimensions, expected_dimensions):
    res = scm_run.to_xarray(dimensions=dimensions)

    do_basic_to_xarray_checks(
        res,
        scm_run,
        expected_dimensions,
        (),
    )
    do_basic_check_of_data_points(res, scm_run, expected_dimensions)

    # no extras
    assert not set(res.coords) - set(res.dims)


@pytest.mark.parametrize(
    "extras",
    (
        ("model",),
        ("climate_model",),
        ("climate_model", "model"),
    ),
)
def test_to_xarray_extras_no_id_coord(scm_run, extras):
    dimensions = ("scenario", "region", "time")
    res = scm_run.to_xarray(dimensions=dimensions, extras=extras)

    do_basic_to_xarray_checks(res, scm_run, dimensions, extras)
    do_basic_check_of_data_points(res, scm_run, dimensions)

    assert set(extras) == set(res.coords) - set(res.dims)

    scm_run_meta = scm_run.meta
    for extra_col in extras:
        xarray_vals = res[extra_col].values
        extra_dims = res[extra_col].dims
        assert len(extra_dims) == 1
        extra_dims = extra_dims[0]
        xarray_coords = res[extra_col][extra_dims].values

        for xarray_extra_val, extra_xarray_coord in zip(xarray_vals, xarray_coords):
            scm_run_extra_val = (
                scm_run_meta[scm_run_meta[extra_dims] == extra_xarray_coord][extra_col]
                .unique()
                .tolist()
            )
            assert len(scm_run_extra_val) == 1
            scm_run_extra_val = scm_run_extra_val[0]

            assert scm_run_extra_val == xarray_extra_val


@pytest.mark.parametrize("extras", (("scenario", "model", "random_key"),))
@pytest.mark.parametrize(
    "dimensions,expected_dimensions",
    (
        (("climate_model", "run_id"), ("climate_model", "run_id", "time", "_id")),
        (("run_id", "climate_model"), ("run_id", "climate_model", "time", "_id")),
        (
            ("run_id", "climate_model", "time"),
            ("run_id", "climate_model", "time", "_id"),
        ),
        (
            ("run_id", "time", "climate_model"),
            ("run_id", "time", "climate_model", "_id"),
        ),
        (
            ("run_id", "climate_model", "time", "_id"),
            ("run_id", "climate_model", "time", "_id"),
        ),
        (
            ("_id", "run_id", "time", "climate_model"),
            ("_id", "run_id", "time", "climate_model"),
        ),
        (
            ("run_id", "_id", "climate_model"),
            ("run_id", "_id", "climate_model", "time"),
        ),
    ),
)
def test_to_xarray_extras_with_id_coord(
    scm_run, extras, dimensions, expected_dimensions
):
    df = scm_run.timeseries()
    val_cols = df.columns.tolist()
    df = df.reset_index()

    df["climate_model"] = "base_m"
    df["run_id"] = 1
    df.loc[:, val_cols] = np.random.rand(df.shape[0], len(val_cols))

    big_df = [df]
    for climate_model in ["abc_m", "def_m", "ghi_m"]:
        for run_id in range(10):
            new_df = df.copy()
            new_df["run_id"] = run_id
            new_df["climate_model"] = climate_model
            new_df.loc[:, val_cols] = np.random.rand(df.shape[0], len(val_cols))

            big_df.append(new_df)

    big_df = pd.concat(big_df).reset_index(drop=True)
    big_df["random_key"] = (100 * np.random.random(big_df.shape[0])).astype(int)
    scm_run = scm_run.__class__(big_df)

    res = scm_run.to_xarray(dimensions=dimensions, extras=extras)

    do_basic_to_xarray_checks(res, scm_run, expected_dimensions, extras)

    assert set(extras) == set(res.coords) - set(res.dims)

    # check a couple of data points to make sure the translation is correct
    # and well-defined
    scm_run_meta = scm_run.meta
    for id_val in res["_id"].values[::10]:
        xarray_timeseries = res.sel(_id=id_val)
        fkwargs = {}
        for extra_col in extras:
            val = xarray_timeseries[extra_col].values.tolist()
            if isinstance(val, list):
                assert len(set(val)) == 1
                fkwargs[extra_col] = val[0]
            else:
                fkwargs[extra_col] = val

        for i, (key, value) in enumerate(fkwargs.items()):
            if i < 1:
                keep_meta_rows = scm_run_meta[key] == value
            else:
                keep_meta_rows &= scm_run_meta[key] == value

        meta_timeseries = scm_run_meta[keep_meta_rows]
        for _, row in meta_timeseries.iterrows():
            scm_run_filter = row.to_dict()
            scm_run_spot = scm_run.filter(**scm_run_filter)

            xarray_sel = {
                k: v for k, v in scm_run_filter.items() if k in xarray_timeseries.dims
            }
            xarray_spot = xarray_timeseries.sel(**xarray_sel)[
                scm_run_filter["variable"]
            ]

            npt.assert_array_equal(
                scm_run_spot.values.squeeze(), xarray_spot.values.squeeze()
            )


@pytest.mark.parametrize("ch", "!@#$%^&*()~`+={}]<>,;:'\".")
@pytest.mark.parametrize("weird_idx", (0, -1, 5))
def test_to_xarray_weird_names(scm_run, ch, weird_idx):
    new_vars = []
    for i, variable_name in enumerate(scm_run.get_unique_meta("variable")):
        if i < 1:
            new_name = list(variable_name)
            new_name.insert(weird_idx, ch)
            new_name = "".join(new_name)
            new_vars.append(new_name)
        else:
            new_vars.append(variable_name)

    dimensions = ("region", "scenario", "time")
    res = scm_run.to_xarray(dimensions=dimensions)

    do_basic_to_xarray_checks(
        res,
        scm_run,
        dimensions,
        (),
    )
    do_basic_check_of_data_points(res, scm_run, dimensions)


def get_multiple_units_scm_run(scm_run, new_unit, new_unit_alternate):
    first_var = scm_run.get_unique_meta("variable")[0]
    scm_run_first_var = scm_run.filter(variable=first_var)
    scm_run_first_var["unit"] = [
        v if i >= 1 else new_unit if v != new_unit else new_unit_alternate
        for i, v in enumerate(scm_run_first_var["unit"].tolist())
    ]
    scm_run_other_vars = scm_run.filter(variable=first_var, keep=False)

    return scmdata.run_append([scm_run_first_var, scm_run_other_vars])


def test_to_xarray_multiple_units_error(scm_run):
    scm_run = get_multiple_units_scm_run(scm_run, "J/yr", "MJ/yr")

    variable_unit_table = scm_run.meta[["variable", "unit"]].drop_duplicates()
    variable_counts = variable_unit_table["variable"].value_counts()
    more_than_one_unit_variables = variable_counts[variable_counts > 1]
    error_msg = re.escape(
        "The following variables are reported in more than one unit. "
        "Found variable-unit combinations are:\n{}".format(
            variable_unit_table[
                variable_unit_table["variable"].isin(
                    more_than_one_unit_variables.index.values
                )
            ]
        )
    )

    with pytest.raises(ValueError, match=error_msg):
        scm_run.to_xarray(dimensions=("region", "scenario", "time"), unify_units=False)


def test_to_xarray_unify_multiple_units(scm_run):
    scm_run = get_multiple_units_scm_run(scm_run, "J/yr", "MJ/yr")

    dimensions = ("region", "scenario", "time")
    res = scm_run.to_xarray(dimensions=dimensions, unify_units=True)
    do_basic_to_xarray_checks(
        res,
        scm_run,
        dimensions,
        (),
    )
    do_basic_check_of_data_points(res, scm_run, dimensions)


def test_to_xarray_unify_multiple_units_incompatible_units(scm_run):
    scm_run = get_multiple_units_scm_run(scm_run, "kg", "g")

    dimensions = ("region", "scenario", "time")

    first_var = scm_run.get_unique_meta("variable")[0]
    error_msg = re.escape(
        "Variable `{}` cannot be converted to a common unit. "
        "Units in the provided dataset: {}.".format(
            first_var, scm_run.filter(variable=first_var).get_unique_meta("unit")
        )
    )
    with pytest.raises(ValueError, match=error_msg):
        scm_run.to_xarray(dimensions=dimensions, unify_units=True)


@pytest.mark.parametrize(
    "dimensions,extras",
    (
        (
            ("junk",),
            (),
        ),
        (
            ("junk",),
            ("climate_model"),
        ),
        (("scenario", "junk_1"), ("junk",)),
        (("scenario",), ("junk",)),
        (("scenario",), ("junk", "climate_model")),
        (("scenario",), ("junk", "junk_2", "climate_model")),
    ),
)
def test_dimension_and_or_extra_not_in_metadata(scm_run, dimensions, extras):
    with pytest.raises(KeyError):
        scm_run.to_xarray(dimensions=dimensions, extras=extras)


def test_to_xarray_dimensions_extra_overlap(scm_run):
    dimensions = ("scenario", "region")
    extras = ("scenario",)

    error_msg = re.escape(
        "dimensions and extras cannot have any overlap. "
        "Current values in both dimensions and extras: {}".format({"scenario"})
    )
    with pytest.raises(ValueError, match=error_msg):
        scm_run.to_xarray(dimensions=dimensions, extras=extras)


def test_to_xarray_non_unique_timeseries(scm_run):
    dimensions = ("region",)

    error_msg = re.escape(
        "dimensions: `{}` and extras: `[]` do not uniquely define the timeseries, "
        "please add extra dimensions and/or extras".format(list(dimensions))
    )
    with pytest.raises(ValueError, match=error_msg):
        scm_run.to_xarray(dimensions=dimensions)


def test_nan_in_dimension(scm_run):
    run_id = np.arange(scm_run.shape[0]).astype(float)
    run_id[-2] = np.nan
    scm_run["run_id"] = run_id

    with pytest.raises(AssertionError, match="nan in dimension: `run_id`"):
        scm_run.to_xarray(dimensions=("run_id",))


def test_non_unique_meta(scm_run):
    scm_run["climate_model"] = ["b_model", "a_model", "a_model"]

    error_msg = re.escape(
        "Other metadata is not unique for dimensions: `{}` and extras: `{}`. "
        "Please add meta columns with more than one value to dimensions or "
        "extras.".format(["scenario"], [])
    )
    error_msg = (
        "{}\nNumber of unique values in each column:\n.*\n(\\s|\\S)*"
        "Existing values in the other metadata:.*".format(error_msg)
    )

    with pytest.raises(ValueError, match=error_msg):
        scm_run.to_xarray(dimensions=("scenario",))
