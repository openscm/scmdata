import itertools
import os.path
import random
import string

import numpy as np
import pytest

import scmdata
import scmdata.testing


@pytest.fixture(params=[10, 10**2, 10**3, 10**3.5, 10**4, 10**4.5])
def big_scmrun(request):
    length = int(request.param)
    t_steps = 750
    variables = [
        "Surface Air Temperature Change",
        "Surface Air Ocean Blended Temperature Change",
        "Effective Radiative Forcing",
        "Atmospheric Concentrations|CO2",
    ]
    scenarios = ["ssp119", "ssp245", "ssp434", "ssp460", "ssp126", "esm-1pctCo2"]
    regions = [
        "World",
        "World|R5.2ASIA",
        "World|R5.2LAM",
        "World|R5.2MAF",
        "World|R5.2REF",
        "World|R5.2OECD",
        "World|Bunkers",
    ]
    climate_models = ["MAGICC7", "MAGICC6", "MAGICC5", "Bill", "Bloggs"]
    to_squash = ["aa", "ab", "ba", "bb"]

    return scmdata.ScmRun(
        np.random.random((length, t_steps)).T,
        index=range(1750, 1750 + t_steps),
        columns={
            "model": "unspecified",
            "variable": random.choices(variables, k=length),
            "unit": "unknown",
            "scenario": random.choices(scenarios, k=length),
            "region": random.choices(regions, k=length),
            "climate_model": random.choices(climate_models, k=length),
            "to_squash": random.choices(to_squash, k=length),
            "ensemble_member": range(length),
        },
    )


def test_recreate_from_timeseries(benchmark, big_scmrun):
    def recreate():
        return scmdata.ScmRun(big_scmrun.timeseries())

    benchmark.pedantic(recreate, iterations=1, rounds=5)


def test_filter(benchmark, big_scmrun):
    v = big_scmrun.get_unique_meta("variable")[0]

    def variable_filter():
        return big_scmrun.filter(variable=v, year=range(1850, 1910 + 1))

    result = benchmark.pedantic(variable_filter, iterations=1, rounds=2)
    assert result.get_unique_meta("variable", no_duplicates=True) == v
    assert result["year"].iloc[0] == 1850
    assert result["year"].iloc[-1] == 1910


def test_get_unique_meta(benchmark, big_scmrun):
    def retrieve_unique_meta():
        return big_scmrun.get_unique_meta("variable")

    result = benchmark.pedantic(retrieve_unique_meta, iterations=1, rounds=2)
    assert len(result) > 0


def test_append(benchmark, big_scmrun):
    def setup():
        other = big_scmrun.copy()
        other["ensemble_member"] += max(other["ensemble_member"]) + 1

        return (), {"other": other}

    def append_runs(other):
        return big_scmrun.append(other)

    res = benchmark.pedantic(append_runs, setup=setup, iterations=1, rounds=2)
    assert len(res) == 2 * len(big_scmrun)


def test_append_other_time_axis(benchmark, big_scmrun):
    def setup():
        other = big_scmrun.timeseries(time_axis="year")
        other.columns = other.columns.map(lambda x: x + max(other.columns) + 1)
        other = scmdata.ScmRun(other)
        other["ensemble_member"] += max(other["ensemble_member"]) + 1

        return (), {"other": other}

    def append_runs_different_times(other):
        return scmdata.run_append([big_scmrun, other])

    res = benchmark.pedantic(
        append_runs_different_times, setup=setup, iterations=1, rounds=2
    )
    assert len(res) == 2 * len(big_scmrun)
    assert res.shape[1] == 2 * big_scmrun.shape[1]


def test_set_meta_single(benchmark, big_scmrun):
    def set_meta():
        big_scmrun["model"] = "new value"

        return big_scmrun

    res = benchmark.pedantic(set_meta, iterations=1, rounds=2)
    assert res.get_unique_meta("model", no_duplicates=True) == "new value"


def test_set_meta_all_with_apply(benchmark, big_scmrun):
    def set_meta():
        big_scmrun["variable"] = big_scmrun["variable"].apply(
            lambda x: "New|{}".format(x)
        )

        return big_scmrun

    res = benchmark.pedantic(set_meta, iterations=1, rounds=2)
    assert res["variable"].str.startswith("New|").all()


def test_set_meta_reduce_uniqueness(benchmark, big_scmrun):
    def set_meta():
        big_scmrun["to_squash"] = big_scmrun["to_squash"].apply(
            lambda x: "a" if x.startswith("a") else "b"
        )

        return big_scmrun

    original_set = big_scmrun.get_unique_meta("to_squash")
    res = benchmark.pedantic(set_meta, iterations=1, rounds=2)
    assert not any([v in original_set for v in res.get_unique_meta("to_squash")])


def test_interpolate(benchmark, big_scmrun):
    def interp():
        return big_scmrun.interpolate(range(1500, 2500))

    res = benchmark.pedantic(interp, iterations=1, rounds=1)
    assert res.shape == (len(big_scmrun), 1000)


@pytest.mark.parametrize("n_to_append", (10, 10**2, 10**3, 10**4))
def test_append_multiple_same_time(benchmark, big_scmrun, n_to_append):
    total_size = n_to_append * big_scmrun.shape[0]
    if total_size >= 10**4:
        pytest.skip("this could be very slow...")

    to_append = []
    for i in range(n_to_append):
        tmp = big_scmrun.copy()
        tmp["ensemble_member"] = range(
            i * big_scmrun.shape[0], (i + 1) * big_scmrun.shape[0]
        )
        to_append.append(tmp)

    def append():
        return scmdata.run_append(to_append)

    res = benchmark.pedantic(append, iterations=1, rounds=1)

    assert res.shape[0] == big_scmrun.shape[0] * n_to_append
    assert res.shape[1] == big_scmrun.shape[1]


@pytest.mark.parametrize("n_models", (2, 3, 4))
@pytest.mark.parametrize("n_ensemble_members", (10**1, 20, 30))
def test_to_from_nc(benchmark, tmpdir, n_models, n_ensemble_members):
    t_steps = 75

    ensemble_members = list(range(n_ensemble_members))
    variables = [
        "Surface Air Temperature Change",
    ]
    models = [v for v in string.ascii_lowercase[:n_models]]

    regions = [
        "World",
        "World|R5.2ASIA",
        "World|R5.2LAM",
        "World|R5.2MAF",
        "World|R5.2REF",
        "World|R5.2OECD",
        "World|Bunkers",
    ]
    climate_models = ["MAGICC7"]

    n_ts = n_models * len(regions) * len(variables) * n_ensemble_members

    tmp = itertools.product(models, regions, variables, ensemble_members)
    models_long, regions_long, variables_long, ensemble_members_long = zip(*tmp)
    scenarios_long = [v * 2 for v in models_long]

    start = scmdata.ScmRun(
        np.random.random((n_ts, t_steps)).T,
        index=range(1750, 1750 + t_steps),
        columns={
            "variable": variables_long,
            "unit": "unknown",
            "model": models_long,
            "scenario": scenarios_long,
            "region": regions_long,
            "climate_model": climate_models,
            "ensemble_member": ensemble_members_long,
        },
    )

    def round_trip():
        nc_file = os.path.join(tmpdir, "nc_dump.nc")
        start.to_nc(
            nc_file,
            dimensions=("variable", "region", "model", "ensemble_member"),
            extras=("scenario",),
        )

        res = scmdata.ScmRun.from_nc(nc_file)

        return res

    res = benchmark.pedantic(round_trip, iterations=1, rounds=1)
    scmdata.testing.assert_scmdf_almost_equal(
        start, res, allow_unordered=True, check_ts_names=False
    )
