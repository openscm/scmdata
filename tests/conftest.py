"""
Fixtures and data for tests.
"""

from collections import namedtuple
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
import pytest

from scmdata import ScmDataFrame

try:
    from pyam import IamDataFrame
except ImportError as e:
    IamDataFrame = None

TEST_DATA = join(dirname(abspath(__file__)), "test_data")

TEST_DF_LONG_TIMES = pd.DataFrame(
    [
        [
            "a_model",
            "a_iam",
            "a_scenario",
            "World",
            "Primary Energy",
            "EJ/yr",
            1,
            6.0,
            6.0,
        ],
        [
            "a_model",
            "a_iam",
            "a_scenario",
            "World",
            "Primary Energy|Coal",
            "EJ/yr",
            0.5,
            3,
            3.0,
        ],
        [
            "a_model",
            "a_iam",
            "a_scenario2",
            "World",
            "Primary Energy",
            "EJ/yr",
            2,
            7,
            7,
        ],
    ],
    columns=[
        "climate_model",
        "model",
        "scenario",
        "region",
        "variable",
        "unit",
        datetime(1005, 1, 1),
        datetime(2010, 1, 1),
        datetime(3010, 12, 31),
    ],
)

TEST_DF = pd.DataFrame(
    [
        [
            "a_model",
            "a_iam",
            "a_scenario",
            "World",
            "Primary Energy",
            "EJ/yr",
            1,
            6.0,
            6.0,
        ],
        [
            "a_model",
            "a_iam",
            "a_scenario",
            "World",
            "Primary Energy|Coal",
            "EJ/yr",
            0.5,
            3,
            3.0,
        ],
        [
            "a_model",
            "a_iam",
            "a_scenario2",
            "World",
            "Primary Energy",
            "EJ/yr",
            2,
            7,
            7.0,
        ],
    ],
    columns=[
        "climate_model",
        "model",
        "scenario",
        "region",
        "variable",
        "unit",
        2005,
        2010,
        2015,
    ],
)

TEST_TS = np.array([[1, 6.0, 6], [0.5, 3, 3], [2, 7, 7]]).T


@pytest.fixture
def test_data_path():
    return TEST_DATA


@pytest.fixture
def doesnt_warn():
    @contextmanager
    def check_context():
        with pytest.warns(None) as record:
            yield
        if record:
            pytest.fail(
                "The following warnings were raised: {}".format(
                    [w.message for w in record.list]
                )
            )

    return check_context


@pytest.fixture(scope="function")
def test_pd_df():
    yield TEST_DF.copy()


@pytest.fixture(scope="function")
def test_scm_datetime_df():
    tdf = TEST_DF.copy()
    tdf.rename(
        {
            2005: datetime(2005, 6, 17, 12),
            2010: datetime(2010, 1, 3, 0),
            2015: datetime(2015, 1, 4, 0),
        },
        axis="columns",
        inplace=True,
    )

    yield ScmDataFrame(tdf)


@pytest.fixture(scope="function")
def test_ts():
    yield TEST_TS.copy()


@pytest.fixture(scope="function")
def test_iam_df():
    if IamDataFrame is None:
        pytest.skip("pyam is not installed")
    yield IamDataFrame(TEST_DF.copy())


@pytest.fixture(
    scope="function",
    params=[
        {"data": TEST_DF.copy()},
        pytest.param(
            {"data": IamDataFrame(TEST_DF.copy()).data},
            marks=pytest.mark.skipif(
                IamDataFrame is None, reason="pyam is not available"
            ),
        ),
        pytest.param(
            {"data": IamDataFrame(TEST_DF.copy()).timeseries()},
            marks=pytest.mark.skipif(
                IamDataFrame is None, reason="pyam is not available"
            ),
        ),
        {
            "data": TEST_TS.copy(),
            "columns": {
                "model": ["a_iam"],
                "climate_model": ["a_model"],
                "scenario": ["a_scenario", "a_scenario", "a_scenario2"],
                "region": ["World"],
                "variable": ["Primary Energy", "Primary Energy|Coal", "Primary Energy"],
                "unit": ["EJ/yr"],
            },
            "index": [2005, 2010, 2015],
        },
    ],
)
def test_scm_df(request):
    if IamDataFrame is None:
        pytest.skip("pyam is not installed")
    yield ScmDataFrame(**request.param)


@pytest.fixture(scope="function")
def test_processing_scm_df():
    yield ScmDataFrame(
        data=np.array([[1, 6.0, 7], [0.5, 3, 2], [2, 7, 0], [-1, -2, 3]]).T,
        columns={
            "model": ["a_iam"],
            "climate_model": ["a_model"],
            "scenario": ["a_scenario", "a_scenario", "a_scenario2", "a_scenario3"],
            "region": ["World"],
            "variable": [
                "Primary Energy",
                "Primary Energy|Coal",
                "Primary Energy",
                "Primary Energy",
            ],
            "unit": ["EJ/yr"],
        },
        index=[datetime(2005, 1, 1), datetime(2010, 1, 1), datetime(2015, 6, 12)],
    )


append_scm_df_pairs_cols = {
    "model": ["a_iam"],
    "climate_model": ["a_model"],
    "region": ["World"],
    "unit": ["EJ/yr"],
}
append_scm_df_pairs_scens = ["a_scenario", "a_scenario", "a_scenario2", "a_scenario3"]
append_scm_df_pairs_vars = [
    "Primary Energy",
    "Primary Energy|Coal",
    "Primary Energy",
    "Primary Energy",
]
append_scm_df_pairs_times = [
    datetime(2005, 1, 1),
    datetime(2010, 1, 1),
    datetime(2015, 6, 12),
]
append_scm_df_base = ScmDataFrame(
    data=np.array([[1, 6.0, 7], [0.5, 3, 2], [2, 7, 0], [-1, -2, 3]]).T,
    index=append_scm_df_pairs_times,
    columns={
        "scenario": append_scm_df_pairs_scens,
        "variable": append_scm_df_pairs_vars,
        **append_scm_df_pairs_cols,
    },
)
append_scm_df_pairs = [
    {
        "base": append_scm_df_base,
        "other": ScmDataFrame(
            data=np.array([[-1, 0, 1]]).T,
            index=append_scm_df_pairs_times,
            columns={
                "scenario": ["a_scenario"],
                "variable": ["Primary Energy"],
                **append_scm_df_pairs_cols,
            },
        ),
        "duplicate_rows": 1,
        "expected": ScmDataFrame(
            data=np.array([[0, 3.0, 4], [0.5, 3, 2], [2, 7, 0], [-1, -2, 3]]).T,
            index=append_scm_df_pairs_times,
            columns={
                "scenario": append_scm_df_pairs_scens,
                "variable": append_scm_df_pairs_vars,
                **append_scm_df_pairs_cols,
            },
        ),
    },
    {
        "base": append_scm_df_base,
        "other": ScmDataFrame(
            data=np.array([[3, 3.5, 3.7], [1, 7, 11], [-2, 1, -1.4]]).T,
            index=append_scm_df_pairs_times,
            columns={
                "scenario": ["a_scenario", "b_scenario", "b_scenario2"],
                "variable": ["Primary Energy", "Primary Energy|Coal", "Primary Energy"],
                **append_scm_df_pairs_cols,
            },
        ),
        "duplicate_rows": 1,
        "expected": ScmDataFrame(
            data=np.array(
                [
                    [2, 4.75, 5.35],
                    [0.5, 3, 2],
                    [2, 7, 0],
                    [-1, -2, 3],
                    [1, 7, 11],
                    [-2, 1, -1.4],
                ]
            ).T,
            index=append_scm_df_pairs_times,
            columns={
                "scenario": append_scm_df_pairs_scens + ["b_scenario", "b_scenario2"],
                "variable": append_scm_df_pairs_vars
                            + ["Primary Energy|Coal", "Primary Energy"],
                **append_scm_df_pairs_cols,
            },
        ),
    },
    {
        "base": append_scm_df_base,
        "other": ScmDataFrame(
            data=np.array([[3, 3.5, 3.7], [1, 7, 11], [-2, 1, -1.4], [-3, -4, -5]]).T,
            index=append_scm_df_pairs_times,
            columns={
                "scenario": ["a_scenario", "b_scenario", "b_scenario2", "a_scenario3"],
                "variable": [
                    "Primary Energy",
                    "Primary Energy|Coal",
                    "Primary Energy",
                    "Primary Energy",
                ],
                **append_scm_df_pairs_cols,
            },
        ),
        "duplicate_rows": 2,
        "expected": ScmDataFrame(
            data=np.array(
                [
                    [2, 4.75, 5.35],
                    [0.5, 3, 2],
                    [2, 7, 0],
                    [-2, -3, -1],
                    [1, 7, 11],
                    [-2, 1, -1.4],
                ]
            ).T,
            index=append_scm_df_pairs_times,
            columns={
                "scenario": append_scm_df_pairs_scens + ["b_scenario", "b_scenario2"],
                "variable": append_scm_df_pairs_vars
                            + ["Primary Energy|Coal", "Primary Energy"],
                **append_scm_df_pairs_cols,
            },
        ),
    },
    {
        "base": append_scm_df_base,
        "other": ScmDataFrame(
            data=np.array([[-1, 0, 1], [3, 4, 4.5], [0.1, 0.2, 0.3], [-4, -8, 10]]).T,
            index=append_scm_df_pairs_times,
            columns={
                "scenario": append_scm_df_pairs_scens,
                "variable": append_scm_df_pairs_vars,
                **append_scm_df_pairs_cols,
            },
        ),
        "duplicate_rows": 4,
        "expected": ScmDataFrame(
            data=np.array(
                [[0, 3, 4], [1.75, 3.5, 3.25], [1.05, 3.6, 0.15], [-2.5, -5, 6.5]]
            ).T,
            index=append_scm_df_pairs_times,
            columns={
                "scenario": append_scm_df_pairs_scens,
                "variable": append_scm_df_pairs_vars,
                **append_scm_df_pairs_cols,
            },
        ),
    },
]


@pytest.fixture(params=append_scm_df_pairs)
def test_append_scm_dfs(request):
    return request.param


@pytest.fixture
def iamdf_type():
    return IamDataFrame


@pytest.fixture(scope="module")
def rcp26():
    fname = join(TEST_DATA, "rcp26_emissions.csv")
    return ScmDataFrame(fname)


possible_source_values = [[1, 5, 3, 5, 7, 3, 2, 9]]

possible_target_values = [  # TODO: Use years here after create_time_points has been fixed
    dict(
        source_start_time=np.datetime64("2000-01-01"),
        source_period_length=np.timedelta64(10, "D"),
        target_start_time=np.datetime64("2000-01-01") - np.timedelta64(5, "D"),
        target_period_length=np.timedelta64(5, "D"),
        source_values=possible_source_values[0],
        target_values=[-1, 1, 3, 5, 4, 3, 4, 5, 6, 7, 5, 3, 2.5, 2, 5.5, 9, 12.5],
        interpolation_type='linear',
        extrapolation_type='linear',
    ),
    dict(
        source_start_time=np.datetime64("2000-01-01"),
        source_period_length=np.timedelta64(10, "D"),
        target_start_time=np.datetime64("2000-01-01") - np.timedelta64(50, "D"),
        target_period_length=np.timedelta64(50, "D"),
        source_values=possible_source_values[0],
        target_values=[1, 1, 3, 9],
        interpolation_type='linear',
        extrapolation_type='constant',
    )
]

test_combinations = []


def create_time_points(
        start_time: np.datetime64,
        period_length: np.timedelta64,
        points_num: int
):
    end_time_output = start_time + (points_num - 1) * period_length
    return np.linspace(
        start_time.astype("datetime64[s]").astype(float),
        end_time_output.astype("datetime64[s]").astype(float),
        points_num,
        dtype="datetime64[s]",
    )


Combination = namedtuple(
    "TestCombination",
    [
        "source",
        "source_values",
        "target",
        "target_values",
        "interpolation_type",
        "extrapolation_type",
    ],
)
for index in possible_target_values:
    combination = Combination(
        source_values=np.array(index["source_values"]),
        source=create_time_points(
            index["source_start_time"],
            index["source_period_length"],
            len(index["source_values"]),
        ),
        target_values=np.array(index["target_values"]),
        target=create_time_points(
            index["target_start_time"],
            index["target_period_length"],
            len(index["target_values"]),
        ),
        interpolation_type=index["interpolation_type"],
        extrapolation_type=index["extrapolation_type"],
    )
    test_combinations.append(combination)


@pytest.fixture(params=test_combinations)
def combo(request):
    return request.param


@pytest.fixture(params=test_combinations, scope="function")
def combo_df(request):
    combination = deepcopy(request.param)
    vals = combination._asdict()
    source = combination.source

    df = ScmDataFrame(
        combination.source_values,
        columns={
            "scenario": ["a_scenario"],
            "model": ["a_model"],
            "region": ["World"],
            "variable": ["Emissions|BC"],
            "unit": ["Mg /yr"],
        },
        index=source,
    )

    return Combination(**vals), df
