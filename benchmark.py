import scmdata
import numpy as np


def create_run():
    years = np.arange(1990, 2010)
    n = 5000
    return scmdata.ScmRun(
        np.ones((len(years), n), dtype=float),
        years,
        {
            "variable": [str(x) for x in range(n)],
            "unit": ["Mt CO2 / year"] * n,
            "category": ["0"] * n,
            "model": ["TEST"] * n,
            "scenario": ["TEST"] * n,
            "region": ["WORLD"] * n,
            "stage": ["test"] * n,
        },
    )


import timeit

single_existing = timeit.timeit(
    "a['scenario'] = 'This is not a test'",
    setup="a = create_run()",
    globals=globals(),
    number=1000,
)
print(f"{single_existing=}")
multi_existing = timeit.timeit(
    "a['scenario'] = b",
    setup="a = create_run(); b = ['This is not a test'] * 5000",
    globals=globals(),
    number=1000,
)
print(f"{multi_existing=}")
