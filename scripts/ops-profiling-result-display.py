import pstats
from pstats import SortKey

for key in [
    "op_scmrun.prof",
    "op_pandas_align_then_pint.prof",
    "op_pandas_align_then_pint_no_duplicate_check.prof",
]:
    print(key)
    print("-" * len(key))
    res_stats = pstats.Stats(key)
    res_stats.sort_stats(SortKey.CUMULATIVE).print_stats(10)
    res_stats.sort_stats(SortKey.CUMULATIVE).print_callees("timeseries")
    res_stats.sort_stats(SortKey.TIME).print_stats(10)
    # res_stats.sort_stats(SortKey.TIME).print_callers(10)
    print("-" * len(key))
    print()
