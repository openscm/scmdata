"""
Testing utilities
"""

import numpy as np
import numpy.testing as npt
import packaging.version
import pandas as pd
import pandas.testing as pdt


def _check_pandas_less_110():
    return packaging.version.parse(pd.__version__) < packaging.version.Version("1.1.0")


def _assert_frame_equal(left, right, **kwargs):
    if _check_pandas_less_110():
        kwargs.pop("rtol", None)
        kwargs.pop("atol", None)

    pdt.assert_frame_equal(left, right, **kwargs)


def assert_scmdf_almost_equal(
    left, right, allow_unordered=False, check_ts_names=True, rtol=1e-5, atol=1e-8
):
    """
    Check that left and right :obj:`ScmRun` are equal.

    Parameters
    ----------
    left : :obj:`ScmRun`

    right : :obj:`ScmRun`

    allow_unordered : bool
        If true, the order of the timeseries is not checked

    check_ts_names : bool
        If true, check that the meta names are the same

    rtol : float
        Relative tolerance on numeric comparisons

    atol : float
        Absolute tolerance on numeric comparisons

    Raises
    ------
    AssertionError
        ``left`` and ``right`` are not equal
    """
    # Check that the meta data is close
    if allow_unordered or not check_ts_names:

        # Checks that all the timeseries are named the same
        if check_ts_names:
            df1_index = np.argsort(left.meta.index)
            df2_index = np.argsort(right.meta.index)
            _assert_frame_equal(left.meta, right.meta, check_like=True)
            npt.assert_allclose(
                left.values[df1_index], right.values[df2_index], rtol=rtol, atol=atol
            )

        else:
            # ignore differing meta index labels
            # instead sort by meta values then check equality
            left_sorted = left.timeseries().sort_index()
            right_ts = right.timeseries()
            # check metadata columns are same set
            if set(left_sorted.index.names) != set(right_ts.index.names):
                raise AssertionError(
                    "{} != {}".format(
                        set(left_sorted.index.names), set(right_ts.index.names)
                    )
                )

            right_sorted = right.timeseries(left_sorted.index.names).sort_index()
            # this checks both the index (i.e. sorted meta) and values are the same
            _assert_frame_equal(left_sorted, right_sorted, rtol=rtol, atol=atol)

    else:
        _assert_frame_equal(left.meta, right.meta)
        npt.assert_allclose(left.values, right.values, rtol=rtol, atol=atol)
