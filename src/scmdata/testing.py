"""
Testing utilities
"""

import numpy as np
import numpy.testing as npt
import pandas.testing as pdt


def assert_scmdf_almost_equal(left, right, allow_unordered=False, check_ts_names=True):
    """
    Check that left and right :obj:`ScmDataFrame` or :obj:`ScmRun` are equal.

    Parameters
    ----------
    left : :obj:`ScmDataFrame`, :obj:`ScmRun`
    right : :obj:`ScmDataFrame`, :obj:`ScmRun`
    allow_unordered : bool
        If true, the order of the timeseries is not checked
    check_ts_names : bool
        If true, only check that the meta names are the same

    """
    # Check that the meta data is close
    if allow_unordered or not check_ts_names:
        df1_index = np.argsort(left.meta.index)
        df2_index = np.argsort(right.meta.index)
        if check_ts_names:
            pdt.assert_frame_equal(left.meta, right.meta, check_like=True)
        else:
            assert (
                left.meta.values[df1_index]
                == right.meta[left.meta.columns].values[df2_index]
            ).all()
        npt.assert_allclose(left.values[df1_index], right.values[df2_index])
    else:
        pdt.assert_frame_equal(left.meta, right.meta)
        npt.assert_allclose(left.values, right.values)
