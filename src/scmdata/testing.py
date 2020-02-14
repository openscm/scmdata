import numpy as np
import numpy.testing as npt
import pandas.testing as pdt


def assert_scmdf_almost_equal(df1, df2, allow_unordered=False, check_ts_names=True):
    """
    Assert that two :obj:`ScmDataFrame`s are almost equal to each other.

    Parameters
    ----------
    df1, df2 : :obj:`ScmDataFrame`
        :obj:`ScmDataFrame` instances to compare

    allow_unordered : bool
        Can the data in ``df1`` and ``df2`` be in any order and still pass?

    check_ts_names : bool
        Do ``df1`` and ``df2``'s :attr:`meta` attributes have to be the same?
        TODO: decide whether to rename `check_ts_names` to `check_meta` for clarity.
    """
    # Check that the meta data is close
    if allow_unordered or not check_ts_names:
        df1_index = np.argsort(df1.meta.index)
        df2_index = np.argsort(df2.meta.index)
        if check_ts_names:
            pdt.assert_frame_equal(df1.meta, df2.meta, check_like=True)
        else:
            assert (df1.meta.values[df1_index] == df2.meta[df1.meta.columns].values[df2_index]).all()
        npt.assert_allclose(df1.values[df1_index], df2.values[df2_index])
    else:
        pdt.assert_frame_equal(df1.meta, df2.meta)
        npt.assert_allclose(df1.values, df2.values)
