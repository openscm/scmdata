import numpy as np
import numpy.testing as npt
import pandas.testing as pdt


def assert_scmdf_almost_equal(df1, df2, allow_unordered=False):
    # Check that the meta data is close

    if allow_unordered:
        pdt.assert_frame_equal(df1.meta, df2.meta, check_like=True)
        df1_index = df1.meta.index
        df2_index = df2.meta.index
        npt.assert_allclose(df1.values[np.argsort(df1_index)], df2.values[np.argsort(df2_index)])
    else:
        pdt.assert_frame_equal(df1.meta, df2.meta)
        npt.assert_allclose(df1.values, df2.values)
