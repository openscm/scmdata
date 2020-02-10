import pandas.testing as pdt
import numpy.testing as npt


def assert_scmdf_almost_equal(df1, df2):
    # Check that the meta data is close

    pdt.assert_frame_equal(df1.meta, df2.meta, check_like=True)
