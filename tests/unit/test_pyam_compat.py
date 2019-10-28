import re
import sys
from unittest import mock

import pytest

from scmdata.pyam_compat import LongDatetimeIamDataFrame


def test_to_int_value_error(test_iam_df):
    idf = test_iam_df.data.rename({"year": "time"}, axis="columns").reset_index()
    idf.loc[:, "time"] = "2003/1/1"
    bad_val = "20311/123/1"
    idf.loc[4, "time"] = bad_val

    error_msg = re.escape(
        "All time values must be convertible to datetime. The following values are "
        "not:\n"
    )
    with pytest.raises(ValueError, match=error_msg):
        LongDatetimeIamDataFrame(idf)


@mock.patch("scmdata.dataframe.rst.LongDatetimeIamDataFrame", None)
def test_pyam_missing(test_scm_df):
    with pytest.raises(ImportError):
        test_scm_df.to_iamdataframe()


def test_pyam_missing_loading():
    with mock.patch.dict(sys.modules, {"pyam": None}):
        # not sure whether deleting like this is fine because of the context manager
        # or a terrible idea...
        del sys.modules["scmdata.pyam_compat"]
        from scmdata.pyam_compat import IamDataFrame as res
        from scmdata.pyam_compat import LongDatetimeIamDataFrame as res_3

        assert all([r is None for r in [res, res_3]])

    with mock.patch.dict(sys.modules, {"matplotlib.axes": None}):
        # not sure whether deleting like this is fine because of the context manager
        # or a terrible idea...
        del sys.modules["scmdata.pyam_compat"]
        from scmdata.pyam_compat import Axes as res_2

        assert all([r is None for r in [res_2]])

    from scmdata.pyam_compat import IamDataFrame as res
    from scmdata.pyam_compat import Axes as res_2
    from scmdata.pyam_compat import LongDatetimeIamDataFrame as res_3

    assert all([r is not None for r in [res, res_2, res_3]])
