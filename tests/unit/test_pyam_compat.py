import re
from unittest import mock

import pytest

from scmdata.pyam_compat import LongDatetimeIamDataFrame


def test_to_int_value_error(test_iam_df):
    idf = test_iam_df.data.rename({"year": "time"}, axis="columns").reset_index()
    idf.loc[:, "time"] = "2003/1/1"
    bad_val = "20311/123/1"
    postion = 4
    idf.loc[postion, "time"] = bad_val

    error_msg = re.escape(
        f"Unknown string format: {bad_val} present at position {postion}"
    )
    with pytest.raises(ValueError, match=error_msg):
        LongDatetimeIamDataFrame(idf)


@mock.patch("scmdata.run.LongDatetimeIamDataFrame", None)
def test_pyam_missing(scm_run):
    with pytest.raises(ImportError):
        scm_run.to_iamdataframe()
