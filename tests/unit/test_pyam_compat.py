import re
from unittest import mock

import pandas as pd
import pytest

from scmdata.pyam_compat import LongDatetimeIamDataFrame


def test_to_int_value_error(test_iam_df):
    idf = test_iam_df.data.rename({"year": "time"}, axis="columns").reset_index()
    idf.loc[:, "time"] = "2003/1/1"
    bad_val = "20311/123/1"
    postion = 4
    idf.loc[postion, "time"] = bad_val

    if pd.__version__.startswith("1"):
        error_msg = re.escape(
            f"Unknown string format: {bad_val} present at position {postion}"
        )

    else:
        error_msg = re.escape(
            f'time data "{bad_val}" '
            "doesn't match format "
            '"%Y/%m/%d", at position 4. You might want to try:'
        )

    with pytest.raises(ValueError, match=error_msg):
        LongDatetimeIamDataFrame(idf)


@mock.patch("scmdata.run.LongDatetimeIamDataFrame", None)
def test_pyam_missing(scm_run):
    with pytest.raises(ImportError):
        scm_run.to_iamdataframe()
