import re

import numpy as np
import pytest

from scmdata.time import InsufficientDataError, TimeseriesConverter


def test_short_data(combo):
    timeseriesconverter = TimeseriesConverter(
        combo.source, combo.target, combo.interpolation_type, combo.extrapolation_type,
    )
    for a in [[], [0], [0, 1]]:
        with pytest.raises(InsufficientDataError):
            timeseriesconverter._convert(np.array(a), combo.source, combo.target)


def test_none_extrapolation_error(combo):
    target = np.asarray(
        [
            combo.source[0] - np.timedelta64(1, "s"),
            combo.source[0],
            combo.source[-1] + np.timedelta64(1, "s"),
        ],
        dtype=np.datetime64,
    )

    error_msg = re.escape(
        "Target time points are outside the source time points, use an "
        "extrapolation type other than None"
    )
    with pytest.raises(InsufficientDataError, match=error_msg):
        TimeseriesConverter(
            combo.source, target, combo.interpolation_type, None
        )


def test_really_long_timespan():
    source = np.asarray(
        [
            np.datetime64("1000-01-01"),
            np.datetime64("2000-01-01"),
            np.datetime64("3500-01-01"),
        ],
        dtype=np.datetime64,
    )
    source_vals = [1.0, 2.0, 3.5]
    target = np.asarray(
        [
            np.datetime64("1000-01-01"),
            np.datetime64("2000-01-01"),
            np.datetime64("3000-01-01"),
        ],
        dtype=np.datetime64,
    )
    target_vals = [1.0, 2.0, 3.0]
    c = TimeseriesConverter(source, target,)

    np.testing.assert_allclose(c.convert_to(target_vals), source_vals, rtol=1e-3)
    np.testing.assert_allclose(c.convert_from(source_vals), target_vals, rtol=1e-3)
