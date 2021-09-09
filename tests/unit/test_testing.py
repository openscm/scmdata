import pytest

import scmdata.testing


def test_almost_equal_message(scm_run):
    other = scmdata.run_append(
        [
            scm_run.filter(scenario="a_scenario") * 2,
            scm_run.filter(scenario="a_scenario2"),
        ]
    )

    # matches the first item even though multiple are wrong
    msg = r"\'a_scenario\', \'EJ/yr\', \'Primary Energy\'"

    with pytest.raises(AssertionError, match=msg):
        scmdata.testing.assert_scmdf_almost_equal(
            scm_run, other, allow_unordered=True, check_ts_names=False
        )
