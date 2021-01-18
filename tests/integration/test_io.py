import os.path

import scmdata
import scmdata.testing


def test_read_write_circularity(test_scm_df_monthly, tmpdir):
    tempfile = os.path.join(tmpdir, "test.csv")

    test_scm_df_monthly.to_csv(tempfile)

    read = scmdata.ScmRun(tempfile)

    scmdata.testing.assert_scmdf_almost_equal(read, test_scm_df_monthly)
