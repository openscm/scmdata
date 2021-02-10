import pytest

from scmdata import ScmRun


sample_quantiles_plumes = pytest.mark.parametrize("quantiles_plumes", (
    (((0.05, 0.95), 0.5), ((0.5,), 1.0),),
    (((0.17, 0.83), 0.7),),
))


def test_plumeplot_default(plumeplot_scmrun):
    plumeplot_scmrun.plumeplot()


@sample_quantiles_plumes
def test_plumeplot(plumeplot_scmrun, quantiles_plumes):
    plumeplot_scmrun.plumeplot(quantiles_plumes=quantiles_plumes)


@sample_quantiles_plumes
def test_plumeplot_pre_calculated(plumeplot_scmrun, quantiles_plumes):
    quantiles = [v for qv in quantiles_plumes for v in qv[0]]
    summary_stats = ScmRun(
        plumeplot_scmrun.quantiles_over("ensemble_member", quantiles=quantiles)
    )
    summary_stats.plumeplot(
        quantiles_plumes=quantiles_plumes,
        pre_calculated=True,
    )


def test_plumeplot_warns_dashes_without_lines(scm_run):
    with pytest.warns(UserWarning) as record:
        scm_run.plumeplot(
            quantiles_plumes=(((0.17, 0.83), 0.7),),
            quantile_over="ensemble_member",
            dashes={"Surface Air Temperature Change": "--"},
        )

    assert len(record) == 1
    assert record[0].message.args[0] == (
        "`dashes` was passed but no lines were plotted, the style "
        "settings will not be used"
    )
