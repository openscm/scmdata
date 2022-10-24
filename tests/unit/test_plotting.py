import builtins
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scmdata.run import ScmRun

pytest.importorskip("seaborn")


def test_plotting_injected_methods(scm_run):
    assert hasattr(scm_run, "line_plot")
    assert hasattr(scm_run, "lineplot")


def test_plotting_long_data(scm_run):
    long_data = scm_run.long_data()

    assert "value" in long_data.columns
    assert len(long_data) == scm_run.shape[0] * scm_run.shape[1]

    exp = scm_run.filter(year=2005, scenario="a_scenario2").values.squeeze()
    obs = long_data[
        (long_data.scenario == "a_scenario2") & (long_data.time.dt.year == 2005)
    ].value.squeeze()

    assert exp == obs


def hide_import(import_name):
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == import_name:
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    return mocked_import


@pytest.fixture
def hide_matplotlib(monkeypatch):
    monkeypatch.setattr(builtins, "__import__", hide_import("matplotlib.pyplot"))


@pytest.fixture
def hide_seaborn(monkeypatch):
    monkeypatch.setattr(builtins, "__import__", hide_import("seaborn"))


def test_no_matplotlib(scm_run, hide_matplotlib):
    with pytest.raises(
        ImportError, match="matplotlib is not installed. Run 'pip install matplotlib'"
    ):
        scm_run.plumeplot()


def test_no_seaborn(scm_run, hide_seaborn):
    with pytest.raises(
        ImportError, match="seaborn is not installed. Run 'pip install seaborn'"
    ):
        scm_run.lineplot()


@patch("seaborn.lineplot")
@patch.object(ScmRun, "long_data")
def test_lineplot(mock_long_data, mock_seaborn_lineplot, scm_run):
    trv = "test long_data return value"
    mock_long_data.return_value = trv

    scm_run.lineplot(time_axis="year")

    mock_long_data.assert_called_with(time_axis="year")

    mock_seaborn_lineplot.assert_called_with(
        ci="sd", data=trv, estimator=np.median, hue="scenario", x="time", y="value"
    )


@patch("seaborn.lineplot")
@patch.object(ScmRun, "long_data")
def test_lineplot_kwargs(mock_long_data, mock_seaborn_lineplot, scm_run):
    tkwargs = {
        "x": "x",
        "y": "y",
        "hue": "hue",
        "ci": "ci",
        "estimator": "estimator",
    }
    trv = "test long_data return value"
    mock_long_data.return_value = trv

    scm_run.lineplot(time_axis="time_axis", **tkwargs)

    mock_long_data.assert_called_with(time_axis="time_axis")

    mock_seaborn_lineplot.assert_called_with(data=trv, **tkwargs)


@pytest.mark.parametrize("single_unit", (True, False))
@patch("seaborn.lineplot")
def test_lineplot_units(mock_seaborn_lineplot, single_unit, scm_run):
    units = scm_run["unit"].values

    if single_unit:
        units[:] = "J/yr"
    else:
        units[-1] = "J/yr"

    scm_run["unit"] = units

    mock_ax = MagicMock()
    mock_seaborn_lineplot.return_value = mock_ax

    scm_run.lineplot(time_axis="year")

    mock_seaborn_lineplot.assert_called()

    if single_unit:
        mock_ax.set_ylabel.assert_called_with(units[0])
    else:
        mock_ax.set_ylabel.assert_not_called()


@patch("seaborn.lineplot")
def test_lineplot_base(mock_seaborn_lineplot, base_scm_run, scm_run):
    mock_ax = MagicMock()
    mock_seaborn_lineplot.return_value = mock_ax

    base_scm_run.lineplot(time_axis="year")
    mock_seaborn_lineplot.assert_called()
    _, call_kwargs = mock_seaborn_lineplot.call_args_list[0]
    assert "hue" not in call_kwargs

    mock_seaborn_lineplot.reset_mock()
    scm_run.lineplot(time_axis="year")
    _, call_kwargs = mock_seaborn_lineplot.call_args_list[0]
    assert "hue" in call_kwargs
