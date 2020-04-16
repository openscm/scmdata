def test_plotting_injected_methods(test_scm_df, test_scm_run):
    for obj in [test_scm_run, test_scm_df]:
        assert hasattr(obj, "line_plot")
        assert hasattr(obj, "lineplot")


def test_plotting_long_data(test_scm_run):
    long_data = test_scm_run.long_data()

    assert "value" in long_data.columns
    assert len(long_data) == test_scm_run.shape[0] * test_scm_run.shape[1]

    exp = test_scm_run.filter(year=2005, scenario="a_scenario2").values.squeeze()
    obs = long_data[(long_data.scenario == "a_scenario2") & (long_data.time.dt.year == 2005)].value.squeeze()

    assert exp == obs
