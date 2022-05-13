import scmdata
from scmdata.remote import read_api_facets, read_api_timeseries


def test_read_ndcs():
    ndcs_url = "https://api.climateresource.com.au/ndcs/v1"

    res = read_api_timeseries(
        ndcs_url,
        **{
            "version": "14Feb2022b_CR",
            "variable": "Emissions|Total *",
            "hot_air": "exclude",
            "category": "Current",
        },
    )

    assert len(res)
    assert isinstance(res, scmdata.ScmRun)


def test_read_facets():
    ndcs_url = "https://api.climateresource.com.au/ndcs/v1/"

    res = read_api_facets(
        ndcs_url,
        **{
            "version": "14Feb2022b_CR",
            "variable": "Emissions|Total *",
            "hot_air": "exclude",
            "category": "Current",
        },
    )

    assert (res.columns == ["name", "value", "count"]).all()
    assert len(res)
