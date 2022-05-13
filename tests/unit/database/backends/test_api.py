from scmdata.database.backends import APIDatabaseBackend
from scmdata.database import ScmDatabase


def test_api_backend():
    backend = APIDatabaseBackend()

    pass


def test_api_database():
    db = ScmDatabase(
        backend="api",
        backend_config={"url": "https://api.climateresource.com.au/ndcs/v1/"},
    )
    db.load()
