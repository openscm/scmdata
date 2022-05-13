from scmdata.database.backends import BaseDatabaseBackend


class APIDatabaseBackend(BaseDatabaseBackend):
    """
    Fetch data from a supported API

    Supported APIs provide two endpoints
    """

    def save(self, sr):
        """
        Save data

        This is not possible for an API backend

        Parameters
        ----------
        sr: scmdata.ScmRun

        Raises
        ------
        NotImplementedError
            API backends are read-only
        """
        raise NotImplementedError("API backends are read-only")

    def load(self, key):
        """
        Load data at a given key

        Parameters
        ----------
        key : str
            Key to load

        Returns
        -------
        scmdata.ScmRun
        """
        raise NotImplementedError("API backends are read-only")

    def delete(self, key):
        """
        Delete a given key

        Parameters
        ----------
        key: str
        """
        raise NotImplementedError("API backends are read-only")

    def get(self, filters):
        """
        Get all matching keys for a given filter

        Parameters
        ----------
        filters: dict of str
            String filters
            If a level is missing then all values are fetched

        Returns
        -------
        list of str or dict
            Each item is a key which may contain data which is of interest
        """
        pass
