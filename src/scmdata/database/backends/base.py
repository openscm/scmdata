"""
Interface for the base database backend

All other database backends should be based upon this interface
"""


from abc import ABC, abstractmethod


class BaseDatabaseBackend(ABC):
    """
    Abstract backend for serialising/deserialising data

    Data is stored as objects represented by keys. These keys can be used later
    to load data.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def save(self, sr):
        """
        Save data

        Parameters
        ----------
        sr: scmdata.ScmRun

        Returns
        -------
        str
            Key where the data is stored
        """
        pass

    @abstractmethod
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
        pass

    def delete(self, key):
        """
        Delete a given key

        Parameters
        ----------
        key: str
        """
        pass

    @abstractmethod
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
        list of str
            Each item is a key which may contain data which is of interest
        """
        pass
