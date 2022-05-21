"""
Database for handling large datasets in a performant, but flexible way
"""
import os
import os.path

import pandas as pd
import six
import tqdm.autonotebook as tqdman

from scmdata import run_append
from scmdata.database._utils import _check_is_subdir
from scmdata.database.backends import BaseDatabaseBackend, backend_classes


class ScmDatabase:
    """
    On-disk database handler for outputs from SCMs

    Data is split into groups as specified by :attr:`levels`. This allows for fast
    reading and writing of new subsets of data when a single output file is no longer
    performant or data cannot all fit in memory.
    """

    def __init__(
        self,
        root_dir,
        levels=("climate_model", "variable", "region", "scenario"),
        backend="netcdf",
        backend_config=None,
    ):
        """
        Initialise the database

        .. note::

            Creating a new :class:`ScmDatabase` does not modify any existing data on
            disk. To load an existing database ensure that the :attr:`root_dir`.
            :attr:`levels` and backend settings are the same as the previous instance.

        Parameters
        ----------
        root_dir : str
            The root directory of the database

        levels : tuple of str
            Specifies how the runs should be stored on disk.

            The data will be grouped by ``levels``. These levels should be adapted to
            best match the input data and desired access pattern. If there are any
            additional varying dimensions, they will be stored as dimensions.

        backend: str or :class:`BaseDatabaseBackend<scmdata.database.backends.BaseDatabaseBackend>`
            Determine the backend to serialize and deserialize data

            Defaults to using :class:`NetCDFDatabaseBackend<scmdata.database.backends.NetCDFDatabaseBackend>`
            which reads and writes data as netCDF files. Note that this requires the
            optional dependency of netCDF4 to be installed.

            If a custom backend class is being used, it must extend the
            :class:`BaseDatabaseBackend<scmdata.database.backends.BaseDatabaseBackend>` class.

        backend_config: dict
            Additional configuration to pass to the backend

            See the documentation for the target backend to determine which configuration
            options are available.


        """
        self._root_dir = root_dir
        self.levels = tuple(levels)

        backend_config = backend_config if backend_config else {}
        for key in ["levels", "root_dir"]:
            if key in backend_config:
                raise ValueError("backend_config cannot contain key `{}`".format(key))
        backend_config["levels"] = self.levels
        backend_config["root_dir"] = root_dir

        self._backend = self._get_backend(backend, backend_config)

    def _get_backend(self, backend, backend_config):
        if isinstance(backend, six.string_types):
            try:
                cls = backend_classes[backend.lower()]
                return cls(**backend_config)
            except KeyError:
                raise ValueError("Unknown database backend: {}".format(backend))
        else:
            if not isinstance(backend, BaseDatabaseBackend):
                raise ValueError(
                    "Backend must be an instance of scmdata.database.BaseDatabaseBackend"
                )
            return backend

    def __repr__(self):
        return "<scmdata.database.SCMDatabase (root_dir: {}, levels: {})>".format(
            self._root_dir, self.levels
        )

    @property
    def root_dir(self):
        """
        Root directory of the database.

        Returns
        -------
        str
        """
        return self._root_dir

    def _clean_filters(self, filters):
        for level in filters:
            if level not in self.levels:
                raise ValueError("Unknown level: {}".format(level))
            if os.sep in filters[level]:
                filters[level] = filters[level].replace(os.sep, "_")
        return filters

    def save(self, scmrun, disable_tqdm=False):
        """
        Save data to the database

        The results are saved with one file for each unique combination of
        :attr:`levels` in a directory structure underneath ``root_dir``.

        Use :meth:`available_data` to see what data is available. Subsets of
        data can then be loaded as an :class:`scmdata.ScmRun <scmdata.run.ScmRun>` using :meth:`load`.

        Parameters
        ----------
        scmrun : :class:`scmdata.ScmRun <scmdata.run.ScmRun>`
            Data to save.

            The timeseries in this run should have valid metadata for each
            of the columns specified in ``levels``.
        disable_tqdm: bool
            If True, do not show the progress bar

        Raises
        ------
        KeyError
            If a filter for a level not in :attr:`levels` is specified
        """
        for r in tqdman.tqdm(
            scmrun.groupby(self.levels),
            leave=False,
            desc="Saving to database",
            disable=disable_tqdm,
        ):
            self._backend.save(r)

    def load(self, disable_tqdm=False, **filters):
        """
        Load data from the database

        Parameters
        ----------
        disable_tqdm: bool
            If True, do not show the progress bar
        filters: dict of str : [str, list[str]]
            Filters for the data to load.

            Defaults to loading all values for a level if it isn't specified.

            If a filter is a list then OR logic is applied within the level.
            For example, if we have ``scenario=["ssp119", "ssp126"]`` then
            both the ssp119 and ssp126 scenarios will be loaded.

        Returns
        -------
        :class:`scmdata.ScmRun`
            Loaded data

        Raises
        ------
        ValueError
            If a filter for a level not in :attr:`levels` is specified

            If no data matching ``filters`` is found
        """
        filters = self._clean_filters(filters)

        load_files = self._backend.get(filters)

        return run_append(
            [
                self._backend.load(f)
                for f in tqdman.tqdm(
                    load_files,
                    desc="Loading files",
                    leave=False,
                    disable=disable_tqdm,
                )
            ]
        )

    def delete(self, **filters):
        """
        Delete data from the database

        Parameters
        ----------
        filters: dict of str
            Filters for the data to load.

            Defaults to deleting all data if nothing is specified.

        Raises
        ------
        ValueError
            If a filter for a level not in :attr:`levels` is specified
        """
        filters = self._clean_filters(filters)
        targets = self._backend.get(filters)

        for t in targets:
            _check_is_subdir(self._root_dir, t)
            self._backend.delete(t)

    def available_data(self):
        """
        Get all the data which is available to be loaded

        If metadata includes non-alphanumeric characters then it
        might appear modified in the returned table. The original
        metadata values can still be used to filter data.

        Returns
        -------
        :class:`pd.DataFrame`
        """
        all_files = self._backend.get({})

        file_meta = []
        for f in all_files:
            dirnames = f.split(os.sep)[:-1]
            file_meta.append(dirnames[-len(self.levels) :])

        data = pd.DataFrame(file_meta, columns=self.levels)

        return data.sort_values(by=data.columns.to_list()).reset_index(drop=True)
