"""
Database for handling large datasets in a performant, but flexible way

Data is chunked using unique combinations of metadata. This allows for the
database to expand as new data is added without having to change any of the
existing data.

Subsets of data are also able to be read without having to load all the data
and then filter. For example, one could save model results from a number of different
climate models and then load just the ``Surface Temperature`` data for all models.
"""
import glob
import itertools
import os
import os.path
import pathlib
from abc import ABC, abstractmethod

import pandas as pd
import six
import tqdm.autonotebook as tqdman

from scmdata import ScmRun, run_append


def ensure_dir_exists(fp):
    """
    Ensure directory exists

    Parameters
    ----------
    fp : str
        Filepath of which to ensure the directory exists
    """
    dir_to_check = os.path.dirname(fp)
    if not os.path.isdir(dir_to_check):
        try:
            os.makedirs(dir_to_check)
        except OSError:  # pragma: no cover
            # Prevent race conditions if multiple threads attempt to create dir at same time
            if not os.path.exists(dir_to_check):
                raise


def _check_is_subdir(root, d):
    root_path = pathlib.Path(root).resolve()
    out_path = pathlib.Path(d).resolve()

    is_subdir = root_path in out_path.parents
    # Sanity check that we never mangle anything outside of the root dir
    if not is_subdir:  # pragma: no cover
        raise AssertionError("{} not in {}".format(d, root))


def _get_safe_filename(inp):
    def safe_char(c):
        if c.isalnum() or c in "-/*_.":
            return c
        else:
            return "-"

    return "".join(safe_char(c) for c in inp)


class DatabaseBackend(ABC):
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


class NetCDFBackend(DatabaseBackend):
    """
    On-disk database handler for outputs from SCMs

    Data is split into groups as specified by :attr:`levels`. This allows for fast
    reading and writing of new subsets of data when a single output file is no longer
    performant or data cannot all fit in memory.
    """

    def get_key(self, sr):
        """
        Get key where the data will be stored

        The key is the root directory joined with the other information provided. The filepath
        is also cleaned to remove spaces and special characters.

        Parameters
        ----------
        sr : :class:`scmdata.ScmRun`
            Data to save

        Raises
        ------
        ValueError
            If non-unique metadata is found for each of :attr:`self.kwargs["levels"]`

        KeyError
            If missing metadata is found for each of :attr:`self.kwargs["levels"]`

        Returns
        -------
        str
            Path in which to save the data without spaces or special characters
        """
        levels = {
            database_level: sr.get_unique_meta(
                database_level, no_duplicates=True
            ).replace(os.sep, "_")
            for database_level in self.kwargs["levels"]
        }

        return self._get_out_filepath(**levels)

    def _get_out_filepath(self, **data_levels):
        out_levels = []
        for database_level in self.kwargs["levels"]:
            if database_level not in data_levels:  # pragma: no cover # emergency valve
                raise KeyError("expected level: {}".format(database_level))
            out_levels.append(str(data_levels[database_level]))

        out_path = os.path.join(self.kwargs["root_dir"], *out_levels)
        out_fname = "__".join(out_levels) + ".nc"
        out_fname = os.path.join(out_path, out_fname)

        _check_is_subdir(self.kwargs["root_dir"], out_fname)

        return _get_safe_filename(out_fname)

    def save(self, sr):
        """
        Save a ScmRun to the database

        The dataset should not contain any duplicate metadata for the
        database levels

        Parameters
        ----------
        sr : :class:`scmdata.ScmRun`
            Data to save

        Raises
        ------
        ValueError
            If duplicate metadata are present for the requested database levels

        KeyError
            If metadata for the requested database levels are not found

        Returns
        -------
        str
            Key where the data is saved
        """
        key = self.get_key(sr)

        ensure_dir_exists(key)
        if os.path.exists(key):
            existing_run = ScmRun.from_nc(key)

            sr = run_append([existing_run, sr])

        # Check for required extra dimensions
        dimensions = self.kwargs.get("dimensions", None)
        if not dimensions:
            nunique_meta_vals = sr.meta.nunique()
            dimensions = nunique_meta_vals[nunique_meta_vals > 1].index.tolist()
        sr.to_nc(key, dimensions=dimensions)
        return key

    def load(self, key):
        """

        Parameters
        ----------
        key: str

        Returns
        -------
        :class:`scmdata.ScmRun`

        """
        return ScmRun.from_nc(key)

    def delete(self, key):
        """
        Delete a key

        Parameters
        ----------
        key: str

        """
        os.remove(key)

    def get(self, filters):
        """
        Get all matching objects for a given filter

        Parameters
        ----------
        filters: dict of str
            String filters
            If a level is missing then all values are fetched

        Returns
        -------
        list of str
        """
        level_options = []
        for level in self.kwargs["levels"]:
            level_values = filters.get(level, ["*"])
            if isinstance(level_values, str):
                level_values = [level_values]

            level_options.append(level_values)

        # AND logic across levels, OR logic within levels
        level_options_product = itertools.product(*level_options)
        globs_to_check = [
            _get_safe_filename(os.path.join(self.kwargs["root_dir"], *levels, "*.nc"))
            for levels in level_options_product
        ]

        load_files = [
            v
            for vlist in [glob.glob(g, recursive=True) for g in globs_to_check]
            for v in vlist
        ]

        return load_files


"""
Loaded backends for ScmDatabase

Additional backends should be based upon :class:`DatabaseBackend`
"""
backend_classes = {"netcdf": NetCDFBackend}


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

        Parameters
        ----------
        root_dir : str
            The root directory of the database

        levels : tuple of str
            Specifies how the runs should be stored on disk.

            The data will be grouped by ``levels``. These levels should be adapted to
            best match the input data and desired access pattern. If there are any
            additional varying dimensions, they will be stored as dimensions.

        backend: str or :class:`DatabaseBackend`
            Determine the backend to serialize and deserialize data

            Defaults to using :class:`NetCDFBackend` which reads and writes data as
            netCDF files. Note that this requires the optional dependency of netCDF4 to
            be installed.

            If a custom backend class is being used, it must be extend the
            :class:`DatabaseBackend` class.

        backend_config: dict
            Additional configuration to pass to the backend

            See the documentation for the target backend to determine what configuration
            options are available.

        .. note::

            Creating a new :class:`ScmDatabase` does not modify any existing data on
            disk. To load an existing database ensure that the :attr:`root_dir` and
            :attr:`levels` are the same as the previous instance.
        """
        self._root_dir = root_dir
        self.levels = tuple(levels)

        backend_config = backend_config if backend_config else {}
        for key in ["levels", "root_dir"]:
            if key in backend_config:
                raise ValueError(
                    "backend_config cannot contain key of `{}`".format(key)
                )
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
            if not isinstance(backend, DatabaseBackend):
                raise ValueError(
                    "Backend should be an instance of scmdata.database.DatabaseBackend"
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
                    load_files, desc="Loading files", leave=False, disable=disable_tqdm,
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
