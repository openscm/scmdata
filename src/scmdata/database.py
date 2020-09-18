"""
Database of results handling
"""
import glob
import os
import os.path

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
        except OSError:
            # Prevent race conditions if multiple threads attempt to create dir at same time
            if not os.path.exists(dir_to_check):
                raise


class SCMDatabase:
    """
    On-disk database handler for outputs from SCMs

    Data is split into groups as specified by :attr:`levels`. This allows for fast
    reading and writing of new subsets of data when a single output file is no longer
    performant.
    """

    def __init__(
        self, root_dir, levels=("climate_model", "variable", "region", "scenario"),
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
            best match the input data and desired access pattern. If there are any additional
            varying dimensions, they will be stored as dimensions.

        .. note::

            Creating a new :class:`ScmDatabase` does not modify any existing data on disk. To
            load an existing database ensure that the :attr:`root_dir` and :attr:`levels` are
            the same as the previous instance.
        """
        self._root_dir = root_dir
        self.levels = levels

    def __repr__(self):
        return "<scmdata.database.SCMDatabase (root_dir: {})>".format(self._root_dir)

    @staticmethod
    def _get_disk_filename(inp):
        return inp.replace("|", "-").replace(" ", "-")

    def save_to_database(self, scmrun):
        """
        Save a set of results to the database

        The results are saved with one file for each unique combination of
        :attr:`levels`.

        Parameters
        ----------
        scmrun : :obj:`scmdata.ScmRun`
            Results to save
        """
        for r in tqdman.tqdm(
            scmrun.groupby(self.levels), leave=False, desc="Saving to database",
        ):
            self._save_to_database_single_file(r)

    def get_out_filepath(self, **levels):
        """
        Get filepath in which data has been saved

        The filepath is the root directory joined with the other information provided. The filepath
        is also cleaned to remove spaces and special characters.

        Parameters
        ----------
        levels: dict of str
            The unique value for each level in :attr:`levels'

        Returns
        -------
        str
            Path in which to save the data without spaces or special characters.

        Raises
        ------
        ValueError
            If no value is provided for level in :attr:`levels'
        """
        out_levels = []
        for l in self.levels:
            if l not in levels:
                raise ValueError("expected value for level: {}".format(l))
            out_levels.append(str(levels[l]))

        out_path = os.path.join(self._root_dir, *out_levels)

        out_fname = "_".join(out_levels) + ".nc"
        return self._get_disk_filename(os.path.join(out_path, out_fname))

    def save_condensed_file(self, scmrun):
        """
        Save results which have multiple ensemble members

        Parameters
        ----------
        scmrun : :obj:`scmdata.ScmRun`
            Results to save in the database

        Raises
        ------
        AssertionError
            ``ensemble_member`` is not included in ``scmrun``'s metadata
        """
        climate_model = scmrun.get_unique_meta("climate_model", no_duplicates=True)
        variable = scmrun.get_unique_meta("variable", no_duplicates=True)
        region = scmrun.get_unique_meta("region", no_duplicates=True)
        scenario = scmrun.get_unique_meta("scenario", no_duplicates=True)

        if "ensemble_member" not in scmrun.meta:
            raise AssertionError("`scmrun` must contain ensemble_member metadata")
        out_file = self.get_out_filepath(
            climate_model, variable, region, scenario, ensemble_member=None
        )
        ensure_dir_exists(out_file)
        scmrun.to_nc(out_file, dimensions=("ensemble_member",))

    def _save_to_database_single_file(self, scmrun):
        levels = {l: scmrun.get_unique_meta(l, no_duplicates=True) for l in self.levels}
        out_file = self.get_out_filepath(**levels)

        ensure_dir_exists(out_file)
        if os.path.exists(out_file):
            existing_run = ScmRun.from_nc(out_file)

            scmrun = run_append([existing_run, scmrun])

        # Check for required extra dimensions
        nunique_meta_vals = scmrun.meta.nunique()
        dimensions = nunique_meta_vals[nunique_meta_vals > 1].index.tolist()
        scmrun.to_nc(out_file, dimensions=dimensions)

    def load_data(self, **filters):
        """
        Load data from the database

        Parameters
        ----------
        filters: dict of str
            Filters for the data to load.

            Defaults to loading all values for a level if it isn't specified.

        Returns
        -------
        :obj: `scmdata.ScmRun`
            Loaded data

        Raises
        ------
        ValueError
            If a filter for a level not in :attr:`levels` is specified

            If no data matching ``filters`` is found
        """
        for k in filters:
            if k not in self.levels:
                raise ValueError("Unknown level: {}".format(k))

        paths_to_load = [filters.get(l, "*") for l in self.levels]
        load_path = os.path.join(self._root_dir, *paths_to_load)
        glob_to_use = self._get_disk_filename(os.path.join(load_path, "**", "*.nc"))
        load_files = glob.glob(glob_to_use, recursive=True)

        return run_append(
            [
                ScmRun.from_nc(f)
                for f in tqdman.tqdm(load_files, desc="Loading files", leave=False)
            ]
        )
