"""
Database of results handling
"""
import glob
import os
import os.path

import pandas as pd
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
    """

    def __init__(self, root_dir):
        """
        Initialise the database handler

        Parameters
        ----------
        root_dir : str
            The root directory of the database
        """
        self._root_dir = root_dir

    def __repr__(self):
        return "<scmdata.database.SCMDatabase (root_dir: {})>".format(self._root_dir)

    @staticmethod
    def _get_disk_filename(inp):
        return inp.replace("|", "-").replace(" ", "-")

    def save_to_database(self, scmrun):
        """
        Save a set of results to the database

        The results are saved with one file for each
        ``["climate_model", "variable", "region", "scenario", "ensemble_member"]``
        combination.

        Parameters
        ----------
        scmrun : :obj:`scmdata.ScmRun`
            Results to save
        """
        for r in tqdman.tqdm(
            scmrun.groupby(
                ["climate_model", "variable", "region", "scenario", "ensemble_member"]
            ),
            leave=False,
            desc="Saving to database",
        ):
            self._save_to_database_single_file(r)

    def get_out_filepath(
        self, climate_model, variable, region, scenario, ensemble_member=None
    ):
        """
        Get filepath in which data has been saved

        The filepath is the root directory joined with the other information provided. The filepath
        is also cleaned to remove spaces and special characters.

        Parameters
        ----------
        climate_model : str
            Climate model to retrieve data for

        variable : str
            Variable to retrieve data for

        region : str
            Region to retrieve data for

        scenario : str
            Scenario to retrieve data for

        ensemble_member : str or None
            Ensemble member to retrieve data for

        Returns
        -------
        str
            Path in which to save the data. If ``ensemble_member`` is ``None`` then it is not
            included in the filename.
        """
        out_dir = os.path.join(
            self._root_dir, climate_model, variable, region, scenario
        )
        if ensemble_member is None:
            out_fname = "{}_{}_{}_{}.nc".format(
                climate_model, variable, region, scenario
            )
        else:
            out_fname = "{}_{}_{}_{}_{}.nc".format(
                climate_model, variable, region, scenario, ensemble_member
            )

        return self._get_disk_filename(os.path.join(out_dir, out_fname))

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
        climate_model = scmrun.get_unique_meta("climate_model", no_duplicates=True)
        variable = scmrun.get_unique_meta("variable", no_duplicates=True)
        region = scmrun.get_unique_meta("region", no_duplicates=True)
        scenario = scmrun.get_unique_meta("scenario", no_duplicates=True)
        ensemble_member = scmrun.get_unique_meta("ensemble_member", no_duplicates=True)
        out_file = self.get_out_filepath(
            climate_model, variable, region, scenario, ensemble_member
        )

        ensure_dir_exists(out_file)

        scmrun.to_nc(out_file)

    def load_data(self, variable, region, scenario):
        """
        Load data from the database

        Parameters
        ----------
        variable : str
            Variable to load

        region : str
            Region to load

        scenario : str
            Scenario to load

        Returns
        -------
        :obj: `scmdata.ScmRun`
            Loaded data
        """
        load_path = os.path.join(self._root_dir, "*", variable, region, scenario)
        glob_to_use = self._get_disk_filename(os.path.join(load_path, "**", "*.nc"))
        load_files = glob.glob(glob_to_use, recursive=True)

        return run_append(
            [
                ScmRun.from_nc(f)
                for f in tqdman.tqdm(load_files, desc="Loading files", leave=False)
            ]
        )
