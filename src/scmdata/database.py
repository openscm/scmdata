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


class Database:
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
        return "<utils.scmdata.Database (root_dir: {})>".format(self._root_dir)

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

    def save_model_reported(self, res, key="all"):
        """
        Save model reported data into the database

        Parameters
        ----------
        res : :obj:`pd.DataFrame`
            Model reported results to save

        key : str
            Identifier to use in the filename

        Raises
        ------
        AssertionError
            The columns of res are not as expected (i.e.
            ``{"value", "ensemble_member", "RCMIP name", "unit", "climate_model"}``)
            or more than one climate model is included in ``res``.
        """
        expected_columns = {
            "value",
            "ensemble_member",
            "RCMIP name",
            "unit",
            "climate_model",
        }
        correct_columns = set(res.columns) == expected_columns
        if not correct_columns:
            raise AssertionError(
                "Input columns: {}. Expected columns: {}.".format(
                    set(res.columns), expected_columns
                )
            )

        climate_model = res["climate_model"].unique().tolist()
        if len(climate_model) != 1:
            raise AssertionError(
                "More than one climate model: {}".format(climate_model)
            )
        climate_model = climate_model[0]

        outfile = self._get_disk_filename(
            os.path.join(
                self._root_dir,
                climate_model,
                "model_reported_metrics_{}.csv".format(key),
            )
        )
        ensure_dir_exists(outfile)
        res.to_csv(outfile, index=False)

    def load_model_reported(self):
        """
        Load all model reported results

        Returns
        -------
        :obj:`pd.DataFrame`
            All model reported results
        """
        glob_path = self._get_disk_filename(
            os.path.join(self._root_dir, "**", "model_reported_metrics*.csv")
        )
        to_load = glob.glob(glob_path, recursive=True)

        return pd.concat([pd.read_csv(f) for f in to_load])

    def save_summary_table(self, res, file_id):
        """
        Save summary table

        Parameters
        ----------
        res : :obj:`pd.DataFrame`
            Summary table to save

        file_id : str
            Identifier to use in the filename

        Raises
        ------
        AssertionError
            Columns of ``res`` are not as expected (i.e. not equal to
            ``{"assessed_range_label", "assessed_range_value", "climate_model", "climate_model_value", "metric", "percentage_difference", "unit"}``)
        """
        expected_columns = {
            "assessed_range_label",
            "assessed_range_value",
            "climate_model",
            "climate_model_value",
            "metric",
            "percentage_difference",
            "unit",
        }
        if set(res.columns) != expected_columns:
            raise AssertionError(
                "Input columns: {}. Expected columns: {}.".format(
                    set(res.columns), expected_columns
                )
            )

        outfile = self._get_disk_filename(
            os.path.join(
                self._root_dir,
                "climate_model_assessed_ranges_summary_table_{}.csv".format(file_id),
            )
        )

        res.to_csv(outfile, index=False)

    def load_summary_tables(self):
        """
        Load all summary tables

        Returns
        -------
        :obj:`pd.DataFrame`
            All summary tables
        """
        load_path = os.path.join(
            self._root_dir, "climate_model_assessed_ranges_summary_table_*.csv",
        )
        load_files = glob.glob(load_path)

        return pd.concat([pd.read_csv(f) for f in load_files])
