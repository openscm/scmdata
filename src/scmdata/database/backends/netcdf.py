"""
Database backend for handling local files stored as NetCDF
"""

import glob
import itertools
import os
import os.path

from scmdata import ScmRun, run_append
from scmdata.database._utils import _check_is_subdir, ensure_dir_exists
from scmdata.database.backends import BaseDatabaseBackend


def _get_safe_filename(inp, include_glob=False):
    def safe_char(c):
        accepted_chars = "-_."
        if include_glob:
            accepted_chars = accepted_chars + "*"

        if c.isalnum() or c in accepted_chars or c == os.sep:
            return c

        return "-"

    return "".join(safe_char(c) for c in inp)


class NetCDFDatabaseBackend(BaseDatabaseBackend):
    """
    Database backend for handling local files stored as NetCDF
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

            If any metadata end with '.'

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

        # Windows does not support directories or filenames which end in a '.'
        if any([level.endswith(".") for level in levels.values()]):
            raise ValueError("Metadata cannot end in a '.'")

        return self._get_out_filepath(**levels)

    def _get_out_filepath(self, **data_levels):
        out_levels = []
        for database_level in self.kwargs["levels"]:
            if database_level not in data_levels:  # pragma: no cover # emergency valve
                raise KeyError("expected level: {}".format(database_level))
            out_levels.append(str(data_levels[database_level]))

        out_path = os.path.join(*out_levels)
        out_fname = "__".join(out_levels) + ".nc"
        out_fname = _get_safe_filename(os.path.join(out_path, out_fname))
        out_fname = os.path.join(self.kwargs["root_dir"], out_fname)

        _check_is_subdir(self.kwargs["root_dir"], out_fname)

        return out_fname

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
            _get_safe_filename(os.path.join(*levels, "*.nc"), include_glob=True)
            for levels in level_options_product
        ]

        load_files = [
            v
            for vlist in [
                glob.glob(os.path.join(self.kwargs["root_dir"], g), recursive=True)
                for g in globs_to_check
            ]
            for v in vlist
        ]

        return load_files
