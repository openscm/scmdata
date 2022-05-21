import os
import os.path
import pathlib


def ensure_dir_exists(fp):
    """
    Ensure directory exists

    Parameters
    ----------
    fp : str
        Filepath of which to ensure the directory exists
    """
    dir_to_check = os.path.dirname(fp)

    if os.path.isfile(dir_to_check):
        raise AssertionError(f"Expected {dir_to_check} to not be a file")

    os.makedirs(dir_to_check, exist_ok=True)


def _check_is_subdir(root, d):
    root_path = pathlib.Path(root).resolve()
    out_path = pathlib.Path(d).resolve()

    is_subdir = root_path in out_path.parents
    # Sanity check that we never mangle anything outside of the root dir
    if not is_subdir:  # pragma: no cover
        raise AssertionError("{} not in {}".format(d, root))
