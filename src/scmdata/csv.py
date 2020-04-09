"""
CSV processing tools
"""

import pandas as pd


def df_to_csv(df, fname: str, **kwargs: Any) -> None:
    """
    Write timeseries data to a csv file

    Parameters
    ----------
    path
        Path to write the file into
    """
    df.timeseries().reset_index().to_csv(fname)