"""
Custom errors and exceptions used by scmdata
"""
import pandas as pd


class NonUniqueMetadataError(ValueError):
    """
    Error raised when metadata is not unique
    """

    def __init__(self, meta):
        self.meta = meta
        # format table to show the metadata clash
        dup = meta.astype(str).groupby(meta.columns.tolist(), as_index=False).size()
        if isinstance(dup, pd.Series):
            # pandas < 1.1 Groupby.size returns a series
            dup.name = "repeats"
            dup = dup.to_frame().reset_index()
        else:
            dup = dup.rename(columns={"size": "repeats"})

        dup = dup[dup.repeats > 1]
        msg = (
            "Duplicate metadata (numbers show how many times the given "
            "metadata is repeated).\n{}".format(dup)
        )

        super().__init__(msg)


class MissingRequiredColumnError(ValueError):
    """
    Error raised when an operation produces missing metadata columns
    """

    def __init__(self, columns):
        self.columns = columns
        msg = "Missing required columns `{}`!".format(columns)

        super().__init__(msg)


class DuplicateTimesError(ValueError):
    """
    Error raised when times are duplicated
    """

    def __init__(self, time_index):
        self.time_index = time_index
        dup = time_index.value_counts()
        dup = dup[dup > 1]

        msg = (
            "Duplicate times (numbers show how many times the given time is "
            "repeated):\n{}".format(dup)
        )

        super().__init__(msg)


class InsufficientDataError(Exception):
    """
    Insufficient data is available to interpolate/extrapolate
    """
