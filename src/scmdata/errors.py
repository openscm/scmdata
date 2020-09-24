"""
Custom errors and exceptions used by scmdata
"""
import pandas as pd


class NonUniqueMetadataError(ValueError):
    """
    Error raised when metadata is not unique
    """

    def __init__(self, meta):
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
        msg = "missing required columns `{}`!".format(columns)

        super().__init__(msg)
