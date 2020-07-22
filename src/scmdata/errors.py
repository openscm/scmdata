"""
Custom errors and exceptions used by scmdata
"""


class NonUniqueMetadataError(ValueError):
    """
    Error raised when metadata is not unique
    """

    def __init__(self, meta):
        # format table to show the metadata clash
        dup = meta.groupby(meta.columns.tolist(), as_index=False).size()
        dup = dup[dup > 1]
        dup.name = "repeats"
        dup = dup.to_frame().reset_index()
        msg = (
            "Duplicate metadata (numbers show how many times the given "
            "metadata is repeated).\n{}".format(dup)
        )

        super().__init__(msg)
