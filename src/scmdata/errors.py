class NonUniqueMetadata(ValueError):
    """
    Error raised when metadata is not unique
    """
    def __init__(self, df):
        import pdb
        pdb.set_trace()
        # format table to show the metadata clash
        super().__init__(msg)
