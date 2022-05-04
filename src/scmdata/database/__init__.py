"""
Database for handling large datasets in a performant, but flexible way

Data is chunked using unique combinations of metadata. This allows for the
database to expand as new data is added without having to change any of the
existing data.

Subsets of data are also able to be read without having to load all the data
and then filter. For example, one could save model results from a number of different
climate models and then load just the ``Surface Temperature`` data for all models.
"""

from ._database import ScmDatabase  # noqa: F401, E402
