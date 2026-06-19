"""
Commonly used typehints
"""

from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from typing import Union

import pandas as pd

FilePath = Union[str, "PathLike[str]"]
MetadataValue = Union[str, int, float]
MetadataType = dict[str, MetadataValue]
ApplyCallable = Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series, float]]
