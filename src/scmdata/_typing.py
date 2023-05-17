"""
Commonly used typehints
"""
from __future__ import annotations

from os import PathLike  # noqa
from typing import Callable, Dict, Union

import pandas as pd

FilePath = Union[str, "PathLike[str]"]
MetadataValue = Union[str, int, float]
MetadataType = Dict[str, MetadataValue]
ApplyCallable = Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series, float]]
