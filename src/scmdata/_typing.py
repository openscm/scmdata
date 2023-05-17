"""
Commonly used typehints
"""
from __future__ import annotations

from os import PathLike  # noqa
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, Union, Literal, TypeVar, Protocol, Generic, Any

import pandas as pd

AnyScmRun = TypeVar("AnyScmRun", bound="BaseScmRun")


class ScmRunCallbackWithArgs(Protocol, Generic[AnyScmRun]):
    def __call__(self, value: AnyScmRun, *args: Any, **kwargs: Any) -> AnyScmRun | None:
        ...


class ScmRunCallback(Protocol, Generic[AnyScmRun]):
    def __call__(self, value: AnyScmRun) -> AnyScmRun | None:
        ...


class NumericCallbackWithArgs(Protocol):
    def __call__(
        self, value: pd.DataFrame | NDArray[np.float_], *args: Any, **kwargs: Any
    ) -> pd.DataFrame | NDArray[np.float_] | None:
        ...


FilePath = Union[str, "PathLike[str]"]
MetadataValue = Union[str, int, float]
MetadataType = Dict[str, MetadataValue]
ApplyCallable = Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series, float]]

TimeAxisOptions = Literal[
    "year",
    "year-month",
    "days since 1970-01-01",
    "seconds since 1970-01-01",
]
