"""
:mod:`scmdata` uses :mod:`openscm_units` to support unit handling and conversion. :mod:`openscm_units` is
built on top of :mod:`pint` and includes some additional quantity definitions to support the tracking of
emissions timeseries.
"""
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import openscm_units

UNIT_REGISTRY: openscm_units.ScmUnitRegistry = openscm_units.unit_registry
"""
Unit registry used for when converting units in :mod:`scmdata`

This defaults to :attr:`openscm_units.unit_registry` so any additional units added to
:attr:`openscm_units.unit_registry` will also be available in :mod:`scmdata. Alternatively,
this attribute can be overridden with a custom :class:`openscm.ScmUnitRegistry` instance
if required.
"""


class UnitConverter:
    """
    Converts numbers between two units.
    """

    def __init__(self, source: str, target: str, context: Optional[str] = None):
        """
        Initialize.

        Parameters
        ----------
        source
            Unit to convert **from**
        target
            Unit to convert **to**
        context
            Context to use for the conversion i.e. which metric to apply when performing
            CO2-equivalent calculations. If ``None``, no metric will be applied and
            CO2-equivalent calculations will raise :class:`DimensionalityError`.

        Raises
        ------
        pint.errors.DimensionalityError
            Units cannot be converted into each other.
        pint.errors.UndefinedUnitError
            Unit undefined.
        """
        self._source = source
        self._target = target
        self._ur = get_unit_registry()

        source_unit = self._ur.Unit(source)
        target_unit = self._ur.Unit(target)

        s1 = self._ur.Quantity(1, source_unit)
        s2 = self._ur.Quantity(-1, source_unit)

        if context is None:
            t1 = s1.to(target_unit)
            t2 = s2.to(target_unit)
        else:
            with self._ur.context(context):
                t1 = s1.to(target_unit)
                t2 = s2.to(target_unit)

        if np.isnan(t1) or np.isnan(t2):
            warn_msg = (
                "No conversion from {} to {} available, nan will be returned "
                "upon conversion".format(source, target)
            )
            warnings.warn(warn_msg)

        self._scaling = float(t2.m - t1.m) / float(s2.m - s1.m)
        self._offset = t1.m - self._scaling * s1.m

    def convert_from(self, v: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert value **from** source unit to target unit.

        Parameters
        ----------
        value
            Value in source unit

        Returns
        -------
        Union[float, np.ndarray]
            Value in target unit
        """
        return self._offset + v * self._scaling

    def convert_to(self, v: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert value from target unit **to** source unit.

        Parameters
        ----------
        value
            Value in target unit

        Returns
        -------
        Union[float, np.ndarray]
            Value in source unit
        """
        return (v - self._offset) / self._scaling

    @property
    def contexts(self) -> Sequence[str]:
        """
        Available contexts for unit conversions
        """
        return list(self._ur._contexts.keys())  # pylint: disable=protected-access

    @property
    def unit_registry(self) -> openscm_units.ScmUnitRegistry:
        """
        Underlying unit registry
        """
        return self._ur

    @property
    def source(self) -> str:
        """
        Source unit
        """
        return self._source

    @property
    def target(self) -> str:
        """
        Target unit
        """
        return self._target


def get_unit_registry() -> openscm_units.ScmUnitRegistry:
    """
    Retrieve the global unit registry
    """
    return UNIT_REGISTRY
