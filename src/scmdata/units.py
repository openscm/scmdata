"""
Unit handling
"""
import warnings
from typing import Optional, Sequence, Union

import numpy as np
from openscm_units import ScmUnitRegistry

_unit_registry = ScmUnitRegistry()
"""
SCMData standard unit registry

The unit registry contains all of the recognised units.
"""
_unit_registry.add_standards()


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

        source_unit = _unit_registry.Unit(source)
        target_unit = _unit_registry.Unit(target)

        s1 = _unit_registry.Quantity(1, source_unit)
        s2 = _unit_registry.Quantity(-1, source_unit)

        if context is None:
            t1 = s1.to(target_unit)
            t2 = s2.to(target_unit)
        else:
            with _unit_registry.context(context):
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
        return list(_unit_registry._contexts.keys())  # pylint: disable=protected-access

    @property
    def unit_registry(self) -> ScmUnitRegistry:
        """
        Underlying unit registry
        """
        return _unit_registry

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
