"""SI base units and the physical quantities they measure.

The catalog follows the `BIPM SI base units
<https://www.bipm.org/en/measurement-units/si-base-units>`_.

>>> BaseUnit.KELVIN.symbol
'K'
>>> BaseUnit.KELVIN.quantity is BaseQuantity.THERMODYNAMIC_TEMPERATURE
True
"""

from dataclasses import dataclass
from enum import Enum, StrEnum

# %% Base quantities


class BaseQuantity(StrEnum):
    """The seven dimensionally independent base quantities of the SI."""

    LENGTH = "length"
    """Spatial extent, including distance and displacement magnitude."""

    MASS = "mass"
    """Mass of a physical system."""

    TIME = "time"
    """Duration and temporal separation."""

    ELECTRIC_CURRENT = "electric current"
    """Rate of electric charge flow."""

    THERMODYNAMIC_TEMPERATURE = "thermodynamic temperature"
    """Absolute temperature measured from absolute zero."""

    AMOUNT_OF_SUBSTANCE = "amount of substance"
    """Amount of specified elementary entities."""

    LUMINOUS_INTENSITY = "luminous intensity"
    """Luminous flux per solid angle in a given direction."""


# %% Base units


@dataclass(frozen=True)
class BaseUnitDefinition:
    """The immutable symbol and measured quantity of a base unit."""

    symbol: str
    """Case-sensitive SI unit symbol."""

    quantity: BaseQuantity
    """Physical base quantity measured by the unit."""


class BaseUnit(BaseUnitDefinition, Enum):
    """The seven SI base units with their symbols and measured quantities."""

    METRE = ("m", BaseQuantity.LENGTH)
    """SI base unit of length."""

    KILOGRAM = ("kg", BaseQuantity.MASS)
    """SI base unit of mass."""

    SECOND = ("s", BaseQuantity.TIME)
    """SI base unit of time."""

    AMPERE = ("A", BaseQuantity.ELECTRIC_CURRENT)
    """SI base unit of electric current."""

    KELVIN = ("K", BaseQuantity.THERMODYNAMIC_TEMPERATURE)
    """SI base unit of thermodynamic temperature."""

    MOLE = ("mol", BaseQuantity.AMOUNT_OF_SUBSTANCE)
    """SI base unit of amount of substance."""

    CANDELA = ("cd", BaseQuantity.LUMINOUS_INTENSITY)
    """SI base unit of luminous intensity."""
