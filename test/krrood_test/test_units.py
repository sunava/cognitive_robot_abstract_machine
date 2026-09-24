"""
The seven SI base units retain their symbols, quantities, and identities.
"""

from dataclasses import FrozenInstanceError

import pytest

from krrood.adapters.json_serializer import from_json, to_json
from krrood.units import BaseQuantity, BaseUnit

# %% Base unit definitions


@pytest.mark.parametrize(
    "unit,symbol,quantity",
    [
        (BaseUnit.METRE, "m", BaseQuantity.LENGTH),
        (BaseUnit.KILOGRAM, "kg", BaseQuantity.MASS),
        (BaseUnit.SECOND, "s", BaseQuantity.TIME),
        (BaseUnit.AMPERE, "A", BaseQuantity.ELECTRIC_CURRENT),
        (BaseUnit.KELVIN, "K", BaseQuantity.THERMODYNAMIC_TEMPERATURE),
        (BaseUnit.MOLE, "mol", BaseQuantity.AMOUNT_OF_SUBSTANCE),
        (BaseUnit.CANDELA, "cd", BaseQuantity.LUMINOUS_INTENSITY),
    ],
)
def test_base_unit_definition(
    unit: BaseUnit, symbol: str, quantity: BaseQuantity
) -> None:
    """
    Each base unit uses its standard SI symbol and measures its base quantity.
    """
    assert unit.symbol == symbol
    assert unit.quantity is quantity


def test_every_base_quantity_has_one_unit() -> None:
    """
    The catalog has one distinct unit for each base quantity without aliases.
    """
    assert {unit.quantity for unit in BaseUnit} == set(BaseQuantity)
    assert len(BaseUnit.__members__) == len(BaseUnit) == len(BaseQuantity)


def test_base_unit_symbol_is_immutable() -> None:
    """
    A shared unit cannot be relabelled by one consumer.
    """
    with pytest.raises(FrozenInstanceError):
        BaseUnit.KELVIN.symbol = BaseUnit.METRE.symbol


def test_base_unit_quantity_is_immutable() -> None:
    """
    A shared unit cannot be reassigned to another physical quantity.
    """
    with pytest.raises(FrozenInstanceError):
        BaseUnit.KELVIN.quantity = BaseQuantity.LENGTH


# %% Persistence


@pytest.mark.parametrize("definition", [*BaseUnit, *BaseQuantity])
def test_unit_definition_roundtrip(definition: BaseUnit | BaseQuantity) -> None:
    """
    Existing enum serialization restores the canonical catalog member.
    """
    assert from_json(to_json(definition)) is definition
