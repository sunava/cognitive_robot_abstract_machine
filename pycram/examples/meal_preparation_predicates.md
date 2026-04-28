---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Meal Preparation Predicates

This example shows how to use the meal-preparation EQL predicates with semantic annotations instead of the old SPARQL
query helpers.

```{code-cell} ipython3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Colander,
    Knife,
    Peel,
    Peeler,
    Potato,
)
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Sphere
from semantic_digital_twin.world_description.world_entity import Body

from krrood.entity_query_language.factories import an, and_, entity, variable

from pycram.querying.meal_preparation_predicates import (
    HasCuttingRepetition,
    PouringAngleInRange,
    RequiresRemovingPart,
    UsesCuttingTool,
    UsesPeelingTool,
    UsesPouringTool,
    get_cutting_tool,
    get_max_pouring_angle,
)
```

We first create a small world with an apple and a potato and attach semantic annotations to them.

```{code-cell} ipython3
world = World()

with world.modify_world():
    root = Body(name=PrefixedName("root"))
    world.add_body(root)

    apple_body = Body(name=PrefixedName("apple_body"))
    apple_shape = Sphere(
        radius=0.05,
        origin=HomogeneousTransformationMatrix(reference_frame=apple_body),
    )
    apple_body.collision = [apple_shape]
    apple_body.visual = [apple_shape]
    apple_connection = Connection6DoF.create_with_dofs(
        parent=root, child=apple_body, world=world
    )
    world.add_connection(apple_connection)
    world.add_semantic_annotation(Apple(root=apple_body, name=PrefixedName("apple")))

    potato_body = Body(name=PrefixedName("potato_body"))
    potato_shape = Sphere(
        radius=0.06,
        origin=HomogeneousTransformationMatrix(reference_frame=potato_body),
    )
    potato_body.collision = [potato_shape]
    potato_body.visual = [potato_shape]
    potato_connection = Connection6DoF.create_with_dofs(
        parent=root, child=potato_body, world=world
    )
    world.add_connection(potato_connection)
    world.state[potato_connection.y.id].position = 0.2
    world.add_semantic_annotation(
        Potato(root=potato_body, name=PrefixedName("potato"))
    )
```

The predicates operate on semantic-annotation types and techniques.

```{code-cell} ipython3
food = variable(type_=Apple, domain=world.semantic_annotations)
quarterable_apples = an(
    entity(food).where(
        and_(
            UsesCuttingTool(food, Knife, technique="Quartering"),
            HasCuttingRepetition(food, "Quartering", "exactly 1"),
        )
    )
)

print(list(quarterable_apples.evaluate()))
print(get_cutting_tool(Apple, "Quartering"))
```

You can also ask for food that requires removing a certain part or that needs a peeling tool.

```{code-cell} ipython3
produce = variable(type_=Potato, domain=world.semantic_annotations)
potatoes_to_peel = an(
    entity(produce).where(
        and_(
            RequiresRemovingPart(produce, Peel),
            UsesPeelingTool(produce, Peeler),
        )
    )
)

print(list(potatoes_to_peel.evaluate()))
```

For pouring, the predicates can check tool compatibility and parameter ranges.

```{code-cell} ipython3
drainable_food = variable(type_=Potato, domain=world.semantic_annotations)
draining_query = an(
    entity(drainable_food).where(
        and_(
            UsesPouringTool(drainable_food, "Draining", Colander),
            PouringAngleInRange(drainable_food, "Draining", 45.0),
        )
    )
)

print(list(draining_query.evaluate()))
print(get_max_pouring_angle(Potato, "Draining"))
```
