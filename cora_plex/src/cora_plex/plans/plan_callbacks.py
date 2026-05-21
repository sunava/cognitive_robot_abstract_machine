from dataclasses import dataclass

from cora_plex.plans.plan import Plan
from cora_plex.plans.plan_entity import PlanEntity
from cora_plex.plans.plan_node import PlanNode


@dataclass
class PlanCallback(PlanEntity):

    def on_start(self, node: PlanNode): ...

    def on_end(self, node: PlanNode): ...
