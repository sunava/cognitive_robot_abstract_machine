"""
Every connection of the world with when it last changed and who changed it: what a TF
tree view shows, for a world that has no TF.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field, replace

from typing_extensions import TYPE_CHECKING, Dict, Optional, Set, Tuple

from cramera_bridge.geometry import POSE_PRECISION
from cramera_protocol.live_messages import (
    ConnectionKind,
    ConnectionState,
    TransformFreshness,
    TransformGraph,
    TransformWriter,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    FixedConnection,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Connection

MOVING_SECONDS = 0.5
"""
How long after a change a connection reads as moving.
"""

SETTLED_SECONDS = 5.0
"""
How long after a change a connection reads as recently settled.
"""

SIGNATURE_SEPARATOR = "|"
"""
Separates the connections of a graph's signature.
"""

# %% one connection


@dataclass(frozen=True)
class ConnectionActivity:
    """
    One connection, its degree-of-freedom values and when they last changed.
    """

    name: str
    """
    The connection's prefixed name.
    """

    parent: str
    """
    The name of the frame it hangs from.
    """

    child: str
    """
    The name of the frame it carries.
    """

    kind: ConnectionKind
    """
    How it holds its child.
    """

    values: Tuple[float, ...]
    """
    Its degree-of-freedom values as last observed; empty when it has none.
    """

    changed_at: Optional[float] = None
    """
    When the values last changed, or None while they never did.
    """

    writer: TransformWriter = TransformWriter.NOBODY
    """
    Who made that change.
    """

    @staticmethod
    def kind_of(connection: Connection) -> ConnectionKind:
        """
        :param connection: A world connection.
        :return: How it holds its child.
        """
        if isinstance(connection, FixedConnection):
            return ConnectionKind.FIXED
        if isinstance(connection, ActiveConnection1DOF):
            return ConnectionKind.ACTUATED
        if isinstance(connection, Connection6DoF):
            return ConnectionKind.FREE
        return ConnectionKind.OTHER

    def freshness(self, now: float) -> TransformFreshness:
        """
        :param now: The time to age the last change against.
        :return: How recently the connection changed.
        """
        if not self.values:
            return TransformFreshness.STATIC
        if self.changed_at is None:
            return TransformFreshness.STALE
        age = now - self.changed_at
        if age <= MOVING_SECONDS:
            return TransformFreshness.MOVING
        if age <= SETTLED_SECONDS:
            return TransformFreshness.SETTLED
        return TransformFreshness.STALE

    def state(self, now: float) -> ConnectionState:
        """
        :param now: The time ages are computed against.
        :return: The connection as the protocol writes it.
        """
        return ConnectionState(
            name=self.name,
            parent=self.parent,
            child=self.child,
            kind=self.kind,
            writer=self.writer,
            freshness=self.freshness(now),
            age_seconds=(
                None
                if self.changed_at is None
                else round(now - self.changed_at, POSE_PRECISION)
            ),
        )


# %% tracker


@dataclass
class TransformTracker:
    """
    Notes, tick by tick, which connections changed; the simulation thread observes and
    HTTP threads read the graph.
    """

    activities: Dict[str, ConnectionActivity] = field(default_factory=dict)
    """
    The newest entry per connection name.
    """

    viewer_written: Set[str] = field(default_factory=set)
    """
    Connections a person changed through the viewer since the last observation.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards :attr:`activities` and :attr:`viewer_written`.
    """

    def note_viewer_write(self, connection_name: str) -> None:
        """
        :param connection_name: A connection a person just changed through the viewer.
        """
        with self.lock:
            self.viewer_written.add(connection_name)

    def observe(self, world: World, now: float) -> None:
        """
        Read every connection's values and stamp those that changed.

        :param world: The world.
        :param now: The time a change observed now is stamped with.
        """
        with self.lock:
            written, self.viewer_written = self.viewer_written, set()
            previous_activities = self.activities
        observed: Dict[str, ConnectionActivity] = {}
        for connection in world.connections:
            name = str(connection.name)
            values = tuple(
                round(float(world.state[dof.id].position), POSE_PRECISION)
                for dof in connection.dofs
            )
            previous = previous_activities.get(name)
            if previous is None:
                observed[name] = ConnectionActivity(
                    name=name,
                    parent=str(connection.parent.name),
                    child=str(connection.child.name),
                    kind=ConnectionActivity.kind_of(connection),
                    values=values,
                )
            elif values == previous.values and name not in written:
                observed[name] = previous
            else:
                observed[name] = replace(
                    previous,
                    values=values,
                    changed_at=now,
                    writer=(
                        TransformWriter.VIEWER
                        if name in written
                        else TransformWriter.DEMO
                    ),
                )
        with self.lock:
            self.activities = observed

    def graph(self, now: float) -> TransformGraph:
        """
        :param now: The time ages are computed against.
        :return: Every connection sorted by name, with the structure's signature.
        """
        with self.lock:
            ordered = sorted(self.activities.values(), key=lambda entry: entry.name)
        structure = SIGNATURE_SEPARATOR.join(
            f"{entry.name}:{entry.parent}:{entry.child}:{entry.kind}"
            for entry in ordered
        )
        return TransformGraph(
            signature=hashlib.sha1(structure.encode("utf-8")).hexdigest(),
            connections=[entry.state(now) for entry in ordered],
        )
