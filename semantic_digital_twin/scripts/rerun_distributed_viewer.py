from __future__ import annotations

import threading
import time

import rclpy

from semantic_digital_twin.adapters.rerun import RerunSink, RerunVisualizer
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    ModelSynchronizer,
    StateSynchronizer,
)


def main() -> None:
    """Mirror the published world and render it in Rerun until interrupted."""
    rclpy.init()
    node = rclpy.create_node("semdt_viewer")
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    world = fetch_world_from_service(node)
    StateSynchronizer(_world=world, node=node)
    ModelSynchronizer(_world=world, node=node)
    RerunVisualizer(
        _world=world,
        application_id="semdt_rerun_viewer",
        sink=RerunSink.SPAWN,
        state_history=False,
        memory_limit="1GB",
    )

    print("Viewer running; updates follow the publisher. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
