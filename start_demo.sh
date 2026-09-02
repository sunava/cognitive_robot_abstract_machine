#!/usr/bin/env bash
# Start a coraplex demo with the live cramera bridge (port 8765).
# Usage:  ./start_demo.sh [path/to/demo.py]
# Default: the PR2 bullet-world breakfast demo.
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="${1:-coraplex/demos/coraplex_bullet_world_demo/demo.py}"

# ROS workspace: sets AMENT_PREFIX_PATH + rclpy so robot descriptions (URDF/meshes) resolve
source ~/ros2_ws/install/setup.bash 2>/dev/null || echo "warn: ~/ros2_ws/install/setup.bash not found"

cd "$REPO"
export CORAPLEX_VISUALIZATION=cramera   # browser backend -> starts the live bridge on :8765
echo "▶ running demo: $DEMO"
echo "  (first run caches meshes ~1 min; bridge stays up after — Ctrl-C to quit)"
exec ./.venv/bin/cramera-live "$DEMO"
