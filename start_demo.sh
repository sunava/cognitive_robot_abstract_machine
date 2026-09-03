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
# Resolve the cramera-live CLI without hardcoding a venv (honor $VENV, else repo .venv,
# else PATH) — so a fresh checkout with a different venv works without editing this file.
LIVE="${VENV:+$VENV/bin/cramera-live}"
[ -n "$LIVE" ] && [ -x "$LIVE" ] || LIVE="$REPO/.venv/bin/cramera-live"
[ -x "$LIVE" ] || LIVE="$(command -v cramera-live || true)"
[ -n "$LIVE" ] || { echo "cramera-live not found — activate your venv or set VENV=/path/to/venv"; exit 1; }
echo "▶ running demo: $DEMO"
echo "  (first run caches meshes ~1 min; bridge stays up after — Ctrl-C to quit)"
exec "$LIVE" "$DEMO"
