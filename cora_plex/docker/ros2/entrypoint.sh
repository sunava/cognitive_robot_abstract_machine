#!/bin/bash

set -e

source /opt/ros/overlay_ws/install/setup.bash
source /opt/ros/overlay_ws/src/cora_plex/cora_plex-venv/bin/activate

exec "$@"