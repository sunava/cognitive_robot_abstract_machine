#!/usr/bin/env bash
# Start the cramera web viewer (http://localhost:8711). Open it, click "◉ Live",
# then the Plan tab to see the running demo. Run alongside ./start_demo.sh.
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ~/ros2_ws/install/setup.bash 2>/dev/null || true
cd "$REPO"
echo "▶ viewer on http://localhost:8711/  (Ctrl-C to quit)"
exec ./.venv/bin/cramera --no-browser
