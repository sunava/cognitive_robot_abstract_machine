# Locate one of cramera's console scripts, for the start scripts that source this file.
#
# Prefers the copy the active environment offers, so ./start_*.sh run cramera from
# whatever venv the shell is in, and falls back to the repo venv a `uv sync` checkout
# provides. Mirrors cramera.paths.console_script, which resolves the same way when the
# viewer starts a demo of its own.
cramera_command() {
  local name="$1" found
  found="$(command -v "$name" || true)"
  if [ ! -x "$found" ]; then
    found="$REPO/.venv/bin/$name"
  fi
  if [ ! -x "$found" ]; then
    echo "$name not found: neither on PATH nor in $REPO/.venv/bin." >&2
    echo "Activate the environment cramera is installed in, or run 'uv sync'." >&2
    return 1
  fi
  printf '%s\n' "$found"
}
