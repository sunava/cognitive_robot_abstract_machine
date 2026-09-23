#!/usr/bin/env bash
# Keep the full CRAMERA server private while sharing its laboratory through HTTPS.
set -euo pipefail
umask 077

# %% configuration
share_repository="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
share_state="${CRAMERA_SHARE_STATE:-${XDG_STATE_HOME:-$HOME/.local/state}/cramera/share}"
share_cloudflared="${CRAMERA_CLOUDFLARED:-$HOME/.local/bin/cloudflared}"
share_python="$share_repository/.venv/bin/python"
share_gateway_url="http://127.0.0.1:8717"
share_backend_url="http://127.0.0.1:8716"
share_scene_query="index.html?scene=precision_lab_pr2_physics&layout=scene&offline=1"
share_start_timeout=55
share_stop_timeout=5

# %% process ownership
owned_process() {
    local kind="$1" process_id arguments
    [[ -f "$share_state/$kind.pid" ]] || return 1
    read -r process_id < "$share_state/$kind.pid"
    [[ "$process_id" =~ ^[1-9][0-9]*$ && -r "/proc/$process_id/cmdline" ]] || return 1
    arguments="$(tr '\0' '\n' < "/proc/$process_id/cmdline")"
    if [[ "$kind" == gateway ]]; then
        [[ "$arguments" == *$'\ncramera.laboratory_share\n'* &&
           "$arguments" == *$'\n'"$share_state/password"$'\n'* ]] || return 1
    else
        [[ "$arguments" == "$share_cloudflared"$'\n'* &&
           "$arguments" == *$'\n'"$share_gateway_url"* ]] || return 1
    fi
    kill -0 "$process_id" 2>/dev/null
}

stop_process() {
    local kind="$1" process_id deadline
    [[ -f "$share_state/$kind.pid" ]] || return 0
    read -r process_id < "$share_state/$kind.pid"
    if ! owned_process "$kind"; then
        if [[ "$process_id" =~ ^[1-9][0-9]*$ ]] && kill -0 "$process_id" 2>/dev/null; then
            echo "Refusing to stop unrelated process in $kind.pid." >&2
            return 1
        fi
        rm -f -- "$share_state/$kind.pid"
        return 0
    fi
    kill "$process_id"
    deadline=$((SECONDS + share_stop_timeout))
    while owned_process "$kind" && ((SECONDS < deadline)); do sleep 0.1; done
    if owned_process "$kind"; then
        echo "The $kind process is still shutting down; run stop again." >&2
        return 1
    fi
    rm -f -- "$share_state/$kind.pid"
}

# %% status and shutdown
show_status() {
    local origin
    if ! owned_process gateway || ! owned_process tunnel || [[ ! -s "$share_state/origin" ]]; then
        echo "The laboratory share is not running. Start it with: $0 start"
        return 1
    fi
    read -r origin < "$share_state/origin"
    printf 'Laboratory: %s/%s\nPassword: ' "$origin" "$share_scene_query"
    cat -- "$share_state/password"
    printf '\nRuntime files: %s\n' "$share_state"
}

stop_share() {
    local result=0
    stop_process tunnel || result=1
    stop_process gateway || result=1
    if ((result == 0)) && [[ -d "$share_state" ]]; then rm -f -- "$share_state/origin"; fi
    return "$result"
}

cleanup_start() {
    local preserve_tunnel="$1"
    if ((preserve_tunnel)); then
        stop_process gateway
        return
    fi
    stop_share
}

# %% startup
start_share() {
    local deadline origin gateway_ready=0 preserve_tunnel=0
    if owned_process gateway && owned_process tunnel && [[ -s "$share_state/origin" ]]; then
        show_status
        return
    fi
    [[ ! -L "$share_state" ]] || { echo "Runtime directory cannot be a symlink." >&2; return 1; }
    [[ -x "$share_python" && -x "$share_cloudflared" ]] || {
        echo "The project .venv/bin/python and $share_cloudflared are required." >&2
        return 1
    }
    curl --fail --silent --max-time 3 "$share_backend_url/api/laboratory/physics/robot/state" >/dev/null || {
        echo "Start the local laboratory server first:" >&2
        printf 'cd %q && CRAMERA_SCENE=precision_lab .venv/bin/python -m cramera.server 8716 --no-browser\n' "$share_repository" >&2
        return 1
    }
    mkdir -p -- "$share_state"
    chmod 700 -- "$share_state"
    if owned_process tunnel && [[ -s "$share_state/origin" ]]; then
        preserve_tunnel=1
        stop_process gateway || return 1
    else
        stop_share || return 1
    fi
    if [[ ! -s "$share_state/password" ]]; then openssl rand -hex 24 > "$share_state/password"; fi
    chmod 600 -- "$share_state/password"
    trap "cleanup_start $preserve_tunnel >/dev/null 2>&1 || true" ERR
    cd -- "$share_repository"
    nohup "$share_python" -m cramera.laboratory_share \
        --port 8717 --backend-port 8716 \
        --password-file "$share_state/password" \
        --public-origin-file "$share_state/origin" \
        > "$share_state/gateway.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$share_state/gateway.pid"
    deadline=$((SECONDS + share_start_timeout))
    while ((SECONDS < deadline)); do
        if ! owned_process gateway; then
            echo "Gateway failed; see $share_state/gateway.log" >&2
            cleanup_start "$preserve_tunnel"
            return 1
        fi
        if curl --silent --output /dev/null --max-time 1 "$share_gateway_url/"; then
            gateway_ready=1
            break
        fi
        sleep 0.2
    done
    if ((gateway_ready == 0)); then
        echo "Gateway startup timed out; see $share_state/gateway.log" >&2
        cleanup_start "$preserve_tunnel"
        return 1
    fi
    if ((preserve_tunnel)); then
        trap - ERR
        show_status
        return
    fi
    nohup "$share_cloudflared" tunnel --no-autoupdate --metrics 127.0.0.1:0 \
        --url "$share_gateway_url" > "$share_state/tunnel.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$share_state/tunnel.pid"
    while ((SECONDS < deadline)); do
        if ! owned_process tunnel; then
            echo "Tunnel failed; see $share_state/tunnel.log" >&2
            stop_share
            return 1
        fi
        origin="$(rg -o 'https://[a-z0-9-]+\.trycloudflare\.com' "$share_state/tunnel.log" | sed -n '1p' || true)"
        if [[ -n "$origin" ]]; then
            printf '%s\n' "$origin" > "$share_state/origin.new"
            mv -- "$share_state/origin.new" "$share_state/origin"
            trap - ERR
            show_status
            return
        fi
        sleep 0.2
    done
    echo "Tunnel startup timed out; see $share_state/tunnel.log" >&2
    stop_share
    return 1
}

# %% command line
case "${1:-status}" in
    start) start_share ;;
    status) show_status ;;
    stop) stop_share ;;
    *) echo "Usage: $0 {start|status|stop}" >&2; exit 2 ;;
esac
