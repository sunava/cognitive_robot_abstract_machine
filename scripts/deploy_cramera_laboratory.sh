#!/usr/bin/env bash
# Stage or activate the isolated password-protected laboratory on its existing host.
set -euo pipefail
umask 077

deployment_repository="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
deployment_host="${CRAMERA_DEPLOY_HOST:-root@cramera.informatik.uni-bremen.de}"
deployment_key="${CRAMERA_DEPLOY_IDENTITY:?Set CRAMERA_DEPLOY_IDENTITY to an existing SSH identity file}"
deployment_password="${CRAMERA_DEPLOY_PASSWORD_FILE:-${XDG_STATE_HOME:-$HOME/.local/state}/cramera/share/password}"
deployment_scenes="${CRAMERA_DEPLOY_SCENES:-$HOME/.cramera/scenes}"
deployment_ssh=(ssh -i "$deployment_key" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10)
printf -v deployment_transport '%q ' "${deployment_ssh[@]}"

case "${1:-stage}" in
    stage)
        [[ -r "$deployment_password" ]] || { echo 'The laboratory password file is missing.' >&2; exit 1; }
        "${deployment_ssh[@]}" "$deployment_host" bash -s < "$deployment_repository/scripts/laboratory-deployment/prepare.sh"
        cd -- "$deployment_repository"
        rsync -azR --exclude='__pycache__' --exclude='ormatic_interface.py' -e "$deployment_transport" \
            cramera/src/cramera coraplex/src/coraplex \
            cramera/scenes/pr2_breakfast_c/pr2_with_ft2_cableguide.urdf scripts/laboratory-deployment \
            "$deployment_host:/srv/cramera-laboratory/"
        deployment_assets="$(mktemp)"
        trap 'rm -f -- "$deployment_assets"' EXIT
        .venv/bin/python scripts/laboratory-deployment/robot_assets.py "$deployment_repository" \
            cramera/scenes/pr2_breakfast_c/pr2_with_ft2_cableguide.urdf > "$deployment_assets"
        rsync -az --files-from="$deployment_assets" -e "$deployment_transport" ./ "$deployment_host:/srv/cramera-laboratory/"
        rsync -az --exclude='__pycache__' --exclude='*.blend' -e "$deployment_transport" \
            "$deployment_scenes/precision_lab" "$deployment_scenes/precision_lab_pr2" \
            "$deployment_scenes/precision_lab_pr2_physics" "$deployment_host:/var/lib/cramera-laboratory/scenes/"
        "${deployment_ssh[@]}" "$deployment_host" 'umask 077; cat > /etc/cramera-laboratory/password' < "$deployment_password"
        ;;
    activate)
        "${deployment_ssh[@]}" "$deployment_host" bash /srv/cramera-laboratory/scripts/laboratory-deployment/activate.sh
        ;;
    *) echo "Usage: $0 {stage|activate}" >&2; exit 2 ;;
esac
