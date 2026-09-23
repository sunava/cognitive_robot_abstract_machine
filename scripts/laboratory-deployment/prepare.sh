#!/usr/bin/env bash
# Prepare the isolated runtime without changing an existing site's services.
set -euo pipefail

if ! getent passwd cramera-lab >/dev/null; then
    useradd --system --home-dir /var/lib/cramera-laboratory --shell /usr/sbin/nologin cramera-lab
fi
install -d -m 0755 /srv/cramera-laboratory
install -d -m 0750 -o cramera-lab -g cramera-lab /var/lib/cramera-laboratory
install -d -m 0750 -o root -g cramera-lab /etc/cramera-laboratory
