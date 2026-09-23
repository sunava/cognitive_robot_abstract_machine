#!/usr/bin/env bash
# Activate only the isolated laboratory and its dedicated nginx mount.
set -euo pipefail

deployment_root=/srv/cramera-laboratory/scripts/laboratory-deployment
deployment_site=/etc/nginx/sites-available/cramera
deployment_include='    include /etc/nginx/snippets/cramera-laboratory.conf;'
deployment_backup="/etc/cramera-laboratory/nginx-before-laboratory-$(date +%Y%m%d%H%M%S)"

chown -R cramera-lab:cramera-lab /var/lib/cramera-laboratory
chown cramera-lab:cramera-lab /etc/cramera-laboratory/password
chmod 600 /etc/cramera-laboratory/password
printf '%s\n' https://cramera.informatik.uni-bremen.de > /etc/cramera-laboratory/origin
chmod 644 /etc/cramera-laboratory/origin
install -m 644 "$deployment_root/cramera-laboratory-backend.service" "$deployment_root/cramera-laboratory-gateway.service" /etc/systemd/system/
systemd-analyze verify /etc/systemd/system/cramera-laboratory-backend.service /etc/systemd/system/cramera-laboratory-gateway.service
systemctl daemon-reload
systemctl restart cramera-laboratory-backend cramera-laboratory-gateway

deployment_deadline=$((SECONDS + 55))
deployment_ready=0
while ((SECONDS < deployment_deadline)); do
    if curl --fail --silent --max-time 2 http://127.0.0.1:8716/api/laboratory/physics/robot/state >/dev/null &&
       curl --fail --silent --max-time 2 -H 'Host: cramera.informatik.uni-bremen.de' http://127.0.0.1:8717/laboratory/ >/dev/null; then
        deployment_ready=1
        break
    fi
    sleep 0.5
done
if ((deployment_ready == 0)); then
    echo 'The isolated laboratory services did not become ready; nginx was not changed.' >&2
    exit 1
fi

cp -p -- "$deployment_site" "$deployment_backup"
install -m 644 "$deployment_root/nginx-laboratory.conf" /etc/nginx/snippets/cramera-laboratory.conf
if ! grep -Fqx "$deployment_include" "$deployment_site"; then
    sed -i '\|    location / {|i\    include /etc/nginx/snippets/cramera-laboratory.conf;\n' "$deployment_site"
fi
if ! nginx -t; then
    cp -p -- "$deployment_backup" "$deployment_site"
    echo 'Nginx rejected the laboratory mount; its previous configuration was restored.' >&2
    exit 1
fi
systemctl reload nginx
systemctl enable cramera-laboratory-backend cramera-laboratory-gateway
printf '%s\n' 'Laboratory: https://cramera.informatik.uni-bremen.de/laboratory/'
