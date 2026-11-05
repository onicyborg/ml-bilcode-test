#!/usr/bin/env bash
set -euo pipefail
export IMAGE="$1"
docker network inspect siklus >/dev/null 2>&1 || docker network create siklus >/dev/null
docker compose pull ml
docker compose up -d --remove-orphans ml
container=$(docker compose ps -q ml)
for attempt in $(seq 1 60); do
  if docker exec "$container" python -c "import urllib.request,json; assert json.load(urllib.request.urlopen('http://127.0.0.1:8000/health'))['ready']" >/dev/null 2>&1; then exit 0; fi
  sleep 5
done
echo 'Service readiness failed; inspect container logs on the VPS.' >&2
exit 1
