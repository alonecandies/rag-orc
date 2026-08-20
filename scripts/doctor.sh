#!/usr/bin/env bash
# Diagnose "the stack is up but nothing connects".
#
# This exists because the failure signature is genuinely confusing: every
# container reports healthy (their healthchecks run *inside* the container), the
# host ports show as LISTEN (Docker's forwarder binds them before it works), and
# `nc -z` succeeds (the forwarder accepts the TCP connection). Yet every client
# times out. The distinguishing test is whether a *sibling container* can reach
# the service — if it can, the services are fine and the host publish path is
# broken, which is a Docker Desktop problem and not a configuration one.
set -uo pipefail

# Read .env so the port overrides configured there are honoured — otherwise this
# script probes 5432 while the stack publishes 5433 and reports a false failure.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

pass() { printf '  \033[32mOK\033[0m    %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }
info() { printf '        %s\n' "$1"; }

host_http() { curl -s -o /dev/null -w '%{http_code}' -m 5 "$1" 2>/dev/null; }

PG_PORT="${RAGORC_PG_PORT:-5432}"
QD_HTTP="${RAGORC_QDRANT_HTTP_PORT:-6333}"
QD_GRPC="${RAGORC_QDRANT_GRPC_PORT:-6334}"
NEO_HTTP="${RAGORC_NEO4J_HTTP_PORT:-7474}"

echo "docker"
if ! docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
  fail "the Docker daemon is not reachable — start Docker Desktop"
  exit 2
fi
pass "daemon reachable ($(docker version --format '{{.Server.Version}}'))"

echo
echo "containers"
docker compose ps --format '{{.Service}} {{.State}} {{.Health}}' 2>/dev/null | while read -r svc state health; do
  if [ "$state" = "running" ]; then pass "$svc $state ${health:-(no healthcheck)}"
  else fail "$svc $state — check: docker compose logs $svc"; fi
done

echo
echo "from the host"
host_ok=0
for probe in "qdrant|http://localhost:${QD_HTTP}/readyz" "neo4j|http://localhost:${NEO_HTTP}"; do
  name="${probe%%|*}"; url="${probe#*|}"
  code=$(host_http "$url")
  if [ "$code" = "200" ]; then pass "$name reachable (HTTP $code)"; host_ok=$((host_ok+1))
  else fail "$name NOT reachable (HTTP ${code:-000}) at $url"; fi
done
# Probed with the project's own psycopg rather than the psql CLI. macOS does not
# ship psql, and the previous check treated "the client is missing" as "the server
# is unreachable" — reporting a failure for something it had not measured, which is
# worse than reporting nothing. psycopg is a base dependency, so it is always here.
PY_BIN=""
for candidate in .venv/bin/python python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import psycopg' >/dev/null 2>&1; then
    PY_BIN="$candidate"; break
  fi
done

if [ -z "$PY_BIN" ]; then
  info "postgres: not probed (no python with psycopg available)"
elif "$PY_BIN" - "$PG_PORT" <<'PYEOF' >/dev/null 2>&1
import sys, psycopg
port = sys.argv[1]
with psycopg.connect(f"postgresql://ragorc:ragorc@localhost:{port}/ragorc", connect_timeout=5) as conn:
    conn.execute("select 1")
PYEOF
then
  pass "postgres reachable on ${PG_PORT}"; host_ok=$((host_ok+1))
else
  fail "postgres NOT reachable on ${PG_PORT} from the host"
fi

echo
echo "from inside the docker network"
net="$(docker compose ps --format '{{.Name}}' 2>/dev/null | head -1)"
if [ -z "$net" ]; then
  fail "no containers running — run: docker compose up -d"
  exit 1
fi
sibling=$(docker run --rm --network ragorc_default curlimages/curl:latest \
  -s -o /dev/null -w '%{http_code}' -m 5 "http://qdrant:6333/readyz" 2>/dev/null)
if [ "$sibling" = "200" ]; then pass "qdrant reachable from a sibling container (HTTP 200)"
else fail "qdrant NOT reachable inside the network either (HTTP ${sibling:-000})"; fi

echo
echo "verdict"
if [ "$sibling" = "200" ] && [ "$host_ok" -eq 0 ]; then
  fail "the services are healthy but the HOST cannot reach them"
  info "Docker Desktop's port forwarder is wedged. This is not a configuration"
  info "problem — most often it happens after the machine sleeps."
  info ""
  info "Fix: restart Docker Desktop (quit it fully, then reopen), then:"
  info "     docker compose up -d && make doctor"
  exit 1
fi
if [ "$host_ok" -gt 0 ]; then
  pass "host can reach the stack — you are good to run: make schema && make seed"
  exit 0
fi
fail "the services themselves are not ready; check: docker compose logs"
exit 1
