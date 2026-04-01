#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORTAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE="${PORTAL_ROOT}/portal.pid"
PORT="${PORT:-8090}"

if [ -f "${PID_FILE}" ]; then
  PID="$(cat "${PID_FILE}")"
  if [ -n "${PID}" ] && kill -0 "${PID}" >/dev/null 2>&1; then
    kill "${PID}" >/dev/null 2>&1 || true
    echo "stopped portal ${PID}"
    rm -f "${PID_FILE}"
    exit 0
  fi
fi

FALLBACK_PID="$(lsof -t -iTCP:${PORT} -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
if [ -n "${FALLBACK_PID}" ]; then
  kill "${FALLBACK_PID}" >/dev/null 2>&1 || true
  echo "stopped portal ${FALLBACK_PID} (matched running process)"
  rm -f "${PID_FILE}"
  exit 0
fi

echo "portal was not running"
rm -f "${PID_FILE}"
