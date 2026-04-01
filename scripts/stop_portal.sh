#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORTAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE="${PORTAL_ROOT}/portal.pid"

if [ ! -f "${PID_FILE}" ]; then
  echo "portal pid file not found"
  exit 0
fi

PID="$(cat "${PID_FILE}")"
if kill "${PID}" >/dev/null 2>&1; then
  echo "stopped portal ${PID}"
else
  echo "portal ${PID} was not running"
fi
rm -f "${PID_FILE}"
