#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORTAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILES=("${PORTAL_ROOT}/ngrok.pid" "${PORTAL_ROOT}/cloudflared.pid")

for PID_FILE in "${PID_FILES[@]}"; do
  if [ ! -f "${PID_FILE}" ]; then
    continue
  fi
  PID="$(cat "${PID_FILE}")"
  if kill "${PID}" >/dev/null 2>&1; then
    echo "stopped tunnel ${PID}"
  else
    echo "tunnel ${PID} was not running"
  fi
  rm -f "${PID_FILE}"
done
