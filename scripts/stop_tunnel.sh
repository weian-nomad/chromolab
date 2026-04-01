#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORTAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILES=("${PORTAL_ROOT}/ngrok.pid" "${PORTAL_ROOT}/cloudflared.pid")
NGROK_URL="${NGROK_URL:-https://tmu-chromosome.ngrok.pizza}"
NGROK_HOST="${NGROK_URL#https://}"
NGROK_HOST="${NGROK_HOST#http://}"

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

FALLBACK_NGROK_PID="$(pgrep -fo "ngrok http .*${NGROK_HOST}" || true)"
if [ -n "${FALLBACK_NGROK_PID}" ]; then
  kill "${FALLBACK_NGROK_PID}" >/dev/null 2>&1 || true
  echo "stopped tunnel ${FALLBACK_NGROK_PID} (matched ngrok process)"
fi

FALLBACK_CLOUDFLARED_PID="$(pgrep -fo "cloudflared.*${PORTAL_ROOT}" || true)"
if [ -n "${FALLBACK_CLOUDFLARED_PID}" ]; then
  kill "${FALLBACK_CLOUDFLARED_PID}" >/dev/null 2>&1 || true
  echo "stopped tunnel ${FALLBACK_CLOUDFLARED_PID} (matched cloudflared process)"
fi
