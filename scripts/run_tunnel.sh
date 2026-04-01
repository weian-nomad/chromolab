#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORTAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHROMOSOME_ROOT="$(cd "${PORTAL_ROOT}/.." && pwd)"
BINARY="${NGROK_BIN:-${CHROMOSOME_ROOT}/ngrok}"
PORT="${PORT:-8090}"
NGROK_URL="${NGROK_URL:-https://tmu-chromosome.ngrok.pizza}"
NGROK_HOST="${NGROK_URL#https://}"
NGROK_HOST="${NGROK_HOST#http://}"
LOG_FILE="${PORTAL_ROOT}/ngrok.log"
PID_FILE="${PORTAL_ROOT}/ngrok.pid"
DOWNLOAD_TGZ="${CHROMOSOME_ROOT}/ngrok-v3-stable-linux-amd64.tgz"
EXISTING_PID=""

if [ -f "${PID_FILE}" ]; then
  EXISTING_PID="$(cat "${PID_FILE}")"
  if [ -n "${EXISTING_PID}" ] && kill -0 "${EXISTING_PID}" >/dev/null 2>&1; then
    echo "ngrok already running with pid ${EXISTING_PID}"
    exit 0
  fi
fi

EXISTING_PID="$(pgrep -fo "ngrok http ${PORT} --url=.*${NGROK_HOST}" || true)"
if [ -n "${EXISTING_PID}" ]; then
  echo "${EXISTING_PID}" > "${PID_FILE}"
  echo "ngrok already running with pid ${EXISTING_PID}"
  exit 0
fi

if [ -x "${BINARY}" ] && file "${BINARY}" | grep -q 'ELF 64-bit'; then
  :
else
  rm -f "${BINARY}" "${DOWNLOAD_TGZ}"
  curl -L --fail -o "${DOWNLOAD_TGZ}" https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
  tar xvzf "${DOWNLOAD_TGZ}" -C "${CHROMOSOME_ROOT}" ngrok >/dev/null
  chmod +x "${BINARY}"
fi

if [ -n "${NGROK_AUTHTOKEN:-}" ]; then
  "${BINARY}" config add-authtoken "${NGROK_AUTHTOKEN}" >/dev/null
fi

nohup "${BINARY}" http "${PORT}" --url="${NGROK_URL}" > "${LOG_FILE}" 2>&1 < /dev/null &
echo $! > "${PID_FILE}"
echo "ngrok started with pid $(cat "${PID_FILE}")"
