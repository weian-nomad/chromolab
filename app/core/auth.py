from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import Header, HTTPException

from app.config import DEFAULT_USERS_FILE, SESSIONS_FILE, USERS_FILE
from app.core.storage import STORE

SESSION_TTL_DAYS = 7
ALLOWED_ROLES = {"admin", "annotator", "viewer"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pbkdf2_hash(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    )
    return base64.b64encode(digest).decode("utf-8")


def _load_users() -> list[dict[str, Any]]:
    ensure_default_users()
    return STORE.load(USERS_FILE, [])


def _save_users(users: list[dict[str, Any]]) -> None:
    STORE.save(USERS_FILE, users)


def _load_sessions() -> list[dict[str, Any]]:
    return STORE.load(SESSIONS_FILE, [])


def _save_sessions(sessions: list[dict[str, Any]]) -> None:
    STORE.save(SESSIONS_FILE, sessions)


def ensure_default_users() -> None:
    if USERS_FILE.exists():
        return
    seed_users = STORE.load(DEFAULT_USERS_FILE, [])
    users: list[dict[str, Any]] = []
    for item in seed_users:
        salt = secrets.token_hex(16)
        users.append(
            {
                "id": secrets.token_hex(8),
                "username": item["username"],
                "display_name": item.get("display_name") or item["username"],
                "role": item["role"],
                "password_hash": _pbkdf2_hash(item["password"], salt),
                "password_salt": salt,
                "created_at": now_iso(),
            }
        )
    _save_users(users)


def list_users() -> list[dict[str, Any]]:
    return [
        {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
            "role": user["role"],
            "created_at": user["created_at"],
        }
        for user in _load_users()
    ]


def create_user(username: str, display_name: str, role: str, password: str) -> dict[str, Any]:
    users = _load_users()
    if any(user["username"] == username for user in users):
        raise HTTPException(status_code=409, detail="Username already exists")
    if role not in ALLOWED_ROLES:
        raise HTTPException(status_code=400, detail=f"Unsupported role: {role}")
    salt = secrets.token_hex(16)
    record = {
        "id": secrets.token_hex(8),
        "username": username,
        "display_name": display_name or username,
        "role": role,
        "password_hash": _pbkdf2_hash(password, salt),
        "password_salt": salt,
        "created_at": now_iso(),
    }
    users.append(record)
    _save_users(users)
    return {
        "id": record["id"],
        "username": record["username"],
        "display_name": record["display_name"],
        "role": record["role"],
        "created_at": record["created_at"],
    }


def _sanitize_user(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "username": record["username"],
        "display_name": record["display_name"],
        "role": record["role"],
        "created_at": record["created_at"],
    }


def authenticate(username: str, password: str) -> dict[str, Any]:
    for user in _load_users():
        expected = _pbkdf2_hash(password, user["password_salt"])
        if user["username"] == username and hmac.compare_digest(expected, user["password_hash"]):
            return _sanitize_user(user)
    raise HTTPException(status_code=401, detail="Invalid credentials")


def create_session(user: dict[str, Any]) -> str:
    token = secrets.token_urlsafe(32)
    sessions = [item for item in _load_sessions() if item["expires_at"] > now_iso()]
    sessions.append(
        {
            "token": token,
            "user_id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
            "role": user["role"],
            "created_at": now_iso(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS)).isoformat(),
        }
    )
    _save_sessions(sessions)
    return token


def revoke_session(token: str) -> None:
    sessions = [item for item in _load_sessions() if item["token"] != token]
    _save_sessions(sessions)


def get_session(token: str) -> dict[str, Any]:
    sessions = _load_sessions()
    active_sessions = []
    current = None
    now = now_iso()
    for item in sessions:
        if item["expires_at"] <= now:
            continue
        active_sessions.append(item)
        if item["token"] == token:
            current = item
    if sessions != active_sessions:
        _save_sessions(active_sessions)
    if not current:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    return current


def session_from_token(token: str) -> dict[str, Any]:
    return get_session(token)


def extract_token(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return authorization.split(" ", 1)[1].strip()


def current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = extract_token(authorization)
    return get_session(token)


def require_roles(*roles: str):
    def dependency(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        user = current_user(authorization)
        if roles and user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return user

    return dependency
