import hashlib
import re
import secrets
import time
import uuid

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from app.storage.database import database


PASSWORD_HASHER = PasswordHasher()
SESSION_TTL_SECONDS = 3600

DUMMY_HASH = PASSWORD_HASHER.hash(secrets.token_urlsafe(32))


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_user(
    username: str,
    password: str,
    role: str = "user",
) -> dict:
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,30}", username) is None:
        raise ValueError(
            "Имя: от 1 до 30 символов, латинские буквы, цифры, _, . или -."
        )

    if not 8 <= len(password) <= 32:
        raise ValueError("Пароль должен содержать от 8 до 32 символов.")

    if role not in {"administrator", "developer", "user"}:
        raise ValueError("Неизвестная роль.")

    user_id = str(uuid.uuid4())
    password_hash = PASSWORD_HASHER.hash(password)
    collection_id = None

    with database() as connection:
        connection.execute(
            """
            INSERT INTO users
                (id, username, password_hash, role)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, username, password_hash, role),
        )

        if role != "developer":
            collection_id = str(uuid.uuid4())

            connection.execute(
                """
                INSERT INTO collections
                    (id, name, kind, owner_id)
                VALUES (?, ?, 'personal', ?)
                """,
                (collection_id, f"Личная коллекция {username}", user_id),
            )

    return {
        "id": user_id,
        "username": username,
        "role": role,
        "personal_collection_id": collection_id,
    }


def login(username: str, password: str) -> dict | None:
    with database() as connection:
        user = connection.execute(
            """
            SELECT id, password_hash, is_active
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()

    stored_hash = user["password_hash"] if user else DUMMY_HASH

    try:
        PASSWORD_HASHER.verify(stored_hash, password)
    except (VerificationError, InvalidHashError):
        return None

    if user is None or not user["is_active"]:
        return None

    token = secrets.token_urlsafe(32)
    now = int(time.time())
    expires_at = now + SESSION_TTL_SECONDS

    with database() as connection:
        connection.execute(
            """
            INSERT INTO sessions
                (token_hash, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (hash_token(token), user["id"], now, expires_at),
        )

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": SESSION_TTL_SECONDS,
    }


def resolve_session(token: str) -> dict | None:
    with database() as connection:
        user = connection.execute(
            """
            SELECT
                users.id,
                users.username,
                users.role,
                sessions.token_hash
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = ?
              AND sessions.expires_at > ?
              AND users.is_active = 1
            """,
            (hash_token(token), int(time.time())),
        ).fetchone()

    return dict(user) if user else None


def logout(token_hash: str) -> None:
    with database() as connection:
        connection.execute(
            "DELETE FROM sessions WHERE token_hash = ?",
            (token_hash,),
        )