import hashlib
import os
from pathlib import Path
import tempfile
import time
from typing import BinaryIO
import uuid

from app.core.access import can_access_collection
from app.core.config import PROJECT_ROOT, load_config
from app.storage.database import database


DOCUMENT_DIRECTORY = PROJECT_ROOT / "data" / "documents"

MAX_FILE_BYTES = int(
    load_config().get("limits", {}).get(
        "file_bytes", 10 * 1024 * 1024
    )
)

ALLOWED_SUFFIXES = {".txt", ".docx", ".pdf"}


class DocumentError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def require_collection_access(
    connection,
    user,
    collection_id,
    level,
):
    session = connection.execute(
        """
        SELECT 1 FROM sessions
        WHERE token_hash = ?
          AND user_id = ?
          AND expires_at > ?
        """,
        (
            user["token_hash"],
            user["id"],
            int(time.time()),
        ),
    ).fetchone()

    if session is None:
        raise DocumentError(401, "Сеанс недействителен.")

    if not can_access_collection(
        connection,
        user["id"],
        collection_id,
        level,
    ):
        raise DocumentError(403, "Коллекция недоступна.")
    

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def save_document(
    user: dict,
    collection_id: str,
    filename: str,
    stream: BinaryIO,
) -> dict:
    with database() as connection:
        require_collection_access(
            connection, user, collection_id, "write"
        )

    original_name = filename.replace("\\", "/").rsplit("/", 1)[-1]

    if (
        not original_name
        or len(original_name) > 255
        or any(ord(char) < 32 for char in original_name)
    ):
        raise DocumentError(400, "Недопустимое имя файла.")

    suffix = Path(original_name).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        raise DocumentError(
            415,
            "Допустимы расширения TXT, DOCX и PDF.",
        )

    DOCUMENT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
        mode=0o700,
    )

    document_id = str(uuid.uuid4())
    storage_name = document_id + suffix
    final_path = DOCUMENT_DIRECTORY / storage_name

    temporary_path = None
    final_created = False
    size = 0
    digest = hashlib.sha256()

    try:
        with tempfile.NamedTemporaryFile(
            dir=DOCUMENT_DIRECTORY,
            prefix=".upload-",
            suffix=".part",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)

            while True:
                block = stream.read(
                    min(65536, MAX_FILE_BYTES - size + 1)
                )

                if not block:
                    break

                size += len(block)

                if size > MAX_FILE_BYTES:
                    raise DocumentError(
                        413, "Превышен размер файла."
                    )

                destination.write(block)
                digest.update(block)

        if size == 0:
            raise DocumentError(400, "Пустой файл.")
        
        created_at = int(time.time())

        with database() as connection:
            connection.execute("BEGIN IMMEDIATE")

            require_collection_access(
                connection, user, collection_id, "write"
            )

            existing = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE collection_id = ? AND sha256 = ?
                ORDER BY created_at, id
                LIMIT 1
                """,
                (collection_id, digest.hexdigest()),
            ).fetchone()

            if existing is not None:
                stored_path = (
                    DOCUMENT_DIRECTORY / existing["storage_name"]
                )

                if (
                    stored_path.is_symlink()
                    or not stored_path.is_file()
                    or stored_path.stat().st_size != existing["size_bytes"]
                    or file_sha256(stored_path) != existing["sha256"]
                ):
                    raise DocumentError(
                        409,
                        "Сохранённый документ требует восстановления.",
                    )

                fields = (
                    "id",
                    "collection_id",
                    "original_name",
                    "extension",
                    "size_bytes",
                    "sha256",
                    "status",
                    "created_at",
                )

                result = {
                    key: existing[key]
                    for key in fields
                }
                result["already_exists"] = True

            else:
                os.replace(temporary_path, final_path)
                temporary_path = None
                final_created = True

                connection.execute(
                    """
                    INSERT INTO documents (
                        id, collection_id, created_by,
                        original_name, storage_name, extension,
                        size_bytes, sha256, status, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'uploaded', ?)
                    """,
                    (
                        document_id,
                        collection_id,
                        user["id"],
                        original_name,
                        storage_name,
                        suffix,
                        size,
                        digest.hexdigest(),
                        created_at,
                    ),
                )

                result = {
                    "id": document_id,
                    "collection_id": collection_id,
                    "original_name": original_name,
                    "extension": suffix,
                    "size_bytes": size,
                    "sha256": digest.hexdigest(),
                    "status": "uploaded",
                    "created_at": created_at,
                    "already_exists": False,
                }

    except BaseException:
        if final_created:
            try:
                with database() as connection:
                    registered = connection.execute(
                        "SELECT 1 FROM documents WHERE id = ?",
                        (document_id,),
                    ).fetchone()
            except Exception:
                registered = True

            if not registered:
                final_path.unlink(missing_ok=True)

        raise

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return result
