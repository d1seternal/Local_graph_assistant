from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
from threading import Lock
import time
from typing import Callable
import uuid

import numpy as np

from app.core.config import PROJECT_ROOT, load_config
from app.services.documents import (
    DOCUMENT_DIRECTORY, DocumentError, file_sha256, require_collection_access,
)
from app.services.parsing import ParsingError, parse_document
from app.storage.database import database


CHUNK_TOKENS = 256
OVERLAP_TOKENS = 32
MAX_CHUNKS = 5000
INDEX_LOCK = Lock()


@lru_cache(maxsize=1)
def _get_embedder():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    path = PROJECT_ROOT / load_config()["embeddings"]["model_path"]
    assets = sorted(p for p in path.rglob("*") if p.is_file())
    if not assets or not any(p.suffix in {".safetensors", ".bin"} for p in assets):
        raise DocumentError(503, "Локальные веса E5 не найдены.")

    fingerprint = hashlib.sha256()
    for asset in assets:
        fingerprint.update(asset.relative_to(path).as_posix().encode())
        fingerprint.update(b"\0")
        fingerprint.update(file_sha256(asset).encode())

    model = SentenceTransformer(
        str(path), device="cpu", local_files_only=True, trust_remote_code=False,
    )
    if model.get_embedding_dimension() != 384:
        raise DocumentError(503, "Ожидается multilingual-e5-small с размерностью 384.")
    if not model.tokenizer.is_fast:
        raise DocumentError(503, "Для точных границ нужен fast-токенизатор E5.")
    return model, fingerprint.hexdigest()


def _split_units(units, tokenizer):
    chunks = []
    for unit in units:
        text = unit["text"]
        offsets = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True,
            truncation=False,
        )["offset_mapping"]
        for start in range(0, len(offsets), CHUNK_TOKENS - OVERLAP_TOKENS):
            end = min(start + CHUNK_TOKENS, len(offsets))
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            fragment = text[char_start:char_end]
            source = dict(unit["source"])
            if source["type"] == "txt":
                source["line_start"] = (
                    unit["source"]["line_start"] + text[:char_start].count("\n")
                )
                source["line_end"] = (
                    unit["source"]["line_start"] + text[:max(char_end - 1, 0)].count("\n")
                )
            source.update(char_start=char_start, char_end=char_end)
            if fragment.strip():
                if len(tokenizer.encode("passage: " + fragment)) > 512:
                    raise DocumentError(400, "Фрагмент превышает вход E5.")
                chunks.append({"text": fragment, "source": source})
            if len(chunks) > MAX_CHUNKS:
                raise DocumentError(400, "Получено более 5000 фрагментов.")
            if end == len(offsets):
                break
    if not chunks:
        raise DocumentError(400, "Не удалось получить текстовые фрагменты.")
    return chunks

def _begin_indexing(user: dict, collection_id: str, document_id: str) -> dict:
    with database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        require_collection_access(connection, user, collection_id, "write")
        row = connection.execute(
            "SELECT * FROM documents WHERE id = ? AND collection_id = ?",
            (document_id, collection_id),
        ).fetchone()
        if row is None:
            raise DocumentError(404, "Документ не найден в коллекции.")
        document = dict(row)
        connection.execute(
            "UPDATE documents SET status = 'processing' WHERE id = ?",
            (document_id,),
        )
    return document

def _validate_stored_document(document: dict) -> Path:
    storage_name = document["storage_name"]
    if Path(storage_name).name != storage_name:
        raise DocumentError(409, "Некорректный внутренний путь документа.")
    path = DOCUMENT_DIRECTORY / storage_name
    if path.is_symlink() or not path.is_file():
        raise DocumentError(409, "Сохранённый файл недоступен.")
    maximum = int(load_config().get("limits", {}).get("file_bytes", 10 * 1024 * 1024))
    actual_size = path.stat().st_size
    if actual_size != document["size_bytes"]:
        raise DocumentError(409, "Размер файла отличается от сохранённого.")
    if actual_size > maximum:
        raise DocumentError(409, "Файл превышает текущий предел размера.")
    if file_sha256(path) != document["sha256"]:
        raise DocumentError(409, "Контрольная сумма документа не совпадает.")
    return path


def _build_embedding_rows(chunks, model, index_id: str,
                          check_deadline: Callable[[], None]) -> list[tuple]:
    batch_size = int(load_config()["embeddings"].get("batch_size", 8))
    if not 1 <= batch_size <= 32:
        raise DocumentError(503, "Размер пакета должен быть от 1 до 32.")
    rows = []
    for start in range(0, len(chunks), batch_size):
        check_deadline()
        batch = chunks[start:start + batch_size]
        vectors = np.asarray(model.encode(
            ["passage: " + chunk["text"] for chunk in batch],
            batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False, prompt="",
        ), dtype="<f4")
        if vectors.shape != (len(batch), 384) or not np.isfinite(vectors).all():
            raise DocumentError(500, "Получены некорректные векторы.")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
            raise DocumentError(500, "Векторы не нормализованы.")
        for number, (chunk, vector) in enumerate(zip(batch, vectors), start=start):
            rows.append((
                str(uuid.uuid4()), index_id, number, chunk["text"],
                json.dumps(chunk["source"], ensure_ascii=False), vector.tobytes(),
            ))
    check_deadline()
    return rows


def _index_settings() -> dict:
    return {
        "chunk_tokens": CHUNK_TOKENS, "overlap_tokens": OVERLAP_TOKENS,
        "normalization": "l2", "vector_dtype": "float32-le", "parser_version": 1,
        "versions": {name: version(name) for name in [
            "sentence-transformers", "torch", "transformers", "tokenizers",
            "numpy", "pypdf", "python-docx",
        ]},
    }


def _publish_index(user: dict, collection_id: str, document: dict,
                   index_id: str, fingerprint: str, rows: list[tuple],
                   settings: dict) -> None:
    with database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        require_collection_access(connection, user, collection_id, "write")
        connection.execute(
            "DELETE FROM document_indexes WHERE document_id = ?", (document["id"],),
        )
        connection.execute(
            """INSERT INTO document_indexes
               (id, document_id, document_sha256, embedding_model,
                model_fingerprint, dimensions, chunk_count, settings_json, indexed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (index_id, document["id"], document["sha256"],
             "intfloat/multilingual-e5-small", fingerprint, 384, len(rows),
             json.dumps(settings), int(time.time())),
        )
        connection.executemany(
            """INSERT INTO document_chunks
               (id, index_id, ordinal, text, source_json, embedding)
               VALUES (?, ?, ?, ?, ?, ?)""", rows,
        )
        connection.execute(
            "UPDATE documents SET status = 'ready' WHERE id = ?", (document["id"],),
        )


def _mark_indexing_failed(document_id: str) -> None:
    try:
        with database() as connection:
            connection.execute(
                "UPDATE documents SET status = 'failed' WHERE id = ? AND status = 'processing'",
                (document_id,),
            )
    except Exception:
        pass 

def index_document(user: dict, collection_id: str, document_id: str) -> dict:
    if not INDEX_LOCK.acquire(blocking=False):
        raise DocumentError(503, "Индексация уже выполняется. Повторите позже.")
    try:
        started = time.monotonic()
        deadline = started + 180
        processing = False

        def check_deadline():
            if time.monotonic() >= deadline:
                raise DocumentError(504, "Истекло время индексации.")

        try:
            document = _begin_indexing(user, collection_id, document_id)
            processing = True
            path = _validate_stored_document(document)
            check_deadline()

            try:
                units = parse_document(path, document["extension"], timeout=30)
            except ParsingError as error:
                raise DocumentError(400, str(error)) from error
            except TimeoutError as error:
                raise DocumentError(504, str(error)) from error
            check_deadline()

            model, fingerprint = _get_embedder()
            check_deadline()
            chunks = _split_units(units, model.tokenizer)
            check_deadline()

            index_id = str(uuid.uuid4())
            rows = _build_embedding_rows(chunks, model, index_id, check_deadline)
            settings = _index_settings()
            check_deadline()
            _publish_index(user, collection_id, document, index_id, fingerprint, rows, settings)
        except Exception:
            if processing:
                _mark_indexing_failed(document_id)
            raise

        return {
            "document_id": document_id, "index_id": index_id, "status": "ready",
            "chunks": len(chunks), "dimensions": 384,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    finally:
        INDEX_LOCK.release()