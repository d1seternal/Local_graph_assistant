import json
import time

import numpy as np

from app.services import indexing
from app.services.documents import DocumentError, require_collection_access
from app.storage.database import database


MODEL_NAME = "intfloat/multilingual-e5-small"
DIMENSIONS = 384
MAX_QUERY_CHARACTERS = 2000
MAX_QUERY_TOKENS = 512
MAX_SEARCH_CHUNKS = 10_000
SCAN_BATCH_SIZE = 256
SEARCH_SECONDS = 60


def _validate_query(query: str, top_k: int) -> str:
    if not isinstance(query, str):
        raise DocumentError(400, "Вопрос должен быть строкой.")
    if len(query) > MAX_QUERY_CHARACTERS:
        raise DocumentError(400, "Вопрос превышает 2000 символов.")
    query = query.strip()
    if not query:
        raise DocumentError(400, "Вопрос не должен быть пустым.")
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise DocumentError(400, "top_k должен быть целым числом от 1 до 10.")
    return query


def _encode_query(model, query: str) -> np.ndarray:
    text = "query: " + query
    if len(model.tokenizer.encode(text)) > MAX_QUERY_TOKENS:
        raise DocumentError(400, "Вопрос вместе с префиксом превышает 512 токенов E5.")
    vectors = np.asarray(model.encode(
        [text], batch_size=1, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False, prompt="",
    ), dtype="<f4")
    if vectors.shape != (1, DIMENSIONS) or not np.isfinite(vectors).all():
        raise DocumentError(500, "Получен некорректный вектор вопроса.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
        raise DocumentError(500, "Вектор вопроса не нормализован.")
    return vectors[0]


def _check_indexes(connection, collection_id: str, fingerprint: str) -> None:
    indexes = connection.execute(
        """SELECT d.sha256, i.id, i.document_sha256, i.embedding_model,
                  i.model_fingerprint, i.dimensions, i.settings_json
           FROM documents AS d
           LEFT JOIN document_indexes AS i ON i.document_id = d.id
           WHERE d.collection_id = ? AND d.status = 'ready'""",
        (collection_id,),
    ).fetchall()
    for index in indexes:
        try:
            settings = json.loads(index["settings_json"])
        except (TypeError, ValueError):
            raise DocumentError(409, "Метаданные индекса повреждены; нужна переиндексация.") from None
        if (
            index["id"] is None
            or index["document_sha256"] != index["sha256"]
            or index["embedding_model"] != MODEL_NAME
            or index["model_fingerprint"] != fingerprint
            or index["dimensions"] != DIMENSIONS
            or not isinstance(settings, dict)
            or settings.get("normalization") != "l2"
            or settings.get("vector_dtype") != "float32-le"
        ):
            raise DocumentError(409, "Индексы коллекции несовместимы с текущей E5; нужна переиндексация.")

def _rank_chunks(connection, collection_id, query_vector, top_k, check_deadline):
    count = connection.execute(
        """SELECT count(*) FROM document_chunks AS c
           JOIN document_indexes AS i ON i.id = c.index_id
           JOIN documents AS d ON d.id = i.document_id
           WHERE d.collection_id = ? AND d.status = 'ready'""",
        (collection_id,),
    ).fetchone()[0]
    if count > MAX_SEARCH_CHUNKS:
        raise DocumentError(413, "Поиск ограничен 10000 фрагментов в выбранной коллекции.")

    cursor = connection.execute(
        """SELECT c.id AS chunk_id, c.ordinal, c.text, c.source_json, c.embedding,
                  i.id AS index_id, d.id AS document_id, d.original_name
           FROM document_chunks AS c
           JOIN document_indexes AS i ON i.id = c.index_id
           JOIN documents AS d ON d.id = i.document_id
           WHERE d.collection_id = ? AND d.status = 'ready'
           ORDER BY d.id, c.ordinal, c.id""",
        (collection_id,),
    )
    best = []
    scanned = 0
    while True:
        check_deadline()
        batch = cursor.fetchmany(SCAN_BATCH_SIZE)
        if not batch:
            break
        vectors = []
        for row in batch:
            blob = row["embedding"]
            if not isinstance(blob, bytes) or len(blob) != DIMENSIONS * 4:
                raise DocumentError(409, "Повреждён сохранённый вектор; нужна переиндексация.")
            vectors.append(np.frombuffer(blob, dtype="<f4"))
        matrix = np.stack(vectors)
        if not np.isfinite(matrix).all() or not np.allclose(
            np.linalg.norm(matrix, axis=1), 1, atol=1e-3,
        ):
            raise DocumentError(409, "Некорректные сохранённые векторы; нужна переиндексация.")
        scores = np.clip(matrix @ query_vector, -1.0, 1.0)
        best.extend((float(score), row) for score, row in zip(scores, batch))
        best.sort(key=lambda item: (
            -item[0], item[1]["document_id"], item[1]["ordinal"], item[1]["chunk_id"],
        ))
        del best[top_k:]
        scanned += len(batch)

    matches = []
    for score, row in best:
        try:
            source = json.loads(row["source_json"])
            if not isinstance(source, dict):
                raise ValueError("source must be a dictionary")
        except (TypeError, ValueError):
            raise DocumentError(409, "Повреждены сведения об источнике фрагмента.") from None
        matches.append({
            "document_id": row["document_id"], "document_name": row["original_name"],
            "index_id": row["index_id"], "chunk_id": row["chunk_id"],
            "ordinal": row["ordinal"], "text": row["text"], "source": source,
            "score": round(score, 6),
        })
    return matches, scanned



def search_collection(user: dict, collection_id: str, query: str, top_k: int = 5) -> dict:
    query = _validate_query(query, top_k)
    with database() as connection:
        require_collection_access(connection, user, collection_id, "read")

    if not indexing.INDEX_LOCK.acquire(blocking=False):
        raise DocumentError(503, "Модель занята индексацией или поиском. Повторите позже.")
    try:
        started = time.monotonic()

        def check_deadline():
            if time.monotonic() - started >= SEARCH_SECONDS:
                raise DocumentError(504, "Истекло время поиска.")

        model, fingerprint = indexing._get_embedder()
        check_deadline()
        vector = _encode_query(model, query)
        check_deadline()
        with database() as connection:
            connection.execute("BEGIN")
            require_collection_access(connection, user, collection_id, "read")
            _check_indexes(connection, collection_id, fingerprint)
            matches, scanned = _rank_chunks(
                connection, collection_id, vector, top_k, check_deadline,
            )
        check_deadline()
        with database() as connection:
            require_collection_access(connection, user, collection_id, "read")
        return {
            "collection_id": collection_id, "query": query, "top_k": top_k,
            "searched_chunks": scanned, "matches": matches,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    finally:
        indexing.INDEX_LOCK.release()
