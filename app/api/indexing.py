import json

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.auth import current_user
from app.api.documents import document_http_error
from app.services.documents import DocumentError, require_collection_access
from app.services.indexing import index_document
from app.storage.database import database


router = APIRouter(
    prefix="/collections/{collection_id}/documents/{document_id}",
    tags=["indexing"],
)


@router.post("/index")
def build_index(collection_id: str, document_id: str, user: dict = Depends(current_user)):
    try:
        return index_document(user, collection_id, document_id)
    except DocumentError as error:
        raise document_http_error(error) from error


@router.get("/index")
def get_index(collection_id: str, document_id: str, user: dict = Depends(current_user)):
    try:
        with database() as connection:
            require_collection_access(connection, user, collection_id, "read")
            document = connection.execute(
                "SELECT status FROM documents WHERE id = ? AND collection_id = ?",
                (document_id, collection_id),
            ).fetchone()
            if document is None:
                raise HTTPException(404, "Документ не найден в коллекции.")
            row = connection.execute(
                """SELECT id, embedding_model, model_fingerprint, dimensions,
                          chunk_count, indexed_at
                   FROM document_indexes WHERE document_id = ?""", (document_id,),
            ).fetchone()
        return {"status": document["status"],
                "index": dict(row) if row and document["status"] == "ready" else None}
    except DocumentError as error:
        raise document_http_error(error) from error


@router.get("/chunks")
def get_chunks(
    collection_id: str, document_id: str,
    limit: int = Query(default=5, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(current_user),
):
    try:
        with database() as connection:
            require_collection_access(connection, user, collection_id, "read")
            document = connection.execute(
                "SELECT status FROM documents WHERE id = ? AND collection_id = ?",
                (document_id, collection_id),
            ).fetchone()
            if document is None:
                raise HTTPException(404, "Документ не найден в коллекции.")
            if document["status"] != "ready":
                raise HTTPException(409, "Индекс документа ещё не готов.")
            rows = connection.execute(
                """SELECT c.id, c.ordinal, c.text, c.source_json
                   FROM document_chunks c
                   JOIN document_indexes i ON i.id = c.index_id
                   WHERE i.document_id = ? ORDER BY c.ordinal LIMIT ? OFFSET ?""",
                (document_id, limit, offset),
            ).fetchall()
        return [{"id": row["id"], "ordinal": row["ordinal"], "text": row["text"],
                 "source": json.loads(row["source_json"])} for row in rows]
    except DocumentError as error:
        raise document_http_error(error) from error
