from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    UploadFile,
    Response
)

from app.api.auth import current_user
from app.services.documents import (
    DocumentError,
    require_collection_access,
    save_document,
)
from app.storage.database import database


router = APIRouter(
    prefix="/collections/{collection_id}/documents",
    tags=["documents"],
)


def document_http_error(error: DocumentError) -> HTTPException:
    headers = (
        {"WWW-Authenticate": "Bearer"}
        if error.status_code == 401
        else None
    )

    return HTTPException(
        status_code=error.status_code,
        detail=str(error),
        headers=headers,
    )


@router.post("", status_code=201)
def upload_document(
    collection_id: str,
    file: UploadFile,
    response: Response,
    user: dict = Depends(current_user),
):
    try:
        result = save_document(
            user=user,
            collection_id=collection_id,
            filename=file.filename or "",
            stream=file.file,
        )

        response.status_code = (
            200 if result["already_exists"] else 201
        )

        return result

    except DocumentError as error:
        raise document_http_error(error) from error

    finally:
        file.file.close()


@router.get("")
def list_documents(
    collection_id: str,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(current_user),
):
    try:
        with database() as connection:
            require_collection_access(
                connection, user, collection_id, "read"
            )

            rows = connection.execute(
                """
                SELECT
                    id, original_name, extension,
                    size_bytes, status, created_at
                FROM documents
                WHERE collection_id = ?
                ORDER BY created_at DESC, id
                LIMIT ? OFFSET ?
                """,
                (collection_id, limit, offset),
            ).fetchall()

        return [dict(row) for row in rows]

    except DocumentError as error:
        raise document_http_error(error) from error
    