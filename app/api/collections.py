from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import current_user
from app.core.access import can_access_collection
from app.storage.database import database


router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("/{collection_id}")
def get_collection(
    collection_id: str,
    user: dict = Depends(current_user),
):
    with database() as connection:
        allowed = can_access_collection(
            connection,
            user_id=user["id"],
            collection_id=collection_id,
            required_level="read",
        )

        if not allowed:
            raise HTTPException(
                status_code=403,
                detail="Коллекция недоступна.",
            )

        collection = connection.execute(
            """
            SELECT id, name, kind
            FROM collections
            WHERE id = ?
            """,
            (collection_id,),
        ).fetchone()

    return dict(collection)