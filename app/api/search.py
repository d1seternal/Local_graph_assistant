"""HTTP-интерфейс поиска. Пользователь определяется только по сеансу."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.api.auth import current_user
from app.api.documents import document_http_error
from app.services.documents import DocumentError
from app.services.search import search_collection


router = APIRouter(prefix="/collections/{collection_id}", tags=["search"])


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10, strict=True)


@router.post("/search")
def search(collection_id: str, body: SearchRequest, user: dict = Depends(current_user)):
    try:
        return search_collection(user, collection_id, body.query, body.top_k)
    except DocumentError as error:
        raise document_http_error(error) from error
