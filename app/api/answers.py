from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.api.auth import current_user
from app.api.documents import document_http_error
from app.services.answers import answer_collection
from app.services.documents import DocumentError


router = APIRouter(prefix="/collections/{collection_id}", tags=["answers"])


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10, strict=True)


@router.post("/ask")
def ask(collection_id: str, body: QuestionRequest, user: dict = Depends(current_user)):
    try:
        return answer_collection(user, collection_id, body.question, body.top_k)
    except DocumentError as error:
        raise document_http_error(error) from error
