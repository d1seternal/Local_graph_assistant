from fastapi import FastAPI

from app.api.auth import router as auth_router
from app.api.collections import router as collections_router
from app.api.health import router as health_router
from app.api.documents import router as documents_router
from app.api.indexing import router as indexing_router
from app.api.search import router as search_router
from app.api.answers import router as answers_router

app = FastAPI(
    title="Локальный ассистент по документам",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(collections_router)
app.include_router(documents_router)
app.include_router(indexing_router)
app.include_router(search_router)
app.include_router(answers_router)

