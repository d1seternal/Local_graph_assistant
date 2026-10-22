from fastapi import FastAPI

from app.api.auth import router as auth_router
from app.api.collections import router as collections_router
from app.api.health import router as health_router
from app.api.documents import router as documents_router


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
