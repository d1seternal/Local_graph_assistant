from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.core.auth import (
    login as create_session,
    logout as delete_session,
    resolve_session,
)


router = APIRouter(prefix="/auth", tags=["auth"])
bearer = HTTPBearer(auto_error=False)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=30)
    password: SecretStr = Field(min_length=8, max_length=32)


def unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Требуется действующий сеанс.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict:
    if credentials is None:
        raise unauthorized()

    token = credentials.credentials

    if len(token) > 128:
        raise unauthorized()

    user = resolve_session(token)

    if user is None:
        raise unauthorized()

    return user


@router.post("/login")
def login(payload: LoginRequest):
    session = create_session(
        payload.username,
        payload.password.get_secret_value(),
    )

    if session is None:
        raise HTTPException(
            status_code=401,
            detail="Неверные учётные данные.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return session


@router.get("/me")
def me(user: dict = Depends(current_user)):
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
    }


@router.post("/logout")
def logout(user: dict = Depends(current_user)):
    delete_session(user["token_hash"])
    return {"status": "logged_out"}