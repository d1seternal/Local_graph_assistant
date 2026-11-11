import time

from app.services import indexing, search
from app.services.documents import DocumentError, require_collection_access
from app.services.generation import generate_answer
from app.storage.database import database
from app.core.deadlines import Deadline, sqlite_budget
from app.core.config import load_config



def _check_context(user: dict, collection_id: str, matches: list[dict], deadline) -> None:
    with database() as connection, sqlite_budget(connection, deadline):
        connection.execute("BEGIN")
        require_collection_access(connection, user, collection_id, "read")
        indexes = {(match["document_id"], match["index_id"]) for match in matches}
        for document_id, index_id in indexes:
            row = connection.execute(
                """SELECT 1 FROM documents AS d
                   JOIN document_indexes AS i ON i.document_id = d.id
                   WHERE d.id = ? AND d.collection_id = ?
                     AND d.status = 'ready' AND i.id = ?""",
                (document_id, collection_id, index_id),
            ).fetchone()
            if row is None:
                raise DocumentError(409, "Документы контекста изменились. Повторите вопрос.")
            
def answer_collection(user: dict, collection_id: str, question: str, top_k: int = 3) -> dict:
    started = time.monotonic()
    deadline = Deadline.after(load_config()["limits"]["request_deadline_seconds"])
    if not indexing.INDEX_LOCK.acquire(blocking=False):
        raise DocumentError(503, "Исполнитель занят.")
    try:
        retrieval = search.search_collection(user, collection_id, question, top_k,
                                             deadline=deadline, _lock_held=True)
        deadline.check()
        matches = retrieval["matches"]
        if not matches:
            return {"collection_id": collection_id, "question": retrieval["query"],
                    "answer": "В коллекции нет проиндексированных фрагментов для ответа.",
                    "sources": [], "model": None, "finish_reason": "no_context",
                    "answer_truncated": False, "searched_chunks": retrieval["searched_chunks"],
                    "duration_seconds": round(time.monotonic() - started, 3)}
        _check_context(user, collection_id, matches, deadline)
        generated = generate_answer(retrieval["query"], matches, deadline=deadline)
        deadline.check()
        _check_context(user, collection_id, matches, deadline)
        deadline.check()
        return {"collection_id": collection_id, "question": retrieval["query"],
                **generated, "searched_chunks": retrieval["searched_chunks"],
                "duration_seconds": round(time.monotonic() - started, 3)}
    finally:
        indexing.INDEX_LOCK.release()