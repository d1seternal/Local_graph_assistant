from functools import lru_cache
import hashlib
import os

from app.core.config import PROJECT_ROOT, load_config
from app.core.deadlines import Deadline
from app.services.documents import DocumentError, file_sha256
from app.services.process_worker import ProcessWorker


@lru_cache(maxsize=1)
def _get_embedder():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = load_config()["embeddings"]
    import torch
    from sentence_transformers import SentenceTransformer


    torch.set_num_threads(config["cpu_threads"])
    path = PROJECT_ROOT / config["model_path"]
    assets = sorted(p for p in path.rglob("*") if p.is_file())
    if not assets or not any(p.suffix in {".safetensors", ".bin"} for p in assets):
        raise DocumentError(503, "Локальные веса E5 не найдены.")

    fingerprint = hashlib.sha256()
    for asset in assets:
        fingerprint.update(asset.relative_to(path).as_posix().encode())
        fingerprint.update(b"\0")
        fingerprint.update(file_sha256(asset).encode())


    model = SentenceTransformer(
        str(path),
        device=config["device"],
        local_files_only=True,
        trust_remote_code=False,
    )

    if model.get_embedding_dimension() != 384:
        raise DocumentError(503, "Ожидается multilingual-e5-small с размерностью 384.")
    if not model.tokenizer.is_fast:
        raise DocumentError(503, "Для точных границ нужен fast-токенизатор E5.")
    return model, fingerprint.hexdigest()


def _embedding_job(operation, payload, expires_at):
    deadline = Deadline(expires_at)
    deadline.check()
    model, fingerprint = _get_embedder()
    deadline.check()
    if operation == "index":
        from app.services.indexing import _split_units, _build_embedding_rows
        chunks = _split_units(payload["units"], model.tokenizer)
        rows = _build_embedding_rows(chunks, model, payload["index_id"], deadline.check)
        result = {"rows": rows, "fingerprint": fingerprint}
    elif operation == "query":
        from app.services.search import _encode_query
        result = {"vector": _encode_query(model, payload["query"]), "fingerprint": fingerprint}
    else:
        raise ValueError("Неизвестная операция E5")
    deadline.check()
    return result


EMBEDDINGS = ProcessWorker(_embedding_job)
