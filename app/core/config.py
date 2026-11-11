from functools import lru_cache
import math
from pathlib import Path
import tomllib
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "app.toml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with CONFIG_PATH.open("rb") as file:
        config = tomllib.load(file)

    generation = config["generation"]
    indexing = config["indexing"]

    if not 0 < generation["num_predict"] < generation["num_ctx"]:
        raise ValueError("0 < num_predict < num_ctx")

    if not 0 <= indexing["overlap_tokens"] < indexing["chunk_tokens"]:
        raise ValueError("0 <= overlap_tokens < chunk_tokens")

    if config["embeddings"]["batch_size"] < 1:
        raise ValueError("batch_size должен быть положительным")

    if config["embeddings"]["batch_size"] > 32:
        raise ValueError("batch_size не должен превышать 32")
    for section, key in [
        ("limits", "request_deadline_seconds"),
        ("indexing", "timeout_seconds"),
        ("indexing", "parse_timeout_seconds"),
        ("generation", "socket_timeout_seconds"),
        ("execution", "process_stop_grace_seconds"),
        ("execution", "cleanup_timeout_seconds"),
    ]:
        value = config[section][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{section}.{key} должен быть конечным положительным числом")
    if generation["stream"] is not False:
        raise ValueError("Текущий адаптер поддерживает только generation.stream=false")
    url = urlsplit(generation["url"])
    if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.path != "/api/chat" or url.username or url.password or url.query or url.fragment):
        raise ValueError("generation.url должен указывать на локальный http://.../api/chat")

    return config
