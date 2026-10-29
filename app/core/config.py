from functools import lru_cache
from pathlib import Path
import tomllib

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

    return config
