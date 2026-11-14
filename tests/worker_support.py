from pathlib import Path
import re
import time
from unittest.mock import patch

import numpy as np


class FakeTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}

    def encode(self, text):
        return list(range(len(text.split()) + 2))


class SlowModel:
    tokenizer = FakeTokenizer()

    def __init__(self, marker_dir):
        self.marker_dir = marker_dir

    def encode(self, texts, **kwargs):
        if any(re.search(r"\bSLOW\b", text) for text in texts):
            Path(self.marker_dir, "started").write_text("started")
            time.sleep(0.6)
            Path(self.marker_dir, "completed").write_text("completed")
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        vectors[:, 0] = 1
        return vectors


def fake_embedding_job(operation, payload, expires_at, *, marker_dir):
    from app.services import embeddings
    with patch.object(embeddings, "_get_embedder", return_value=(SlowModel(marker_dir), "test-fingerprint")):
        return embeddings._embedding_job(operation, payload, expires_at)


def hanging_job(marker_dir, expires_at):
    Path(marker_dir, "started").write_text("started")
    time.sleep(0.6)
    Path(marker_dir, "completed").write_text("completed")
    return "done"


def hanging_parser(path, extension, limits, expires_at):
    return hanging_job(str(Path(path).parent.parent), expires_at)
