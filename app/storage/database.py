from contextlib import contextmanager
from pathlib import Path
import sqlite3

from app.core.config import PROJECT_ROOT


DATABASE_PATH = PROJECT_ROOT / "data" / "app.sqlite"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@contextmanager
def database(path=DATABASE_PATH):
    connection = sqlite3.connect(path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    schema = SCHEMA_PATH.read_text(encoding="utf-8")

    with database() as connection:
        connection.executescript(schema)
