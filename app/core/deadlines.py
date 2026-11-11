from dataclasses import dataclass
from contextlib import contextmanager
import sqlite3
import time

from app.services.documents import DocumentError


@dataclass(frozen=True)
class Deadline:
    expires_at: float

    @classmethod
    def after(cls, seconds: float, parent=None):
        end = time.monotonic() + seconds
        return cls(min(end, parent.expires_at) if parent is not None else end)

    def remaining(self) -> float:
        left = self.expires_at - time.monotonic()
        if left <= 0:
            raise DocumentError(504, "Истёк общий срок выполнения операции.")
        return left

    def check(self) -> None:
        self.remaining()


@contextmanager
def sqlite_budget(connection, deadline):
    left = deadline.remaining()
    connection.execute(f"PRAGMA busy_timeout = {max(1, int(min(left, 5) * 1000))}")
    connection.set_progress_handler(lambda: int(time.monotonic() >= deadline.expires_at), 1000)
    try:
        yield
        deadline.check()
    except sqlite3.OperationalError:
        deadline.check()
        raise
    finally:
        connection.set_progress_handler(None, 0)
