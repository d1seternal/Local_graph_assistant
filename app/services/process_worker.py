import atexit
import multiprocessing
from multiprocessing.connection import wait
from threading import Lock

from app.core.config import load_config
from app.services.documents import DocumentError


def _serve(connection, handler):
    try:
        connection.send({"ready": True})
        while True:
            args = connection.recv()
            try:
                result = handler(*args)
                reply = {"result": result}
            except DocumentError as error:
                reply = {"error": str(error), "status": error.status_code}
            except Exception as error:
                reply = {"error": str(error), "type": type(error).__name__}
            connection.send(reply)
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


class ProcessWorker:
    def __init__(self, handler):
        self.handler = handler
        self._process = None
        self._connection = None
        self._guard = Lock()
        self._poisoned = False
        self.last_dispatched = False
        self.last_stopped_pid = None
        atexit.register(self.close)

    def state(self):
        alive = self._process is not None and self._process.is_alive()
        return {"alive": alive, "pid": self._process.pid if alive else None,
                "busy": self._guard.locked(), "cleanup_failed": self._poisoned,
                "last_stopped_pid": self.last_stopped_pid}

    def _receive(self, deadline):
        ready = wait([self._connection, self._process.sentinel], timeout=deadline.remaining())
        deadline.check()
        if self._connection not in ready:
            if self._process.sentinel in ready:
                raise DocumentError(503, "Дочерний исполнитель аварийно завершился.")
            raise DocumentError(504, "Истёк срок ожидания дочернего исполнителя.")
        reply = self._connection.recv()
        deadline.check()
        return reply

    def _start(self, deadline):
        if self._poisoned:
            raise DocumentError(503, "Предыдущий исполнитель ещё не завершён; требуется проверка процесса.")
        if self._process is not None and self._process.is_alive():
            return
        self.close()
        deadline.check()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_serve, args=(child, self.handler), daemon=True)
        self._connection, self._process = parent, process
        try:
            process.start()
        except BaseException:
            parent.close()
            self._connection = self._process = None
            raise
        finally:
            child.close()
        self._receive(deadline)

    def call(self, args, deadline):
        if not self._guard.acquire(blocking=False):
            raise DocumentError(503, "Исполнитель занят.")
        self.last_dispatched = False
        try:
            try:
                self._start(deadline)
                deadline.check()
                self.last_dispatched = True
                self._connection.send(args)
                reply = self._receive(deadline)
            except (EOFError, BrokenPipeError, OSError):
                self.close()
                raise DocumentError(503, "Потеряно соединение с дочерним исполнителем.") from None
            except BaseException:
                self.close()
                raise
            if "status" in reply:
                raise DocumentError(reply["status"], reply["error"])
            if "error" in reply:
                raise RuntimeError(f'{reply.get("type", "WorkerError")}: {reply["error"]}')
            return reply["result"]
        finally:
            self._guard.release()

    def close(self):
        process = self._process
        if process is not None and process.pid is not None:
            grace = load_config().get("execution", {}).get("process_stop_grace_seconds", 0.1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=grace)
            if process.is_alive():
                process.kill()
                process.join(timeout=grace)
            if process.is_alive():
                self._poisoned = True
                return 
            process.join(timeout=0)
            self.last_stopped_pid = process.pid
            process.close()
        if self._connection is not None:
            self._connection.close()
        self._process = self._connection = None
        self._poisoned = False
