from functools import partial
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event, Lock, Thread
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.auth import current_user
from app.core.config import PROJECT_ROOT, load_config
from app.core.deadlines import Deadline
from app.main import app
from app.services import answers, generation, indexing, parsing, search
from app.services.documents import DocumentError
from app.services.process_worker import ProcessWorker
from app.storage.database import database, SCHEMA_PATH
from tests.worker_support import fake_embedding_job, hanging_parser


class OllamaStub(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.last_payload = payload
        self.server.last_path = self.path
        if "MISSING_MODEL" in json.dumps(payload):
            self.send_error(404)
            return
        if "SLOW_HTTP" in json.dumps(payload):
            self.server.slow_started.set()
            time.sleep(0.6)
            self.server.slow_finished.set()
        body = json.dumps({"done": True, "done_reason": "stop",
                           "message": {"content": "30 дней [1]."},
                           "prompt_eval_count": 40, "eval_count": 7}).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


class DeadlineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.db_path = self.root / "test.sqlite"
        self.files = self.root / "documents"
        self.files.mkdir()
        self.path = self.files / "source.txt"
        self.path.write_text("Резервные копии хранятся 30 дней.", encoding="utf-8")
        self.user = {"id": "alice", "token_hash": "synthetic-test-session"}
        with self.open_db() as connection:
            connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            connection.execute("INSERT INTO users VALUES ('alice','alice','test-only','user',1)")
            connection.execute("INSERT INTO collections VALUES ('personal','Личная','personal','alice')")
            connection.execute("INSERT INTO sessions VALUES (?,?,?,?)",
                               (self.user["token_hash"], "alice", 0, int(time.time()) + 3600))
            connection.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?)",
                               ("doc", "personal", "alice", "demo.txt", "source.txt", ".txt",
                                self.path.stat().st_size, hashlib.sha256(self.path.read_bytes()).hexdigest(),
                                "uploaded", 0))
        self.worker = ProcessWorker(partial(fake_embedding_job, marker_dir=str(self.root)))
        self.addCleanup(self.worker.close)
        for item in [
            patch.object(indexing, "database", self.open_db),
            patch.object(search, "database", self.open_db),
            patch.object(answers, "database", self.open_db),
            patch.object(indexing, "DOCUMENT_DIRECTORY", self.files),
            patch.object(indexing, "INDEX_LOCK", Lock()),
            patch.object(indexing, "EMBEDDINGS", self.worker),
            patch.object(search, "EMBEDDINGS", self.worker),
            patch.object(indexing, "version", return_value="test-version"),
            patch.object(indexing, "parse_document", side_effect=lambda p, e, timeout: parsing._extract(p, e, {})),
            patch.object(generation, "_OLLAMA_UNCERTAIN", False),
        ]:
            item.start()
            self.addCleanup(item.stop)
        app.dependency_overrides[current_user] = lambda: self.user
        self.addCleanup(app.dependency_overrides.clear)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
    
        self.worker.call(("query", {"query": "warmup"}, time.monotonic() + 10), Deadline.after(10))

    def open_db(self):
        return database(self.db_path)

    def make_ready(self):
        return indexing.index_document(self.user, "personal", "doc")

    def start_http(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), OllamaStub)
        server.slow_started, server.slow_finished = Event(), Event()
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        thread.start()
        def stop():
            server.shutdown()
            server.server_close()
            thread.join()
        self.addCleanup(stop)
        worker = ProcessWorker(generation._http_job)
        self.addCleanup(worker.close)
        for item in [patch.object(generation, "HTTP_WORKER", worker),
                     patch.object(generation, "OLLAMA_URL", f"http://127.0.0.1:{server.server_port}/api/chat")]:
            item.start()
            self.addCleanup(item.stop)
        generation._post_ollama({"model": generation.MODEL, "messages": []}, deadline=Deadline.after(10))
        return server, worker

    def test_slow_index_encode_is_killed_and_resource_reusable(self):
        old = self.make_ready()
        self.path.write_text("SLOW резервные копии", encoding="utf-8")
        with self.open_db() as connection:
            connection.execute("UPDATE documents SET size_bytes=?, sha256=? WHERE id='doc'",
                               (self.path.stat().st_size, hashlib.sha256(self.path.read_bytes()).hexdigest()))
        pid = self.worker.state()["pid"]
        started = time.monotonic()
        with patch.dict(indexing.indexing_config, {"timeout_seconds": 0.08}):
            response = self.client.post("/collections/personal/documents/doc/index")
        elapsed = time.monotonic() - started
        self.assertEqual(response.status_code, 504)
        self.assertLess(elapsed, 0.4) 
        self.assertTrue((self.root / "started").exists())
        self.assertFalse((self.root / "completed").exists())
        self.assertFalse(self.worker.state()["alive"])
        self.assertEqual(self.worker.last_stopped_pid, pid)
        self.assertFalse(indexing.INDEX_LOCK.locked())
        with self.open_db() as connection:
            self.assertEqual(connection.execute("SELECT status FROM documents").fetchone()[0], "failed")
            self.assertEqual(connection.execute("SELECT id FROM document_indexes").fetchone()[0], old["index_id"])

        result = self.worker.call(("query", {"query": "next"}, time.monotonic() + 10), Deadline.after(10))
        self.assertEqual(result["vector"].shape, (384,))
        self.assertNotEqual(self.worker.state()["pid"], pid)
        self.assertFalse((self.root / "completed").exists())
        print(f"index timeout={elapsed:.3f}s; worker {pid} stopped; next worker={self.worker.state()['pid']}")

    def test_total_answer_deadline_includes_slow_http(self):
        self.make_ready()
        server, http_worker = self.start_http()
        warm = self.client.post("/collections/personal/ask", json={"question": "warmup"})
        self.assertEqual(warm.status_code, 200)
        started = time.monotonic()
        with patch.dict(load_config()["limits"], {"request_deadline_seconds": 0.04}):
            response = self.client.post("/collections/personal/ask", json={"question": "SLOW_HTTP срок?"})
        elapsed = time.monotonic() - started
        self.assertEqual(response.status_code, 504)
        self.assertLess(elapsed, 0.35)
        self.assertFalse(http_worker.state()["alive"])
        self.assertFalse(indexing.INDEX_LOCK.locked())
        self.assertEqual(generation.generation_state()["server_state"], "unconfirmed_after_timeout")
        started = time.monotonic()
        retry = self.client.post("/collections/personal/ask", json={"question": "повтор"})
        self.assertEqual(retry.status_code, 503)
        self.assertLess(time.monotonic() - started, 0.3)
    
        self.assertTrue(server.slow_started.is_set())
        self.assertFalse(server.slow_finished.is_set())
        self.assertTrue(server.slow_finished.wait(timeout=2))
        generation.acknowledge_ollama_stopped(confirmed_stopped=True)
        success = self.client.post("/collections/personal/ask", json={"question": "срок?"})
        self.assertEqual(success.status_code, 200)
        self.assertEqual(success.json()["usage"]["total_tokens"], 47)
        print(f"answer timeout={elapsed:.3f}s; client stopped; unsafe retry rejected with 503")

    def test_missing_ollama_model_reports_unavailability(self):
        self.make_ready()
        self.start_http()
        response = self.client.post("/collections/personal/ask", json={"question": "MISSING_MODEL срок?"})
        self.assertEqual(response.status_code, 503)
        self.assertIn(generation.MODEL, response.json()["detail"])
        self.assertFalse(indexing.INDEX_LOCK.locked())
        self.assertEqual(self.client.post("/collections/personal/ask", json={"question": "срок?"}).status_code, 200)

    def test_slow_query_encode_stops_before_generation(self):
        self.make_ready()
        with patch.dict(load_config()["limits"], {"request_deadline_seconds": 0.08}):
            started = time.monotonic()
            response = self.client.post("/collections/personal/ask", json={"question": "SLOW question"})
        self.assertEqual(response.status_code, 504)
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertFalse(self.worker.state()["alive"])
        self.assertFalse(indexing.INDEX_LOCK.locked())

    def test_supported_entrypoints_use_current_config(self):
        self.assertFalse((PROJECT_ROOT / "app" / "rag-demo.py").exists())
        config = load_config()
        self.assertFalse({"base_url", "context_tokens", "output_tokens"} & config["generation"].keys())
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.make_ready()
        server, _ = self.start_http()
        response = self.client.post("/collections/personal/ask", json={"question": "срок?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(server.last_path, "/api/chat")
        self.assertEqual(server.last_payload["model"], config["generation"]["model"])
        self.assertEqual(server.last_payload["options"]["num_ctx"], config["generation"]["num_ctx"])
        self.assertEqual(server.last_payload["options"]["num_predict"], config["generation"]["num_predict"])
        cli = subprocess.run([sys.executable, "-m", "app.create_user", "--help"],
                             cwd=PROJECT_ROOT, capture_output=True, timeout=10)
        self.assertEqual(cli.returncode, 0, cli.stderr.decode())

    def test_parser_hang_is_stopped(self):
        worker = ProcessWorker(hanging_parser)
        self.addCleanup(worker.close)
 
        worker._start(Deadline.after(10))
        with patch.object(parsing, "PARSER", worker):
            with self.assertRaises(TimeoutError):
                parsing.parse_document(self.path, ".txt", timeout=0.08)
        self.assertFalse(worker.state()["alive"])
        self.assertFalse((self.root / "completed").exists())

    def test_expired_request_cannot_return_success(self):
        self.make_ready()
        real_search = search.search_collection
        def delayed_search(*args, **kwargs):
            result = real_search(*args, **kwargs)
            time.sleep(0.06)
            return result
        with patch.object(search, "search_collection", side_effect=delayed_search):
            with patch.dict(load_config()["limits"], {"request_deadline_seconds": 0.04}):
                response = self.client.post("/collections/personal/ask", json={"question": "срок?"})
        self.assertEqual(response.status_code, 504)

    def test_index_parser_uses_remaining_request_budget(self):
        worker = ProcessWorker(hanging_parser)
        self.addCleanup(worker.close)
        worker._start(Deadline.after(10))
        with patch.object(parsing, "PARSER", worker):
            with patch.object(indexing, "parse_document", parsing.parse_document):
                with patch.dict(load_config()["limits"], {"request_deadline_seconds": 0.08}):
                    started = time.monotonic()
                    response = self.client.post("/collections/personal/documents/doc/index")
        self.assertEqual(response.status_code, 504)
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertFalse(worker.state()["alive"])
        self.assertFalse(indexing.INDEX_LOCK.locked())
        with self.open_db() as connection:
            self.assertEqual(connection.execute("SELECT status FROM documents").fetchone()[0], "failed")

    def test_worker_startup_is_included_in_deadline(self):
        worker = ProcessWorker(hanging_parser)
        self.addCleanup(worker.close)
        started = time.monotonic()
        with self.assertRaises(DocumentError) as error:
            worker.call((), Deadline.after(0.001))
        self.assertEqual(error.exception.status_code, 504)
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertFalse(worker.state()["alive"])
        self.assertFalse(worker.last_dispatched)


if __name__ == "__main__":
    unittest.main()
