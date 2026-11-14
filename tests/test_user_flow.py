from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import collections, documents as documents_api
from app.core import auth
from app.main import app
from app.services import documents
from app.storage.database import database, SCHEMA_PATH


class UserFlowTests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        db_path = root / "test.sqlite"

        def open_db():
            return database(db_path)

        with open_db() as conn:
            conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        for module in (auth, collections, documents_api, documents):
            stack.enter_context(patch.object(module, "database", open_db))
        stack.enter_context(patch.object(documents, "DOCUMENT_DIRECTORY", root / "documents"))
        self.alice = auth.create_user("alice", "test-only-password")
        self.bob = auth.create_user("bob", "other-test-password")
        self.client = stack.enter_context(TestClient(app))
        self.url = f"/collections/{self.alice['personal_collection_id']}/documents"

    def sign_in(self):
        response = self.client.post("/auth/login", json={"username": "alice", "password": "test-only-password"})
        self.assertEqual(response.status_code, 200)
        self.client.headers["Authorization"] = f"Bearer {response.json()['access_token']}"

    def test_page_session_and_collection_access(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/assets/app.js").status_code, 200)
        self.assertEqual(self.client.get("/collections").status_code, 401)
        self.client.headers["Authorization"] = "Bearer invalid-token"
        self.assertEqual(self.client.get("/collections").status_code, 401)
        self.sign_in()
        rows = self.client.get("/collections").json()
        self.assertEqual([row["id"] for row in rows], [self.alice["personal_collection_id"]])
        self.assertEqual(self.client.get(f"/collections/{self.bob['personal_collection_id']}/documents").status_code, 403)
        self.assertEqual(self.client.post("/auth/logout").status_code, 200)
        self.assertEqual(self.client.get("/collections").status_code, 401)

    def test_upload_limits_deduplication_and_listing(self):
        self.sign_in()
        self.assertEqual(self.client.post(self.url, files={"file": ("empty.txt", b"")}).status_code, 400)
        with patch.object(documents, "MAX_FILE_BYTES", 4):
            self.assertEqual(self.client.post(self.url, files={"file": ("large.txt", b"12345")}).status_code, 413)
        first = self.client.post(self.url, files={"file": ("demo.txt", b"Backup policy")})
        again = self.client.post(self.url, files={"file": ("renamed.txt", b"Backup policy")})
        self.assertEqual((first.status_code, again.status_code), (201, 200))
        self.assertEqual(first.json()["id"], again.json()["id"])
        self.assertEqual([row["id"] for row in self.client.get(self.url).json()], [first.json()["id"]])
