import os
import unittest
from unittest.mock import patch

from psycopg2 import OperationalError

import backend.services.db as db_module
from backend.services.db import get_connection


class DatabaseConnectionTests(unittest.TestCase):
    def setUp(self):
        if db_module._pool is not None:
            try:
                db_module._pool.closeall()
            except Exception:
                pass
        db_module._pool = None

    def tearDown(self):
        if db_module._pool is not None:
            try:
                db_module._pool.closeall()
            except Exception:
                pass
        db_module._pool = None

    def test_tenant_or_user_not_found_error_has_local_dev_hint(self):
        database_url = "postgresql://wrong-user:pw@127.0.0.1:5432/postgres"
        old = os.environ.get("DATABASE_URL")
        try:
            os.environ["DATABASE_URL"] = database_url
            with patch("backend.services.db._load_dotenv_if_available", return_value=None):
                with patch(
                    "backend.services.db.psycopg2.connect",
                    side_effect=OperationalError("FATAL:  Tenant or user not found"),
                ):
                    with self.assertRaises(RuntimeError) as ctx:
                        get_connection()
            self.assertIn("docker compose up -d db", str(ctx.exception))
            self.assertIn("verify username/password/host/port", str(ctx.exception))
        finally:
            if old is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = old

    def test_generic_connection_error_is_preserved(self):
        database_url = "postgresql://user:pw@127.0.0.1:5432/app"
        old = os.environ.get("DATABASE_URL")
        try:
            os.environ["DATABASE_URL"] = database_url
            with patch("backend.services.db._load_dotenv_if_available", return_value=None):
                with patch(
                    "backend.services.db.psycopg2.connect",
                    side_effect=OperationalError("connection refused"),
                ):
                    with self.assertRaises(RuntimeError) as ctx:
                        get_connection()
            self.assertIn("DATABASE_URL connection failed", str(ctx.exception))
            self.assertIn("connection refused", str(ctx.exception))
        finally:
            if old is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = old


if __name__ == "__main__":
    unittest.main()
