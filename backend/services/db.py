import logging
import os
from typing import Any, Iterable, List, Optional, Tuple

import psycopg2
import psycopg2.extras
import psycopg2.pool
from psycopg2.extensions import connection as PGConnection

from backend.utils.config import _load_dotenv_if_available

logger = logging.getLogger(__name__)

_pool: psycopg2.pool.ThreadedConnectionPool | None = None

POOL_MIN = int(os.getenv("PG_POOL_MIN", "2") or 2)
POOL_MAX = int(os.getenv("PG_POOL_MAX", "12") or 12)


def _dsn_kwargs() -> dict:
    _load_dotenv_if_available()
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return {"dsn": database_url}
    return {
        "host": os.getenv("PGHOST", "127.0.0.1"),
        "port": int(os.getenv("PGPORT", "5432")),
        "user": os.getenv("PGUSER", "scholarrag"),
        "password": os.getenv("PGPASSWORD", "scholarrag"),
        "dbname": os.getenv("PGDATABASE", "scholarrag_db"),
    }


def _database_connection_hint(database_url: str, exc: psycopg2.OperationalError) -> str:
    message = str(exc)
    if "tenant or user not found" in message.lower():
        return (
            "DATABASE_URL was rejected with `Tenant or user not found`. "
            "For local development, set DATABASE_URL to "
            "`postgresql://scholarrag:scholarrag@127.0.0.1:5432/scholarrag_db` and run "
            "`docker compose up -d db`. Then verify username/password/host/port values in your "
            "DATABASE_URL."
        )
    return f"DATABASE_URL connection failed: {message}"


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None or _pool.closed:
        kwargs = _dsn_kwargs()
        try:
            _pool = psycopg2.pool.ThreadedConnectionPool(POOL_MIN, POOL_MAX, **kwargs)
        except psycopg2.OperationalError as exc:
            raise RuntimeError(_database_connection_hint(kwargs.get("dsn", ""), exc)) from exc
    return _pool


def get_connection() -> PGConnection:
    """Return a connection from the pool. Caller must putconn() after use."""
    return _get_pool().getconn()


def _putconn(conn: PGConnection) -> None:
    try:
        _get_pool().putconn(conn)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass


def execute(query: str, params: Optional[Iterable[Any]] = None) -> None:
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, params or [])
    finally:
        _putconn(conn)


def execute_batch(query: str, params_list: List[Tuple], page_size: int = 100) -> None:
    """Execute a parameterized query for many rows using psycopg2.extras.execute_batch."""
    if not params_list:
        return
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, query, params_list, page_size=page_size)
    finally:
        _putconn(conn)


def execute_values(query: str, values: List[Tuple], template: Optional[str] = None, fetch: bool = False):
    """Execute a multi-row INSERT using psycopg2.extras.execute_values (fastest bulk insert)."""
    if not values:
        return [] if fetch else None
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                result = psycopg2.extras.execute_values(
                    cur, query, values, template=template, fetch=fetch, page_size=250,
                )
                return result if fetch else None
    finally:
        _putconn(conn)


def fetchall(query: str, params: Optional[Iterable[Any]] = None):
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, params or [])
                cols = [desc[0] for desc in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        _putconn(conn)


def fetchone(query: str, params: Optional[Iterable[Any]] = None):
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, params or [])
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [desc[0] for desc in cur.description]
                return dict(zip(cols, row))
    finally:
        _putconn(conn)
