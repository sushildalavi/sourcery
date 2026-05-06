"""Tests for the X-Request-ID middleware."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.middleware import REQUEST_ID_HEADER, RequestIDMiddleware


def _client() -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/echo")
    def echo():
        return {"ok": True}

    return TestClient(app)


def test_middleware_mints_request_id_when_absent():
    resp = _client().get("/echo")
    assert resp.status_code == 200
    rid = resp.headers.get(REQUEST_ID_HEADER)
    assert rid and len(rid) >= 16


def test_middleware_preserves_upstream_request_id():
    upstream = "trace-test-xyz-001"
    resp = _client().get("/echo", headers={REQUEST_ID_HEADER: upstream})
    assert resp.headers.get(REQUEST_ID_HEADER) == upstream


def test_middleware_distinct_ids_across_requests():
    c = _client()
    a = c.get("/echo").headers.get(REQUEST_ID_HEADER)
    b = c.get("/echo").headers.get(REQUEST_ID_HEADER)
    assert a and b and a != b
