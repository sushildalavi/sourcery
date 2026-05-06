"""Tests for the WorkspaceMiddleware tenant resolver."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from backend.middleware import WORKSPACE_HEADER, WorkspaceMiddleware


def _client() -> TestClient:
    app = FastAPI()
    app.add_middleware(WorkspaceMiddleware)

    @app.get("/probe")
    def probe(request: Request):
        return {"workspace": request.state.workspace_id}

    return TestClient(app)


def test_default_workspace_when_header_absent():
    resp = _client().get("/probe")
    assert resp.status_code == 200
    assert resp.json()["workspace"] == "default"
    assert resp.headers[WORKSPACE_HEADER] == "default"


def test_valid_workspace_id_pinned_on_request_state():
    resp = _client().get("/probe", headers={WORKSPACE_HEADER: "research-group-42"})
    assert resp.json()["workspace"] == "research-group-42"
    assert resp.headers[WORKSPACE_HEADER] == "research-group-42"


def test_injection_chars_fall_back_to_default():
    """Anything outside [a-zA-Z0-9_.-] gets dropped — defends against
    SQL injection or path-traversal attempts via the workspace header."""
    bad = ["'; DROP TABLE users; --", "../etc/passwd", "ws id with spaces", "A" * 200]
    for value in bad:
        resp = _client().get("/probe", headers={WORKSPACE_HEADER: value})
        assert resp.json()["workspace"] == "default", value


def test_allowed_charset_chars():
    for value in ["abc", "ABC123", "ws_42", "ws-42", "team.alpha"]:
        resp = _client().get("/probe", headers={WORKSPACE_HEADER: value})
        assert resp.json()["workspace"] == value, value
