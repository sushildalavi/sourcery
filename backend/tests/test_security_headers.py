"""Verify the SecurityHeadersMiddleware writes OWASP-aligned headers."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.middleware import SecurityHeadersMiddleware


def _client(*, hsts: bool = False, monkeypatch=None) -> TestClient:
    if monkeypatch is not None:
        if hsts:
            monkeypatch.setenv("ENABLE_HSTS", "true")
        else:
            monkeypatch.delenv("ENABLE_HSTS", raising=False)

    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/probe")
    def probe():
        return {"ok": True}

    return TestClient(app)


def test_default_security_headers(monkeypatch):
    resp = _client(monkeypatch=monkeypatch).get("/probe")
    assert resp.status_code == 200
    h = {k.lower(): v for k, v in resp.headers.items()}
    assert h["x-content-type-options"] == "nosniff"
    assert h["x-frame-options"] == "DENY"
    assert h["referrer-policy"] == "strict-origin-when-cross-origin"
    assert "geolocation=()" in h["permissions-policy"]
    assert h["cross-origin-opener-policy"] == "same-origin"
    assert h["cross-origin-resource-policy"] == "same-site"
    assert "strict-transport-security" not in h, "HSTS must be opt-in"


def test_hsts_when_enabled(monkeypatch):
    resp = _client(hsts=True, monkeypatch=monkeypatch).get("/probe")
    h = {k.lower(): v for k, v in resp.headers.items()}
    assert "max-age=31536000" in h["strict-transport-security"]
    assert "includeSubDomains" in h["strict-transport-security"]


def test_existing_header_is_not_overwritten(monkeypatch):
    """If a route explicitly sets a security header, the middleware respects it."""
    monkeypatch.delenv("ENABLE_HSTS", raising=False)
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/custom")
    def custom():
        from fastapi.responses import JSONResponse

        return JSONResponse({"ok": True}, headers={"X-Frame-Options": "SAMEORIGIN"})

    resp = TestClient(app).get("/custom")
    assert resp.headers["X-Frame-Options"] == "SAMEORIGIN"
