"""ASGI middleware for request tracing, security headers, and structured access logging."""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

REQUEST_ID_HEADER = "X-Request-ID"
WORKSPACE_HEADER = "X-Workspace-Id"
DEFAULT_WORKSPACE = "default"


def current_workspace(request) -> str:
    """Resolve the workspace for the current request.

    Reads `request.state.workspace_id` (set by `WorkspaceMiddleware`).
    Returns `DEFAULT_WORKSPACE` if state is missing or empty so route
    handlers never have to special-case startup-time / test scenarios.
    """
    try:
        ws = getattr(request.state, "workspace_id", None)
    except Exception:
        ws = None
    if not ws:
        return DEFAULT_WORKSPACE
    return str(ws)


_access_log = logging.getLogger("scholarrag.access")


# OWASP-aligned security header defaults. Tunable via env so a deployment
# behind a CDN that already injects HSTS doesn't double-set headers.
_DEFAULT_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=(), payment=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-site",
    # HSTS is only meaningful behind TLS — opt in via env when serving over https.
    # "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Apply security response headers on every response.

    Set ENABLE_HSTS=true (only behind TLS) to add Strict-Transport-Security.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._headers = dict(_DEFAULT_SECURITY_HEADERS)
        if _env_flag("ENABLE_HSTS", default=False):
            self._headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        for k, v in self._headers.items():
            response.headers.setdefault(k, v)
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        start = time.perf_counter()

        # Stash on request.state so route handlers can include it in logs.
        request.state.request_id = rid

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            _access_log.exception(
                "request failed rid=%s method=%s path=%s elapsed_ms=%.1f",
                rid,
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.headers[REQUEST_ID_HEADER] = rid
        # Skip noisy paths from access log; they're for liveness / favicons.
        if request.url.path not in {"/", "/favicon.ico"}:
            _access_log.info(
                "rid=%s method=%s path=%s status=%s elapsed_ms=%.1f",
                rid,
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
        return response


_WORKSPACE_RE = re.compile(r"^[a-zA-Z0-9_.\-]{1,64}$")


def _sanitized_workspace(raw: str | None) -> str:
    """Return a safe workspace id; reject anything that looks injected."""
    if not raw:
        return DEFAULT_WORKSPACE
    raw = raw.strip()
    if not raw or not _WORKSPACE_RE.match(raw):
        return DEFAULT_WORKSPACE
    return raw


class WorkspaceMiddleware(BaseHTTPMiddleware):
    """Resolve the request's tenant from `X-Workspace-Id` and pin it on
    `request.state.workspace_id`.

    Workspace IDs:
      - default to `"default"` when the header is absent (back-compat),
      - are rejected (silently fall back to `"default"`) if they fail a
        strict alphanumeric / `._-` allowlist — prevents header-injection
        attacks against the SQL filter,
      - are echoed back on the response so callers can confirm which
        tenant they were resolved as.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        workspace = _sanitized_workspace(request.headers.get(WORKSPACE_HEADER))
        request.state.workspace_id = workspace
        response = await call_next(request)
        response.headers[WORKSPACE_HEADER] = workspace
        return response
