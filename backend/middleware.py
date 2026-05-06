"""ASGI middleware for request tracing, security headers, and structured access logging."""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

REQUEST_ID_HEADER = "X-Request-ID"

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
            self._headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

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
