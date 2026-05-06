"""ASGI middleware for request tracing + structured access logging.

Each request gets an `X-Request-ID` (preserved from upstream if present, else
freshly minted). The same id is logged with the access line and echoed in
the response header so an operator can correlate UI traces, backend logs,
and downstream service calls.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

REQUEST_ID_HEADER = "X-Request-ID"

_access_log = logging.getLogger("scholarrag.access")


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
