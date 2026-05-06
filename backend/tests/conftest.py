"""Pytest configuration for backend tests.

Keeps the test suite hermetic: tests should not make live LLM calls unless
they explicitly opt in by patching the relevant client or setting the
environment variable directly inside the test.
"""

import os


def _force_disable(env_var: str) -> None:
    os.environ[env_var] = "false"


# Disable the query intent resolver by default. Individual tests that want to
# exercise the resolver path can set INTENT_RESOLVER_ENABLED=true and mock the
# OpenAI client in backend.intent_resolver.
_force_disable("INTENT_RESOLVER_ENABLED")
