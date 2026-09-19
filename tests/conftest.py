"""Shared test configuration.

The MCP tests are async. anyio's pytest plugin (installed transitively with the
MCP SDK) drives them, and it requires this fixture to choose a backend. Pinned
to asyncio rather than parametrised over trio: running the whole MCP suite twice
doubles the subprocess spawns for no additional coverage.
"""

import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
