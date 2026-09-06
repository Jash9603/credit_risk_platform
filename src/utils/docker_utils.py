"""Utilities for behaving correctly whether the code runs on a laptop or inside the
`api` container — used by the ingest script (waiting for Postgres to accept connections)
and by anything that needs to know its runtime environment."""
import socket
import time
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def running_in_docker() -> bool:
    return Path("/.dockerenv").exists()


def wait_for_port(host: str, port: int, timeout: int = 60, interval: float = 1.5) -> None:
    """Block until `host:port` accepts TCP connections. Used before the ingest script
    talks to the `db` compose service, which needs a few seconds to finish initialising
    even after its container process has started."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                logger.info("%s:%s is accepting connections", host, port)
                return
        except OSError as exc:
            last_error = exc
            time.sleep(interval)

    raise TimeoutError(f"{host}:{port} not reachable after {timeout}s") from last_error
