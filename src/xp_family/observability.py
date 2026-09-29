"""Structured logging and lightweight tracing of LLM calls.

Every LLM call emits one `llm_call` event with model, latency, token usage and outcome,
tagged with a per-request trace id so a single user request can be followed end to end.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import structlog

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    logging.basicConfig(format="%(message)s", level=level.upper())
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def new_trace_id() -> str:
    trace_id = uuid.uuid4().hex[:16]
    _trace_id.set(trace_id)
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
    return trace_id


def current_trace_id() -> str | None:
    return _trace_id.get()


class Span:
    """Mutable bag of attributes recorded when the traced block exits."""

    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}

    def set(self, **attributes: Any) -> None:
        self.attributes.update(attributes)


@contextmanager
def trace(event: str, **attributes: Any) -> Iterator[Span]:
    """Time a block and log it as one structured event, including failures."""
    log = get_logger("xp_family.trace")
    span = Span()
    start = time.perf_counter()
    try:
        yield span
    except Exception as exc:
        log.error(
            event,
            status="error",
            error=type(exc).__name__,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            **attributes,
            **span.attributes,
        )
        raise
    log.info(
        event,
        status="ok",
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
        **attributes,
        **span.attributes,
    )
