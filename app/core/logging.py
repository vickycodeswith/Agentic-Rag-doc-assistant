import logging
import re
import sys
from typing import Any, Dict
from rich.console import Console
from rich.logging import RichHandler

from app.config import settings

# Redact sensitive keys from log output
SENSITIVE_PATTERNS = [
    re.compile(r"(api[-_]?key\s*[:=]\s*['\"]?)([\w\-]{8,})(['\"]?)", re.IGNORECASE),
    re.compile(r"(authorization\s*[:=]\s*['\"]?Bearer\s+)([\w\-]{8,})(['\"]?)", re.IGNORECASE),
]


class SensitiveDataFilter(logging.Filter):
    """Filters out API keys and bearer tokens from log messages."""
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern in SENSITIVE_PATTERNS:
                record.msg = pattern.sub(r"\1***REDACTED***\3", record.msg)
        return True


def setup_logger(name: str = "rag_assistant") -> logging.Logger:
    """Configures structured logging with Rich console formatting and security filters."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Rich Handler for clean, colorized terminal output
    console = Console(file=sys.stderr)
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False
    )
    rich_handler.addFilter(SensitiveDataFilter())

    formatter = logging.Formatter("%(name)s - %(message)s")
    rich_handler.setFormatter(formatter)
    logger.addHandler(rich_handler)
    logger.propagate = False

    return logger


logger = setup_logger()


def log_graph_event(step_name: str, details: Dict[str, Any]) -> None:
    """Helper to log structured graph execution steps for observability."""
    logger.info(f"⚡ [GraphStep: {step_name}] -> {details}")
