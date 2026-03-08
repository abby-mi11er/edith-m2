"""
Structured Logging — A3
========================
JSON-formatted logging for production use.
Human-readable in dev, structured JSON in prod.

Usage:
    from server.logging_config import setup_logging
    setup_logging()  # Call once at startup
"""
import logging
import json
import os
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exc"] = self.formatException(record.exc_info)
        # Add extra fields if present
        for key in ("request_id", "user", "endpoint", "duration_ms", "status_code"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        return json.dumps(log_entry, default=str)


class PrettyFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[31;1m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = datetime.now().strftime("%H:%M:%S")
        name = record.name.replace("edith.", "")
        msg = record.getMessage()
        
        extra_parts = []
        for key in ("request_id", "duration_ms", "status_code"):
            val = getattr(record, key, None)
            if val is not None:
                extra_parts.append(f"{key}={val}")
        extra = f" [{', '.join(extra_parts)}]" if extra_parts else ""
        
        return f"{color}{ts} {record.levelname:>7}{self.RESET} {name}: {msg}{extra}"


def setup_logging(level: str = "") -> None:
    """Configure logging based on environment.
    
    - EDITH_LOG_FORMAT=json → JSON output (for log aggregation)
    - EDITH_LOG_FORMAT=pretty (or unset) → colored human output
    - EDITH_LOG_LEVEL → DEBUG/INFO/WARNING/ERROR (default: INFO)
    """
    log_format = os.environ.get("EDITH_LOG_FORMAT", "pretty").lower()
    log_level = level or os.environ.get("EDITH_LOG_LEVEL", "INFO").upper()
    
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Remove existing handlers to avoid duplicate output
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    handler = logging.StreamHandler(sys.stderr)
    
    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(PrettyFormatter())
    
    root.addHandler(handler)
    
    # Quiet noisy libraries
    for noisy in ("httpcore", "httpx", "chromadb", "urllib3", "onnxruntime"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
