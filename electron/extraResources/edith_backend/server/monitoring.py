"""
Sentry error monitoring integration for E.D.I.T.H.

Captures unhandled exceptions, slow queries, and security events.
Set SENTRY_DSN in .env to enable. Disabled silently if not configured.

Usage in main.py:
    from server.monitoring import init_monitoring
    init_monitoring(app)
"""
import os
import logging

log = logging.getLogger("edith.monitoring")

_sentry_enabled = False


def init_monitoring(app=None):
    """Initialize Sentry error monitoring if SENTRY_DSN is set."""
    global _sentry_enabled
    dsn = os.environ.get("SENTRY_DSN", "")

    if not dsn:
        log.info("Sentry DSN not set — error monitoring disabled")
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        sentry_sdk.init(
            dsn=dsn,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                StarletteIntegration(transaction_style="endpoint"),
            ],
            # Performance: sample 10% of transactions in production
            traces_sample_rate=float(os.environ.get("SENTRY_TRACES_RATE", "0.1")),
            # Only send errors, not info/warning
            before_send=_filter_event,
            # Don't send PII
            send_default_pii=False,
            # Tag with environment
            environment=os.environ.get("EDITH_ENV", "development"),
            release=_get_version(),
        )
        _sentry_enabled = True
        log.info("Sentry error monitoring active")
    except ImportError:
        log.info("sentry-sdk not installed — pip install sentry-sdk[fastapi]")
    except Exception as e:
        log.warning(f"Sentry init failed: {e}")


def _filter_event(event, hint):
    """Filter events before sending to Sentry — strip PII, skip noisy errors."""
    # Don't send 404s or client errors
    if "exception" in event:
        exc_type = event.get("exception", {}).get("values", [{}])[0].get("type", "")
        if exc_type in ("HTTPException", "RequestValidationError"):
            return None

    # Strip any potential PII from breadcrumbs
    for breadcrumb in event.get("breadcrumbs", {}).get("values", []):
        if "data" in breadcrumb:
            for key in list(breadcrumb["data"].keys()):
                if any(pii in key.lower() for pii in ("email", "password", "token", "key", "secret")):
                    breadcrumb["data"][key] = "[REDACTED]"

    return event


def _get_version():
    """Get app version for Sentry release tracking."""
    try:
        with open(os.path.join(os.path.dirname(__file__), "..", "package.json")) as f:
            import json
            return json.load(f).get("version", "0.0.0")
    except Exception:
        return "0.0.0"


def capture_error(error, context=None):
    """Manually capture an error with optional context."""
    if not _sentry_enabled:
        return
    try:
        import sentry_sdk
        with sentry_sdk.push_scope() as scope:
            if context:
                for k, v in context.items():
                    scope.set_extra(k, v)
            sentry_sdk.capture_exception(error)
    except Exception:
        pass


def capture_message(message, level="info"):
    """Capture a message (not an error) for alerting."""
    if not _sentry_enabled:
        return
    try:
        import sentry_sdk
        sentry_sdk.capture_message(message, level=level)
    except Exception:
        pass
