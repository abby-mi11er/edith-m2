"""
Event Bus Routes — API for the E.D.I.T.H. nervous system
============================================================
Exposes the EventBus for monitoring, debugging, and manual event emission.
"""

import logging
from fastapi import APIRouter, Body

log = logging.getLogger("edith.routes.bus")

router = APIRouter(prefix="/api/bus", tags=["eventbus"])


@router.get("/status")
async def bus_status():
    """Full status of the EventBus — subscriber count, event stats, errors."""
    from server.event_bus import bus
    return bus.status


@router.get("/wiring")
async def bus_wiring():
    """
    The complete wiring map. Shows every event → subscriber relationship.
    This IS the nervous system diagram.
    """
    from server.event_bus import bus, EVENT_CATALOG
    wiring = bus.wiring_map
    # Annotate with catalog descriptions
    annotated = {}
    for event_name, handlers in wiring.items():
        annotated[event_name] = {
            "handlers": handlers,
            "description": EVENT_CATALOG.get(event_name, ""),
        }
    return {"wiring": annotated, "total_events": len(EVENT_CATALOG)}


@router.get("/history")
async def bus_history(event: str = "", limit: int = 50):
    """Get recent event history, optionally filtered by event name."""
    from server.event_bus import bus
    return {"events": bus.history(event, limit)}


@router.get("/catalog")
async def event_catalog():
    """List all known event types and their descriptions."""
    from server.event_bus import EVENT_CATALOG
    return {"events": EVENT_CATALOG}


@router.post("/emit")
async def manual_emit(body: dict = Body(default={})):
    """
    Manually emit an event (for testing/debugging).
    
    Body: { "event": "paper.indexed", "data": {...}, "source": "manual" }
    """
    from server.event_bus import bus
    event_name = body.get("event", "")
    if not event_name:
        return {"error": "Provide 'event' name"}
    data = body.get("data", {})
    source = body.get("source", "manual")
    event = await bus.emit(event_name, data, source=source)
    return {"emitted": event.to_dict(), "subscribers_notified": len(bus._subscribers.get(event_name, []))}
