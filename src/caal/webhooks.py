"""Webhook server for external triggers (announcements, tool reload, wake word).

This module provides HTTP endpoints that allow external systems (like n8n)
to trigger actions on the running voice agent.

Endpoints:
    POST /announce      - Make the agent speak a message
    POST /reload-tools  - Refresh MCP tool cache and optionally announce
    POST /wake          - Handle wake word detection (greet user)
    GET  /health        - Health check

Usage:
    # Start in a background thread from voice_agent.py:
    import threading
    import uvicorn
    from caal.webhooks import app

    def run_webhook_server():
        uvicorn.run(app, host="0.0.0.0", port=8889, log_level="info")

    webhook_thread = threading.Thread(target=run_webhook_server, daemon=True)
    webhook_thread.start()
"""

from __future__ import annotations

import logging
import random

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Wake word greetings - randomly selected for variety
WAKE_GREETINGS = [
    "Hey, what's up?",
    "Hi there!",
    "Yeah?",
    "What can I do for you?",
    "Hey!",
    "Yo!",
    "What's up?",
]

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CAAL Webhook API",
    description="External triggers for CAAL voice agent",
    version="1.0.0",
)


class AnnounceRequest(BaseModel):
    """Request body for /announce endpoint."""

    message: str
    room_name: str = "voice_assistant_room"


class ReloadToolsRequest(BaseModel):
    """Request body for /reload-tools endpoint."""

    tool_name: str | None = None  # Optional: announce specific tool name
    message: str | None = None  # Optional: custom announcement message (overrides tool_name)
    room_name: str = "voice_assistant_room"


class WakeRequest(BaseModel):
    """Request body for /wake endpoint."""

    room_name: str = "voice_assistant_room"


class WakeResponse(BaseModel):
    """Response body for /wake endpoint."""

    status: str
    room_name: str


class AnnounceResponse(BaseModel):
    """Response body for /announce endpoint."""

    status: str
    room_name: str


class ReloadToolsResponse(BaseModel):
    """Response body for /reload-tools endpoint."""

    status: str
    tool_count: int
    room_name: str


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    active_sessions: list[str]


@app.post("/announce", response_model=AnnounceResponse)
async def announce(req: AnnounceRequest) -> AnnounceResponse:
    """Make the agent speak a message.

    This endpoint injects an announcement into an active voice session.
    The agent will speak the provided message using TTS.

    Args:
        req: AnnounceRequest with message and optional room_name

    Returns:
        AnnounceResponse with status

    Raises:
        HTTPException: 404 if no active session in the specified room
    """
    from . import session_registry

    result = session_registry.get(req.room_name)
    if not result:
        logger.warning(f"Announce failed: no session in room {req.room_name}")
        raise HTTPException(
            status_code=404,
            detail=f"No active session in room: {req.room_name}",
        )

    session, _agent = result
    logger.info(f"Announcing to room {req.room_name}: {req.message[:50]}...")

    # Say the message directly (bypasses LLM for instant response)
    await session.say(req.message)

    return AnnounceResponse(status="announced", room_name=req.room_name)


@app.post("/reload-tools", response_model=ReloadToolsResponse)
async def reload_tools(req: ReloadToolsRequest) -> ReloadToolsResponse:
    """Refresh MCP tool cache and optionally announce new tool availability.

    This endpoint clears the n8n workflow cache and re-discovers available
    workflows. Optionally announces the change:
    - If `message` is provided, speaks that exact message
    - If only `tool_name` is provided, speaks "A new tool called '{tool_name}' is now available."
    - If neither is provided, reloads silently

    Args:
        req: ReloadToolsRequest with optional message, tool_name, and room_name

    Returns:
        ReloadToolsResponse with status and tool count

    Raises:
        HTTPException: 404 if no active session in the specified room
    """
    from . import session_registry
    from .integrations import n8n

    result = session_registry.get(req.room_name)
    if not result:
        logger.warning(f"Reload failed: no session in room {req.room_name}")
        raise HTTPException(
            status_code=404,
            detail=f"No active session in room: {req.room_name}",
        )

    session, agent = result
    logger.info(f"Reloading tools for room {req.room_name}")

    # Clear all caches
    agent._ollama_tools_cache = None
    n8n.clear_caches()

    # Re-discover n8n workflows if MCP is configured
    tool_count = 0
    if agent._n8n_mcp and agent._n8n_base_url:
        try:
            tools, name_map = await n8n.discover_n8n_workflows(
                agent._n8n_mcp, agent._n8n_base_url
            )
            agent._n8n_workflow_tools = tools
            agent._n8n_workflow_name_map = name_map
            tool_count = len(tools)
            logger.info(f"Discovered {tool_count} n8n workflows")
        except Exception as e:
            logger.error(f"Failed to re-discover n8n workflows: {e}")

    # Announce: custom message takes priority, then tool_name format
    if req.message:
        await session.say(req.message)
    elif req.tool_name:
        await session.say(f"A new tool called '{req.tool_name}' is now available.")

    return ReloadToolsResponse(
        status="reloaded",
        tool_count=tool_count,
        room_name=req.room_name,
    )


@app.post("/wake", response_model=WakeResponse)
async def wake(req: WakeRequest) -> WakeResponse:
    """Handle wake word detection - greet the user.

    This endpoint is called by the frontend when the wake word ("Hey Cal")
    is detected. The agent responds with a brief greeting to acknowledge
    the user and indicate readiness.

    Args:
        req: WakeRequest with room_name

    Returns:
        WakeResponse with status

    Raises:
        HTTPException: 404 if no active session in the specified room
    """
    from . import session_registry

    result = session_registry.get(req.room_name)
    if not result:
        logger.warning(f"Wake failed: no session in room {req.room_name}")
        raise HTTPException(
            status_code=404,
            detail=f"No active session in room: {req.room_name}",
        )

    session, _agent = result
    logger.info(f"Wake word detected in room {req.room_name}")

    # Say a random greeting directly (bypasses LLM for instant response)
    greeting = random.choice(WAKE_GREETINGS)
    await session.say(greeting)

    return WakeResponse(status="greeted", room_name=req.room_name)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse with status and list of active session room names
    """
    from . import session_registry

    return HealthResponse(
        status="ok",
        active_sessions=session_registry.list_rooms(),
    )
