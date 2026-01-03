#!/usr/bin/env python3
"""
CAAL Voice Framework - Voice Agent
==================================

A voice assistant with MCP integrations for n8n workflows.

Usage:
    python voice_agent.py dev

Configuration:
    - .env: Environment variables (MCP URL, model settings)
    - prompt/default.md: Agent system prompt

Environment Variables:
    WYOMING_STT_URL     - Wyoming STT service URI (default: "tcp://nabu.home:10300")
    WYOMING_TTS_URL     - Wyoming TTS service URI (default: "tcp://nabu.home:10200")
    TTS_VOICE           - TTS voice name (default: "am_puck")
    OLLAMA_MODEL        - LLM model name (default: "gpt-oss-20b-mxfp4", kept for backward compatibility)
    LLAMACPP_HOST       - llama.cpp server URL (default: "http://llama.home/v1")
    TIMEZONE            - Timezone for date/time (default: "Pacific Time")
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

import requests

# Add src directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv

# Load environment variables from .env
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_script_dir, ".env"))

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, mcp
from livekit.plugins import silero

from caal import LlamaCppLLM
from caal.integrations import (
    load_mcp_config,
    initialize_mcp_servers,
    WebSearchTools,
    discover_n8n_workflows,
)
from caal.llm import llamacpp_llm_node, ToolDataCache
from caal import session_registry
from caal.stt import WakeWordGatedSTT, WyomingSTT
from caal.tts import WyomingTTS

# Configure logging (LiveKit CLI reconfigures root logger, so set our level explicitly)
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

# Suppress verbose logs from dependencies
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)  # MCP client SSE/JSON-RPC spam
logging.getLogger("livekit").setLevel(logging.WARNING)  # LiveKit internal logs
logging.getLogger("livekit_api").setLevel(logging.WARNING)  # Rust bridge logs
logging.getLogger("livekit.agents.voice").setLevel(logging.WARNING)  # Suppress segment sync warnings
logging.getLogger("livekit.plugins.openai.tts").setLevel(logging.WARNING)  # Suppress "no request_id" spam
logging.getLogger("caal").setLevel(logging.INFO)  # Our package - INFO level

# =============================================================================
# Configuration
# =============================================================================

# Infrastructure config (from .env only - URLs, tokens, etc.)
WYOMING_STT_URL = os.getenv("WYOMING_STT_URL", "tcp://nabu.home:10300")
WYOMING_TTS_URL = os.getenv("WYOMING_TTS_URL", "tcp://nabu.home:10200")
LLAMACPP_HOST = os.getenv("LLAMACPP_HOST", os.getenv("OLLAMA_HOST", "http://llama.home/v1"))
TIMEZONE_ID = os.getenv("TIMEZONE", "America/Los_Angeles")
TIMEZONE_DISPLAY = os.getenv("TIMEZONE_DISPLAY", "Pacific Time")

# Import settings module for runtime-configurable values
from caal import settings as settings_module


def get_runtime_settings() -> dict:
    """Get runtime-configurable settings.

    These can be changed via the settings UI without rebuilding.
    Falls back to .env values for backwards compatibility.
    """
    settings = settings_module.load_settings()

    return {
        "tts_voice": settings.get("tts_voice") or os.getenv("TTS_VOICE", "am_puck"),
        "model": settings.get("model") or os.getenv("OLLAMA_MODEL", os.getenv("LLAMACPP_MODEL", "gpt-oss-20b-mxfp4")),
        "temperature": settings.get("temperature", float(os.getenv("OLLAMA_TEMPERATURE", os.getenv("LLAMACPP_TEMPERATURE", "0.7")))),
        "num_ctx": settings.get("num_ctx", int(os.getenv("OLLAMA_NUM_CTX", os.getenv("LLAMACPP_NUM_CTX", "8192")))),
        "max_turns": settings.get("max_turns", int(os.getenv("OLLAMA_MAX_TURNS", os.getenv("LLAMACPP_MAX_TURNS", "20")))),
        "tool_cache_size": settings.get("tool_cache_size", int(os.getenv("TOOL_CACHE_SIZE", "3"))),
    }


def load_prompt() -> str:
    """Load and populate prompt template with date context."""
    return settings_module.load_prompt_with_context(
        timezone_id=TIMEZONE_ID,
        timezone_display=TIMEZONE_DISPLAY,
    )


# =============================================================================
# Agent Definition
# =============================================================================

# Type alias for tool status callback
ToolStatusCallback = callable  # async (bool, list[str], list[dict]) -> None


class VoiceAssistant(WebSearchTools, Agent):
    """Voice assistant with MCP tools and web search."""

    def __init__(
        self,
        llamacpp_llm: LlamaCppLLM,
        mcp_servers: dict[str, mcp.MCPServerHTTP] | None = None,
        n8n_workflow_tools: list[dict] | None = None,
        n8n_workflow_name_map: dict[str, str] | None = None,
        n8n_base_url: str | None = None,
        on_tool_status: ToolStatusCallback | None = None,
        tool_cache_size: int = 3,
        max_turns: int = 20,
    ) -> None:
        super().__init__(
            instructions=load_prompt(),
            llm=llamacpp_llm,  # Satisfies LLM interface requirement
        )

        # All MCP servers (for multi-MCP support)
        # Named _caal_mcp_servers to avoid conflict with LiveKit's internal _mcp_servers handling
        self._caal_mcp_servers = mcp_servers or {}

        # n8n-specific for workflow execution (n8n uses webhook-based execution)
        self._n8n_workflow_tools = n8n_workflow_tools or []
        self._n8n_workflow_name_map = n8n_workflow_name_map or {}
        self._n8n_base_url = n8n_base_url

        # Callback for publishing tool status to frontend
        self._on_tool_status = on_tool_status

        # Context management: tool data cache and sliding window
        self._tool_data_cache = ToolDataCache(max_entries=tool_cache_size)
        self._max_turns = max_turns

    async def llm_node(self, chat_ctx, tools, model_settings):
        """Custom LLM node using llama.cpp server via OpenAI-compatible API."""
        # Access config from LlamaCppLLM instance via self.llm
        async for chunk in llamacpp_llm_node(
            self,
            chat_ctx,
            model=self.llm.model,
            temperature=self.llm.temperature,
            num_ctx=self.llm.num_ctx,
            tool_data_cache=self._tool_data_cache,
            max_turns=self._max_turns,
        ):
            yield chunk


# =============================================================================
# Agent Entrypoint
# =============================================================================

async def entrypoint(ctx: agents.JobContext) -> None:
    """Main entrypoint for the voice agent."""

    # Start webhook server in the same event loop (first job only)
    global _webhook_server_task
    if _webhook_server_task is None:
        _webhook_server_task = asyncio.create_task(start_webhook_server())

    logger.debug(f"Joining room: {ctx.room.name}")
    await ctx.connect()

    # Load MCP servers from config
    mcp_servers = {}
    try:
        mcp_configs = load_mcp_config()
        mcp_servers = await initialize_mcp_servers(mcp_configs)
    except Exception as e:
        logger.warning(f"Failed to load MCP config: {e}")

    # Discover n8n workflows (n8n uses webhook-based execution, not MCP tools)
    n8n_workflow_tools = []
    n8n_workflow_name_map = {}
    n8n_base_url = None
    n8n_mcp = mcp_servers.get("n8n")
    if n8n_mcp:
        try:
            # Extract base URL from n8n MCP server config
            n8n_config = next((c for c in mcp_configs if c.name == "n8n"), None)
            if n8n_config:
                # URL format: http://HOST:PORT/mcp-server/http
                # Base URL: http://HOST:PORT
                url_parts = n8n_config.url.rsplit("/", 2)
                n8n_base_url = url_parts[0] if len(url_parts) >= 2 else n8n_config.url

            n8n_workflow_tools, n8n_workflow_name_map = await discover_n8n_workflows(
                n8n_mcp, n8n_base_url
            )
        except Exception as e:
            logger.warning(f"Failed to discover n8n workflows: {e}")

    # Get runtime settings (from settings.json with .env fallback)
    runtime = get_runtime_settings()

    # Create LlamaCppLLM instance (config lives here, accessed via self.llm in agent)
    llamacpp_llm = LlamaCppLLM(
        model=runtime["model"],
        temperature=runtime["temperature"],
        num_ctx=runtime["num_ctx"],
        base_url=LLAMACPP_HOST,
    )

    # Log configuration
    logger.info("=" * 60)
    logger.info("STARTING VOICE AGENT")
    logger.info("=" * 60)
    logger.info(f"  STT: {WYOMING_STT_URL} (Wyoming)")
    logger.info(f"  TTS: {WYOMING_TTS_URL} (voice={runtime['tts_voice']})")
    logger.info(
        f"  LLM: llama.cpp ({runtime['model']}, num_ctx={runtime['num_ctx']}, base_url={LLAMACPP_HOST})"
    )
    logger.info(f"  MCP: {list(mcp_servers.keys()) or 'None'}")
    logger.info("=" * 60)

    # Build STT - optionally wrapped with wake word detection
    base_stt = WyomingSTT(uri=WYOMING_STT_URL)

    # Load wake word settings
    all_settings = settings_module.load_settings()
    wake_word_enabled = all_settings.get("wake_word_enabled", False)

    # Session reference for wake word callback (set after session creation)
    _session_ref: AgentSession | None = None

    if wake_word_enabled:
        import json
        import random

        wake_word_model = all_settings.get("wake_word_model", "models/hey_jarvis.onnx")
        wake_word_threshold = all_settings.get("wake_word_threshold", 0.5)
        wake_word_timeout = all_settings.get("wake_word_timeout", 3.0)
        wake_greetings = all_settings.get("wake_greetings", ["Hey, what's up?"])

        async def on_wake_detected():
            """Play wake greeting directly via TTS, bypassing agent turn-taking."""
            nonlocal _session_ref
            if _session_ref is None:
                logger.warning("Wake detected but session not ready yet")
                return

            try:
                # Pick a random greeting
                greeting = random.choice(wake_greetings)
                logger.info(f"Wake word detected, playing greeting: {greeting}")

                # Get TTS and audio output from session
                tts = _session_ref.tts
                audio_output = _session_ref.output.audio

                # Synthesize and push audio frames directly (bypasses turn-taking)
                audio_stream = tts.synthesize(greeting)
                async for event in audio_stream:
                    if hasattr(event, "frame") and event.frame:
                        await audio_output.capture_frame(event.frame)

                # Flush to complete the audio segment
                audio_output.flush()

            except Exception as e:
                logger.warning(f"Failed to play wake greeting: {e}")

        async def on_state_changed(state):
            """Publish wake word state to connected clients."""
            payload = json.dumps({
                "type": "wakeword_state",
                "state": state.value,
            })
            try:
                await ctx.room.local_participant.publish_data(
                    payload.encode("utf-8"),
                    reliable=True,
                    topic="wakeword_state",
                )
                logger.debug(f"Published wake word state: {state.value}")
            except Exception as e:
                logger.warning(f"Failed to publish wake word state: {e}")

        stt_instance = WakeWordGatedSTT(
            inner_stt=base_stt,
            model_path=wake_word_model,
            threshold=wake_word_threshold,
            silence_timeout=wake_word_timeout,
            on_wake_detected=on_wake_detected,
            on_state_changed=on_state_changed,
        )
        logger.info(f"  Wake word: ENABLED (model={wake_word_model}, threshold={wake_word_threshold})")
    else:
        stt_instance = base_stt
        logger.info("  Wake word: disabled")

    # Create session with Wyoming STT and TTS
    logger.info(f"  STT instance type: {type(stt_instance).__name__}")
    logger.info(f"  STT capabilities: streaming={stt_instance.capabilities.streaming}")
    session = AgentSession(
        stt=stt_instance,
        llm=llamacpp_llm,
        tts=WyomingTTS(
            uri=WYOMING_TTS_URL,
            voice=runtime["tts_voice"],
        ),
        vad=silero.VAD.load(),
        allow_interruptions=False,  # Prevent background noise from interrupting agent
    )
    logger.info(f"  Session STT: {type(session.stt).__name__}")

    # Set session reference for wake word callback
    _session_ref = session

    # ==========================================================================
    # Round-trip latency tracking
    # ==========================================================================

    _transcription_time: float | None = None

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev) -> None:
        nonlocal _transcription_time
        _transcription_time = time.perf_counter()
        logger.debug(f"User said: {ev.transcript[:80]}...")

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev) -> None:
        nonlocal _transcription_time
        if ev.new_state == "speaking" and _transcription_time is not None:
            latency_ms = (time.perf_counter() - _transcription_time) * 1000
            logger.info(f"ROUND-TRIP LATENCY: {latency_ms:.0f}ms (LLM + TTS)")
            _transcription_time = None

        # Notify wake word STT of agent state for silence timer management
        if isinstance(stt_instance, WakeWordGatedSTT):
            stt_instance.set_agent_busy(ev.new_state in ("thinking", "speaking"))

    async def _publish_tool_status(
        tool_used: bool,
        tool_names: list[str],
        tool_params: list[dict],
    ) -> None:
        """Publish tool usage status to frontend via data packet."""
        import json
        payload = json.dumps({
            "tool_used": tool_used,
            "tool_names": tool_names,
            "tool_params": tool_params,
        })

        try:
            await ctx.room.local_participant.publish_data(
                payload.encode("utf-8"),
                reliable=True,
                topic="tool_status",
            )
            logger.debug(f"Published tool status: used={tool_used}, names={tool_names}")
        except Exception as e:
            logger.warning(f"Failed to publish tool status: {e}")

    # ==========================================================================

    # Create agent with LlamaCppLLM and all MCP servers
    assistant = VoiceAssistant(
        llamacpp_llm=llamacpp_llm,
        mcp_servers=mcp_servers,
        n8n_workflow_tools=n8n_workflow_tools,
        n8n_workflow_name_map=n8n_workflow_name_map,
        n8n_base_url=n8n_base_url,
        on_tool_status=_publish_tool_status,
        tool_cache_size=runtime["tool_cache_size"],
        max_turns=runtime["max_turns"],
    )

    # Start session
    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(),
    )

    # Register session for webhook access
    session_registry.register(ctx.room.name, session, assistant)

    # Create event to wait for session close
    close_event = asyncio.Event()

    @session.on("close")
    def on_session_close(ev) -> None:
        logger.info(f"Session closed: {ev.reason}")
        close_event.set()

    try:
        # Send initial greeting
        await session.generate_reply(
            instructions="Greet the user briefly and let them know you're ready to help."
        )

        logger.info("Agent ready - listening for speech...")

        # Wait until session closes (room disconnects, etc.)
        await close_event.wait()

    finally:
        # Unregister session on cleanup
        session_registry.unregister(ctx.room.name)


# =============================================================================
# Model Preloading
# =============================================================================


def preload_models():
    """Preload LLM models on startup.

    Ensures models are ready before first user connection, avoiding
    delays on first request (especially important on HDDs).

    Note: Wyoming STT/TTS services handle their own model loading.
    """
    llamacpp_host = os.getenv("LLAMACPP_HOST", os.getenv("OLLAMA_HOST", "http://llama.home/v1"))
    llamacpp_model = os.getenv("OLLAMA_MODEL", os.getenv("LLAMACPP_MODEL", "gpt-oss-20b-mxfp4"))

    logger.info("Preloading models...")

    # Warm up llama.cpp LLM (OpenAI-compatible endpoint)
    try:
        logger.info(f"  Loading LLM: {llamacpp_model}")
        response = requests.post(
            f"{llamacpp_host}/chat/completions",
            json={
                "model": llamacpp_model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 10,
            },
            timeout=180
        )
        if response.status_code == 200:
            logger.info("  ✓ LLM ready")
        else:
            logger.warning(f"  LLM warmup returned {response.status_code}")
    except Exception as e:
        logger.warning(f"  Failed to preload LLM: {e}")


# =============================================================================
# Webhook Server
# =============================================================================

WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8889"))

# Global reference to webhook server task (started in entrypoint)
_webhook_server_task: asyncio.Task | None = None


async def start_webhook_server():
    """Start FastAPI webhook server in the current event loop.

    This runs the webhook server in the same event loop as the LiveKit agent,
    avoiding cross-thread async issues that cause 200x slower MCP calls.
    """
    import uvicorn
    from caal.webhooks import app

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=WEBHOOK_PORT,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    logger.debug(f"Starting webhook server on port {WEBHOOK_PORT}")
    await server.serve()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Preload models before starting worker
    preload_models()

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Suppress memory warnings (models use ~1GB, this is expected)
            job_memory_warn_mb=0,
        )
    )
