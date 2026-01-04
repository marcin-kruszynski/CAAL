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

from livekit import agents, rtc
from livekit.agents import AgentSession, Agent, mcp
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
from caal.ndjson_log import log_event

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
        "model": settings.get("model") or os.getenv("LLAMACPP_MODEL", "gpt-oss-20b-mxfp4"),
        "temperature": settings.get("temperature", float(os.getenv("LLAMACPP_TEMPERATURE", "0.7"))),
        "num_ctx": settings.get("num_ctx", int(os.getenv("LLAMACPP_NUM_CTX", "8192"))),
        "max_turns": settings.get("max_turns", int(os.getenv("LLAMACPP_MAX_TURNS", "20"))),
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
    try:
        logger.info("Entrypoint called - starting initialization")
        log_event(
            hypothesis_id="S",
            location="voice_agent.py:entrypoint",
            message="Entrypoint called",
            data={"room_name": getattr(ctx.room, "name", None)},
        )
        
        # Start webhook server in the same event loop (first job only)
        global _webhook_server_task
        if _webhook_server_task is None:
            logger.info("Starting webhook server...")
            _webhook_server_task = asyncio.create_task(start_webhook_server())

        logger.info(f"Joining room: {ctx.room.name}")
        await ctx.connect()
        logger.info("Connected to room")
        log_event(
            hypothesis_id="S",
            location="voice_agent.py:entrypoint",
            message="Connected to room",
            data={"room_name": getattr(ctx.room, "name", None)},
        )

        # Load MCP servers from config
        logger.info("Loading MCP servers...")
        mcp_servers = {}
        try:
            logger.info("Calling load_mcp_config()...")
            mcp_configs = load_mcp_config()
            logger.info(f"load_mcp_config() returned {len(mcp_configs)} configs: {[c.name for c in mcp_configs]}")
            logger.info("Calling initialize_mcp_servers()...")
            try:
                mcp_servers = await initialize_mcp_servers(mcp_configs)
                logger.info(f"MCP servers loaded: {list(mcp_servers.keys())}")
            except (RuntimeError, Exception) as mcp_error:
                # Catch any errors during MCP initialization (including SSL/cancel scope errors)
                logger.error(f"Error during MCP server initialization: {mcp_error}", exc_info=True)
                # Continue without MCP servers - agent can still function
                mcp_servers = {}
                logger.warning("Continuing without MCP servers due to initialization errors")
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}", exc_info=True)
            # Continue without MCP servers
            mcp_servers = {}

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
        logger.info("Loading runtime settings...")
        runtime = get_runtime_settings()
        logger.info(f"Runtime settings loaded: model={runtime['model']}, tts_voice={runtime['tts_voice']}")

        # Create LlamaCppLLM instance (config lives here, accessed via self.llm in agent)
        logger.info("Creating LlamaCppLLM instance...")
        llamacpp_llm = LlamaCppLLM(
            model=runtime["model"],
            temperature=runtime["temperature"],
            num_ctx=runtime["num_ctx"],
            base_url=LLAMACPP_HOST,
        )
        logger.info("LlamaCppLLM instance created")

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
        logger.info("Creating AgentSession...")
        logger.info(f"  STT instance type: {type(stt_instance).__name__}")
        logger.info(f"  STT capabilities: streaming={stt_instance.capabilities.streaming}")
        logger.info(f"  TTS URI: {WYOMING_TTS_URL}")
        logger.info(f"  TTS voice: {runtime['tts_voice']}")
        
        try:
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
            logger.info(f"  Session created successfully. STT: {type(session.stt).__name__}")
        except Exception as e:
            logger.error(f"Failed to create AgentSession: {e}", exc_info=True)
            raise

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
            logger.info(f"User said: {ev.transcript[:80]}...")
            log_event(
                hypothesis_id="B",
                location="voice_agent.py:on_user_input_transcribed",
                message="User input transcribed",
                data={"transcript_preview": (ev.transcript or "")[:160]},
            )

        @session.on("agent_state_changed")
        def on_agent_state_changed(ev) -> None:
            nonlocal _transcription_time
            # Always log state transitions (helps debug "hang after tool" where agent never returns to listening)
            log_event(
                hypothesis_id="ST",
                location="voice_agent.py:on_agent_state_changed",
                message="Agent state changed",
                data={"old_state": getattr(ev, "old_state", None), "new_state": getattr(ev, "new_state", None)},
            )
            if ev.new_state == "speaking" and _transcription_time is not None:
                latency_ms = (time.perf_counter() - _transcription_time) * 1000
                logger.info(f"ROUND-TRIP LATENCY: {latency_ms:.0f}ms (LLM + TTS)")
                log_event(
                    hypothesis_id="C",
                    location="voice_agent.py:on_agent_state_changed",
                    message="Round-trip latency",
                    data={"latency_ms": int(latency_ms)},
                )
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

        # Wait for audio tracks to be published and subscribe to them BEFORE starting the session
        # AgentSession needs tracks to be subscribed before it starts to receive audio
        logger.info("Waiting for audio tracks to be published...")
        max_wait_time = 10.0  # Maximum time to wait for tracks
        wait_interval = 0.5
        waited_time = 0.0
        audio_track_found = False
        
        while waited_time < max_wait_time:
            # Check for audio tracks
            for participant in ctx.room.remote_participants.values():
                for publication in participant.track_publications.values():
                    if publication.kind == rtc.TrackKind.KIND_AUDIO:
                        audio_track_found = True
                        if not publication.subscribed:
                            try:
                                publication.set_subscribed(True)
                                logger.info(f"Subscribed to audio track {publication.sid} from {participant.identity} before session start")
                            except Exception as e:
                                logger.error(f"Failed to subscribe to audio track: {e}", exc_info=True)
                        # Wait for track object to be available
                        if publication.track is not None:
                            logger.info(f"Audio track {publication.sid} is ready (subscribed={publication.subscribed}, track available)")
                            # Found a ready track, break out of loops
                            break
                if audio_track_found and any(pub.kind == rtc.TrackKind.KIND_AUDIO and pub.track is not None 
                                            for pub in participant.track_publications.values()):
                    break
            if audio_track_found:
                # Wait a bit more for subscription to complete
                await asyncio.sleep(1.0)
                break
            await asyncio.sleep(wait_interval)
            waited_time += wait_interval
        
        if not audio_track_found:
            logger.warning(f"No audio tracks found after waiting {waited_time}s, starting session anyway")
        else:
            logger.info(f"Audio track(s) found and subscribed, starting session")
        
        # Set up track published/subscribed handlers BEFORE starting session
        # This ensures we catch tracks published after session starts
        @ctx.room.on("track_published")
        def on_track_published(publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
            """Log when tracks are published and subscribe to audio tracks."""
            # Manually subscribe to audio tracks if not already subscribed
            if publication.kind == rtc.TrackKind.KIND_AUDIO and not publication.subscribed:
                try:
                    publication.set_subscribed(True)
                    logger.info(f"Manually subscribed to audio track from {participant.identity}")
                    # Bind AgentSession input to this participant so RoomIO starts forwarding frames
                    try:
                        session.input.audio.set_participant(participant)
                        logger.info(
                            f"AgentSession input participant set to {participant.identity} (track_published)"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to set AgentSession input participant (track_published): {e}",
                            exc_info=True,
                        )
                    # Wait a moment and verify the track is available
                    import asyncio
                    async def verify_and_forward_track():
                        await asyncio.sleep(0.5)
                        if publication.track is not None:
                            track = publication.track
                            logger.info(f"Track {publication.sid} is available after subscription: {type(track)}")
                            # Manually forward audio frames from the track to AgentSession's input
                            # AgentSession might not automatically receive audio from tracks subscribed after it starts
                            if isinstance(track, rtc.RemoteAudioTrack):
                                logger.info(f"Setting up AudioStream to forward frames from track {publication.sid} to AgentSession")
                                frame_count = 0
                                
                                async def forward_audio_frames():
                                    nonlocal frame_count
                                    try:
                                        # Create AudioStream from the track to receive frames
                                        audio_stream = rtc.AudioStream(track)
                                        logger.info(f"AudioStream created for track {publication.sid}, starting to forward frames")
                                        
                                        async for audio_frame in audio_stream:
                                            frame_count += 1
                                            if frame_count == 1 or frame_count % 100 == 0:
                                                logger.info(f"Received audio frame {frame_count} from track {publication.sid}, forwarding to session")
                                            try:
                                                # Forward frame to AgentSession's input
                                                # Forward frame directly to STT stream's input
                                                # AgentSession's input.audio doesn't have capture_frame, so we push to STT stream directly
                                                if frame_count == 1:
                                                    # Debug: log STT and session attributes on first frame
                                                    logger.info(f"=== DEBUG: Frame 1 - Inspecting session ===")
                                                    logger.info(f"Session type: {type(session)}")
                                                    logger.info(f"Session has stt: {hasattr(session, 'stt')}")
                                                    if hasattr(session, 'stt'):
                                                        logger.info(f"STT type: {type(session.stt)}")
                                                        logger.info(f"STT dir (first 20): {[x for x in dir(session.stt) if not x.startswith('__')][:20]}")
                                                        logger.info(f"STT _active_stream: {getattr(session.stt, '_active_stream', 'NOT FOUND')}")
                                                        logger.info(f"STT _stream: {getattr(session.stt, '_stream', 'NOT FOUND')}")
                                                    logger.info(f"Session has input: {hasattr(session, 'input')}")
                                                    if hasattr(session, 'input'):
                                                        logger.info(f"Session input type: {type(session.input)}")
                                                        logger.info(f"Session input dir (first 20): {[x for x in dir(session.input) if not x.startswith('__')][:20]}")
                                                        if hasattr(session.input, 'audio'):
                                                            logger.info(f"Session input.audio type: {type(session.input.audio)}")
                                                            logger.info(f"Session input.audio dir (first 20): {[x for x in dir(session.input.audio) if not x.startswith('__')][:20]}")
                                                    logger.info(f"=== END DEBUG ===")
                                                
                                                if hasattr(session, 'stt') and session.stt is not None:
                                                    # Get the active STT stream - try different attribute names
                                                    # WakeWordGatedSTT uses _active_stream, others might use _stream
                                                    stt_stream = getattr(session.stt, '_active_stream', None) or getattr(session.stt, '_stream', None)
                                                    
                                                    if stt_stream is not None:
                                                        # Try push_frame first
                                                        if hasattr(stt_stream, 'push_frame'):
                                                            stt_stream.push_frame(audio_frame)
                                                        else:
                                                            # Try accessing _input_ch directly
                                                            input_ch = getattr(stt_stream, '_input_ch', None)
                                                            if input_ch is not None:
                                                                try:
                                                                    input_ch.send_nowait(audio_frame)
                                                                except Exception as e:
                                                                    logger.warning(f"Error sending frame to input_ch: {e}")
                                                            else:
                                                                if frame_count <= 5:
                                                                    logger.warning(f"STT stream {type(stt_stream)} has no push_frame or _input_ch")
                                                    else:
                                                        if frame_count <= 5:
                                                            logger.warning(f"STT {type(session.stt)} has no _active_stream or _stream attribute")
                                                else:
                                                    if frame_count <= 5:
                                                        logger.warning("Session STT is not available")
                                            except Exception as e:
                                                logger.error(f"Error forwarding audio frame to session: {e}", exc_info=True)
                                    except Exception as e:
                                        logger.error(f"Error in audio stream forwarding: {e}", exc_info=True)
                                
                                # Start forwarding frames in background
                                asyncio.create_task(forward_audio_frames())
                                logger.info(f"Audio frame forwarding task started for track {publication.sid}")
                        else:
                            logger.warning(f"Track {publication.sid} is NOT available after subscription")
                    asyncio.create_task(verify_and_forward_track())
                except Exception as e:
                    logger.error(f"Failed to subscribe to audio track: {e}", exc_info=True)

        @ctx.room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
            """Log when tracks are subscribed and verify AgentSession receives audio."""
            # AgentSession should automatically receive audio from subscribed tracks
            # Verify that the track is an audio track and log for debugging
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"Audio track {publication.sid} subscribed from {participant.identity}")
                logger.info(f"AgentSession should now receive audio from this track automatically")
                logger.info(f"Session input.audio should be receiving frames from room's subscribed audio tracks")

                # Critical: bind input stream to the participant identity so RoomIO attaches
                try:
                    session.input.audio.set_participant(participant)
                    logger.info(
                        f"AgentSession input participant set to {participant.identity} (track_subscribed)"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to set AgentSession input participant (track_subscribed): {e}",
                        exc_info=True,
                    )
                
                # Also set up AudioStream forwarding as a backup
                if isinstance(track, rtc.RemoteAudioTrack) and session is not None:
                    logger.info(f"Setting up AudioStream backup forwarding for track {publication.sid}")
                    frame_count = 0
                    
                    async def forward_audio_frames_backup():
                        nonlocal frame_count
                        try:
                            # Create AudioStream from the track to receive frames
                            audio_stream = rtc.AudioStream(track)
                            logger.info(f"AudioStream (backup) created for track {publication.sid}, starting to forward frames")
                            
                            async for audio_frame in audio_stream:
                                frame_count += 1
                                if frame_count == 1 or frame_count % 100 == 0:
                                    logger.info(f"Received audio frame {frame_count} from track {publication.sid} (backup), forwarding to session")
                                try:
                                    # Forward frame directly to STT stream's input
                                    # AgentSession's input.audio doesn't have capture_frame, so we push to STT stream directly
                                    if hasattr(session, 'stt') and session.stt is not None:
                                        # Get the active STT stream and push frame to it
                                        # The STT stream should have a push_frame method or we can access _input_ch
                                        # Try to get the stream from the STT
                                        stt_stream = getattr(session.stt, '_stream', None)
                                        if stt_stream is not None and hasattr(stt_stream, 'push_frame'):
                                            stt_stream.push_frame(audio_frame)
                                        else:
                                            # Try accessing _input_ch directly if available
                                            input_ch = getattr(stt_stream, '_input_ch', None)
                                            if input_ch is not None:
                                                input_ch.send_nowait(audio_frame)
                                            else:
                                                logger.warning(f"Could not find way to push frame to STT stream: {type(stt_stream)}")
                                    else:
                                        logger.warning("Session STT is not available")
                                except Exception as e:
                                    logger.error(f"Error forwarding audio frame to session (backup): {e}", exc_info=True)
                        except Exception as e:
                            logger.error(f"Error in audio stream forwarding (backup): {e}", exc_info=True)
                    
                    # Start forwarding frames in background
                    asyncio.create_task(forward_audio_frames_backup())
                    logger.info(f"Audio frame forwarding task (backup) started for track {publication.sid}")
        
        # Also check for existing tracks that might not have been subscribed
        async def check_and_subscribe_existing_tracks():
            """Check for existing audio tracks and subscribe to them."""
            await asyncio.sleep(1)  # Wait a bit for tracks to be published
            for participant in ctx.room.remote_participants.values():
                for publication in participant.track_publications.values():
                    if publication.kind == rtc.TrackKind.KIND_AUDIO and not publication.subscribed:
                        try:
                            publication.set_subscribed(True)
                            logger.info(f"Subscribed to existing audio track from {participant.identity}")
                        except Exception as e:
                            logger.error(f"Failed to subscribe to existing audio track: {e}", exc_info=True)
        
        # Schedule check for existing tracks
        asyncio.create_task(check_and_subscribe_existing_tracks())
        
        # Start session
        logger.info("Starting AgentSession...")
        try:
            log_event(
                hypothesis_id="F",
                location="voice_agent.py:session.start",
                message="About to start AgentSession",
                data={
                    "room_name": getattr(ctx.room, "name", None),
                    "remote_participants": len(getattr(ctx.room, "remote_participants", {}) or {}),
                },
            )
            await session.start(
                room=ctx.room,
                agent=assistant,
            )
            
            # Debug: After session.start(), check if we can access the STT stream
            logger.info("AgentSession started successfully")
            log_event(
                hypothesis_id="F",
                location="voice_agent.py:session.start",
                message="AgentSession started",
                data={"room_name": getattr(ctx.room, "name", None)},
            )
        except Exception as e:
            logger.error(f"Failed to start AgentSession: {e}", exc_info=True)
            log_event(
                hypothesis_id="F",
                location="voice_agent.py:session.start",
                message="AgentSession start failed",
                data={"error": str(e)[:160]},
            )
            raise

        # Bind AgentSession input to the remote participant identity.
        #
        # Runtime evidence:
        # - livekit's `_ParticipantInputStream._on_track_available()` ONLY attaches if
        #   `self._participant_identity == participant.identity`
        # - `_participant_identity` defaults to None until `set_participant(...)` is called
        #   (see livekit.agents.voice.room_io._input._ParticipantInputStream)
        #
        # If we don't call `set_participant`, no room audio will ever be forwarded into the
        # AgentSession STT pipeline.
        async def setup_participant_for_existing_tracks() -> None:
            await asyncio.sleep(0.5)  # allow room publications to populate
            try:
                identities = [p.identity for p in ctx.room.remote_participants.values()]
                logger.info(
                    f"setup_participant_for_existing_tracks: remote_participants={identities}"
                )
                # Bind to the first remote participant (RoomIO will pick the first available mic track)
                participant = next(iter(ctx.room.remote_participants.values()), None)
                if participant is None:
                    logger.warning(
                        "setup_participant_for_existing_tracks: no remote participants yet"
                    )
                    return
                try:
                    session.input.audio.set_participant(participant)
                    logger.info(
                        f"AgentSession input participant set to {participant.identity} (startup)"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to set AgentSession input participant (startup): {e}",
                        exc_info=True,
                    )
            except Exception as e:
                logger.error(f"setup_participant_for_existing_tracks failed: {e}", exc_info=True)

        asyncio.create_task(setup_participant_for_existing_tracks())

        # Listen for track published events and manually subscribe to audio tracks
        @ctx.room.on("track_published")
        def on_track_published(publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
            """Log when tracks are published and subscribe to audio tracks."""
            # Manually subscribe to audio tracks if not already subscribed
            if publication.kind == rtc.TrackKind.KIND_AUDIO and not publication.subscribed:
                try:
                    publication.set_subscribed(True)
                    logger.info(f"Manually subscribed to audio track from {participant.identity}")
                    # Wait a moment and verify the track is available
                    import asyncio
                    async def verify_and_forward_track():
                        await asyncio.sleep(0.5)
                        if publication.track is not None:
                            track = publication.track
                            logger.info(f"Track {publication.sid} is available after subscription: {type(track)}")
                            # Manually forward audio frames from the track to AgentSession's input
                            # AgentSession might not automatically receive audio from tracks subscribed after it starts
                            if isinstance(track, rtc.RemoteAudioTrack):
                                logger.info(f"Setting up AudioStream to forward frames from track {publication.sid} to AgentSession")
                                frame_count = 0
                                
                                async def forward_audio_frames():
                                    nonlocal frame_count
                                    try:
                                        # Create AudioStream from the track to receive frames
                                        audio_stream = rtc.AudioStream(track)
                                        logger.info(f"AudioStream created for track {publication.sid}, starting to forward frames")
                                        
                                        async for audio_frame in audio_stream:
                                            frame_count += 1
                                            if frame_count == 1 or frame_count % 100 == 0:
                                                logger.info(f"Received audio frame {frame_count} from track {publication.sid}, forwarding to session")
                                            try:
                                                # Forward frame to AgentSession's input
                                                # Forward frame directly to STT stream's input
                                                # AgentSession's input.audio doesn't have capture_frame, so we push to STT stream directly
                                                if frame_count == 1:
                                                    # Debug: log STT and session attributes on first frame
                                                    logger.info(f"=== DEBUG: Frame 1 - Inspecting session ===")
                                                    logger.info(f"Session type: {type(session)}")
                                                    logger.info(f"Session has stt: {hasattr(session, 'stt')}")
                                                    if hasattr(session, 'stt'):
                                                        logger.info(f"STT type: {type(session.stt)}")
                                                        logger.info(f"STT dir (first 20): {[x for x in dir(session.stt) if not x.startswith('__')][:20]}")
                                                        logger.info(f"STT _active_stream: {getattr(session.stt, '_active_stream', 'NOT FOUND')}")
                                                        logger.info(f"STT _stream: {getattr(session.stt, '_stream', 'NOT FOUND')}")
                                                    logger.info(f"Session has input: {hasattr(session, 'input')}")
                                                    if hasattr(session, 'input'):
                                                        logger.info(f"Session input type: {type(session.input)}")
                                                        logger.info(f"Session input dir (first 20): {[x for x in dir(session.input) if not x.startswith('__')][:20]}")
                                                        if hasattr(session.input, 'audio'):
                                                            logger.info(f"Session input.audio type: {type(session.input.audio)}")
                                                            logger.info(f"Session input.audio dir (first 20): {[x for x in dir(session.input.audio) if not x.startswith('__')][:20]}")
                                                    logger.info(f"=== END DEBUG ===")
                                                
                                                if hasattr(session, 'stt') and session.stt is not None:
                                                    # Get the active STT stream - try different attribute names
                                                    # WakeWordGatedSTT uses _active_stream, others might use _stream
                                                    stt_stream = getattr(session.stt, '_active_stream', None) or getattr(session.stt, '_stream', None)
                                                    
                                                    if stt_stream is not None:
                                                        # Try push_frame first
                                                        if hasattr(stt_stream, 'push_frame'):
                                                            stt_stream.push_frame(audio_frame)
                                                        else:
                                                            # Try accessing _input_ch directly
                                                            input_ch = getattr(stt_stream, '_input_ch', None)
                                                            if input_ch is not None:
                                                                try:
                                                                    input_ch.send_nowait(audio_frame)
                                                                except Exception as e:
                                                                    logger.warning(f"Error sending frame to input_ch: {e}")
                                                            else:
                                                                if frame_count <= 5:
                                                                    logger.warning(f"STT stream {type(stt_stream)} has no push_frame or _input_ch")
                                                    else:
                                                        if frame_count <= 5:
                                                            logger.warning(f"STT {type(session.stt)} has no _active_stream or _stream attribute")
                                                else:
                                                    if frame_count <= 5:
                                                        logger.warning("Session STT is not available")
                                            except Exception as e:
                                                logger.error(f"Error forwarding audio frame to session: {e}", exc_info=True)
                                    except Exception as e:
                                        logger.error(f"Error in audio stream forwarding: {e}", exc_info=True)
                                
                                # Start forwarding frames in background
                                asyncio.create_task(forward_audio_frames())
                                logger.info(f"Audio frame forwarding task started for track {publication.sid}")
                        else:
                            logger.warning(f"Track {publication.sid} is NOT available after subscription")
                    asyncio.create_task(verify_and_forward_track())
                except Exception as e:
                    logger.error(f"Failed to subscribe to audio track: {e}", exc_info=True)

        @ctx.room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
            """Log when tracks are subscribed."""
            # AgentSession should automatically receive audio from subscribed tracks
            # Verify that the track is an audio track and log for debugging
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"Audio track {publication.sid} subscribed from {participant.identity}")
                logger.info(f"AgentSession should now receive audio from this track automatically")
                logger.info(f"Session input.audio should be receiving frames from room's subscribed audio tracks")
        
        # Also check for existing tracks that might not have been subscribed
        async def check_and_subscribe_existing_tracks():
            """Check for existing audio tracks and subscribe to them."""
            await asyncio.sleep(1)  # Wait a bit for tracks to be published
            for participant in ctx.room.remote_participants.values():
                for publication in participant.track_publications.values():
                    if publication.kind == rtc.TrackKind.KIND_AUDIO and not publication.subscribed:
                        try:
                            publication.set_subscribed(True)
                            logger.info(f"Subscribed to existing audio track from {participant.identity}")
                        except Exception as e:
                            logger.error(f"Failed to subscribe to existing audio track: {e}", exc_info=True)
        
        # Schedule check for existing tracks
        asyncio.create_task(check_and_subscribe_existing_tracks())

        # Register session for webhook access
        logger.info("Registering session in registry...")
        session_registry.register(ctx.room.name, session, assistant)
        logger.info("Session registered")

        # Create event to wait for session close
        close_event = asyncio.Event()

        @session.on("close")
        def on_session_close(ev) -> None:
            logger.info(f"Session closed: {ev.reason}")
            log_event(
                hypothesis_id="D",
                location="voice_agent.py:on_session_close",
                message="Session closed",
                data={"reason": str(getattr(ev, "reason", ""))[:120]},
            )
            close_event.set()

        try:
            # Send initial greeting
            logger.info("Sending initial greeting...")
            await session.generate_reply(
                instructions="Greet the user briefly and let them know you're ready to help.",
                # Critical: while an uninterruptible speech is playing, AgentActivity.push_audio()
                # discards user audio (discard_audio_if_uninterruptible default True). This makes
                # it look like the mic works but "nothing happens".
                allow_interruptions=True,
            )
            logger.info("Initial greeting sent")

            logger.info("Agent ready - listening for speech...")
            log_event(
                hypothesis_id="E",
                location="voice_agent.py:entrypoint",
                message="Agent ready",
                data={"room_name": getattr(ctx.room, "name", None)},
            )

            # Wait until session closes (room disconnects, etc.)
            await close_event.wait()
            logger.info("Session close event received")

        except Exception as e:
            logger.error(f"Error in session main loop: {e}", exc_info=True)
            raise
        finally:
            # Unregister session on cleanup
            logger.info("Unregistering session...")
            session_registry.unregister(ctx.room.name)
            logger.info("Session unregistered")
    except Exception as e:
        logger.error(f"Fatal error in entrypoint: {e}", exc_info=True)
        raise


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
