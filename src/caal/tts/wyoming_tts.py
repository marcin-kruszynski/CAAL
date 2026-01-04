"""Wyoming protocol TTS implementation for LiveKit agents."""

from __future__ import annotations

import asyncio
import logging
import uuid

import numpy as np
from livekit import rtc
from livekit.agents.tts import (
    AudioEmitter,
    SynthesizeStream,
    SynthesizedAudio,
    TTS,
    TTSCapabilities,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
)
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeVoice

from ..ndjson_log import log_event

logger = logging.getLogger(__name__)

# Wyoming typically uses 16kHz mono PCM
WYOMING_SAMPLE_RATE = 16000
WYOMING_CHANNELS = 1
WYOMING_WIDTH = 2  # 16-bit


class WyomingTTS(TTS):
    """Wyoming protocol TTS implementation."""

    def __init__(
        self,
        *,
        uri: str = "tcp://nabu.home:10200",
        voice: str | None = None,
        sample_rate: int = WYOMING_SAMPLE_RATE,
        num_channels: int = WYOMING_CHANNELS,
    ) -> None:
        """Initialize Wyoming TTS.

        Args:
            uri: Wyoming server URI (e.g., "tcp://nabu.home:10200")
            voice: Optional voice name (if not provided, uses server default)
            sample_rate: Audio sample rate (default: 16000)
            num_channels: Number of audio channels (default: 1)
        """
        super().__init__(
            capabilities=TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._uri = uri
        self._voice = voice
        self._info: Info | None = None

    @property
    def model(self) -> str:
        """Return the model name."""
        if self._info and self._info.tts:
            return self._info.tts.model or "wyoming-tts"
        return "wyoming-tts"

    @property
    def provider(self) -> str:
        """Return the provider name."""
        return "wyoming"

    async def _ensure_info(self) -> Info:
        """Get server info if not already cached."""
        if self._info is None:
            try:
                async with AsyncClient.from_uri(self._uri) as client:
                    await client.write_event(Describe().event())
                    event = await client.read_event()
                    if event and isinstance(event, Info):
                        self._info = event
                    else:
                        logger.warning("Failed to get info from Wyoming server")
                        self._info = Info()
            except Exception as e:
                logger.error(f"Error getting Wyoming TTS info: {e}")
                self._info = Info()
        return self._info

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        """Synthesize text to speech."""
        return WyomingTTSSynthesizeStream(
            tts=self,
            uri=self._uri,
            voice=self._voice,
            text=text,
            conn_options=conn_options,
        )

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        """Create a streaming TTS stream that accepts text incrementally.
        
        The framework will call push_text() on the returned stream to provide text.
        """
        return WyomingTTSStream(
            tts=self,
            uri=self._uri,
            voice=self._voice,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        """Close the TTS instance."""
        pass


class WyomingTTSSynthesizeStream(SynthesizeStream):
    """Streaming TTS synthesis using Wyoming protocol."""

    def __init__(
        self,
        tts: WyomingTTS,
        *,
        uri: str,
        voice: str | None,
        text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        """Initialize synthesis stream."""
        super().__init__(
            tts=tts,
            conn_options=conn_options,
        )
        self._uri = uri
        self._voice = voice
        self._text = text
        self._client: AsyncClient | None = None

    async def _run(self, output_emitter: AudioEmitter) -> None:
        """Main processing loop."""
        try:
            self._client = AsyncClient.from_uri(self._uri)
            await self._client.__aenter__()

            # Send synthesize request
            synthesize_event = Synthesize(
                text=self._text,
                voice=SynthesizeVoice(name=self._voice) if self._voice else None,
            )
            await self._client.write_event(synthesize_event.event())

            request_id = str(uuid.uuid4())
            # IMPORTANT: AudioEmitter must be initialized for LiveKit to manage segment end/flush.
            # We keep a stable output sample rate and resample incoming audio if needed.
            output_emitter.initialize(
                request_id=request_id,
                sample_rate=WYOMING_SAMPLE_RATE,
                num_channels=WYOMING_CHANNELS,
                mime_type="audio/pcm",
                stream=True,
            )
            in_rate: int | None = None
            in_width: int | None = None
            in_channels: int | None = None

            # Read audio chunks
            audio_started = False
            current_segment_id: str | None = None
            event_count = 0
            while True:
                event = await self._client.read_event()
                if event is None:
                    break
                event_count += 1
                # Wyoming client returns generic Event; decode by type
                if AudioStart.is_type(event.type):
                    in_rate = getattr(event, "data", {}).get("rate") or WYOMING_SAMPLE_RATE
                    in_width = getattr(event, "data", {}).get("width") or WYOMING_WIDTH
                    in_channels = getattr(event, "data", {}).get("channels") or WYOMING_CHANNELS
                    audio_started = True
                    # Start a new segment
                    current_segment_id = current_segment_id or f"{request_id}:{output_emitter.num_segments}"
                    output_emitter.start_segment(segment_id=current_segment_id)
                elif AudioChunk.is_type(event.type):
                    if not audio_started:
                        # Best-effort fallback if server doesn't send AudioStart first
                        in_rate = in_rate or WYOMING_SAMPLE_RATE
                        in_width = in_width or WYOMING_WIDTH
                        in_channels = in_channels or WYOMING_CHANNELS
                        # Start segment if we haven't already
                        current_segment_id = current_segment_id or f"{request_id}:{output_emitter.num_segments}"
                        output_emitter.start_segment(segment_id=current_segment_id)
                        audio_started = True

                    # AudioEmitter expects raw bytes matching mime_type ("audio/pcm")
                    chunk = AudioChunk.from_event(event)
                    if chunk.audio:
                        output_emitter.push(
                            _convert_pcm(
                                chunk.audio,
                                in_rate=in_rate or WYOMING_SAMPLE_RATE,
                                in_width=in_width or WYOMING_WIDTH,
                                in_channels=in_channels or WYOMING_CHANNELS,
                                out_rate=WYOMING_SAMPLE_RATE,
                                out_channels=WYOMING_CHANNELS,
                            )
                        )
                elif AudioStop.is_type(event.type):
                    # End the segment
                    output_emitter.end_segment()
                    break

        except Exception as e:
            logger.error(f"Wyoming TTS synthesis error: {e}")
            # Error will be handled by the stream infrastructure
        finally:
            if self._client:
                try:
                    await self._client.__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error closing Wyoming TTS client: {e}")

    def _convert_pcm_to_frame(self, pcm_data: bytes) -> rtc.AudioFrame | None:
        """Convert PCM bytes to LiveKit AudioFrame."""
        if not pcm_data:
            return None

        # Convert bytes to numpy array (int16)
        samples = np.frombuffer(pcm_data, dtype=np.int16)

        # Reshape for mono channel
        if WYOMING_CHANNELS == 1:
            samples = samples.reshape(-1, 1)

        # Create AudioFrame
        return rtc.AudioFrame(
            data=samples.tobytes(),
            sample_rate=WYOMING_SAMPLE_RATE,
            num_channels=WYOMING_CHANNELS,
            samples_per_channel=len(samples) // WYOMING_CHANNELS,
        )

    async def aclose(self) -> None:
        """Close the stream."""
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass


class WyomingTTSStream(SynthesizeStream):
    """Streaming TTS that accepts text incrementally via push_text()."""

    def __init__(
        self,
        tts: WyomingTTS,
        *,
        uri: str,
        voice: str | None,
        conn_options: APIConnectOptions,
    ) -> None:
        """Initialize streaming TTS stream."""
        super().__init__(
            tts=tts,
            conn_options=conn_options,
        )
        self._uri = uri
        self._voice = voice
        self._client: AsyncClient | None = None
        self._text_buffer: list[str] = []
        self._output_emitter: AudioEmitter | None = None
        self._text_ready = asyncio.Event()
        self._input_ended = False
        self._request_id: str | None = None
        self._emitter_initialized: bool = False

    async def _run(self, output_emitter: AudioEmitter) -> None:
        """Main processing loop - waits for text via push_text() and synthesizes."""
        self._output_emitter = output_emitter
        # IMPORTANT: AudioEmitter must be initialized, otherwise LiveKit will raise on end_input().
        # We keep a stable output sample rate and resample incoming audio if needed.
        self._request_id = str(uuid.uuid4())
        output_emitter.initialize(
            request_id=self._request_id,
            sample_rate=WYOMING_SAMPLE_RATE,
            num_channels=WYOMING_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        self._emitter_initialized = True
        
        # Wait for text to be pushed and input to end
        while not self._input_ended or self._text_buffer:
            await self._text_ready.wait()
            
            if self._text_buffer:
                await self._synthesize_buffered_text()
            
            self._text_ready.clear()
            
            # If input has ended and buffer is empty, we're done
            if self._input_ended and not self._text_buffer:
                break

    def push_text(self, text: str) -> None:
        """Add text to the synthesis buffer."""
        self._text_buffer.append(text)
        self._text_ready.set()

    def end_input(self) -> None:
        """Signal that text input is complete."""
        self._input_ended = True
        self._text_ready.set()

    async def flush(self) -> None:
        """Flush any pending text and synthesize."""
        if self._text_buffer and self._output_emitter:
            await self._synthesize_buffered_text()

    async def _synthesize_buffered_text(self) -> None:
        """Synthesize the buffered text."""
        if not self._text_buffer or not self._output_emitter:
            return

        # Combine all text chunks
        full_text = "".join(self._text_buffer)
        self._text_buffer.clear()

        if not full_text.strip():
            return

        try:
            log_event(
                hypothesis_id="TTS",
                location="wyoming_tts.py:WyomingTTSStream._synthesize_buffered_text",
                message="TTS synthesize start",
                data={"text_len": len(full_text), "text_preview": full_text.strip()[:80]},
            )
            # Create new client for each synthesis (Wyoming protocol is request/response)
            client = AsyncClient.from_uri(self._uri)
            await client.__aenter__()

            try:
                # Send synthesize request
                synthesize_event = Synthesize(
                    text=full_text,
                    voice=SynthesizeVoice(name=self._voice) if self._voice else None,
                )
                event_obj = synthesize_event.event()
                if event_obj is None:
                    logger.error("Failed to create synthesize event")
                    return
                await client.write_event(event_obj)

                # Read audio chunks
                audio_started = False
                current_segment_id: str | None = None
                event_count = 0
                while True:
                    event = await client.read_event()
                    if event is None:
                        break
                    event_count += 1
                    if event_count <= 3:
                        log_event(
                            hypothesis_id="TTS",
                            location="wyoming_tts.py:WyomingTTSStream._synthesize_buffered_text",
                            message="TTS event received",
                            data={"event_count": event_count, "event_type": getattr(event, "type", None)},
                        )
                    if AudioStart.is_type(event.type):
                        audio_started = True
                        in_rate = getattr(event, "data", {}).get("rate") or WYOMING_SAMPLE_RATE
                        in_width = getattr(event, "data", {}).get("width") or WYOMING_WIDTH
                        in_channels = getattr(event, "data", {}).get("channels") or WYOMING_CHANNELS
                        rid = self._request_id or "wyoming-tts"
                        current_segment_id = current_segment_id or f"{rid}:{self._output_emitter.num_segments}"
                        self._output_emitter.start_segment(segment_id=current_segment_id)
                    elif AudioChunk.is_type(event.type):
                        if not audio_started:
                            # Best-effort fallback if server doesn't send AudioStart first
                            in_rate = WYOMING_SAMPLE_RATE
                            in_width = WYOMING_WIDTH
                            in_channels = WYOMING_CHANNELS
                            rid = self._request_id or "wyoming-tts"
                            current_segment_id = current_segment_id or f"{rid}:{self._output_emitter.num_segments}"
                            self._output_emitter.start_segment(segment_id=current_segment_id)
                            audio_started = True

                        # AudioEmitter expects raw bytes matching mime_type ("audio/pcm")
                        chunk = AudioChunk.from_event(event)
                        if chunk.audio:
                            self._output_emitter.push(
                                _convert_pcm(
                                    chunk.audio,
                                    in_rate=in_rate,
                                    in_width=in_width,
                                    in_channels=in_channels,
                                    out_rate=WYOMING_SAMPLE_RATE,
                                    out_channels=WYOMING_CHANNELS,
                                )
                            )
                    elif AudioStop.is_type(event.type):
                        self._output_emitter.end_segment()
                        break
            finally:
                # Always close the client
                try:
                    await client.__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error closing Wyoming TTS client: {e}")
        except Exception as e:
            logger.error(f"Wyoming TTS stream synthesis error: {e}", exc_info=True)

    def _convert_pcm_to_frame(self, pcm_data: bytes) -> rtc.AudioFrame | None:
        """Convert PCM bytes to LiveKit AudioFrame."""
        if not pcm_data:
            return None

        # Convert bytes to numpy array (int16)
        samples = np.frombuffer(pcm_data, dtype=np.int16)

        # Reshape for mono channel
        if WYOMING_CHANNELS == 1:
            samples = samples.reshape(-1, 1)

        # Create AudioFrame
        return rtc.AudioFrame(
            data=samples.tobytes(),
            sample_rate=WYOMING_SAMPLE_RATE,
            num_channels=WYOMING_CHANNELS,
            samples_per_channel=len(samples) // WYOMING_CHANNELS,
        )


def _convert_pcm(
    pcm: bytes,
    *,
    in_rate: int,
    in_width: int,
    in_channels: int,
    out_rate: int,
    out_channels: int,
) -> bytes:
    """
    Convert incoming PCM16 (Wyoming AudioChunk) to the output format expected by AudioEmitter.
    - Handles channel downmixing to mono
    - Handles resampling via linear interpolation
    """
    if not pcm:
        return b""
    # Only support 16-bit PCM for now
    if in_width != 2:
        return pcm

    samples = np.frombuffer(pcm, dtype=np.int16)

    if in_channels > 1:
        try:
            samples = samples.reshape(-1, in_channels)
            # Downmix to mono by averaging
            samples = samples.mean(axis=1)
        except Exception:
            # If reshape fails, just pass through
            pass

    # Ensure 1-D
    samples = np.asarray(samples).astype(np.float32).reshape(-1)

    if in_rate != out_rate and len(samples) > 1:
        num_out = int(len(samples) * out_rate / in_rate)
        if num_out > 1:
            x_old = np.linspace(0, len(samples) - 1, num=len(samples))
            x_new = np.linspace(0, len(samples) - 1, num=num_out)
            samples = np.interp(x_new, x_old, samples).astype(np.float32)

    # Channels: we currently only emit mono
    if out_channels != 1:
        # Best-effort: duplicate mono to N channels
        samples_i16 = np.clip(samples, -32768, 32767).astype(np.int16)
        samples_i16 = np.repeat(samples_i16[:, None], out_channels, axis=1).reshape(-1)
        return samples_i16.tobytes()

    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

    async def aclose(self) -> None:
        """Close the stream."""
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass

