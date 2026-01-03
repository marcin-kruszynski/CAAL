"""Wyoming protocol STT implementation for LiveKit agents."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from livekit import rtc
from livekit.agents.stt import (
    RecognizeStream,
    SpeechEvent,
    SpeechEventType,
    STT,
    STTCapabilities,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import aio
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.info import Describe, Info
from wyoming.stt import Transcript

logger = logging.getLogger(__name__)

# Wyoming typically uses 16kHz mono PCM
WYOMING_SAMPLE_RATE = 16000
WYOMING_CHANNELS = 1
WYOMING_WIDTH = 2  # 16-bit


class WyomingSTT(STT):
    """Wyoming protocol STT implementation."""

    def __init__(
        self,
        *,
        uri: str = "tcp://nabu.home:10300",
        model: str | None = None,
    ) -> None:
        """Initialize Wyoming STT.

        Args:
            uri: Wyoming server URI (e.g., "tcp://nabu.home:10300")
            model: Optional model name (if not provided, uses server default)
        """
        super().__init__(
            capabilities=STTCapabilities(streaming=True, interim_results=False)
        )
        self._uri = uri
        self._model = model
        self._info: Info | None = None

    @property
    def model(self) -> str:
        """Return the model name."""
        if self._model:
            return self._model
        if self._info and self._info.stt:
            return self._info.stt.model or "wyoming-stt"
        return "wyoming-stt"

    @property
    def provider(self) -> str:
        """Return the provider name."""
        return "wyoming"

    async def _ensure_info(self) -> Info:
        """Get server info if not already cached."""
        if self._info is None:
            async with AsyncClient(self._uri) as client:
                await client.write_event(Describe().event())
                event = await client.read_event()
                if event and isinstance(event, Info):
                    self._info = event
                else:
                    logger.warning("Failed to get info from Wyoming server")
                    self._info = Info()
        return self._info

    async def _recognize_impl(
        self,
        buffer: rtc.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        """Perform non-streaming recognition."""
        # Convert audio buffer to PCM format
        audio_data = self._convert_audio_to_pcm(buffer)

        async with AsyncClient(self._uri) as client:
            # Send audio start
            await client.write_event(
                AudioStart(
                    rate=WYOMING_SAMPLE_RATE,
                    width=WYOMING_WIDTH,
                    channels=WYOMING_CHANNELS,
                ).event()
            )

            # Send audio data
            await client.write_event(AudioChunk(audio=audio_data).event())

            # Send audio stop
            await client.write_event(AudioStop().event())

            # Read transcription
            while True:
                event = await client.read_event()
                if event is None:
                    break
                if isinstance(event, Transcript):
                    return SpeechEvent(
                        type=SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[SpeechEvent.Alternative(text=event.text)],
                        language=event.language,
                    )

        # No transcription received
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechEvent.Alternative(text="")],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        """Create a streaming recognition stream."""
        return WyomingSTTStream(
            stt=self,
            uri=self._uri,
            model=self._model,
            language=language,
            conn_options=conn_options,
        )

    def _convert_audio_to_pcm(self, buffer: rtc.AudioBuffer) -> bytes:
        """Convert LiveKit AudioBuffer to PCM bytes."""
        # Get audio data as numpy array
        samples = np.frombuffer(buffer.data, dtype=np.int16)

        # Handle multi-channel by taking first channel or converting to mono
        if buffer.num_channels > 1:
            # Reshape to channels x samples
            samples = samples.reshape(-1, buffer.num_channels)
            # Take first channel or average (simple mono conversion)
            samples = samples[:, 0]

        # Resample if needed (simple linear interpolation for now)
        if buffer.sample_rate != WYOMING_SAMPLE_RATE:
            # Simple resampling: linear interpolation
            num_samples = int(
                len(samples) * WYOMING_SAMPLE_RATE / buffer.sample_rate
            )
            indices = np.linspace(0, len(samples) - 1, num_samples)
            samples = np.interp(indices, np.arange(len(samples)), samples)

        # Convert to int16 PCM bytes
        return samples.astype(np.int16).tobytes()

    async def aclose(self) -> None:
        """Close the STT instance."""
        pass


class WyomingSTTStream(RecognizeStream):
    """Streaming STT using Wyoming protocol."""

    def __init__(
        self,
        stt: WyomingSTT,
        *,
        uri: str,
        model: str | None,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> None:
        """Initialize streaming STT."""
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=WYOMING_SAMPLE_RATE,
        )
        self._uri = uri
        self._model = model
        self._language = language
        self._client: AsyncClient | None = None
        self._audio_buffer: list[bytes] = []

    async def _run(self) -> None:
        """Main processing loop."""
        try:
            self._client = AsyncClient(self._uri)
            await self._client.__aenter__()

            # Send audio start
            await self._client.write_event(
                AudioStart(
                    rate=WYOMING_SAMPLE_RATE,
                    width=WYOMING_WIDTH,
                    channels=WYOMING_CHANNELS,
                ).event()
            )

            # Process audio frames
            async def _process_audio() -> None:
                """Process incoming audio frames."""
                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        # Flush: send accumulated audio and stop
                        if self._audio_buffer:
                            audio_data = b"".join(self._audio_buffer)
                            await self._client.write_event(
                                AudioChunk(audio=audio_data).event()
                            )
                            self._audio_buffer = []
                        await self._client.write_event(AudioStop().event())
                        continue

                    frame: rtc.AudioFrame = data
                    audio_data = self._convert_frame_to_pcm(frame)
                    self._audio_buffer.append(audio_data)

                    # Send chunks periodically (every ~100ms)
                    if len(b"".join(self._audio_buffer)) >= (
                        WYOMING_SAMPLE_RATE * WYOMING_WIDTH * WYOMING_CHANNELS * 0.1
                    ):
                        audio_chunk = b"".join(self._audio_buffer)
                        await self._client.write_event(
                            AudioChunk(audio=audio_chunk).event()
                        )
                        self._audio_buffer = []

            async def _read_transcripts() -> None:
                """Read transcription events from Wyoming."""
                while True:
                    try:
                        event = await self._client.read_event()
                        if event is None:
                            break
                        if isinstance(event, Transcript):
                            if event.text:
                                # Emit final transcript
                                self._event_ch.send_nowait(
                                    SpeechEvent(
                                        type=SpeechEventType.FINAL_TRANSCRIPT,
                                        alternatives=[
                                            SpeechEvent.Alternative(text=event.text)
                                        ],
                                        language=event.language,
                                    )
                                )
                    except Exception as e:
                        logger.error(f"Error reading transcript: {e}")
                        break

            # Run both tasks
            await asyncio.gather(
                _process_audio(),
                _read_transcripts(),
                return_exceptions=True,
            )

        except Exception as e:
            logger.error(f"Wyoming STT stream error: {e}")
        finally:
            if self._client:
                try:
                    # Send final audio stop if we have buffered audio
                    if self._audio_buffer:
                        audio_data = b"".join(self._audio_buffer)
                        await self._client.write_event(
                            AudioChunk(audio=audio_data).event()
                        )
                    await self._client.write_event(AudioStop().event())
                    await self._client.__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error closing Wyoming client: {e}")

    def _convert_frame_to_pcm(self, frame: rtc.AudioFrame) -> bytes:
        """Convert LiveKit AudioFrame to PCM bytes."""
        # Get audio data as numpy array
        samples = np.frombuffer(frame.data, dtype=np.int16)

        # Handle multi-channel
        if frame.num_channels > 1:
            samples = samples.reshape(-1, frame.num_channels)
            samples = samples[:, 0]  # Take first channel

        # Resample if needed
        if frame.sample_rate != WYOMING_SAMPLE_RATE:
            num_samples = int(
                len(samples) * WYOMING_SAMPLE_RATE / frame.sample_rate
            )
            indices = np.linspace(0, len(samples) - 1, num_samples)
            samples = np.interp(indices, np.arange(len(samples)), samples)

        return samples.astype(np.int16).tobytes()

    async def aclose(self) -> None:
        """Close the stream."""
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass

