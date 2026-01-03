"""Wyoming protocol TTS implementation for LiveKit agents."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from livekit import rtc
from livekit.agents.tts import (
    SynthesisEvent,
    SynthesisEventType,
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
from wyoming.tts import Synthesize

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
    ) -> None:
        """Initialize Wyoming TTS.

        Args:
            uri: Wyoming server URI (e.g., "tcp://nabu.home:10200")
            voice: Optional voice name (if not provided, uses server default)
        """
        super().__init__(
            capabilities=TTSCapabilities(streaming=True)
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
                async with AsyncClient(self._uri) as client:
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
    ) -> TTS.SynthesizeStream:
        """Synthesize text to speech."""
        return WyomingTTSSynthesizeStream(
            tts=self,
            uri=self._uri,
            voice=self._voice,
            text=text,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        """Close the TTS instance."""
        pass


class WyomingTTSSynthesizeStream(TTS.SynthesizeStream):
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
            sample_rate=WYOMING_SAMPLE_RATE,
            num_channels=WYOMING_CHANNELS,
        )
        self._uri = uri
        self._voice = voice
        self._text = text
        self._client: AsyncClient | None = None

    async def _run(self) -> None:
        """Main processing loop."""
        try:
            self._client = AsyncClient(self._uri)
            await self._client.__aenter__()

            # Send synthesize request
            synthesize_event = Synthesize(text=self._text)
            if self._voice:
                synthesize_event.voice = self._voice
            await self._client.write_event(synthesize_event.event())

            # Read audio chunks
            audio_started = False
            while True:
                event = await self._client.read_event()
                if event is None:
                    break

                if isinstance(event, AudioStart):
                    audio_started = True
                    # Emit start event
                    self._event_ch.send_nowait(
                        SynthesisEvent(
                            type=SynthesisEventType.STARTED,
                        )
                    )
                elif isinstance(event, AudioChunk):
                    if not audio_started:
                        # Emit start if we haven't already
                        self._event_ch.send_nowait(
                            SynthesisEvent(
                                type=SynthesisEventType.STARTED,
                            )
                        )
                        audio_started = True

                    # Convert PCM bytes to AudioFrame
                    audio_frame = self._convert_pcm_to_frame(event.audio)
                    if audio_frame:
                        self._event_ch.send_nowait(
                            SynthesisEvent(
                                type=SynthesisEventType.AUDIO,
                                audio=audio_frame,
                            )
                        )
                elif isinstance(event, AudioStop):
                    # Emit finished event
                    self._event_ch.send_nowait(
                        SynthesisEvent(
                            type=SynthesisEventType.FINISHED,
                        )
                    )
                    break

        except Exception as e:
            logger.error(f"Wyoming TTS synthesis error: {e}")
            self._event_ch.send_nowait(
                SynthesisEvent(
                    type=SynthesisEventType.ERROR,
                    error=str(e),
                )
            )
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

