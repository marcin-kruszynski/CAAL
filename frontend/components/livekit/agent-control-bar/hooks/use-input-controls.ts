import { useCallback, useMemo } from 'react';
import { Track } from 'livekit-client';
import {
  type TrackReferenceOrPlaceholder,
  useLocalParticipant,
  usePersistentUserChoices,
  useTrackToggle,
} from '@livekit/components-react';

export interface UseInputControlsProps {
  saveUserChoices?: boolean;
  onDisconnect?: () => void;
  onDeviceError?: (error: { source: Track.Source; error: Error }) => void;
}

export interface UseInputControlsReturn {
  micTrackRef: TrackReferenceOrPlaceholder;
  microphoneToggle: ReturnType<typeof useTrackToggle<Track.Source.Microphone>>;
  cameraToggle: ReturnType<typeof useTrackToggle<Track.Source.Camera>>;
  screenShareToggle: ReturnType<typeof useTrackToggle<Track.Source.ScreenShare>>;
  handleAudioDeviceChange: (deviceId: string) => void;
  handleVideoDeviceChange: (deviceId: string) => void;
  handleMicrophoneDeviceSelectError: (error: Error) => void;
  handleCameraDeviceSelectError: (error: Error) => void;
}

export function useInputControls({
  saveUserChoices = true,
  onDeviceError,
}: UseInputControlsProps = {}): UseInputControlsReturn {
  const { microphoneTrack, localParticipant } = useLocalParticipant();

  const {
    saveAudioInputEnabled,
    saveVideoInputEnabled,
    saveAudioInputDeviceId,
    saveVideoInputDeviceId,
  } = usePersistentUserChoices({ preventSave: !saveUserChoices });

  const microphoneToggle = useTrackToggle({
    source: Track.Source.Microphone,
    onDeviceError: (error) => {
      // Handle NotFoundError: device not found (e.g., device disconnected)
      if (error instanceof DOMException && error.name === 'NotFoundError') {
        console.warn(
          '[useInputControls] Microphone device not found in onDeviceError, resetting to default device'
        );
        // Clear saved deviceId to force use of default device
        saveAudioInputDeviceId('default');
      }
      onDeviceError?.({ source: Track.Source.Microphone, error });
    },
  });

  const cameraToggle = useTrackToggle({
    source: Track.Source.Camera,
    onDeviceError: (error) => {
      // Handle NotFoundError: device not found (e.g., device disconnected)
      if (error instanceof DOMException && error.name === 'NotFoundError') {
        console.warn(
          '[useInputControls] Camera device not found in onDeviceError, resetting to default device'
        );
        // Clear saved deviceId to force use of default device
        saveVideoInputDeviceId('default');
      }
      onDeviceError?.({ source: Track.Source.Camera, error });
    },
  });

  const screenShareToggle = useTrackToggle({
    source: Track.Source.ScreenShare,
    onDeviceError: (error) => onDeviceError?.({ source: Track.Source.ScreenShare, error }),
  });

  const micTrackRef = useMemo(() => {
    return {
      participant: localParticipant,
      source: Track.Source.Microphone,
      publication: microphoneTrack,
    };
  }, [localParticipant, microphoneTrack]);

  const handleAudioDeviceChange = useCallback(
    (deviceId: string) => {
      saveAudioInputDeviceId(deviceId ?? 'default');
    },
    [saveAudioInputDeviceId]
  );

  const handleVideoDeviceChange = useCallback(
    (deviceId: string) => {
      saveVideoInputDeviceId(deviceId ?? 'default');
    },
    [saveVideoInputDeviceId]
  );

  const handleToggleCamera = useCallback(
    async (enabled?: boolean) => {
      try {
        if (screenShareToggle.enabled) {
          screenShareToggle.toggle(false);
        }
        await cameraToggle.toggle(enabled);
        // persist video input enabled preference
        saveVideoInputEnabled(!cameraToggle.enabled);
      } catch (error) {
        // Handle NotFoundError: device not found (e.g., device disconnected)
        if (error instanceof DOMException && error.name === 'NotFoundError') {
          console.warn('[useInputControls] Camera device not found, resetting to default device');
          // Clear saved deviceId and retry with default device
          saveVideoInputDeviceId('default');
          try {
            // Retry with default device
            await cameraToggle.toggle(enabled);
            saveVideoInputEnabled(!cameraToggle.enabled);
          } catch (retryError) {
            console.error(
              '[useInputControls] Failed to toggle camera even with default device:',
              retryError
            );
            onDeviceError?.({ source: Track.Source.Camera, error: retryError as Error });
          }
        } else {
          console.error('[useInputControls] Failed to toggle camera:', error);
          onDeviceError?.({ source: Track.Source.Camera, error: error as Error });
        }
      }
    },
    [cameraToggle, screenShareToggle, saveVideoInputEnabled, saveVideoInputDeviceId, onDeviceError]
  );

  const handleToggleMicrophone = useCallback(
    async (enabled?: boolean) => {
      try {
        await microphoneToggle.toggle(enabled);
        // persist audio input enabled preference
        saveAudioInputEnabled(!microphoneToggle.enabled);
      } catch (error) {
        // Handle NotFoundError: device not found (e.g., device disconnected)
        if (error instanceof DOMException && error.name === 'NotFoundError') {
          console.warn(
            '[useInputControls] Microphone device not found, resetting to default device'
          );
          // Clear saved deviceId and retry with default device
          saveAudioInputDeviceId('default');
          try {
            // Retry with default device
            await microphoneToggle.toggle(enabled);
            saveAudioInputEnabled(!microphoneToggle.enabled);
          } catch (retryError) {
            console.error(
              '[useInputControls] Failed to toggle microphone even with default device:',
              retryError
            );
            onDeviceError?.({ source: Track.Source.Microphone, error: retryError as Error });
          }
        } else {
          console.error('[useInputControls] Failed to toggle microphone:', error);
          onDeviceError?.({ source: Track.Source.Microphone, error: error as Error });
        }
      }
    },
    [microphoneToggle, saveAudioInputEnabled, saveAudioInputDeviceId, onDeviceError]
  );

  const handleToggleScreenShare = useCallback(
    async (enabled?: boolean) => {
      if (cameraToggle.enabled) {
        cameraToggle.toggle(false);
      }
      await screenShareToggle.toggle(enabled);
    },
    [cameraToggle, screenShareToggle]
  );
  const handleMicrophoneDeviceSelectError = useCallback(
    (error: Error) => onDeviceError?.({ source: Track.Source.Microphone, error }),
    [onDeviceError]
  );

  const handleCameraDeviceSelectError = useCallback(
    (error: Error) => onDeviceError?.({ source: Track.Source.Camera, error }),
    [onDeviceError]
  );

  return {
    micTrackRef,
    cameraToggle: {
      ...cameraToggle,
      toggle: handleToggleCamera,
    },
    microphoneToggle: {
      ...microphoneToggle,
      toggle: handleToggleMicrophone,
    },
    screenShareToggle: {
      ...screenShareToggle,
      toggle: handleToggleScreenShare,
    },
    handleAudioDeviceChange,
    handleVideoDeviceChange,
    handleMicrophoneDeviceSelectError,
    handleCameraDeviceSelectError,
  };
}
