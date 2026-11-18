import { useCallback, useEffect, useRef, useState } from 'react';

export interface RecordingResult {
  blob: Blob;
  mimeType: string;
  extension: string;
  durationMs: number;
}

interface UseAudioRecorderOptions {
  microphoneId?: string;
  onComplete: (result: RecordingResult) => void;
}

const BAR_COUNT = 28;
const MIN_DURATION_MS = 1000; // Skip clips under 1s to prevent Whisper hallucinations
const SILENCE_RMS_THRESHOLD = 0.008; // Minimum RMS amplitude to count as non-silent speech

function pickSupportedMimeType(): { mimeType: string; extension: string } {
  if (typeof MediaRecorder === 'undefined') {
    return { mimeType: '', extension: 'webm' };
  }
  const candidates = [
    { mimeType: 'audio/webm;codecs=opus', extension: 'webm' },
    { mimeType: 'audio/webm', extension: 'webm' },
    { mimeType: 'audio/mp4', extension: 'mp4' },
    { mimeType: 'audio/ogg;codecs=opus', extension: 'ogg' },
  ];
  for (const item of candidates) {
    if (MediaRecorder.isTypeSupported(item.mimeType)) {
      return item;
    }
  }
  return { mimeType: '', extension: 'webm' };
}

function getFriendlyMicError(err: unknown): string {
  if (err instanceof DOMException) {
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      return 'Microphone access was denied. Please allow microphone permission in your browser settings and try again.';
    }
    if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      return 'No microphone was found. Connect a microphone or check your device audio settings.';
    }
    if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
      return 'Your microphone is busy or unavailable. Close other apps using the mic and try again.';
    }
    if (err.name === 'OverconstrainedError') {
      return 'The selected microphone is no longer connected. Switching back to default in Settings may help.';
    }
  }
  return 'Could not start the microphone. Please check browser permissions and try again.';
}

/**
 * Custom hook managing browser MediaRecorder + Web Audio AnalyserNode for
 * live waveform visualization, timer tracking, and short/silent audio filtering.
 */
export function useAudioRecorder({ microphoneId = 'default', onComplete }: UseAudioRecorderOptions) {
  const [isRecording, setIsRecording] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [waveformLevels, setWaveformLevels] = useState<number[]>(() => Array(BAR_COUNT).fill(0.08));
  const [recorderError, setRecorderError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startTimeRef = useRef<number>(0);
  const maxRmsRef = useRef<number>(0);
  const isStartingRef = useRef<boolean>(false);
  const stopRequestedBeforeStartRef = useRef<boolean>(false);
  const onCompleteRef = useRef(onComplete);

  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  const cleanupMedia = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    mediaRecorderRef.current = null;
    setWaveformLevels(Array(BAR_COUNT).fill(0.08));
  }, []);

  useEffect(() => {
    return () => {
      cleanupMedia();
    };
  }, [cleanupMedia]);

  const clearError = useCallback(() => {
    setRecorderError(null);
  }, []);

  const startRecording = useCallback(async () => {
    if (isRecording || isStartingRef.current) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setRecorderError('Voice recording is not supported in this browser. Try Chrome on desktop or Android.');
      return;
    }

    isStartingRef.current = true;
    stopRequestedBeforeStartRef.current = false;
    setRecorderError(null);
    setElapsedMs(0);
    maxRmsRef.current = 0;

    try {
      const audioConstraints: MediaTrackConstraints = {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      };
      if (microphoneId && microphoneId !== 'default') {
        audioConstraints.deviceId = { ideal: microphoneId };
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });

      // If the user already released the button while permission prompt / stream setup resolved
      if (stopRequestedBeforeStartRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        isStartingRef.current = false;
        setRecorderError('Hold the microphone button for at least 1 second while speaking.');
        return;
      }

      streamRef.current = stream;

      // Set up Web Audio AnalyserNode for live waveform & silence detection
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioCtx();
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.75;
      source.connect(analyser);

      audioContextRef.current = audioCtx;
      analyserRef.current = analyser;

      const { mimeType, extension } = pickSupportedMimeType();
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);

      chunksRef.current = [];
      startTimeRef.current = Date.now();
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const durationMs = Date.now() - startTimeRef.current;
        const peakRms = maxRmsRef.current;
        const actualMime = recorder.mimeType || mimeType || 'audio/webm';
        const blob = new Blob(chunksRef.current, { type: actualMime });

        cleanupMedia();
        setIsRecording(false);

        if (durationMs < MIN_DURATION_MS) {
          setRecorderError('Recording was shorter than 1 second. Hold the button a little longer while speaking.');
          return;
        }

        if (peakRms < SILENCE_RMS_THRESHOLD || blob.size < 500) {
          setRecorderError('No speech detected — audio sounded silent. Check your microphone and try speaking clearly.');
          return;
        }

        onCompleteRef.current({
          blob,
          mimeType: actualMime,
          extension,
          durationMs,
        });
      };

      recorder.start(150);
      isStartingRef.current = false;
      setIsRecording(true);

      // Update live timer every 50ms
      timerRef.current = window.setInterval(() => {
        setElapsedMs(Date.now() - startTimeRef.current);
      }, 50);

      // Sample time-domain & frequency data for smooth waveform + RMS silence tracking
      const timeDomain = new Float32Array(analyser.fftSize);
      const freqData = new Uint8Array(analyser.frequencyBinCount);

      const tickWaveform = () => {
        if (!analyserRef.current) return;

        analyserRef.current.getFloatTimeDomainData(timeDomain);
        let sumSquares = 0;
        for (let i = 0; i < timeDomain.length; i++) {
          sumSquares += timeDomain[i] * timeDomain[i];
        }
        const rms = Math.sqrt(sumSquares / timeDomain.length);
        if (rms > maxRmsRef.current) {
          maxRmsRef.current = rms;
        }

        analyserRef.current.getByteFrequencyData(freqData);
        const usableBins = Math.min(freqData.length, 48);
        const nextBars: number[] = [];
        for (let i = 0; i < BAR_COUNT; i++) {
          // Symmetric distribution from center outward looks responsive and clean
          const distFromCenter = Math.abs(i - (BAR_COUNT - 1) / 2) / ((BAR_COUNT - 1) / 2);
          const binIdx = Math.min(usableBins - 1, Math.floor(distFromCenter * usableBins * 0.7));
          const normalized = freqData[binIdx] / 255;
          // Boost voice energy slightly while keeping a crisp resting floor
          const level = Math.max(0.1, Math.min(1, normalized * 1.25 + rms * 2.5));
          nextBars.push(level);
        }
        setWaveformLevels(nextBars);
        rafRef.current = requestAnimationFrame(tickWaveform);
      };

      rafRef.current = requestAnimationFrame(tickWaveform);
    } catch (err) {
      isStartingRef.current = false;
      cleanupMedia();
      setIsRecording(false);
      setRecorderError(getFriendlyMicError(err));
    }
  }, [isRecording, microphoneId, cleanupMedia]);

  const stopRecording = useCallback(() => {
    if (isStartingRef.current) {
      stopRequestedBeforeStartRef.current = true;
      return;
    }
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop();
    }
  }, []);

  return {
    isRecording,
    elapsedMs,
    waveformLevels,
    recorderError,
    startRecording,
    stopRecording,
    clearError,
  };
}
