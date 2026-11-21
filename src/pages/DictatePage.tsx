import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Mic,
  Copy,
  Check,
  Trash2,
  AlertCircle,
  Loader2,
  X,
} from 'lucide-react';
import { Waveform } from '../components/Waveform';
import { useAudioRecorder, type RecordingResult } from '../hooks/useAudioRecorder';
import type { DictationEntry, LanguageCode, SnippetItem, WritingStyle } from '../types';

interface DictatePageProps {
  style: WritingStyle;
  onChangeStyle: (style: WritingStyle) => void;
  language: LanguageCode;
  microphoneId: string;
  dictionary: string[];
  snippets: SnippetItem[];
  onSaveDictation: (entry: DictationEntry) => void;
  onUpdateDictation: (id: string, patch: Partial<DictationEntry>) => void;
}

const STYLES: WritingStyle[] = ['Professional', 'Casual', 'Simple'];

function isTextInputFocused(): boolean {
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if ((el as HTMLElement).isContentEditable) return true;
  return false;
}

export const DictatePage: React.FC<DictatePageProps> = ({
  style,
  onChangeStyle,
  language,
  microphoneId,
  dictionary,
  snippets,
  onSaveDictation,
  onUpdateDictation,
}) => {
  const [viewMode, setViewMode] = useState<'cleaned' | 'raw'>('cleaned');
  const [cleanedText, setCleanedText] = useState('');
  const [rawText, setRawText] = useState('');
  const [activeEntryId, setActiveEntryId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const spaceHeldRef = useRef(false);

  // Use refs for values needed inside async transcription callback
  const styleRef = useRef(style);
  const languageRef = useRef(language);
  const dictionaryRef = useRef(dictionary);
  const snippetsRef = useRef(snippets);

  useEffect(() => {
    styleRef.current = style;
    languageRef.current = language;
    dictionaryRef.current = dictionary;
    snippetsRef.current = snippets;
  }, [style, language, dictionary, snippets]);

  const handleRecordingComplete = useCallback(
    async (recording: RecordingResult) => {
      setIsProcessing(true);
      setApiError(null);

      try {
        // Step 1: Send recorded audio to POST /api/transcribe
        const formData = new FormData();
        formData.append('audio', recording.blob, `recording.${recording.extension}`);
        formData.append('language', languageRef.current);
        formData.append('dictionary', JSON.stringify(dictionaryRef.current));

        const transcribeRes = await fetch('/api/transcribe', {
          method: 'POST',
          body: formData,
        });

        const transcribeData = await transcribeRes.json().catch(() => ({}));
        if (!transcribeRes.ok) {
          throw new Error(
            transcribeData.error || 'Could not transcribe audio. Please try again.'
          );
        }

        const transcript = (transcribeData.rawText || '').trim();
        if (!transcript) {
          setApiError('No speech detected in that recording. Please try speaking clearly.');
          return;
        }

        // Step 2: Send raw transcript to POST /api/cleanup
        const cleanupRes = await fetch('/api/cleanup', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            rawText: transcript,
            style: styleRef.current,
            dictionary: dictionaryRef.current,
            snippets: snippetsRef.current,
          }),
        });

        const cleanupData = await cleanupRes.json().catch(() => ({}));
        if (!cleanupRes.ok) {
          // Preserve raw transcript so the user doesn't lose their words on cleanup failure
          setRawText(transcript);
          setCleanedText(transcript);
          setViewMode('raw');
          throw new Error(
            cleanupData.error || 'Cleanup failed, showing raw transcript instead.'
          );
        }

        const cleaned = (cleanupData.cleanedText || transcript).trim();
        const entryId =
          typeof crypto !== 'undefined' && crypto.randomUUID
            ? crypto.randomUUID()
            : `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

        setRawText(transcript);
        setCleanedText(cleaned);
        setActiveEntryId(entryId);
        setViewMode('cleaned');

        onSaveDictation({
          id: entryId,
          rawText: transcript,
          cleanedText: cleaned,
          style: styleRef.current,
          language: languageRef.current,
          durationMs: recording.durationMs,
          createdAt: Date.now(),
          approved: false,
          provider: (cleanupData.provider as 'groq' | 'local') || 'groq',
        });
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : 'Something went wrong while processing your dictation.';
        setApiError(message);
      } finally {
        setIsProcessing(false);
      }
    },
    [onSaveDictation]
  );

  const {
    isRecording,
    elapsedMs,
    waveformLevels,
    recorderError,
    startRecording,
    stopRecording,
    clearError: clearRecorderError,
  } = useAudioRecorder({
    microphoneId,
    onComplete: handleRecordingComplete,
  });

  // Ensure pointer release anywhere on the screen stops recording cleanly
  useEffect(() => {
    if (!isRecording) return;

    const handleGlobalRelease = () => {
      stopRecording();
    };

    window.addEventListener('pointerup', handleGlobalRelease);
    window.addEventListener('pointercancel', handleGlobalRelease);
    window.addEventListener('blur', handleGlobalRelease);

    return () => {
      window.removeEventListener('pointerup', handleGlobalRelease);
      window.removeEventListener('pointercancel', handleGlobalRelease);
      window.removeEventListener('blur', handleGlobalRelease);
    };
  }, [isRecording, stopRecording]);

  // Hold Space key to dictate when no text input is focused
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || e.repeat) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (isTextInputFocused()) return;
      if (isProcessing) return;

      e.preventDefault();
      if (!spaceHeldRef.current) {
        spaceHeldRef.current = true;
        setApiError(null);
        startRecording();
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return;
      if (spaceHeldRef.current) {
        e.preventDefault();
        spaceHeldRef.current = false;
        stopRecording();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [isProcessing, startRecording, stopRecording]);

  const activeText = viewMode === 'cleaned' ? cleanedText : rawText;

  const handleTextChange = (value: string) => {
    if (viewMode === 'cleaned') {
      setCleanedText(value);
      if (activeEntryId) {
        onUpdateDictation(activeEntryId, { cleanedText: value });
      }
    } else {
      setRawText(value);
      if (activeEntryId) {
        onUpdateDictation(activeEntryId, { rawText: value });
      }
    }
  };

  const handleCopy = async () => {
    if (!activeText.trim()) return;
    try {
      await navigator.clipboard.writeText(activeText);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setApiError('Unable to copy automatically. Select the text and copy manually.');
    }
  };

  const handleClear = () => {
    setCleanedText('');
    setRawText('');
    setActiveEntryId(null);
    setApiError(null);
    clearRecorderError();
  };

  const displayedError = recorderError || apiError;
  const wordCount = activeText.trim() ? activeText.trim().split(/\s+/).length : 0;
  const charCount = activeText.length;

  return (
    <div className="space-y-6">
      {/* Page Header & Writing Style Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
            Dictate
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
            Hold to speak, release to transcribe and format automatically.
          </p>
        </div>

        {/* Segmented Writing Style Selector */}
        <div
          role="group"
          aria-label="Writing style"
          className="inline-flex items-center p-1 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-100/80 dark:bg-[#111827] self-start sm:self-auto"
        >
          {STYLES.map((item) => {
            const active = style === item;
            return (
              <button
                key={item}
                type="button"
                onClick={() => onChangeStyle(item)}
                className={`px-3.5 py-2 min-h-[40px] rounded-lg text-xs sm:text-sm font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  active
                    ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                {item}
              </button>
            );
          })}
        </div>
      </div>

      {/* Friendly Error Banner */}
      {displayedError && (
        <div
          role="alert"
          className="flex items-start justify-between gap-3 rounded-xl border border-red-200 dark:border-red-900/60 bg-red-50 dark:bg-red-950/40 px-4 py-3 text-sm text-red-800 dark:text-red-200"
        >
          <div className="flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0 text-red-600 dark:text-red-400" />
            <span>{displayedError}</span>
          </div>
          <button
            type="button"
            onClick={() => {
              setApiError(null);
              clearRecorderError();
            }}
            aria-label="Dismiss error"
            className="p-1 rounded-md text-red-600 dark:text-red-300 hover:bg-red-100 dark:hover:bg-red-900/50 cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Primary Focal Anchor: Hold-to-Speak Microphone Stage */}
      <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] px-6 py-8 sm:py-10 flex flex-col items-center gap-6">
        <Waveform
          levels={waveformLevels}
          isRecording={isRecording}
          isProcessing={isProcessing}
          elapsedMs={elapsedMs}
        />

        <div className="relative flex items-center justify-center">
          {/* Pulsing halo strictly while recording */}
          {isRecording && (
            <span
              aria-hidden="true"
              className="absolute inset-0 rounded-full bg-red-500/30 animate-ping"
            />
          )}

          <button
            type="button"
            disabled={isProcessing}
            onPointerDown={(e) => {
              if (e.button !== 0 && e.pointerType === 'mouse') return;
              e.preventDefault();
              setApiError(null);
              startRecording();
            }}
            onContextMenu={(e) => e.preventDefault()}
            aria-label={
              isRecording
                ? 'Recording in progress. Release to stop.'
                : isProcessing
                ? 'Processing recording'
                : 'Hold to dictate'
            }
            className={`relative z-10 w-24 h-24 sm:w-28 sm:h-28 rounded-full flex items-center justify-center transition-all touch-none select-none cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-blue-600 ${
              isRecording
                ? 'bg-red-600 hover:bg-red-600 text-white shadow-xl shadow-red-600/30 scale-105'
                : isProcessing
                ? 'bg-slate-200 dark:bg-slate-800 text-slate-500 dark:text-slate-400 cursor-wait'
                : 'bg-blue-600 hover:bg-blue-700 active:scale-95 text-white shadow-xl shadow-blue-600/25'
            }`}
          >
            {isProcessing ? (
              <Loader2 className="w-9 h-9 animate-spin" />
            ) : (
              <Mic className="w-9 h-9 sm:w-10 sm:h-10" />
            )}
          </button>
        </div>

        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 text-center">
          {isRecording ? (
            <span className="font-medium text-red-600 dark:text-red-400">
              Release to finish recording
            </span>
          ) : (
            <>
              <span className="hidden sm:inline">
                Press and hold the mic button or hold{' '}
                <kbd className="px-1.5 py-0.5 rounded border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-800 font-mono text-xs text-slate-700 dark:text-slate-300">
                  Space
                </kbd>{' '}
                to dictate
              </span>
              <span className="sm:hidden">Press and hold the mic button to speak</span>
            </>
          )}
        </p>
      </section>

      {/* Result Editor Card */}
      <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] overflow-hidden">
        {/* Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 border-b border-slate-200 dark:border-slate-800">
          <div className="flex items-center gap-3">
            {/* Raw vs Cleaned Toggle */}
            <div
              role="group"
              aria-label="Transcript version"
              className="inline-flex items-center p-1 rounded-lg bg-slate-100 dark:bg-slate-800/80"
            >
              <button
                type="button"
                onClick={() => setViewMode('cleaned')}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  viewMode === 'cleaned'
                    ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                Cleaned
              </button>
              <button
                type="button"
                onClick={() => setViewMode('raw')}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  viewMode === 'raw'
                    ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                Raw
              </button>
            </div>

            {/* Clean unboxed metadata */}
            <div className="hidden sm:flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
              <span>{wordCount} words</span>
              <span aria-hidden="true">·</span>
              <span>{charCount} chars</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {activeText && (
              <button
                type="button"
                onClick={handleClear}
                className="inline-flex items-center gap-1.5 min-h-[40px] px-3 py-1.5 rounded-lg text-xs font-medium text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Clear</span>
              </button>
            )}

            <button
              type="button"
              onClick={handleCopy}
              disabled={!activeText.trim()}
              className="inline-flex items-center gap-1.5 min-h-[40px] px-4 py-1.5 rounded-lg bg-slate-900 dark:bg-blue-600 hover:bg-slate-800 dark:hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5" />
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Editable Textarea */}
        <textarea
          value={activeText}
          onChange={(e) => handleTextChange(e.target.value)}
          placeholder={
            viewMode === 'cleaned'
              ? 'Your cleaned dictation will appear here ready to copy or edit…'
              : 'The unmodified raw speech transcript will appear here…'
          }
          rows={7}
          aria-label={viewMode === 'cleaned' ? 'Cleaned dictation text' : 'Raw transcript text'}
          className="w-full p-4 sm:p-5 bg-transparent text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 text-base leading-relaxed resize-y focus:outline-none"
        />
      </section>
    </div>
  );
};
