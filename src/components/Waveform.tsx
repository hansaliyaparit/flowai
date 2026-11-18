import React from 'react';

interface WaveformProps {
  levels: number[];
  isRecording: boolean;
  isProcessing: boolean;
  elapsedMs: number;
}

function formatDuration(ms: number): string {
  const totalTenths = Math.floor(ms / 100);
  const tenths = totalTenths % 10;
  const totalSeconds = Math.floor(totalTenths / 10);
  const seconds = totalSeconds % 60;
  const minutes = Math.floor(totalSeconds / 60);

  const mm = String(minutes).padStart(2, '0');
  const ss = String(seconds).padStart(2, '0');
  return `${mm}:${ss}.${tenths}`;
}

/**
 * Live audio waveform + tabular timer using compositor-only transform: scaleY() for 60fps rendering.
 */
export const Waveform: React.FC<WaveformProps> = ({
  levels,
  isRecording,
  isProcessing,
  elapsedMs,
}) => {
  return (
    <div className="flex flex-col items-center gap-3 w-full">
      <div
        className="flex items-center justify-center gap-1 h-12 px-4"
        aria-hidden="true"
      >
        {levels.map((level, idx) => {
          const scale = isRecording ? Math.max(0.12, Math.min(1, level)) : 0.14;
          return (
            <div
              key={idx}
              className={`w-1 h-10 rounded-full origin-center transition-transform duration-75 ${
                isRecording
                  ? 'bg-red-500 dark:bg-red-400'
                  : isProcessing
                  ? 'bg-blue-500/60 dark:bg-blue-400/60'
                  : 'bg-slate-300 dark:bg-slate-700'
              }`}
              style={{ transform: `scaleY(${scale})` }}
            />
          );
        })}
      </div>

      <div className="flex items-center gap-2.5 text-sm">
        {isRecording ? (
          <>
            <span className="inline-block w-2 h-2 rounded-full bg-red-600 animate-pulse" />
            <span className="font-medium text-red-600 dark:text-red-400">Recording</span>
            <span className="text-slate-400 dark:text-slate-600" aria-hidden="true">·</span>
            <span className="font-mono tabular-nums font-medium text-slate-900 dark:text-slate-100">
              {formatDuration(elapsedMs)}
            </span>
          </>
        ) : isProcessing ? (
          <span className="font-medium text-blue-600 dark:text-blue-400">
            Transcribing and cleaning…
          </span>
        ) : (
          <span className="text-slate-500 dark:text-slate-400">
            Ready to dictate
          </span>
        )}
      </div>
    </div>
  );
};
