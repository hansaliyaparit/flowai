import React, { useMemo, useState } from 'react';
import { Search, Copy, Check, Trash2, Mic } from 'lucide-react';
import type { DictationEntry } from '../types';

interface HistoryPageProps {
  entries: DictationEntry[];
  onDeleteEntry: (id: string) => void;
  onClearAll: () => void;
  onGoToDictate: () => void;
}

function formatTimestamp(ts: number): string {
  const date = new Date(ts);
  const now = new Date();
  const isToday = date.toDateString() === now.toDateString();

  const time = date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  if (isToday) return `Today, ${time}`;

  const day = date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  return `${day}, ${time}`;
}

export const HistoryPage: React.FC<HistoryPageProps> = ({
  entries,
  onDeleteEntry,
  onClearAll,
  onGoToDictate,
}) => {
  const [query, setQuery] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [rawVisibleIds, setRawVisibleIds] = useState<Record<string, boolean>>({});
  const [confirmClear, setConfirmClear] = useState(false);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return entries;
    return entries.filter(
      (entry) =>
        entry.cleanedText.toLowerCase().includes(q) ||
        entry.rawText.toLowerCase().includes(q)
    );
  }, [entries, query]);

  const handleCopy = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      window.setTimeout(() => setCopiedId(null), 1600);
    } catch {
      // Clipboard fallback not needed
    }
  };

  const toggleRaw = (id: string) => {
    setRawVisibleIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
            History
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
            Saved locally in your browser across sessions.
          </p>
        </div>

        {entries.length > 0 && (
          <div className="flex items-center gap-2 self-start sm:self-auto">
            {confirmClear ? (
              <>
                <button
                  type="button"
                  onClick={() => {
                    onClearAll();
                    setConfirmClear(false);
                  }}
                  className="min-h-[40px] px-3.5 py-2 rounded-lg bg-red-600 hover:bg-red-700 text-white text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                >
                  Confirm Clear All
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmClear(false)}
                  className="min-h-[40px] px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 text-xs font-medium hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                >
                  Cancel
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={() => setConfirmClear(true)}
                className="min-h-[40px] px-3.5 py-2 rounded-lg border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                Clear All
              </button>
            )}
          </div>
        )}
      </div>

      {/* Search Bar */}
      {entries.length > 0 && (
        <div className="relative">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search past dictations…"
            aria-label="Search past dictations"
            className="w-full min-h-[44px] pl-10 pr-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] text-sm text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600/30 focus:border-blue-600"
          />
        </div>
      )}

      {/* Empty States */}
      {entries.length === 0 ? (
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-10 text-center space-y-4">
          <p className="text-base font-medium text-slate-900 dark:text-slate-100">
            No dictations yet
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400 max-w-sm mx-auto">
            Every dictation you record is automatically saved here so you can search or copy it later.
          </p>
          <button
            type="button"
            onClick={onGoToDictate}
            className="inline-flex items-center gap-2 min-h-[44px] px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium transition-colors cursor-pointer"
          >
            <Mic className="w-4 h-4" />
            <span>Start Dictating</span>
          </button>
        </div>
      ) : filtered.length === 0 ? (
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-8 text-center">
          <p className="text-sm text-slate-500 dark:text-slate-400">
            No dictations match &ldquo;{query}&rdquo;.
          </p>
        </div>
      ) : (
        <div className="divide-y divide-slate-200 dark:divide-slate-800 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] overflow-hidden">
          {filtered.map((entry) => {
            const showRaw = Boolean(rawVisibleIds[entry.id]);
            const displayedText = showRaw ? entry.rawText : entry.cleanedText;
            const durationSec = (entry.durationMs / 1000).toFixed(1);

            return (
              <article key={entry.id} className="p-4 sm:p-5 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  {/* Unboxed typographic metadata */}
                  <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
                    <span>{formatTimestamp(entry.createdAt)}</span>
                    <span aria-hidden="true">·</span>
                    <span>{durationSec}s</span>
                    <span aria-hidden="true">·</span>
                    <span className="font-sans">{entry.style}</span>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => toggleRaw(entry.id)}
                      className="min-h-[36px] px-2.5 py-1 rounded-lg text-xs font-medium text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer whitespace-nowrap"
                    >
                      {showRaw ? 'Show Cleaned' : 'Show Raw'}
                    </button>

                    <button
                      type="button"
                      onClick={() => handleCopy(entry.id, displayedText)}
                      aria-label="Copy dictation"
                      className="inline-flex items-center gap-1.5 min-h-[36px] px-3 py-1 rounded-lg text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer whitespace-nowrap"
                    >
                      {copiedId === entry.id ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                          <span>Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5" />
                          <span>Copy</span>
                        </>
                      )}
                    </button>

                    <button
                      type="button"
                      onClick={() => onDeleteEntry(entry.id)}
                      aria-label="Delete dictation"
                      className="min-w-[36px] min-h-[36px] flex items-center justify-center rounded-lg text-slate-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/40 transition-colors cursor-pointer"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                <p className="text-sm sm:text-base text-slate-800 dark:text-slate-100 whitespace-pre-wrap leading-relaxed">
                  {displayedText}
                </p>
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
};
