import React, { useMemo, useRef, useState } from 'react';
import {
  Search,
  Copy,
  Check,
  Trash2,
  Mic,
  Download,
  Upload,
  ShieldAlert,
  Edit3,
  CheckSquare,
  Square,
  FileCode,
  FileSpreadsheet,
  Cpu,
  Cloud,
} from 'lucide-react';
import type { DictationEntry } from '../types';

interface HistoryPageProps {
  entries: DictationEntry[];
  onDeleteEntry: (id: string) => void;
  onClearAll: () => void;
  onGoToDictate: () => void;
  onUpdateEntry: (id: string, patch: Partial<DictationEntry>) => void;
  onImportEntries: (newEntries: DictationEntry[]) => void;
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
  onUpdateEntry,
  onImportEntries,
}) => {
  const [query, setQuery] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [rawVisibleIds, setRawVisibleIds] = useState<Record<string, boolean>>({});
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState('');
  const [confirmClear, setConfirmClear] = useState(false);
  const [importStatus, setImportStatus] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Search filter
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return entries;
    return entries.filter(
      (entry) =>
        entry.cleanedText.toLowerCase().includes(q) ||
        entry.rawText.toLowerCase().includes(q)
    );
  }, [entries, query]);

  // Approved count calculations
  const approvedCount = useMemo(() => {
    return entries.filter((e) => e.approved).length;
  }, [entries]);

  const handleCopy = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      window.setTimeout(() => setCopiedId(null), 1600);
    } catch {
      // Fallback
    }
  };

  const toggleRaw = (id: string) => {
    setRawVisibleIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const toggleApproved = (entry: DictationEntry) => {
    onUpdateEntry(entry.id, { approved: !entry.approved });
  };

  const startEditing = (entry: DictationEntry) => {
    setEditingId(entry.id);
    setEditText(entry.cleanedText);
  };

  const saveEdit = (id: string) => {
    if (editText.trim()) {
      onUpdateEntry(id, { cleanedText: editText.trim() });
    }
    setEditingId(null);
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditText('');
  };

  // Export as .jsonl (Only approved entries, format: {"style":"...","raw":"...","cleaned":"..."})
  const handleExportJSONL = () => {
    const approvedItems = entries.filter((e) => e.approved);
    if (approvedItems.length === 0) {
      alert('No approved dictations to export! Check the "Approved" box on entries you want in your dataset.');
      return;
    }

    const lines = approvedItems.map((item) =>
      JSON.stringify({
        style: item.style || 'Professional',
        raw: item.rawText,
        cleaned: item.cleanedText,
      })
    );

    const blob = new Blob([lines.join('\n')], { type: 'application/x-jsonlines;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `flowai-dataset-${Date.now()}.jsonl`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Export as .csv
  const handleExportCSV = () => {
    const approvedItems = entries.filter((e) => e.approved);
    if (approvedItems.length === 0) {
      alert('No approved dictations to export! Check the "Approved" box on entries you want in your dataset.');
      return;
    }

    const escapeCSV = (str: string) => `"${str.replace(/"/g, '""')}"`;
    const rows = [
      ['style', 'raw', 'cleaned'],
      ...approvedItems.map((item) => [
        escapeCSV(item.style || 'Professional'),
        escapeCSV(item.rawText),
        escapeCSV(item.cleanedText),
      ]),
    ];

    const csvContent = rows.map((r) => r.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `flowai-dataset-${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Import dataset (.jsonl)
  const handleImportFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        const lines = text.split('\n').filter((l) => l.trim().length > 0);
        const importedEntries: DictationEntry[] = [];

        for (let i = 0; i < lines.length; i++) {
          const parsed = JSON.parse(lines[i]);
          if (parsed.raw && parsed.cleaned) {
            importedEntries.push({
              id: `imported-${Date.now()}-${i}-${Math.random().toString(36).slice(2, 6)}`,
              rawText: String(parsed.raw),
              cleanedText: String(parsed.cleaned),
              style: (parsed.style as any) || 'Professional',
              language: 'auto',
              durationMs: 3000,
              createdAt: Date.now() - (lines.length - i) * 60000,
              approved: true,
              provider: 'local',
            });
          }
        }

        if (importedEntries.length > 0) {
          onImportEntries(importedEntries);
          setImportStatus(`Successfully imported ${importedEntries.length} dictation entries.`);
          setTimeout(() => setImportStatus(null), 4000);
        } else {
          alert('No valid dictation rows found. Expected format: {"style":"...","raw":"...","cleaned":"..."}');
        }
      } catch (err) {
        alert('Failed to parse JSONL file. Please ensure each line is valid JSON.');
      }
      if (fileInputRef.current) fileInputRef.current.value = '';
    };
    reader.readAsText(file);
  };

  return (
    <div className="space-y-6">
      {/* Top Header & Global Actions */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
            History & Dataset Hub
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
            Manage your past dictations and curate high-quality fine-tuning datasets.
          </p>
        </div>

        {entries.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            {/* Import JSONL */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImportFile}
              accept=".jsonl,.json,.txt"
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center gap-1.5 min-h-[38px] px-3.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 text-xs font-medium transition-colors cursor-pointer"
              title="Import approved dictations from a .jsonl file"
            >
              <Upload className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
              <span>Import .jsonl</span>
            </button>

            {/* Export JSONL */}
            <button
              type="button"
              onClick={handleExportJSONL}
              disabled={approvedCount === 0}
              className={`inline-flex items-center gap-1.5 min-h-[38px] px-3.5 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
                approvedCount > 0
                  ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-xs'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-400 cursor-not-allowed'
              }`}
              title="Export approved items as .jsonl for fine-tuning"
            >
              <FileCode className="w-3.5 h-3.5" />
              <span>Export dataset (.jsonl)</span>
            </button>

            {/* Export CSV */}
            <button
              type="button"
              onClick={handleExportCSV}
              disabled={approvedCount === 0}
              className={`inline-flex items-center gap-1.5 min-h-[38px] px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors cursor-pointer ${
                approvedCount > 0
                  ? 'border-slate-200 dark:border-slate-700 bg-white dark:bg-[#111827] text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800'
                  : 'border-slate-200 dark:border-slate-800 text-slate-400 cursor-not-allowed'
              }`}
              title="Export approved items as .csv"
            >
              <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
              <span>Export .csv</span>
            </button>

            {/* Clear All */}
            {confirmClear ? (
              <div className="flex items-center gap-1 ml-auto">
                <button
                  type="button"
                  onClick={() => {
                    onClearAll();
                    setConfirmClear(false);
                  }}
                  className="min-h-[38px] px-3 py-1.5 rounded-lg bg-red-600 hover:bg-red-700 text-white text-xs font-medium transition-colors cursor-pointer"
                >
                  Confirm Clear
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmClear(false)}
                  className="min-h-[38px] px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 text-xs font-medium hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setConfirmClear(true)}
                className="min-h-[38px] px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-800 text-slate-500 hover:text-red-600 dark:text-slate-400 hover:bg-red-50 dark:hover:bg-red-950/30 text-xs font-medium transition-colors cursor-pointer"
              >
                Clear
              </button>
            )}
          </div>
        )}
      </div>

      {/* Dataset Progress Counter & Privacy Notice Banner */}
      {entries.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {/* Approved Counter */}
          <div className="md:col-span-1 rounded-xl p-3.5 bg-blue-50/70 dark:bg-blue-950/20 border border-blue-200/80 dark:border-blue-900/40 flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-blue-600/10 text-blue-600 dark:text-blue-400 flex items-center justify-center shrink-0">
              <CheckSquare className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs font-medium text-blue-900 dark:text-blue-200">
                <span className="font-bold text-sm text-blue-700 dark:text-blue-300">
                  {approvedCount} approved
                </span>{' '}
                of {entries.length} total
              </div>
              <div className="text-[11px] text-blue-600/80 dark:text-blue-400/80">
                Recommended: 200+ for fine-tuning
              </div>
            </div>
          </div>

          {/* Privacy Notice Banner */}
          <div className="md:col-span-2 rounded-xl p-3.5 bg-amber-50/70 dark:bg-amber-950/20 border border-amber-200/80 dark:border-amber-900/40 flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-amber-600/10 text-amber-600 dark:text-amber-400 flex items-center justify-center shrink-0">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div className="text-xs text-amber-900 dark:text-amber-200 leading-relaxed">
              <span className="font-semibold">Privacy notice before export:</span> Remove private info (passwords, API keys, client names, personal phone numbers) before sharing datasets or training public models.
            </div>
          </div>
        </div>
      )}

      {/* Import Success Banner */}
      {importStatus && (
        <div className="p-3 rounded-xl bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-800 text-xs text-emerald-800 dark:text-emerald-300 flex items-center gap-2">
          <Check className="w-4 h-4 text-emerald-600 shrink-0" />
          <span>{importStatus}</span>
        </div>
      )}

      {/* Search Bar */}
      {entries.length > 0 && (
        <div className="relative">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search past dictations by raw or cleaned text…"
            aria-label="Search past dictations"
            className="w-full min-h-[44px] pl-10 pr-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] text-sm text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600/30 focus:border-blue-600"
          />
        </div>
      )}

      {/* Entries List */}
      {entries.length === 0 ? (
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-10 text-center space-y-4">
          <p className="text-base font-medium text-slate-900 dark:text-slate-100">
            No dictations yet
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400 max-w-sm mx-auto">
            Every dictation you record is automatically saved here. You can approve high-quality samples to build your personal fine-tuning dataset.
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
            const isEditing = editingId === entry.id;
            const durationSec = (entry.durationMs / 1000).toFixed(1);
            const isApproved = Boolean(entry.approved);

            return (
              <article
                key={entry.id}
                className={`p-4 sm:p-5 space-y-3 transition-colors ${
                  isApproved ? 'bg-blue-50/20 dark:bg-blue-950/10' : ''
                }`}
              >
                {/* Header bar: Metadata + Provider Badge + Actions */}
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
                    {/* Approved Checkbox */}
                    <button
                      type="button"
                      onClick={() => toggleApproved(entry)}
                      className="inline-flex items-center gap-1.5 py-1 px-2 rounded-md font-sans font-medium text-xs border transition-colors cursor-pointer"
                      title={isApproved ? 'Remove from approved dataset' : 'Mark approved for training dataset'}
                    >
                      {isApproved ? (
                        <>
                          <CheckSquare className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
                          <span className="text-blue-700 dark:text-blue-300">Approved</span>
                        </>
                      ) : (
                        <>
                          <Square className="w-3.5 h-3.5 text-slate-400" />
                          <span className="text-slate-500 dark:text-slate-400">Approve</span>
                        </>
                      )}
                    </button>

                    <span aria-hidden="true">·</span>
                    <span>{formatTimestamp(entry.createdAt)}</span>
                    <span aria-hidden="true">·</span>
                    <span>{durationSec}s</span>
                    <span aria-hidden="true">·</span>
                    <span className="font-sans px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-[11px]">
                      {entry.style}
                    </span>

                    {/* Provider Badge (Feature 4 requirement) */}
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-sans font-medium ${
                        entry.provider === 'local'
                          ? 'bg-purple-100 text-purple-700 dark:bg-purple-950/50 dark:text-purple-300 border border-purple-200 dark:border-purple-800'
                          : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800'
                      }`}
                    >
                      {entry.provider === 'local' ? (
                        <>
                          <Cpu className="w-3 h-3" />
                          <span>Local Model</span>
                        </>
                      ) : (
                        <>
                          <Cloud className="w-3 h-3" />
                          <span>Groq Cloud</span>
                        </>
                      )}
                    </span>
                  </div>

                  {/* Actions: Toggle Raw, Edit, Copy, Delete */}
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => toggleRaw(entry.id)}
                      className="min-h-[34px] px-2.5 py-1 rounded-lg text-xs font-medium text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer whitespace-nowrap"
                    >
                      {showRaw ? 'Show Cleaned' : 'Show Raw'}
                    </button>

                    {!isEditing && (
                      <button
                        type="button"
                        onClick={() => startEditing(entry)}
                        title="Edit cleaned text"
                        className="inline-flex items-center gap-1 min-h-[34px] px-2.5 py-1 rounded-lg text-xs font-medium text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer whitespace-nowrap"
                      >
                        <Edit3 className="w-3.5 h-3.5" />
                        <span>Edit</span>
                      </button>
                    )}

                    <button
                      type="button"
                      onClick={() => handleCopy(entry.id, showRaw ? entry.rawText : entry.cleanedText)}
                      aria-label="Copy dictation"
                      className="inline-flex items-center gap-1.5 min-h-[34px] px-3 py-1 rounded-lg text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer whitespace-nowrap"
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
                      className="min-w-[34px] min-h-[34px] flex items-center justify-center rounded-lg text-slate-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/40 transition-colors cursor-pointer"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Body Content / Inline Editor */}
                {isEditing ? (
                  <div className="space-y-2 pt-1">
                    <label className="block text-xs font-medium text-slate-500 dark:text-slate-400">
                      Edit cleaned text (will be saved to dataset):
                    </label>
                    <textarea
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                      rows={3}
                      className="w-full p-3 rounded-xl border border-blue-500/50 bg-white dark:bg-[#0D1320] text-sm text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500/30"
                    />
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => saveEdit(entry.id)}
                        className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium transition-colors cursor-pointer"
                      >
                        Save Edits
                      </button>
                      <button
                        type="button"
                        onClick={cancelEdit}
                        className="px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 text-xs font-medium hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <div>
                    {showRaw ? (
                      <div className="space-y-1">
                        <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                          Raw Whisper Transcript
                        </span>
                        <p className="text-sm sm:text-base text-slate-700 dark:text-slate-300 italic whitespace-pre-wrap leading-relaxed">
                          {entry.rawText}
                        </p>
                      </div>
                    ) : (
                      <p className="text-sm sm:text-base text-slate-900 dark:text-slate-100 whitespace-pre-wrap leading-relaxed">
                        {entry.cleanedText}
                      </p>
                    )}
                  </div>
                )}
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
};
