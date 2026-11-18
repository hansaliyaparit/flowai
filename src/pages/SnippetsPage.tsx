import React, { useState } from 'react';
import { Plus, Trash2, ArrowRight } from 'lucide-react';
import type { SnippetItem } from '../types';

interface SnippetsPageProps {
  snippets: SnippetItem[];
  onAddSnippet: (trigger: string, expansion: string) => void;
  onRemoveSnippet: (id: string) => void;
}

export const SnippetsPage: React.FC<SnippetsPageProps> = ({
  snippets,
  onAddSnippet,
  onRemoveSnippet,
}) => {
  const [trigger, setTrigger] = useState('');
  const [expansion, setExpansion] = useState('');
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const cleanTrigger = trigger.trim();
    const cleanExpansion = expansion.trim();
    if (!cleanTrigger || !cleanExpansion) return;

    const duplicate = snippets.some(
      (s) => s.trigger.toLowerCase() === cleanTrigger.toLowerCase()
    );
    if (duplicate) {
      setError(`A snippet for "${cleanTrigger}" already exists.`);
      return;
    }

    onAddSnippet(cleanTrigger, cleanExpansion);
    setTrigger('');
    setExpansion('');
    setError(null);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
          Snippets
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
          Voice shortcuts that automatically expand spoken phrases into saved text.
        </p>
      </div>

      {/* Add Snippet Form */}
      <form
        onSubmit={handleSubmit}
        className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-4 sm:p-5 space-y-4"
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <label
              htmlFor="snippet-trigger"
              className="block text-xs font-medium text-slate-600 dark:text-slate-300"
            >
              When I say
            </label>
            <input
              id="snippet-trigger"
              type="text"
              value={trigger}
              onChange={(e) => {
                setTrigger(e.target.value);
                if (error) setError(null);
              }}
              placeholder="e.g. my email"
              className="w-full min-h-[44px] px-3.5 py-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/60 text-sm text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600/30 focus:border-blue-600"
            />
          </div>

          <div className="space-y-1.5">
            <label
              htmlFor="snippet-expansion"
              className="block text-xs font-medium text-slate-600 dark:text-slate-300"
            >
              Insert this text
            </label>
            <input
              id="snippet-expansion"
              type="text"
              value={expansion}
              onChange={(e) => setExpansion(e.target.value)}
              placeholder="e.g. alex@company.com"
              className="w-full min-h-[44px] px-3.5 py-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/60 text-sm text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600/30 focus:border-blue-600"
            />
          </div>
        </div>

        {error && (
          <p className="text-xs text-red-600 dark:text-red-400">{error}</p>
        )}

        <div className="flex justify-end">
          <button
            type="submit"
            disabled={!trigger.trim() || !expansion.trim()}
            className="inline-flex items-center gap-2 min-h-[44px] px-5 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Plus className="w-4 h-4" />
            <span>Add Snippet</span>
          </button>
        </div>
      </form>

      {/* Snippets List */}
      {snippets.length === 0 ? (
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-8 text-center space-y-2">
          <p className="text-sm font-medium text-slate-900 dark:text-slate-100">
            No voice snippets yet
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400 max-w-md mx-auto">
            Create shortcuts for emails, calendar links, addresses, or sign-offs. Say &ldquo;my email&rdquo; and FlowAI inserts your full address.
          </p>
        </div>
      ) : (
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] divide-y divide-slate-200 dark:divide-slate-800 overflow-hidden">
          {snippets.map((snippet) => (
            <div
              key={snippet.id}
              className="flex items-center justify-between gap-4 px-4 sm:px-5 py-3.5"
            >
              <div className="min-w-0 flex flex-col sm:flex-row sm:items-center gap-1.5 sm:gap-3">
                <span className="text-sm font-medium text-slate-900 dark:text-slate-100">
                  &ldquo;{snippet.trigger}&rdquo;
                </span>
                <ArrowRight className="hidden sm:inline w-3.5 h-3.5 text-slate-400 shrink-0" />
                <span className="text-sm text-slate-600 dark:text-slate-300 font-mono truncate">
                  {snippet.expansion}
                </span>
              </div>

              <button
                type="button"
                onClick={() => onRemoveSnippet(snippet.id)}
                aria-label={`Delete snippet ${snippet.trigger}`}
                className="min-w-[40px] min-h-[40px] flex items-center justify-center rounded-lg text-slate-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/40 transition-colors shrink-0 cursor-pointer"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
