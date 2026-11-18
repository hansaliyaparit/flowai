import React, { useState } from 'react';
import { Plus, X } from 'lucide-react';

interface DictionaryPageProps {
  words: string[];
  onAddWord: (word: string) => void;
  onRemoveWord: (word: string) => void;
}

export const DictionaryPage: React.FC<DictionaryPageProps> = ({
  words,
  onAddWord,
  onRemoveWord,
}) => {
  const [input, setInput] = useState('');
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed) return;

    const alreadyExists = words.some(
      (w) => w.toLowerCase() === trimmed.toLowerCase()
    );
    if (alreadyExists) {
      setError(`"${trimmed}" is already in your dictionary.`);
      return;
    }

    onAddWord(trimmed);
    setInput('');
    setError(null);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
          Dictionary
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
          Add names, product terms, or acronyms so FlowAI spells them accurately every time.
        </p>
      </div>

      {/* Add Word Form */}
      <form onSubmit={handleSubmit} className="space-y-2">
        <div className="flex flex-col sm:flex-row gap-2.5">
          <input
            type="text"
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              if (error) setError(null);
            }}
            placeholder="e.g. Kubernetes, Wispr, Hansaliya, PostgreSQL"
            aria-label="New dictionary word"
            className="flex-1 min-h-[44px] px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] text-sm text-slate-900 dark:text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600/30 focus:border-blue-600"
          />
          <button
            type="submit"
            disabled={!input.trim()}
            className="inline-flex items-center justify-center gap-2 min-h-[44px] px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Plus className="w-4 h-4" />
            <span>Add Word</span>
          </button>
        </div>
        {error && (
          <p className="text-xs text-red-600 dark:text-red-400">{error}</p>
        )}
      </form>

      {/* Word List */}
      {words.length === 0 ? (
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-8 text-center space-y-2">
          <p className="text-sm font-medium text-slate-900 dark:text-slate-100">
            Your personal dictionary is empty
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400 max-w-md mx-auto">
            Add custom spellings like teammate names, company brands, or technical libraries above.
          </p>
        </div>
      ) : (
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] divide-y divide-slate-200 dark:divide-slate-800 overflow-hidden">
          {words.map((word) => (
            <div
              key={word}
              className="flex items-center justify-between px-4 py-3 hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors"
            >
              <span className="text-sm font-medium text-slate-900 dark:text-slate-100">
                {word}
              </span>
              <button
                type="button"
                onClick={() => onRemoveWord(word)}
                aria-label={`Remove ${word}`}
                className="min-w-[40px] min-h-[40px] flex items-center justify-center rounded-lg text-slate-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/40 transition-colors cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
