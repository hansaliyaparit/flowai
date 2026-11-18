import type { AppSettings, DictationEntry, SnippetItem } from '../types';

const KEYS = {
  HISTORY: 'flowai_history',
  DICTIONARY: 'flowai_dictionary',
  SNIPPETS: 'flowai_snippets',
  SETTINGS: 'flowai_settings',
} as const;

export const DEFAULT_SETTINGS: AppSettings = {
  language: 'auto',
  microphoneId: 'default',
  defaultStyle: 'Professional',
  theme: 'system',
};

function readJSON<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function writeJSON<T>(key: string, value: T): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Ignore storage quota errors cleanly
  }
}

export function loadHistory(): DictationEntry[] {
  const items = readJSON<DictationEntry[]>(KEYS.HISTORY, []);
  return Array.isArray(items) ? items : [];
}

export function saveHistory(entries: DictationEntry[]): void {
  writeJSON(KEYS.HISTORY, entries);
}

export function loadDictionary(): string[] {
  const words = readJSON<string[]>(KEYS.DICTIONARY, []);
  return Array.isArray(words) ? words : [];
}

export function saveDictionary(words: string[]): void {
  writeJSON(KEYS.DICTIONARY, words);
}

export function loadSnippets(): SnippetItem[] {
  const snippets = readJSON<SnippetItem[]>(KEYS.SNIPPETS, []);
  return Array.isArray(snippets) ? snippets : [];
}

export function saveSnippets(snippets: SnippetItem[]): void {
  writeJSON(KEYS.SNIPPETS, snippets);
}

export function loadSettings(): AppSettings {
  const saved = readJSON<Partial<AppSettings>>(KEYS.SETTINGS, {});
  return {
    ...DEFAULT_SETTINGS,
    ...saved,
  };
}

export function saveSettings(settings: AppSettings): void {
  writeJSON(KEYS.SETTINGS, settings);
}
