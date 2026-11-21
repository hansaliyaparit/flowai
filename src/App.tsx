import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { Sidebar } from './components/Sidebar';
import { OfflineIndicator } from './components/OfflineIndicator';
import { DictatePage } from './pages/DictatePage';
import { HistoryPage } from './pages/HistoryPage';
import { DictionaryPage } from './pages/DictionaryPage';
import { SnippetsPage } from './pages/SnippetsPage';
import { SettingsPage } from './pages/SettingsPage';
import { ModelLabPage } from './pages/ModelLabPage';
import {
  loadDictionary,
  loadHistory,
  loadSettings,
  loadSnippets,
  saveDictionary,
  saveHistory,
  saveSettings,
  saveSnippets,
} from './lib/storage';
import type {
  AppSettings,
  DictationEntry,
  PageId,
  SnippetItem,
  ThemeMode,
  WritingStyle,
} from './types';

export default function App() {
  const [activePage, setActivePage] = useState<PageId>('dictate');

  // Persistent local state (stored in browser localStorage)
  const [history, setHistory] = useState<DictationEntry[]>(() => loadHistory());
  const [dictionary, setDictionary] = useState<string[]>(() => loadDictionary());
  const [snippets, setSnippets] = useState<SnippetItem[]>(() => loadSnippets());
  const [settings, setSettings] = useState<AppSettings>(() => loadSettings());

  // Current writing style starts from user's default setting
  const [activeStyle, setActiveStyle] = useState<WritingStyle>(
    () => loadSettings().defaultStyle
  );

  // Track resolved dark mode (supports 'light', 'dark', and 'system')
  const [resolvedDark, setResolvedDark] = useState<boolean>(() => {
    if (settings.theme === 'dark') return true;
    if (settings.theme === 'light') return false;
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false;
  });

  // Check if server has GROQ_API_KEY configured
  const [missingApiKey, setMissingApiKey] = useState(false);

  useEffect(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then((data) => {
        if (data && data.hasApiKey === false) {
          setMissingApiKey(true);
        }
      })
      .catch(() => {
        // Ignore offline/network startup checks
      });
  }, []);

  // Sync theme setting with <html class="dark"> and listen for OS theme changes
  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)');

    const applyTheme = () => {
      const isDark =
        settings.theme === 'dark' ||
        (settings.theme === 'system' && media.matches);
      setResolvedDark(isDark);
      document.documentElement.classList.toggle('dark', isDark);
    };

    applyTheme();
    media.addEventListener('change', applyTheme);
    return () => media.removeEventListener('change', applyTheme);
  }, [settings.theme]);

  const handleToggleTheme = () => {
    const nextTheme: ThemeMode = resolvedDark ? 'light' : 'dark';
    const nextSettings: AppSettings = { ...settings, theme: nextTheme };
    setSettings(nextSettings);
    saveSettings(nextSettings);
  };

  const handleUpdateSettings = (patch: Partial<AppSettings>) => {
    const next = { ...settings, ...patch };
    setSettings(next);
    saveSettings(next);
    if (patch.defaultStyle) {
      setActiveStyle(patch.defaultStyle);
    }
  };

  // History handlers
  const handleSaveDictation = useCallback((entry: DictationEntry) => {
    setHistory((prev) => {
      const next = [entry, ...prev];
      saveHistory(next);
      return next;
    });
  }, []);

  const handleUpdateDictation = useCallback(
    (id: string, patch: Partial<DictationEntry>) => {
      setHistory((prev) => {
        const next = prev.map((item) =>
          item.id === id ? { ...item, ...patch } : item
        );
        saveHistory(next);
        return next;
      });
    },
    []
  );

  const handleDeleteHistory = (id: string) => {
    setHistory((prev) => {
      const next = prev.filter((item) => item.id !== id);
      saveHistory(next);
      return next;
    });
  };

  const handleClearHistory = () => {
    setHistory([]);
    saveHistory([]);
  };

  const handleImportHistory = useCallback((imported: DictationEntry[]) => {
    setHistory((prev) => {
      const existingIds = new Set(prev.map((e) => e.id));
      const filteredNew = imported.filter((e) => !existingIds.has(e.id));
      const next = [...filteredNew, ...prev];
      saveHistory(next);
      return next;
    });
  }, []);

  // Dictionary handlers
  const handleAddWord = (word: string) => {
    setDictionary((prev) => {
      const next = [...prev, word].sort((a, b) => a.localeCompare(b));
      saveDictionary(next);
      return next;
    });
  };

  const handleRemoveWord = (word: string) => {
    setDictionary((prev) => {
      const next = prev.filter((w) => w !== word);
      saveDictionary(next);
      return next;
    });
  };

  // Snippet handlers
  const handleAddSnippet = (trigger: string, expansion: string) => {
    const id =
      typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setSnippets((prev) => {
      const next = [{ id, trigger, expansion }, ...prev];
      saveSnippets(next);
      return next;
    });
  };

  const handleRemoveSnippet = (id: string) => {
    setSnippets((prev) => {
      const next = prev.filter((s) => s.id !== id);
      saveSnippets(next);
      return next;
    });
  };

  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-slate-50 dark:bg-[#090D16] text-slate-900 dark:text-slate-100">
      <Sidebar
        activePage={activePage}
        onSelectPage={setActivePage}
        theme={settings.theme}
        resolvedDark={resolvedDark}
        onToggleTheme={handleToggleTheme}
        counts={{
          history: history.length,
          dictionary: dictionary.length,
          snippets: snippets.length,
        }}
      />

      <main className="flex-1 min-w-0 px-4 py-6 sm:px-8 sm:py-8 pb-24 md:pb-10">
        <div className="max-w-4xl mx-auto space-y-6">
          {missingApiKey && (
            <div
              role="status"
              className="flex items-start gap-3 rounded-xl border border-amber-300 dark:border-amber-800/70 bg-amber-50 dark:bg-amber-950/40 px-4 py-3 text-sm text-amber-900 dark:text-amber-200"
            >
              <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0 text-amber-600 dark:text-amber-400" />
              <p>
                <strong>GROQ_API_KEY</strong> is not set on the server yet. Add it to your environment variables (or <code className="font-mono text-xs">.env</code> file) and restart the server to enable dictation.
              </p>
            </div>
          )}

          {activePage === 'dictate' && (
            <DictatePage
              style={activeStyle}
              onChangeStyle={setActiveStyle}
              language={settings.language}
              microphoneId={settings.microphoneId}
              dictionary={dictionary}
              snippets={snippets}
              onSaveDictation={handleSaveDictation}
              onUpdateDictation={handleUpdateDictation}
            />
          )}

          {activePage === 'history' && (
            <HistoryPage
              entries={history}
              onDeleteEntry={handleDeleteHistory}
              onClearAll={handleClearHistory}
              onGoToDictate={() => setActivePage('dictate')}
              onUpdateEntry={handleUpdateDictation}
              onImportEntries={handleImportHistory}
            />
          )}

          {activePage === 'dictionary' && (
            <DictionaryPage
              words={dictionary}
              onAddWord={handleAddWord}
              onRemoveWord={handleRemoveWord}
            />
          )}

          {activePage === 'snippets' && (
            <SnippetsPage
              snippets={snippets}
              onAddSnippet={handleAddSnippet}
              onRemoveSnippet={handleRemoveSnippet}
            />
          )}

          {activePage === 'settings' && (
            <SettingsPage
              settings={settings}
              onUpdateSettings={handleUpdateSettings}
            />
          )}

          {activePage === 'lab' && <ModelLabPage />}
        </div>
      </main>

      <OfflineIndicator />
    </div>
  );
}
