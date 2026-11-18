import React from 'react';
import {
  Mic,
  History,
  BookA,
  Zap,
  Settings,
  Sun,
  Moon,
  AudioLines,
} from 'lucide-react';
import type { PageId, ThemeMode } from '../types';
import { PWAInstallButton } from './PWAInstallButton';

interface SidebarProps {
  activePage: PageId;
  onSelectPage: (page: PageId) => void;
  theme: ThemeMode;
  resolvedDark: boolean;
  onToggleTheme: () => void;
  counts: {
    history: number;
    dictionary: number;
    snippets: number;
  };
}

const NAV_ITEMS: Array<{ id: PageId; label: string; icon: React.ComponentType<{ className?: string }> }> = [
  { id: 'dictate', label: 'Dictate', icon: Mic },
  { id: 'history', label: 'History', icon: History },
  { id: 'dictionary', label: 'Dictionary', icon: BookA },
  { id: 'snippets', label: 'Snippets', icon: Zap },
  { id: 'settings', label: 'Settings', icon: Settings },
];

export const Sidebar: React.FC<SidebarProps> = ({
  activePage,
  onSelectPage,
  resolvedDark,
  onToggleTheme,
  counts,
}) => {
  const getCount = (id: PageId): number | null => {
    if (id === 'history' && counts.history > 0) return counts.history;
    if (id === 'dictionary' && counts.dictionary > 0) return counts.dictionary;
    if (id === 'snippets' && counts.snippets > 0) return counts.snippets;
    return null;
  };

  return (
    <>
      {/* Desktop Left Sidebar */}
      <aside className="hidden md:flex md:flex-col md:w-60 lg:w-64 shrink-0 border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0D1320] min-h-screen sticky top-0 h-screen">
        {/* Brand Header */}
        <div className="flex items-center justify-between px-5 h-16 border-b border-slate-200 dark:border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white shrink-0">
              <AudioLines className="w-4 h-4" />
            </div>
            <span className="text-lg font-bold tracking-tight text-slate-900 dark:text-slate-100 whitespace-nowrap">
              FlowAI
            </span>
          </div>

          <button
            type="button"
            onClick={onToggleTheme}
            aria-label={resolvedDark ? 'Switch to light mode' : 'Switch to dark mode'}
            className="w-9 h-9 rounded-lg flex items-center justify-center text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer"
          >
            {resolvedDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>

        {/* Navigation Links */}
        <nav className="flex-1 p-3 space-y-1" aria-label="Primary navigation">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            const count = getCount(item.id);
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectPage(item.id)}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-medium transition-colors cursor-pointer whitespace-nowrap ${
                  isActive
                    ? 'bg-blue-50 dark:bg-blue-500/15 text-blue-700 dark:text-blue-300'
                    : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800/70 hover:text-slate-900 dark:hover:text-slate-100'
                }`}
              >
                <span className="flex items-center gap-3">
                  <Icon className="w-4 h-4 shrink-0" />
                  <span>{item.label}</span>
                </span>
                {count !== null && (
                  <span className="font-mono tabular-nums text-xs text-slate-400 dark:text-slate-500">
                    {count}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Bottom Install / Hint Area */}
        <div className="p-4 border-t border-slate-200 dark:border-slate-800 space-y-3">
          <PWAInstallButton />
          <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
            Hold <kbd className="px-1.5 py-0.5 rounded border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-800 font-mono text-[11px] text-slate-700 dark:text-slate-300">Space</kbd> on Dictate to speak.
          </p>
        </div>
      </aside>

      {/* Mobile Compact Top Header (<= 52px) */}
      <header className="md:hidden sticky top-0 z-30 h-13 px-4 flex items-center justify-between border-b border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-[#0D1320]/90 backdrop-blur-md">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-blue-600 flex items-center justify-center text-white shrink-0">
            <AudioLines className="w-4 h-4" />
          </div>
          <span className="text-base font-bold tracking-tight text-slate-900 dark:text-slate-100 whitespace-nowrap">
            FlowAI
          </span>
        </div>

        <div className="flex items-center gap-2">
          <PWAInstallButton compact />
          <button
            type="button"
            onClick={onToggleTheme}
            aria-label={resolvedDark ? 'Switch to light mode' : 'Switch to dark mode'}
            className="min-w-[40px] min-h-[40px] rounded-lg flex items-center justify-center text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            {resolvedDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>
      </header>

      {/* Mobile Bottom Tab Bar (5 touch-friendly destinations) */}
      <nav
        aria-label="Mobile navigation"
        className="md:hidden fixed bottom-0 left-0 right-0 z-40 border-t border-slate-200 dark:border-slate-800 bg-white/95 dark:bg-[#0D1320]/95 backdrop-blur-md pb-[env(safe-area-inset-bottom)]"
      >
        <div className="grid grid-cols-5 h-15">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectPage(item.id)}
                className={`flex flex-col items-center justify-center gap-1 min-h-[44px] transition-colors ${
                  isActive
                    ? 'text-blue-600 dark:text-blue-400'
                    : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span className="text-[11px] font-medium leading-none whitespace-nowrap">
                  {item.label}
                </span>
              </button>
            );
          })}
        </div>
      </nav>
    </>
  );
};
