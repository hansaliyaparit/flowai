import React, { useCallback, useEffect, useState } from 'react';
import { RefreshCw, CheckCircle2 } from 'lucide-react';
import { PWAInstallButton } from '../components/PWAInstallButton';
import { usePWAInstall } from '../hooks/usePWAInstall';
import type { AppSettings, LanguageCode, ThemeMode, WritingStyle } from '../types';

interface SettingsPageProps {
  settings: AppSettings;
  onUpdateSettings: (patch: Partial<AppSettings>) => void;
}

const LANGUAGES: Array<{ code: LanguageCode; label: string }> = [
  { code: 'auto', label: 'Auto-detect' },
  { code: 'en', label: 'English' },
  { code: 'hi', label: 'Hindi' },
  { code: 'gu', label: 'Gujarati' },
];

const STYLES: WritingStyle[] = ['Professional', 'Casual', 'Simple'];

const THEMES: Array<{ value: ThemeMode; label: string }> = [
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
  { value: 'system', label: 'System' },
];

export const SettingsPage: React.FC<SettingsPageProps> = ({
  settings,
  onUpdateSettings,
}) => {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceError, setDeviceError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const { isInstalled } = usePWAInstall();

  const loadDevices = useCallback(async (requestPermission = false) => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    setRefreshing(true);
    setDeviceError(null);

    try {
      if (requestPermission && navigator.mediaDevices.getUserMedia) {
        const tempStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        tempStream.getTracks().forEach((t) => t.stop());
      }
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = allDevices.filter((d) => d.kind === 'audioinput');
      setDevices(audioInputs);
    } catch {
      setDeviceError('Allow microphone permission to list device names.');
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadDevices(false);
  }, [loadDevices]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
          Settings
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
          Configure spoken language, microphone input, default writing style, and appearance.
        </p>
      </div>

      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] divide-y divide-slate-200 dark:divide-slate-800">
        {/* Language */}
        <div className="p-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <label
              htmlFor="setting-language"
              className="block text-sm font-medium text-slate-900 dark:text-slate-100"
            >
              Spoken Language
            </label>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Hint Whisper for faster, more accurate transcription.
            </p>
          </div>
          <select
            id="setting-language"
            value={settings.language}
            onChange={(e) =>
              onUpdateSettings({ language: e.target.value as LanguageCode })
            }
            className="min-h-[44px] px-3.5 py-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-sm text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600/30 focus:border-blue-600 sm:w-52"
          >
            {LANGUAGES.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.label}
              </option>
            ))}
          </select>
        </div>

        {/* Microphone Choice */}
        <div className="p-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <label
              htmlFor="setting-mic"
              className="block text-sm font-medium text-slate-900 dark:text-slate-100"
            >
              Microphone
            </label>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Choose which audio input FlowAI records from.
            </p>
            {deviceError && (
              <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
                {deviceError}
              </p>
            )}
          </div>
          <div className="flex items-center gap-2 sm:w-72">
            <select
              id="setting-mic"
              value={settings.microphoneId}
              onChange={(e) =>
                onUpdateSettings({ microphoneId: e.target.value })
              }
              className="flex-1 min-w-0 min-h-[44px] px-3.5 py-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-sm text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600/30 focus:border-blue-600"
            >
              <option value="default">System Default Microphone</option>
              {devices
                .filter((d) => d.deviceId && d.deviceId !== 'default')
                .map((device, index) => (
                  <option key={device.deviceId} value={device.deviceId}>
                    {device.label || `Microphone ${index + 1}`}
                  </option>
                ))}
            </select>
            <button
              type="button"
              onClick={() => loadDevices(true)}
              aria-label="Refresh microphone list"
              title="Refresh microphone list"
              className="min-w-[44px] min-h-[44px] flex items-center justify-center rounded-xl border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors shrink-0 cursor-pointer"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Default Writing Style */}
        <div className="p-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <span className="block text-sm font-medium text-slate-900 dark:text-slate-100">
              Default Writing Style
            </span>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Initial cleanup style when you open FlowAI.
            </p>
          </div>
          <div
            role="group"
            aria-label="Default writing style"
            className="inline-flex items-center p-1 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-900 self-start sm:self-auto"
          >
            {STYLES.map((styleOption) => (
              <button
                key={styleOption}
                type="button"
                onClick={() => onUpdateSettings({ defaultStyle: styleOption })}
                className={`px-3 py-1.5 min-h-[36px] rounded-lg text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  settings.defaultStyle === styleOption
                    ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                {styleOption}
              </button>
            ))}
          </div>
        </div>

        {/* Appearance / Theme */}
        <div className="p-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <span className="block text-sm font-medium text-slate-900 dark:text-slate-100">
              Appearance
            </span>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Choose light, dark, or match your system theme.
            </p>
          </div>
          <div
            role="group"
            aria-label="Appearance theme"
            className="inline-flex items-center p-1 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-900 self-start sm:self-auto"
          >
            {THEMES.map((t) => (
              <button
                key={t.value}
                type="button"
                onClick={() => onUpdateSettings({ theme: t.value })}
                className={`px-3.5 py-1.5 min-h-[36px] rounded-lg text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  settings.theme === t.value
                    ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        {/* PWA Installation */}
        <div className="p-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <span className="block text-sm font-medium text-slate-900 dark:text-slate-100">
              Install FlowAI App
            </span>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Install as a standalone PWA on Windows, macOS, or Android.
            </p>
          </div>
          <div className="sm:w-44">
            {isInstalled ? (
              <div className="inline-flex items-center gap-2 text-xs font-medium text-emerald-600 dark:text-emerald-400">
                <CheckCircle2 className="w-4 h-4 shrink-0" />
                <span>Installed as app</span>
              </div>
            ) : (
              <PWAInstallButton />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
