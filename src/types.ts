export type WritingStyle = 'Professional' | 'Casual' | 'Simple';

export type LanguageCode = 'auto' | 'en' | 'hi' | 'gu';

export type ThemeMode = 'light' | 'dark' | 'system';

export type PageId = 'dictate' | 'history' | 'dictionary' | 'snippets' | 'settings';

export interface DictationEntry {
  id: string;
  rawText: string;
  cleanedText: string;
  style: WritingStyle;
  language: LanguageCode;
  durationMs: number;
  createdAt: number;
}

export interface SnippetItem {
  id: string;
  trigger: string;
  expansion: string;
}

export interface AppSettings {
  language: LanguageCode;
  microphoneId: string; // 'default' or specific MediaDeviceInfo.deviceId
  defaultStyle: WritingStyle;
  theme: ThemeMode;
}
