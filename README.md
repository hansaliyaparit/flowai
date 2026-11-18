# FlowAI

FlowAI is a fast, responsive voice dictation web app inspired by Wispr Flow. Hold a button (or hold the `Space` key) to speak, and FlowAI transcribes your audio with Groq Whisper and cleans up filler words, grammar, self-corrections, spoken lists, custom dictionary spellings, and voice snippets using Groq Llama.

## Tech Stack

- **Frontend:** React, Vite, Tailwind CSS, Lucide Icons
- **Backend:** Node.js + Express (`server.js`)
- **Speech-to-Text:** Groq Whisper (`whisper-large-v3-turbo`)
- **Text Cleanup:** Groq LLM (`openai/gpt-oss-20b` with fallback)
- **Storage:** Browser `localStorage` (history, dictionary, snippets, settings)
- **PWA:** Installable on Chrome (Windows/macOS/Android) and iOS Safari via `vite-plugin-pwa`

## Setup & Running Locally

### 1. Get a Groq API Key
Create a free API key in the [Groq Console](https://console.groq.com/keys).

### 2. Set `GROQ_API_KEY`
Create a `.env` file in the project root (you can copy `.env.example`):

```bash
cp .env.example .env
```

Open `.env` and add your Groq API key:

```env
GROQ_API_KEY="gsk_your_actual_groq_api_key_here"
```

> **Security Note:** `GROQ_API_KEY` is read strictly inside `server.js` on the backend and is never sent to or bundled in the frontend.

### 3. Install Dependencies

```bash
npm install
```

### 4. Run in Development

```bash
npm run dev
```

This starts `server.js` on `http://localhost:3000` with Vite middleware serving the React app and Express handling `/api/transcribe` and `/api/cleanup`.

### 5. Build & Run for Production

```bash
npm run build
npm start
```

## Features

- **Hold-to-Dictate:** Press and hold the microphone button (or hold `Space` when no text field is focused) to record; release to transcribe and clean up.
- **Smart Silence & Short-Clip Protection:** Automatically skips recordings under 1 second or silent audio to avoid Whisper hallucinations.
- **Cleanup Rules:** Strips filler words (`um`, `uh`, `like`, `you know`), resolves mid-sentence self-corrections (`"Monday, actually Tuesday"` → `"Tuesday."`), formats spoken numbered lists, and respects `Professional`, `Casual`, or `Simple` styles.
- **Personal Dictionary:** Add custom names and technical terms so both Whisper and the cleanup model spell them accurately.
- **Voice Snippets:** Define trigger phrases (e.g., `"my email"`) that automatically expand into saved text.
- **Raw vs Cleaned Toggle:** Inspect or edit either the cleaned text or raw Whisper transcript.
- **Installable PWA:** Click **Install App** in the sidebar/header (or **Add to Home Screen** on iOS Safari) to use FlowAI as a standalone app.
