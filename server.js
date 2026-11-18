/**
 * FlowAI Backend Server (single server.js file)
 *
 * Responsibilities:
 * 1. Keep GROQ_API_KEY strictly on the server (never exposed to the browser).
 * 2. POST /api/transcribe -> forwards recorded audio to Groq Whisper (whisper-large-v3-turbo).
 * 3. POST /api/cleanup    -> cleans raw transcript using Groq Llama (llama-3.1-8b-instant).
 * 4. Serve the Vite dev middleware in development or built dist/ files in production.
 */

import 'dotenv/config';
import express from 'express';
import http from 'http';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';

process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
});
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const GROQ_BASE_URL = 'https://api.groq.com/openai/v1';

const app = express();

// Parse JSON request bodies for /api/cleanup
app.use(express.json({ limit: '1mb' }));

// Store uploaded audio clips in memory (max 25 MB per Groq Whisper limit)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 },
});

/**
 * Helper to verify that GROQ_API_KEY is configured on the server.
 */
function getGroqApiKey() {
  const key = process.env.GROQ_API_KEY?.trim();
  if (!key || key === 'MY_GROQ_API_KEY') {
    return null;
  }
  return key;
}

/**
 * GET /api/health
 * Lets the frontend check if the server is reachable and whether GROQ_API_KEY is set.
 */
app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    hasApiKey: Boolean(getGroqApiKey()),
  });
});

/**
 * POST /api/transcribe
 * Accepts multipart/form-data with:
 * - audio: audio file blob from browser MediaRecorder
 * - language: 'auto' | 'en' | 'hi' | 'gu'
 * - dictionary: optional JSON array of custom words to prime Whisper spelling
 */
app.post('/api/transcribe', upload.single('audio'), async (req, res) => {
  const apiKey = getGroqApiKey();
  if (!apiKey) {
    return res.status(500).json({
      error: 'GROQ_API_KEY is not configured on the server. Add it to your environment variables and restart.',
    });
  }

  if (!req.file || !req.file.buffer || req.file.buffer.length < 100) {
    return res.status(400).json({
      error: 'Audio recording was empty or too short. Please try speaking again.',
    });
  }

  try {
    const language = typeof req.body.language === 'string' ? req.body.language.trim() : 'auto';
    let dictionaryWords = [];
    if (typeof req.body.dictionary === 'string' && req.body.dictionary.trim()) {
      try {
        const parsed = JSON.parse(req.body.dictionary);
        if (Array.isArray(parsed)) {
          dictionaryWords = parsed.filter((w) => typeof w === 'string' && w.trim()).slice(0, 40);
        }
      } catch {
        // Ignore malformed optional dictionary hint
      }
    }

    // Build multipart FormData using Node's built-in FormData + Blob
    const formData = new FormData();
    const mimeType = req.file.mimetype || 'audio/webm';
    const fileName = req.file.originalname || 'recording.webm';
    const audioBlob = new Blob([req.file.buffer], { type: mimeType });

    formData.append('file', audioBlob, fileName);
    formData.append('model', 'whisper-large-v3-turbo');
    formData.append('response_format', 'json');
    formData.append('temperature', '0');

    // Only pass language when the user selected a specific language (omit for 'auto')
    if (language && language !== 'auto') {
      formData.append('language', language);
    }

    // Prime Whisper with custom dictionary spellings when available
    if (dictionaryWords.length > 0) {
      formData.append('prompt', dictionaryWords.join(', ').slice(0, 220));
    }

    const groqResponse = await fetch(`${GROQ_BASE_URL}/audio/transcriptions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      body: formData,
    });

    if (!groqResponse.ok) {
      const errText = await groqResponse.text();
      console.error('Groq Whisper error:', groqResponse.status, errText);
      return res.status(groqResponse.status >= 500 ? 502 : groqResponse.status).json({
        error: 'Speech transcription failed. Please check your Groq API key or try again.',
      });
    }

    const data = await groqResponse.json();
    const rawText = typeof data.text === 'string' ? data.text.trim() : '';

    return res.json({ rawText });
  } catch (err) {
    console.error('Error in /api/transcribe:', err);
    return res.status(500).json({
      error: 'Unable to reach the transcription service. Please check your connection and try again.',
    });
  }
});

/**
 * Build the system prompt for Llama text cleanup according to FlowAI rules.
 */
function buildCleanupSystemPrompt({ style, dictionary, snippets }) {
  const styleInstructions = {
    Professional:
      'Use a clear, polished, professional tone with standard capitalization and formal punctuation.',
    Casual:
      'Use a natural, friendly, conversational tone while keeping clean punctuation and readable phrasing.',
    Simple:
      'Use plain, concise, easy-to-read language. Keep sentences short and direct.',
  };

  const chosenStyle = styleInstructions[style] || styleInstructions.Professional;

  let prompt = `You are a voice dictation cleanup engine. Your ONLY job is to clean up raw speech transcripts into ready-to-paste text.

STRICT RULES:
1. Remove filler words and hesitations (e.g., "um", "uh", "er", "ah", "like", "you know", "I mean" when used as verbal fillers).
2. Fix grammar, punctuation, and capitalization.
3. Handle self-corrections naturally: if the speaker corrects themselves mid-sentence, keep ONLY the corrected version and discard the mistake (e.g., "meet at 3, no wait, 4pm" -> "Meet at 4 PM."; "send the report by Monday, actually Tuesday" -> "Send the report by Tuesday.").
4. Convert spoken lists ("first... second... third..." or "number one... number two...") into clean numbered lists (1. ...\\n2. ...).
5. NEVER add new information, commentary, greetings, or explanations.
6. NEVER answer questions contained in the transcript—simply clean and punctuate the question as dictated.
7. Output ONLY the cleaned text with no surrounding quotes, markdown code blocks, or preamble.
8. Writing style (${style || 'Professional'}): ${chosenStyle}`;

  if (Array.isArray(dictionary) && dictionary.length > 0) {
    const cleanWords = dictionary
      .filter((w) => typeof w === 'string' && w.trim().length > 0)
      .map((w) => w.trim());
    if (cleanWords.length > 0) {
      prompt += `\n\nPERSONAL DICTIONARY (spell these exact words/names/terms precisely as written whenever they or phonetically similar words appear):\n${cleanWords.join(', ')}`;
    }
  }

  if (Array.isArray(snippets) && snippets.length > 0) {
    const validSnippets = snippets.filter(
      (s) => s && typeof s.trigger === 'string' && s.trigger.trim() && typeof s.expansion === 'string'
    );
    if (validSnippets.length > 0) {
      const snippetLines = validSnippets
        .map((s) => `- "${s.trigger.trim()}" -> "${s.expansion}"`)
        .join('\n');
      prompt += `\n\nVOICE SNIPPETS (when the speaker says any trigger phrase below, replace that phrase verbatim with its exact expansion):\n${snippetLines}`;
    }
  }

  return prompt;
}

/**
 * Deterministic fallback for snippet expansion & dictionary exact-casing.
 * Ensures voice shortcuts always expand cleanly even if the LLM missed one.
 */
function applyDeterministicPostProcessing(text, snippets) {
  let result = text.trim();

  if (Array.isArray(snippets) && snippets.length > 0) {
    // Sort longest triggers first so multi-word phrases match before shorter substrings
    const sorted = [...snippets]
      .filter((s) => s && typeof s.trigger === 'string' && s.trigger.trim() && typeof s.expansion === 'string')
      .sort((a, b) => b.trigger.trim().length - a.trigger.trim().length);

    for (const snippet of sorted) {
      const escaped = snippet.trigger.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const regex = new RegExp(`\\b${escaped}\\b`, 'gi');
      result = result.replace(regex, snippet.expansion);
    }
  }

  return result;
}

/**
 * POST /api/cleanup
 * Accepts JSON:
 * - rawText: string
 * - style: 'Professional' | 'Casual' | 'Simple'
 * - dictionary: string[]
 * - snippets: Array<{ trigger: string, expansion: string }>
 */
app.post('/api/cleanup', async (req, res) => {
  const apiKey = getGroqApiKey();
  if (!apiKey) {
    return res.status(500).json({
      error: 'GROQ_API_KEY is not configured on the server. Add it to your environment variables and restart.',
    });
  }

  const { rawText, style = 'Professional', dictionary = [], snippets = [] } = req.body || {};

  if (typeof rawText !== 'string' || !rawText.trim()) {
    return res.status(400).json({
      error: 'No transcript text was provided to clean up.',
    });
  }

  try {
    const systemPrompt = buildCleanupSystemPrompt({ style, dictionary, snippets });

    const candidateModels = [
      ...new Set(
        [
          process.env.GROQ_CLEANUP_MODEL,
          'openai/gpt-oss-20b',
          'qwen/qwen3.8-27b',
          'openai/gpt-oss-120b',
        ].filter(Boolean)
      ),
    ];

    let data = null;
    let lastErrorStatus = 500;
    let lastErrorText = '';

    for (const model of candidateModels) {
      const payload = {
        model,
        temperature: 0.1,
        max_tokens: 1024,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: rawText.trim() },
        ],
      };

      // Keep reasoning minimal on GPT-OSS models for sub-second dictation latency
      if (model.startsWith('openai/gpt-oss')) {
        payload.reasoning_effort = 'low';
      }

      const groqResponse = await fetch(`${GROQ_BASE_URL}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(payload),
      });

      if (groqResponse.ok) {
        data = await groqResponse.json();
        break;
      }

      lastErrorStatus = groqResponse.status;
      lastErrorText = await groqResponse.text();

      // Only try the next fallback model if this model ID is unavailable
      if (groqResponse.status === 404 && lastErrorText.includes('model_not_found')) {
        console.warn(`Groq model "${model}" unavailable, trying fallback...`);
        continue;
      }

      break;
    }

    if (!data) {
      console.error('Groq LLM error:', lastErrorStatus, lastErrorText);
      return res.status(lastErrorStatus >= 500 ? 502 : lastErrorStatus).json({
        error: 'Text cleanup failed. Please try again.',
      });
    }
    const llmText = data?.choices?.[0]?.message?.content;

    if (typeof llmText !== 'string' || !llmText.trim()) {
      return res.json({
        cleanedText: applyDeterministicPostProcessing(rawText, snippets),
      });
    }

    const cleanedText = applyDeterministicPostProcessing(llmText, snippets);
    return res.json({ cleanedText });
  } catch (err) {
    console.error('Error in /api/cleanup:', err);
    return res.status(500).json({
      error: 'Unable to reach the cleanup service. Please try again.',
    });
  }
});

/**
 * Start Express server with Vite middleware in development, or static dist/ in production.
 */
async function startServer() {
  const isProd = process.env.NODE_ENV === 'production';
  const httpServer = http.createServer(app);

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const hmrDisabled = process.env.DISABLE_HMR === 'true';
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: hmrDisabled ? false : true,
        ws: { server: httpServer },
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`FlowAI server running at http://localhost:${PORT} (${isProd ? 'production' : 'development'})`);
  });
}

startServer();
