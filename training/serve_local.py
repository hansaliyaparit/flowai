"""
================================================================================
FLOWAI: LOCAL MODEL FASTAPI INFERENCE SERVER
================================================================================
Exposes: POST /cleanup
Payload: { "style": "Professional" | "Casual" | "Simple", "raw": "string" }
Returns: { "cleaned": "string" }

Integrates directly with FlowAI's server.js when CLEANUP_PROVIDER=local!
================================================================================
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="FlowAI Local Cleanup Model Server")

# Model paths (prefers merged model, falls back to base model if not fine-tuned yet)
MERGED_MODEL_DIR = "./flowai-qwen-merged"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"


class CleanupRequest(BaseModel):
    style: str = "Professional"
    raw: str


class CleanupResponse(BaseModel):
    cleaned: str


def build_system_prompt(style: str) -> str:
    return (
        f"You are FlowAI's intelligent dictation cleanup engine. "
        f"Your task is to rewrite the raw speech transcript into polished {style} text. "
        f"Remove verbal fillers (um, uh, like, you know), resolve mid-sentence self-corrections, "
        f"and fix grammar and punctuation. Output ONLY the cleaned text with no preamble."
    )


@app.on_event("startup")
def load_model():
    global model, tokenizer
    model_path = MERGED_MODEL_DIR if os.path.exists(MERGED_MODEL_DIR) else BASE_MODEL_NAME
    print(f"[*] Starting FlowAI Local Model Server...")
    print(f"[*] Loading model from: {model_path} onto {device.upper()}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    print("[✓] Model loaded and ready to serve requests on port 8000!")


@app.get("/health")
def health_check():
    return {"status": "ok", "device": device, "model_loaded": model is not None}


@app.post("/cleanup", response_model=CleanupResponse)
def clean_text(req: CleanupRequest):
    if not req.raw or not req.raw.strip():
        raise HTTPException(status_code=400, detail="raw text cannot be empty")

    sys_prompt = build_system_prompt(req.style)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Raw speech: {req.raw.strip()}"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    cleaned = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return CleanupResponse(cleaned=cleaned)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
