# FlowAI: LoRA Fine-Tuning Guide (Qwen2.5-0.5B)

This folder contains the complete, self-contained pipeline for fine-tuning **Qwen/Qwen2.5-0.5B-Instruct** using **LoRA (Low-Rank Adaptation)** on your approved FlowAI dictations.

---

## 1. Overview & Architecture

- **Base Model:** `Qwen/Qwen2.5-0.5B-Instruct` (~490 Million parameters).
- **LoRA Configuration:**
  - Rank ($r$): 16
  - Alpha ($\alpha$): 32
  - Target Projections: `q_proj`, `k_proj`, `v_proj`, `o_proj` in self-attention layers.
  - Trainable parameters: **~1.2%** of full model (~6 Million parameters).
- **Label Masking:** Prompt tokens (system instruction + user speech) are masked with `-100` so Cross-Entropy loss is only computed on the desired cleaned text.
- **Hardware Requirement:** Can run on a free Google Colab **T4 GPU** (16GB VRAM) or locally on an Nvidia RTX GPU (8GB+ VRAM).

---

## 2. Dataset Requirements

1. In FlowAI, go to the **History** page.
2. Review your past recordings and click the **Approve** checkbox on high-quality examples.
3. Edit the cleaned text if necessary to fix spelling or add custom formatting.
4. Click **Export dataset (.jsonl)**.
5. Save the file into this `training/` folder as **`dataset.jsonl`**.

### How many examples do I need?
- **Minimum for testing:** 30–50 examples (teaches basic formatting).
- **Recommended for solid generalization:** **200+ examples**.
- **Production-grade:** 500+ examples across varied accents and styles.

---

## 3. Running on Free Google Colab (Step-by-Step)

If you don't have a local Nvidia GPU, you can run fine-tuning completely for free on Google Colab:

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Runtime** &rarr; **Change runtime type** &rarr; Select **T4 GPU**.
3. Upload `finetune.py`, `requirements.txt`, and your exported `dataset.jsonl`.
4. In a code cell, install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```
5. Run the training script:
   ```bash
   !python finetune.py
   ```
6. The script will:
   - Print layer names, weight shapes, and parameter counts.
   - Attach LoRA adapters ($r=16, \alpha=32$).
   - Train for 3 epochs with a 15% validation split.
   - Display validation predictions comparing raw vs cleaned text.
   - Save the LoRA adapter in `./flowai-qwen-lora-adapter`.
   - Merge the adapter into the base model and save `./flowai-qwen-merged`.
7. Download the `./flowai-qwen-merged` folder to your local machine.

---

## 4. Serving the Model Locally (FastAPI)

Once fine-tuning is finished:

1. Install local dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI inference server:
   ```bash
   python serve_local.py
   ```
   The server starts at `http://localhost:8000`.

3. Test the server with curl or PowerShell:
   ```bash
   curl -X POST http://localhost:8000/cleanup \
     -H "Content-Type: application/json" \
     -d '{"style": "Professional", "raw": "um please send the figma designs by like tomorrow"}'
   ```
   Expected response:
   ```json
   {
     "cleaned": "Please send the Figma designs by tomorrow."
   }
   ```

---

## 5. Connecting FlowAI to Your Local Model

1. Open your `.env` file in the root `flowai/` directory.
2. Set:
   ```env
   CLEANUP_PROVIDER="local"
   LOCAL_CLEANUP_URL="http://localhost:8000/cleanup"
   ```
3. Restart FlowAI:
   ```bash
   npm run dev
   ```
4. Now, when you dictate in FlowAI:
   - Audio is transcribed by Whisper.
   - Text is sent to your local FastAPI server.
   - If the local server is ever offline, FlowAI will automatically fall back to Groq Cloud and indicate the provider on your History card!
