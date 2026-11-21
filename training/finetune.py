"""
================================================================================
FLOWAI: LORA FINE-TUNING PIPELINE FOR TEXT CLEANUP & STYLE REWRITING
================================================================================
Model: Qwen/Qwen2.5-0.5B-Instruct
Technique: Low-Rank Adaptation (LoRA) via Hugging Face PEFT + Transformers
Task: Takes raw speech transcripts with verbal fillers and self-corrections,
      and cleans them into polished Professional, Casual, or Simple text.

Can be run locally or on a FREE Google Colab T4 GPU instance!
================================================================================
"""

import os
import sys
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, PeftModel

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "dataset.jsonl"  # Path to exported FlowAI approved dataset
OUTPUT_DIR = "./flowai-qwen-lora-adapter"
MERGED_OUTPUT_DIR = "./flowai-qwen-merged"

BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
VAL_SPLIT_PERCENTAGE = 0.15  # 15% validation split


def build_system_prompt(style: str) -> str:
    """
    Constructs the system instructions matching FlowAI's cleanup rules.
    """
    return (
        f"You are FlowAI's intelligent dictation cleanup engine. "
        f"Your task is to rewrite the raw speech transcript into polished {style} text. "
        f"Remove verbal fillers (um, uh, like, you know), resolve mid-sentence self-corrections, "
        f"and fix grammar and punctuation. Output ONLY the cleaned text with no preamble."
    )


def load_flowai_dataset(jsonl_path: str):
    """
    Loads JSONL dataset exported from FlowAI:
    Each line: {"style": "Professional", "raw": "...", "cleaned": "..."}
    """
    if not os.path.exists(jsonl_path):
        print(f"[!] Dataset file '{jsonl_path}' not found.")
        print(
            "    Please export your approved dataset from FlowAI's History page "
            "and place it in this folder as 'dataset.jsonl'."
        )
        # Create a small synthetic dummy dataset so the script can still be test-run
        print("    --> Generating fallback sample data for demonstration...")
        samples = [
            {
                "style": "Professional",
                "raw": "um please send the report to finance by like tomorrow",
                "cleaned": "Please send the report to the finance team by tomorrow.",
            },
            {
                "style": "Casual",
                "raw": "hey um you know let us meet at 3 no wait 4pm",
                "cleaned": "Hey, let's meet at 4:00 PM.",
            },
            {
                "style": "Simple",
                "raw": "so basically the server is down and we need to restart it",
                "cleaned": "The server is down. Please restart it.",
            },
            {
                "style": "Professional",
                "raw": "sun yaar we need to approve this pull request na",
                "cleaned": "Please review and approve this pull request at your earliest convenience.",
            },
        ] * 15
        return Dataset.from_list(samples)

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"[*] Loaded {len(records)} approved examples from {jsonl_path}")
    return Dataset.from_list(records)


def main():
    print("=" * 80)
    print("FLOWAI: QWEN-2.5-0.5B LORA FINE-TUNING")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Compute device: {device.upper()}")
    if device == "cuda":
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"[*] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ==========================================================================
    # 2. LOAD TOKENIZER AND BASE MODEL
    # ==========================================================================
    print(f"\n[*] Loading Tokenizer and Base Model: {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # ==========================================================================
    # 3. PRINT MODEL ARCHITECTURE, WEIGHT SHAPES & PARAMETERS
    # ==========================================================================
    print("\n" + "=" * 50)
    print("MODEL ARCHITECTURE & WEIGHT SHAPES")
    print("=" * 50)
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        # Print first few attention and MLP layers as examples
        if "layers.0" in name or "layers.1" in name or "embed_tokens" in name:
            print(f"  - {name:<50} Shape: {str(list(param.shape)):<20} ({param.numel():,} params)")

    print(f"\n[*] Total Base Model Parameters: {total_params:,} ({total_params / 1e6:.1f} Million)")

    # ==========================================================================
    # 4. ATTACH LORA ADAPTERS (PEFT)
    # ==========================================================================
    print("\n[*] Applying LoRA (Low-Rank Adaptation)...")
    print(f"    - Rank (r): {LORA_R}")
    print(f"    - Alpha: {LORA_ALPHA}")
    print(f"    - Targeted Projections: q_proj, k_proj, v_proj, o_proj")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Freeze base model and inject LoRA matrices A and B
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percent = (trainable_params / total_params) * 100
    print(f"[*] Trainable Parameters: {trainable_params:,} ({trainable_percent:.2f}% of full model)")
    print("    --> Over 99% of parameters are safely frozen! Fast and lightweight.")

    # ==========================================================================
    # 5. DATASET PREPARATION & LABEL MASKING (-100)
    # ==========================================================================
    raw_dataset = load_flowai_dataset(DATASET_PATH)

    def preprocess_and_mask_labels(batch):
        """
        Converts {"style", "raw", "cleaned"} into tokenized conversations.
        CRUCIAL: The prompt tokens are masked with label -100 so Cross-Entropy loss
        is ONLY computed on the assistant's generated cleaned output!
        """
        input_ids_list = []
        labels_list = []

        for style, raw, cleaned in zip(batch["style"], batch["raw"], batch["cleaned"]):
            sys_prompt = build_system_prompt(style)

            # Build prompt tokens (System + User)
            messages_prompt = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Raw speech: {raw}"},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages_prompt, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

            # Build target response tokens
            response_text = f"{cleaned}{tokenizer.eos_token}"
            response_ids = tokenizer.encode(response_text, add_special_tokens=False)

            input_ids = prompt_ids + response_ids
            # Mask prompt tokens with -100 (PyTorch CrossEntropyLoss ignores index -100)
            labels = [-100] * len(prompt_ids) + response_ids

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        return {"input_ids": input_ids_list, "labels": labels_list}

    print("\n[*] Tokenizing and applying -100 prompt loss masks...")
    tokenized_dataset = raw_dataset.map(
        preprocess_and_mask_labels,
        batched=True,
        remove_columns=raw_dataset.column_names,
    )

    # 15% Hold-out Validation Split
    split_dataset = tokenized_dataset.train_test_split(test_size=VAL_SPLIT_PERCENTAGE, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    print(f"[*] Dataset split: {len(train_dataset)} training examples, {len(val_dataset)} validation examples.")

    # ==========================================================================
    # 6. TRAINING ARGUMENTS & TRAINER
    # ==========================================================================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )

    print("\n[*] Starting LoRA Fine-Tuning for 3 Epochs...")
    trainer.train()

    # ==========================================================================
    # 7. EVALUATION ON VALIDATION SAMPLES
    # ==========================================================================
    print("\n" + "=" * 50)
    print("VALIDATION PREDICTION EXAMPLES (AFTER FINE-TUNING)")
    print("=" * 50)
    model.eval()

    sample_tests = [
        ("Professional", "um we should like schedule the demo for tomorrow morning"),
        ("Casual", "sun yaar can you check the server logs na"),
        ("Simple", "basically the checkout service is throwing five hundred error"),
    ]

    for style, raw in sample_tests:
        sys_prompt = build_system_prompt(style)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Raw speech: {raw}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.2,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Extract only newly generated tokens
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"\n[Style: {style}]")
        print(f"  Raw:     \"{raw}\"")
        print(f"  Cleaned: \"{response}\"")

    # ==========================================================================
    # 8. SAVE LORA ADAPTER & STANDALONE MERGED MODEL
    # ==========================================================================
    print(f"\n[*] Saving LoRA adapter weights to: {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"[*] Merging LoRA adapter back into base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    merged_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    merged_model = merged_model.merge_and_unload()

    print(f"[*] Saving full merged standalone model to: {MERGED_OUTPUT_DIR}...")
    merged_model.save_pretrained(MERGED_OUTPUT_DIR)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

    print("\n[✓] Fine-tuning complete! You can now serve your model locally with:")
    print("    python serve_local.py")


if __name__ == "__main__":
    main()
