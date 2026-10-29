# ====================================================IMPORT=================================================================

import os
import sys
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

import openpyxl
from peft import PeftModel
import pandas as pd

import warnings

warnings.simplefilter("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_train import create_prompt

# Paths: run_dir is the output_dir of the SFT run (checkpoints + the test split written by that same run)
base_model = "Qwen/Qwen2.5-3B-Instruct"
run_dir = r"C:\Users\...\logs\peft_128_Tu_Po\fit_..."
lora_path = os.path.join(run_dir, "checkpoint-...")
data_path = os.path.join(run_dir, "test_dataset.json")
use_lora = True  # False: zero-shot baseline (base model without the adapter), same prompt and test set

# Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True
)

# Base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype="auto",
    device_map="auto"
)

# Load LoRA
if use_lora:
    model = PeftModel.from_pretrained(
        model,
        lora_path
    )

model.eval()

# Load Dataset
df = pd.read_json(
    data_path,
    lines=True
)

# Prompt and Inference
predictions = []

for idx, row in df.iterrows():

    tupi_text = row["Tupi"]

    # Same chat template as in training (ends with "<|im_start|>assistant\n")
    prompt = create_prompt(row, src_lng="Tupi", tgt_lng="Português", mode="inference", tokenizer=tokenizer)["full_prompt"]

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # greedy: deterministic
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens, not the prompt
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    predictions.append({
        "Tupi": tupi_text,
        "fonte": row["fonte"],
        "Português_esperado": row["Referência"],
        "Português_gemini": row["Português"] if row["fonte"] == "gemini" else None,
        "Português_predito": response,
    })

    print("=" * 80)
    print("TUPI:")
    print(tupi_text)

    print("\nESPERADO:")
    print(row["Referência"])

    print("\nPREDITO:")
    print(response)

# Save
pred_df = pd.DataFrame(predictions)
pred_df.to_csv(
    os.path.join(run_dir, "predictions.csv" if use_lora else "predictions_base.csv"),
    index=False
)
