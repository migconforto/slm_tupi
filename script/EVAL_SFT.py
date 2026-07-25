# ====================================================IMPORT=================================================================

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

# Paths
base_model = "Qwen/Qwen2.5-3B-Instruct"
data_path = r"C:\Users\...\val_dataset.json"
lora_path = r"C:\Users\...\peft-128"

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

tokenizer.pad_token = tokenizer.eos_token

# Base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype="auto",
    device_map="auto"
)

# Load LoRA
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

    prompt = f"""<|im_start|>system
    Você é um assistente de IA muito útil para traduções.<|im_end|>
    <|im_start|>user
    Traduza o seguinte texto em Tupi para Português. Não inclua informações adicionais ou conteúdo irrelevante.

    {tupi_text}<|im_end|>
    <|im_start|>assistant
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            max_length=50,
            #max_tokens=512,
            top_p=0.9,
            temperature=0.1,
            do_sample=True,
            #repetition_penalty=1.05,
            #eos_token_id=tokenizer.eos_token_id,
            #pad_token_id=tokenizer.eos_token_id,
        )


    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    #response = tokenizer.decode(
    #    outputs[0][inputs["input_ids"].shape[1]:],
    #    skip_special_tokens=True
    #).strip()

    #response = response.replace(
    #    "Fim da tradução.",
    #    ""
    #).strip()

    predictions.append({
        "Tupi": tupi_text,
        "Português_esperado": row["Português"],
        "Português_predito": response,
    })

    print("=" * 80)
    print("TUPI:")
    print(tupi_text)

    print("\nESPERADO:")
    print(row["Português"])

    print("\nPREDITO:")
    print(response)

# Save
pred_df = pd.DataFrame(predictions)
pred_df.to_csv(
    "predictions.csv",
    index=False
)