import pandas as pd
import re

from utils.utils_nlp import normalize_text

def pre_process(df):

    # Compute input length
    df["input_length"] = df["input"].apply(len)

    # Print dataset size before cleaning
    count_before = len(df)
    print(f"Length of inputs before: {count_before}")

    # Drop empty pairs and repeated Tupi sentences, so the same source can't fall in both train and test
    df = df[(df["input"].str.strip() != "") & (df["output"].str.strip() != "")]
    df = df[~df["input"].map(normalize_text).duplicated(keep="first")].reset_index(drop=True)

    # Print dataset size after cleaning
    count_after = len(df)
    print(f"Length of inputs after: {count_after}")
    print(f"Removed rows: {count_before - count_after}")

    return df

def create_prompt(sample, src_lng, tgt_lng, is_prefix =False, is_suffix = False, eos_rep = 1, mode="train", tokenizer=None):

    if tokenizer is None or tokenizer.eos_token is None:
        raise ValueError("A tokenizer with a defined EOS token is required.")

    system_message = f"Você é um assistente de IA muito útil para traduções."
    response_prefix = "Aqui está a tradução: " if is_prefix else ""
    response_suffix = f"\nFim da tradução." if is_suffix else ""

    #input_text = sample[src_lng.capitalize()].strip()  # Extract the input text.
    input_text = sample[src_lng].strip()
    #response = ( sample[tgt_lng.capitalize()].strip() if tgt_lng.capitalize() in sample else "")  # Extract the target text.
    response = sample[tgt_lng].strip() if tgt_lng in sample else ""
    question = f"Traduza o seguinte texto em Tupi para Português. Não inclua informações adicionais ou conteúdo irrelevante.\n\n{input_text}"
    # Get the EOS token from the tokenizer.
    eos_token = tokenizer.eos_token
    if mode == "train":
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response_prefix + response + response_suffix + eos_token* eos_rep}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
            ]

        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return { "full_prompt": full_prompt }


def create_prompt_gemma(sample, src_lng, tgt_lng, is_prefix =False, is_suffix = False, eos_rep = 1, mode="train", tokenizer=None):

    if tokenizer is None or tokenizer.eos_token is None:
        raise ValueError("A tokenizer with a defined EOS token is required.")

    alpaca_prompt = """A seguir, encontra-se uma instrução que descreve uma tarefa, juntamente com uma entrada que fornece contexto adicional. Escreva uma resposta que complete adequadamente a solicitação.

### Instrução:
{}

### Entrada:
{}

### Resposta:
{}"""

    input_text = sample[src_lng.capitalize()].strip()  # Extract the input text.
    response = ( sample[tgt_lng.capitalize()].strip() if tgt_lng.capitalize() in sample else "")  # Extract the target text.
    instruction = "Traduza o seguinte texto em Tupi para Português. Não inclua informações adicionais ou conteúdo irrelevante."
    # Get the EOS token from the tokenizer.
    eos_token = tokenizer.eos_token
    if mode == "train":
        full_prompt = alpaca_prompt.format(instruction, input_text, response) + eos_token
    else:
        full_prompt = alpaca_prompt.format(instruction, input_text, "") + eos_token

    return { "full_prompt": full_prompt }

def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
