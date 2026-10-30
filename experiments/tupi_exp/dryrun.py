"""Checks each experiment without loading the model (T5).

Run from experiments/: python -m tupi_exp.dryrun [--experiment exp01_hparams]
"""
import argparse
import collections
import json
import math
import statistics
from dataclasses import asdict

from tupi_exp.config import BASE_MODEL, BASELINE_VAL_SHA256, EXPERIMENTS, get_experiment, qwen2_lora_params, spec_steps, trainer_steps
from tupi_exp.lexicon import check_no_leak, load_lexicon, render_prompt, select_entries
from tupi_exp.normalize import tupi_key
from tupi_exp.splits import ids_hash, load_split, reproduce_split
from utils.utils_train import create_prompt

CHARS_PER_TOKEN = 2  # pessimistic estimate, used only without the tokenizer


class CharEstimator:
    """Stands for the tokenizer when it is not available offline: same chat format, counted by characters."""

    eos_token = "<|im_end|>"
    estimated = True

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        turns = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        return "".join(turns) + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text):
        return {"input_ids": range(math.ceil(len(text) / CHARS_PER_TOKEN))}


def load_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    except OSError:
        return CharEstimator()


def load_model_config():
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(BASE_MODEL, local_files_only=True).to_dict()
    except OSError:
        return None


def full_prompt(experiment, row, lexicon, tokenizer):
    """The text the notebook of the experiment trains on."""
    sample = {"Tupi": row["input"], "Português": row["output"]}
    arguments = dict(src_lng="Tupi", tgt_lng="Português", eos_rep=0, mode="train", tokenizer=tokenizer)
    if experiment.prompt == "glossario":
        return render_prompt(sample, lexicon, example_id=row["id"], **arguments)["full_prompt"]
    return create_prompt(sample, **arguments)["full_prompt"]


def split_rows(experiment):
    """Train and validation rows as the notebook of the experiment builds them."""
    with open(experiment.dataset_path, encoding="utf-8") as f:
        rows = json.load(f)
    by_id = {row["id"]: row for row in rows}
    train_ids, val_ids = load_split()
    val_set = set(val_ids)
    return [row for row in rows if row["id"] not in val_set], [by_id[row_id] for row_id in val_ids], train_ids


def pick_examples(experiment, train_rows, val_rows, lexicon):
    """Five rows to print: 2 historical, 2 synthetic and 1 without glossary entries."""
    def entries_of(row):
        return select_entries(row["input"], lexicon, example_id=row["id"]) if experiment.prompt == "glossario" else []

    pool = sorted(val_rows, key=lambda row: row["id"])
    added = [row for row in train_rows if row.get("origem") == "split_aug"]
    historical = added or [row for row in pool if row["fonte"] == "gemini"]
    synthetic = [row for row in pool if row["fonte"] == "sintetico"]
    if experiment.prompt == "glossario":
        historical = [row for row in historical if entries_of(row)]
        synthetic = [row for row in synthetic if entries_of(row)]
    chosen = historical[:2] + synthetic[:2]
    no_match = [row for row in pool if not entries_of(row) and row not in chosen]
    return chosen + no_match[:1]


def token_stats(lengths):
    ordered = sorted(lengths)
    return {
        "min": ordered[0],
        "mediana": statistics.median(ordered),
        "p99": ordered[int(0.99 * len(ordered))],
        "max": ordered[-1],
    }


def run_experiment(name, tokenizer, model_config):
    """Prints the checks of one experiment and returns them. Any violation raises AssertionError."""
    experiment = get_experiment(name)
    hp = experiment.profile
    train_rows, val_rows, base_train_ids = split_rows(experiment)
    val_ids = [row["id"] for row in val_rows]

    print(f"\n{'=' * 30} {name} {'=' * 30}")
    print(f"Dataset: {experiment.dataset_path}")
    print(f"Prompt: {experiment.prompt} | max_seq_length: {experiment.max_seq_length}")
    print(f"Perfil resolvido: {asdict(hp)}")

    # Same validation everywhere (P6)
    val_hash = ids_hash(val_ids)
    assert val_hash == BASELINE_VAL_SHA256, f"validation differs from the baseline: {val_hash}"
    assert ids_hash(reproduce_split()[1]) == val_hash, "saved split differs from the code"
    print(f"Validação: {len(val_rows)} exemplos | hash {val_hash} (igual ao baseline)")

    # No leakage (P4)
    train_set = {row["id"] for row in train_rows}
    assert set(base_train_ids) <= train_set, "a train id of the baseline is missing"
    val_keys = {tupi_key(row["input"]) for row in val_rows}
    assert not val_keys & {tupi_key(row["input"]) for row in train_rows}, "validation sentence in train"
    fontes = collections.Counter(row["fonte"] for row in train_rows)
    print(f"Treino: {len(train_rows)} exemplos {dict(fontes)} | interseção com validação: 0")

    lexicon = {}
    if experiment.prompt == "glossario":
        lexicon = load_lexicon()
        check_no_leak(lexicon, val_rows)
        print(f"Glossário: {len(lexicon)} entradas | interseção com validação: 0")

    # Steps and parameters
    steps = trainer_steps(len(train_rows), hp)
    print(f"Passos do Trainer: {steps} | fórmula do SPEC: {spec_steps(len(train_rows), hp)}")
    if hp.eval_steps > steps:
        print(f"AVISO: eval_steps={hp.eval_steps} e logging_steps={hp.logging_steps} são maiores que {steps} passos: "
              "nada é avaliado nem registrado durante o treino")
    trainable = qwen2_lora_params(model_config, hp.r, hp.target_modules) if model_config else None
    print(f"Parâmetros treináveis esperados: {trainable}" + ("" if model_config else " (config do modelo indisponível)"))

    # Lengths and truncation (R2.7)
    lengths = {}
    for split, rows in (("treino", train_rows), ("validação", val_rows)):
        lengths[split] = [len(tokenizer(full_prompt(experiment, row, lexicon, tokenizer))["input_ids"]) for row in rows]
    estimated = getattr(tokenizer, "estimated", False)
    source = "ESTIMATIVA por caracteres" if estimated else "tokenizador do Qwen"
    truncated = sum(length > experiment.max_seq_length for values in lengths.values() for length in values)
    for split, values in lengths.items():
        print(f"Tokens ({source}), {split}: {token_stats(values)}")
    if estimated:
        print("AVISO: tokenizador indisponível offline, truncamento não verificado")
    else:
        assert truncated == 0, f"{truncated} truncated examples"
        print(f"Exemplos truncados com max_seq_length={experiment.max_seq_length}: 0")

    entries_histogram = {}
    if experiment.prompt == "glossario":
        for split, rows in (("treino", train_rows), ("validação", val_rows)):
            entries_histogram[split] = dict(sorted(collections.Counter(
                len(select_entries(row["input"], lexicon, example_id=row["id"])) for row in rows
            ).items()))
            print(f"Entradas por prompt, {split}: {entries_histogram[split]}")

    print("\nPrompts renderizados:")
    for row in pick_examples(experiment, train_rows, val_rows, lexicon):
        print(f"--- {row['id']} ({row['fonte']})")
        print(full_prompt(experiment, row, lexicon, tokenizer))

    return {
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "val_hash": val_hash,
        "steps": steps,
        "truncated": truncated,
        "max_tokens": max(max(values) for values in lengths.values()),
        "entries_histogram": entries_histogram,
    }


def main():
    parser = argparse.ArgumentParser(description="Dry-run of the SFT experiments (the model is not loaded)")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), action="append")
    args = parser.parse_args()

    tokenizer, model_config = load_tokenizer(), load_model_config()
    for name in args.experiment or sorted(EXPERIMENTS):
        run_experiment(name, tokenizer, model_config)
    print("\nDry-run concluído: todas as verificações passaram.")


if __name__ == "__main__":
    main()
