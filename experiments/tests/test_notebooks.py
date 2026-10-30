import json
import os
import re

import pytest

from tupi_exp import ROOT

EXPERIMENTS = [
    ("exp01_hparams", "sft_exp01.ipynb", "eval_exp01.ipynb"),
    ("exp02_lexicon", "sft_exp02.ipynb", "eval_exp02.ipynb"),
    ("exp03_split_aug", "sft_exp03.ipynb", "eval_exp03.ipynb"),
]

LOOSE_VALUES = [
    r"warmup_ratio = \d",
    r"max_grad_norm = \d",
    r"lora_alpha = \d",
    r"lora_dropout = \d",
    r"weight_decay = \d",
    r"MAX_LEN = \d",
    r"eval_steps=\d",
    r"logging_steps = \d",
    r"\br=\d",
    r"learning_rate=\d",
    r"num_train_epochs=\d",
    r"seed=\d",
]


def load(*path):
    with open(os.path.join(ROOT, *path), encoding="utf-8") as f:
        return json.load(f)


def source(notebook):
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


@pytest.mark.parametrize("name, sft, _", EXPERIMENTS)
def test_sft_copy_keeps_the_cells_of_the_original(name, sft, _):
    copy = load("experiments", name, sft)
    assert [cell["id"] for cell in copy["cells"]] == [cell["id"] for cell in load("notebooks", "sft.ipynb")["cells"]]


@pytest.mark.parametrize("name, sft, evaluation", EXPERIMENTS)
def test_copies_carry_no_stale_outputs(name, sft, evaluation):
    for file_name in (sft, evaluation):
        code_cells = [cell for cell in load("experiments", name, file_name)["cells"] if cell["cell_type"] == "code"]
        assert all(cell["outputs"] == [] and cell["execution_count"] is None for cell in code_cells)


@pytest.mark.parametrize("name, sft, _", EXPERIMENTS)
def test_sft_copy_reads_the_profile(name, sft, _):
    text = source(load("experiments", name, sft))
    assert f'get_experiment("{name}")' in text
    assert "check_lora_model(model, hp)" in text
    assert 'assert ids_hash(test_dataset["id"]) == BASELINE_VAL_SHA256' in text
    assert "gradient_accumulation_steps=hp.gradient_accumulation_steps" in text
    assert 'f"logs/{experiment.name}/' in text
    for pattern in LOOSE_VALUES:
        assert not re.search(pattern, text), pattern


def test_only_exp02_changes_the_prompt():
    for name, sft, evaluation in EXPERIMENTS:
        uses_glossary = name == "exp02_lexicon"
        train_text, eval_text = source(load("experiments", name, sft)), source(load("experiments", name, evaluation))
        assert ("render_prompt(sample, lexicon" in train_text) == uses_glossary
        assert ("render_prompt(row, lexicon" in eval_text) == uses_glossary
        assert ("create_prompt(sample" in train_text) != uses_glossary


def test_only_exp03_takes_the_split_from_the_saved_ids():
    for name, sft, _ in EXPERIMENTS:
        text = source(load("experiments", name, sft))
        assert ("load_split()" in text) == (name == "exp03_split_aug")
        assert "dataset.shuffle(seed=SPLIT_SEED)" in text


@pytest.mark.parametrize("name, _, evaluation", EXPERIMENTS)
def test_evaluation_copy_names_its_spreadsheet(name, _, evaluation):
    text = source(load("experiments", name, evaluation))
    assert f"predictions_{name}.csv" in text
    assert '"Subconjunto": SUBCONJUNTO[row["fonte"]]' in text
    assert 'data_path = os.path.join(run_dir, "test_dataset.json")' in text
