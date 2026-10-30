import json
import os
import re
from dataclasses import asdict

import pytest

from tupi_exp import ROOT
from tupi_exp.config import (
    BASELINE,
    EXPERIMENTS,
    HP_V2,
    TARGET_MODULES,
    check_lora_model,
    get_experiment,
    get_profile,
    qwen2_lora_params,
    spec_steps,
    trainer_steps,
)

QWEN_3B = {
    "hidden_size": 2048,
    "intermediate_size": 11008,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "num_hidden_layers": 36,
}


def notebook_source():
    with open(os.path.join(ROOT, "notebooks", "sft.ipynb"), encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells)


def test_hp_v2_matches_the_spec_table():
    assert HP_V2.r == 64
    assert HP_V2.lora_alpha == 64
    assert HP_V2.lora_dropout == 0.05
    assert HP_V2.learning_rate == 1e-4
    assert HP_V2.lr_scheduler_type == "cosine"
    assert HP_V2.warmup_ratio == 0.05
    assert HP_V2.weight_decay == 0.01
    assert HP_V2.max_grad_norm == 1.0
    assert HP_V2.num_train_epochs == 3
    assert HP_V2.per_device_train_batch_size == 1
    assert HP_V2.gradient_accumulation_steps == 8
    assert HP_V2.target_modules == TARGET_MODULES


def test_hp_v2_changes_only_the_listed_fields():
    base, new = asdict(BASELINE), asdict(HP_V2)
    changed = {key for key in base if base[key] != new[key]}
    assert changed == {
        "name",
        "r",
        "lora_alpha",
        "lora_dropout",
        "learning_rate",
        "warmup_ratio",
        "max_grad_norm",
        "gradient_accumulation_steps",
    }


@pytest.mark.parametrize(
    "pattern, value",
    [
        (r"\br=(\d+),", BASELINE.r),
        (r"lora_alpha = (\d+)", BASELINE.lora_alpha),
        (r"lora_dropout = (\d+)", BASELINE.lora_dropout),
        (r"learning_rate=([\d.e-]+),", BASELINE.learning_rate),
        (r"warmup_ratio = ([\d.]+)", BASELINE.warmup_ratio),
        (r"weight_decay = ([\d.]+)", BASELINE.weight_decay),
        (r"max_grad_norm = ([\d.]+)", BASELINE.max_grad_norm),
        (r"num_train_epochs=(\d+),", BASELINE.num_train_epochs),
        (r"eval_steps=(\d+)", BASELINE.eval_steps),
        (r"logging_steps = (\d+)", BASELINE.logging_steps),
        (r"seed=(\d+)\n", BASELINE.seed),
    ],
)
def test_baseline_profile_follows_the_notebook(pattern, value):
    assert float(re.search(pattern, notebook_source()).group(1)) == float(value)


def test_steps_for_the_current_and_the_spec_train_sizes():
    assert spec_steps(1408, HP_V2) == 528
    assert trainer_steps(1408, HP_V2) == 528
    assert spec_steps(1382, HP_V2) == 519
    assert trainer_steps(1382, HP_V2) == 516


def test_trainable_parameters_from_the_config():
    assert qwen2_lora_params(QWEN_3B, 128) == 239_468_544  # observed in the baseline run
    assert qwen2_lora_params(QWEN_3B, 64) == 119_734_272


def test_named_lookups():
    assert get_profile("hp_v2") is HP_V2
    assert get_experiment("exp01_hparams").profile is HP_V2
    assert {exp.profile for exp in EXPERIMENTS.values()} == {HP_V2}


def tiny_lora_model(r, lora_alpha):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    lora_config = LoraConfig(r=r, lora_alpha=lora_alpha, target_modules=list(TARGET_MODULES), task_type="CAUSAL_LM")
    return get_peft_model(Qwen2ForCausalLM(config), lora_config)


def test_check_lora_model_accepts_the_right_profile():
    check_lora_model(tiny_lora_model(HP_V2.r, HP_V2.lora_alpha), HP_V2)


def test_check_lora_model_rejects_a_wrong_rank():
    with pytest.raises(AssertionError):
        check_lora_model(tiny_lora_model(32, HP_V2.lora_alpha), HP_V2)


def test_check_lora_model_rejects_a_wrong_alpha():
    with pytest.raises(AssertionError):
        check_lora_model(tiny_lora_model(HP_V2.r, 8), HP_V2)
