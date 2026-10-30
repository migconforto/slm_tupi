"""Named hyperparameter profiles and experiment definitions (R1.1)."""
import math
import os
from dataclasses import dataclass, replace

from tupi_exp import DATA_DIR

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# Split of the baseline run. Kept apart from HParams.seed: changing it changes the validation set (P6)
SPLIT_SEED = 13
TRAIN_SHARE = 0.7
BASELINE_VAL_SHA256 = "86c9431b2bf3d5265f30bf166dab5b9834dfab2bbb80d05b8882f8b022a4e5f3"

# Glossary of exp02 (R2.3, R2.4)
LEXICON_MAX_TOKENS = 2
LEXICON_MAX_ENTRIES = 8

# Output spreadsheet column (section 6 of the SPEC)
SUBCONJUNTO = {"gemini": "historico", "sintetico": "sintetico"}


@dataclass(frozen=True)
class HParams:
    name: str
    r: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    target_modules: tuple
    seed: int
    evaluation_strategy: str
    eval_steps: int
    logging_steps: int
    save_strategy: str


# Values read from notebooks/sft.ipynb; tests/test_config.py fails if they drift apart
BASELINE = HParams(
    name="baseline",
    r=128,
    lora_alpha=8,
    lora_dropout=0.0,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.5,
    weight_decay=0.01,
    max_grad_norm=0.3,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    target_modules=TARGET_MODULES,
    seed=13,
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_steps=1000,
    save_strategy="epoch",
)

HP_V2 = replace(
    BASELINE,
    name="hp_v2",
    r=64,
    lora_alpha=64,
    lora_dropout=0.05,
    learning_rate=1e-4,
    warmup_ratio=0.05,
    max_grad_norm=1.0,
    gradient_accumulation_steps=8,
)

PROFILES = {profile.name: profile for profile in (BASELINE, HP_V2)}


@dataclass(frozen=True)
class Experiment:
    name: str
    dataset_path: str
    prompt: str  # "atual" or "glossario"
    max_seq_length: int
    profile: HParams


V2_PATH = os.path.join(DATA_DIR, "dataset_sintetico_aumentado.json")
V3_PATH = os.path.join(DATA_DIR, "dataset_augmentation_v3.json")
HISTORICAL_PATH = os.path.join(DATA_DIR, "dataset_historico.json")

EXPERIMENTS = {
    "exp01_hparams": Experiment("exp01_hparams", V2_PATH, "atual", 128, HP_V2),
    "exp02_lexicon": Experiment("exp02_lexicon", V2_PATH, "glossario", 144, HP_V2),  # longest prompt: 142 tokens (R2.7)
    "exp03_split_aug": Experiment("exp03_split_aug", V3_PATH, "atual", 128, HP_V2),
}


def get_profile(name):
    return PROFILES[name]


def get_experiment(name):
    return EXPERIMENTS[name]


def trainer_steps(n_train, hp):
    """Optimizer steps the Trainer will run (transformers 4.44: floor per epoch)."""
    batches = math.ceil(n_train / hp.per_device_train_batch_size)
    return hp.num_train_epochs * max(batches // hp.gradient_accumulation_steps, 1)


def spec_steps(n_train, hp):
    """Step formula written in the SPEC (ceil per epoch)."""
    effective_batch = hp.per_device_train_batch_size * hp.gradient_accumulation_steps
    return math.ceil(n_train / effective_batch) * hp.num_train_epochs


def qwen2_lora_params(config, r, target_modules=TARGET_MODULES):
    """Trainable LoRA parameters for a Qwen2 config, computed from the config alone."""
    hidden = config["hidden_size"]
    inner = config["intermediate_size"]
    head_dim = hidden // config["num_attention_heads"]
    key_value = config["num_key_value_heads"] * head_dim
    in_out = {
        "q_proj": (hidden, hidden),
        "k_proj": (hidden, key_value),
        "v_proj": (hidden, key_value),
        "o_proj": (hidden, hidden),
        "gate_proj": (hidden, inner),
        "up_proj": (hidden, inner),
        "down_proj": (inner, hidden),
    }
    per_layer = sum(r * sum(in_out[module]) for module in target_modules)
    return per_layer * config["num_hidden_layers"]


def check_lora_model(model, hp):
    """Print and assert the r, alpha and trainable parameters actually instantiated (R1.3)."""
    peft_config = model.peft_config["default"]
    ranks = {param.shape[0] for name, param in model.named_parameters() if ".lora_A." in name}
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    expected = qwen2_lora_params(model.config.to_dict(), hp.r, hp.target_modules)

    print(f"LoRA: r={peft_config.r}, alpha={peft_config.lora_alpha}, "
          f"postos instanciados={sorted(ranks)}, treináveis={trainable}, esperados={expected}")

    assert peft_config.r == hp.r, f"r {peft_config.r} != {hp.r}"
    assert peft_config.lora_alpha == hp.lora_alpha, f"alpha {peft_config.lora_alpha} != {hp.lora_alpha}"
    assert ranks == {hp.r}, f"instantiated ranks {sorted(ranks)} != {hp.r}"
    assert trainable == expected, f"trainable parameters {trainable} != {expected}"
