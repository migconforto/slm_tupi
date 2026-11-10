"""Named hyperparameter profiles and experiment definitions (R1.1)."""
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, replace

from tupi_exp import DATA_DIR, SPLITS_DIR

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# Seed of the experiments: change only this line. It drives both the train/validation split and the training.
# The split and the v3 of a new seed are generated the first time the seed is used
SEED = 99
SPLIT_SEED = SEED
TRAIN_SHARE = 0.7

# Validation of the baseline runs already made, by seed. A seed outside this table has no baseline to compare with
BASELINE_RUNS_SHA256 = {13: "86c9431b2bf3d5265f30bf166dab5b9834dfab2bbb80d05b8882f8b022a4e5f3"}

# Glossary of exp02 (R2.3, R2.4)
LEXICON_MAX_TOKENS = 2
LEXICON_MAX_ENTRIES = 8

# Output spreadsheet column (section 6 of the SPEC)
SUBCONJUNTO = {"gemini": "historico", "sintetico": "sintetico"}


def seed_of_run(run_dir):
    """Seed of a training run, read from the -SEED<N> suffix of its folder (the config SEED when there is none)."""
    match = re.search(r"-SEED(\d+)$", os.path.basename(os.path.normpath(run_dir)))
    if match:
        return int(match.group(1))
    print(f"{run_dir} não termina em -SEED<N>: usando a SEED do config.py ({SEED})")
    return SEED


def split_path(name, seed=SPLIT_SEED):
    return os.path.join(SPLITS_DIR, f"seed{seed}_{name}_ids.json")


def ids_hash(ids):
    """SHA-256 of the sorted id list, so the order of the examples does not matter."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def saved_val_hash(seed=SPLIT_SEED):
    """Hash of the saved validation of a seed, or None while its split is not generated."""
    if not os.path.exists(split_path("val", seed)):
        return None
    with open(split_path("val", seed), encoding="utf-8") as f:
        return ids_hash(json.load(f))


def ensure_seed_data(seed=SPLIT_SEED):
    """Generates the split and the v3 of a seed the first time it is used."""
    if all(os.path.exists(path) for path in (split_path("train", seed), split_path("val", seed), v3_path(seed))):
        return
    print(f"Seed {seed}: primeiro uso, gerando o split e o v3 em splits/ e data/")
    from tupi_exp.split_augment import generate  # late import: split_augment imports this module

    generate(seed)


def __getattr__(name):
    # Every experiment of a seed must use this validation (P6). Computed on first use (PEP 562),
    # so that the data of a new seed already exist
    if name == "BASELINE_VAL_SHA256":
        ensure_seed_data()
        globals()[name] = saved_val_hash()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def require_seed_data(*paths):
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found: generate the data of this seed with `python -m tupi_exp.split_augment`, from experiments/")


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


# Values read from notebooks/sft.ipynb, except the seed; tests/test_config.py fails if they drift apart
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
    seed=SEED,
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
HISTORICAL_PATH = os.path.join(DATA_DIR, "dataset_historico.json")


def v3_path(seed=SPLIT_SEED):
    """v3 comes from the train split, so each seed has its own file: the v3 of another seed would leak validation."""
    return os.path.join(DATA_DIR, f"dataset_augmentation_v3_seed{seed}.json")


V3_PATH = v3_path()

EXPERIMENTS = {
    "exp01_hparams": Experiment("exp01_hparams", V2_PATH, "atual", 128, HP_V2),
    "exp02_lexicon": Experiment("exp02_lexicon", V2_PATH, "glossario", 152, HP_V2),  # longest prompt of any seed: 149 tokens (R2.7)
    "exp03_split_aug": Experiment("exp03_split_aug", V3_PATH, "atual", 128, HP_V2),
}


def get_profile(name):
    return PROFILES[name]


def get_experiment(name):
    experiment = EXPERIMENTS[name]
    ensure_seed_data()
    require_seed_data(split_path("val"), experiment.dataset_path)
    return experiment


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
