# ====================================================IMPORT=================================================================
# Script version of notebooks/sft.ipynb (same defaults). Run from the project root:
#   python script/build_datasets.py
#   python script/SFT.py

from datasets import Dataset

import os
import sys
import time
import argparse
import warnings

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import logging as transformers_logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from utils.utils_train import pre_process, create_prompt, print_trainable_parameters

warnings.simplefilter("ignore")
transformers_logging.set_verbosity_error()


# ========================== CMD Argument Parser ==========================
def str2bool(value):
    # argparse's type=bool turns any non-empty string (even "False") into True
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Expected True or False, got {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) with LoRA, Tupi -> Portuguese")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size per device during evaluation")
    parser.add_argument("--src_lng", type=str, default="Tupi", help="Source language")
    parser.add_argument("--tgt_lng", type=str, default="Português", help="Target language")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--project_root", type=str, default=ROOT, help="Path to project root")
    parser.add_argument("--training_dataset_path", type=str, default=os.path.join(ROOT, "data", "dataset_sintetico_aumentado.json"), help="Gemini labels + synthetic pairs (script/build_datasets.py)")
    parser.add_argument("--historical_dataset_path", type=str, default=os.path.join(ROOT, "data", "dataset_historico.json"), help="Human translations of the historical sentences (test reference)")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Path to model")
    parser.add_argument("--resume_from_checkpoint", type=str2bool, default=False, help="Resume training from checkpoint")
    parser.add_argument("--resume_checkpoint_path", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--r", type=int, default=128, help="LoRA rank")
    parser.add_argument("--is_peft", type=str2bool, default=True, help="Use PEFT (LoRA)")
    parser.add_argument("--load_in_4bit", type=str2bool, default=False, help="Load the base model in 4-bit NF4 (QLoRA) to reduce memory usage")
    parser.add_argument("--is_train_response_only", type=str2bool, default=True, help="Compute the loss on the answer only")
    parser.add_argument("--seed", type=int, default=13, help="Seed for the train/test split and the training")
    parser.add_argument("--train_ratio", type=float, default=1.0, help="Fraction of the train and test splits to use (quick tests)")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Loss on the test split every N steps (monitoring only)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Arguments passed:")
    for name, value in vars(args).items():
        print(f"{name}: {value}")

    learning_rate = args.learning_rate
    per_device_train_batch_size = args.per_device_train_batch_size
    per_device_eval_batch_size = args.per_device_eval_batch_size
    num_train_epochs = args.num_train_epochs
    training_dataset_path = args.training_dataset_path
    historical_dataset_path = args.historical_dataset_path
    project_root = args.project_root
    model_path = args.model_path
    resume_from_checkpoint = args.resume_from_checkpoint
    resume_checkpoint_path = args.resume_checkpoint_path
    src_lng = args.src_lng
    tgt_lng = args.tgt_lng
    r = args.r
    is_peft = args.is_peft
    load_in_4bit = args.load_in_4bit
    is_train_response_only = args.is_train_response_only
    train_seed = args.seed

    fp16 = False
    bf16 = False
    train_ratio = args.train_ratio  # Fraction of samples to be used for training and evaluation
    warmup_ratio = 0.5
    logging_steps = 1000
    evaluation_strategy = "steps"  # eval loss on the 30% test split: monitoring only, never used to pick checkpoints or epochs
    save_strategy = "epoch"
    eval_steps = args.eval_steps
    max_grad_norm = 0.3
    MAX_LEN = 128
    weight_decay = 0.01
    dtype = None  # 4-bit compute dtype. None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

    if is_peft:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_alpha = 8
        lora_dropout = 0

    formatted_time = time.strftime("%m_%d_%H_%M", time.localtime(time.time()))

    if resume_from_checkpoint and resume_checkpoint_path is None:
        raise ValueError("Please provide a checkpoint path to resume training from")

    if resume_from_checkpoint:
        output_dir = resume_checkpoint_path
    else:
        input_file_name = os.path.splitext(os.path.basename(training_dataset_path))[0]
        if is_peft:
            output_dir = f"logs/peft_{r}_{src_lng[:2]}_{tgt_lng[:2]}/fit_{formatted_time}_{train_ratio}_{input_file_name}"
        else:
            output_dir = f"logs/full_{src_lng[:2]}_{tgt_lng[:2]}/fit_{formatted_time}_{train_ratio}_{input_file_name}"

    # ========================== dataset preparation ==========================

    sys.path.append(project_root)

    train_dataset_df = pd.read_json(os.path.abspath(training_dataset_path))
    pre_processed_dataset_df = pre_process(train_dataset_df)

    if not isinstance(pre_processed_dataset_df, pd.DataFrame):
        raise TypeError("data_preprocess should return a pandas DataFrame.")

    dataset = Dataset.from_pandas(pre_processed_dataset_df)

    dataset = dataset.shuffle(seed=train_seed)
    n_train = int(0.7 * len(dataset))
    dataset = dataset.map(
        lambda x, idx: {
            "split": "train" if idx < n_train else "test"
        },
        with_indices=True
    )

    # Filter by split
    train_dataset = dataset.filter(lambda x: x["split"] == "train")
    test_dataset = dataset.filter(lambda x: x["split"] == "test")

    # Select subset
    train_dataset = train_dataset.select(range(int(len(train_dataset) * train_ratio)))
    test_dataset = test_dataset.select(range(int(len(test_dataset) * train_ratio)))

    # Test reference: human translation for historical sentences (their training label is Gemini's), template label for synthetic ones
    historical_df = pd.read_json(historical_dataset_path)
    gold_by_id = dict(zip(historical_df["id"], historical_df["output"]))
    test_dataset = test_dataset.map(lambda x: {"Referência": gold_by_id.get(x["id"], x["output"])})

    # Rename columns
    train_dataset = train_dataset.rename_columns({
        "input": "Tupi",
        "output": "Português",
    })

    test_dataset = test_dataset.rename_columns({
        "input": "Tupi",
        "output": "Português",
    })

    # Save the split next to the checkpoints, so the evaluation reads the test set of this same run
    os.makedirs(output_dir, exist_ok=True)
    test_dataset.to_json(os.path.join(output_dir, "test_dataset.json"))
    train_dataset.to_json(os.path.join(output_dir, "train_dataset.json"))

    # Keep Qwen's own pad token (<|endoftext|>): padding with eos (<|im_end|>) masks the end of the answer out of the loss
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"

    # eos_rep=0: the chat template already closes the answer with <|im_end|>
    train_dataset = train_dataset.map(
        lambda sample: {
            "full_prompt": create_prompt(sample, src_lng=src_lng, tgt_lng=tgt_lng, eos_rep=0, mode="train", tokenizer=tokenizer)["full_prompt"]
        }
    ).select_columns(["full_prompt"])

    test_dataset = test_dataset.map(
        lambda sample: {
            "full_prompt": create_prompt(sample, src_lng=src_lng, tgt_lng=tgt_lng, eos_rep=0, mode="train", tokenizer=tokenizer)["full_prompt"]
        }
    ).select_columns(["full_prompt"])

    print(train_dataset["full_prompt"][0])

    def tokenize_function(examples):
        return tokenizer(
            examples["full_prompt"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["full_prompt"])
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["full_prompt"])

    # ========================== model ==========================

    # Seed before the LoRA matrices are initialised (the Trainer only seeds itself later, when it is created)
    set_seed(train_seed)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config if load_in_4bit else None,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if is_peft:
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",  # without it the Trainer finds no "labels" argument and never computes eval_loss
            use_rslora=False,
        )
        model = get_peft_model(model, lora_config)

    print(model)
    print(print_trainable_parameters(model))

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Shape: {param.shape}, Trainable parameters: {param.numel()}")
        else:
            print(f"Layer: {name}, Shape: {param.shape}, Non-trainable parameters: {param.numel()}")

    # ========================== training ==========================

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_ratio=warmup_ratio,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        remove_unused_columns=False,
        disable_tqdm=False,
        seed=train_seed,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        dataset_text_field="full_prompt",
        max_seq_length=MAX_LEN,
        dataset_num_proc=2,
        packing=False,
    )

    # Loss only on the answer: the fixed prompt and the Tupi source are masked out (-100)
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer,
    ) if is_train_response_only else None

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
    )

    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("Finished training SFT.")

    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# The guard matters on Windows: the DataLoader workers re-import this file
if __name__ == "__main__":
    main()
