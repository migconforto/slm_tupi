# Experimentos de SFT

Novos experimentos de *fine-tuning* Tupi Antigo → Português, definidos em `SPEC_experimentos_sft.md`. Tudo aqui é aditivo: nenhum arquivo do pipeline original foi alterado. O relatório completo, com decisões e divergências, está em [CHANGES.md](CHANGES.md).

## Experimentos

| Experimento | Dataset | Prompt | Hiperparâmetros | Fator que muda vs `exp01` |
|---|---|---|---|---|
| Baseline (`notebooks/sft.ipynb`, não tocar) | `dataset_sintetico_aumentado.json` (v2) | atual | atuais | — |
| `exp01_hparams` | v2 | atual | `hp_v2` | referência dos demais |
| `exp02_lexicon` | v2 | atual + glossário | `hp_v2` | prompt |
| `exp03_split_aug` | `dataset_augmentation_v3_seed<N>.json` (treino aumentado) | atual | `hp_v2` | dados |

Dentro de uma mesma semente, todos usam a mesma validação de 593 exemplos. Os notebooks conferem isso pelo hash dos IDs.

O perfil `hp_v2` fica em `tupi_exp/config.py`. Para testar outro valor, edite só esse arquivo.

| Parâmetro | Baseline | `hp_v2` |
|---|---:|---:|
| `r` | 128 | 64 |
| `lora_alpha` | 8 | 64 |
| `lora_dropout` | 0 | 0,05 |
| `learning_rate` | 1e-5 | 1e-4 |
| `warmup_ratio` | 0,5 | 0,05 |
| `max_grad_norm` | 0,3 | 1,0 |
| `gradient_accumulation_steps` | 1 | 8 |
| `num_train_epochs` | 3 | 3 |
| `seed` | `SEED` | `SEED` |

## Estrutura

```
experiments/
  tupi_exp/                  módulo novo
    config.py                perfis de hiperparâmetros e experimentos
    normalize.py             normalize_tupi()
    audit_orthography.py     auditoria de ortografia
    splits.py                reproduz e salva o split
    split_augment.py         partição de frases longas e geração do v3
    lexicon.py               glossário, leave-one-out e render_prompt()
    dryrun.py                verificação sem carregar o modelo
  tests/                     testes unitários
  reports/                   relatórios gerados
  exp01_hparams/             sft_exp01.ipynb e eval_exp01.ipynb
  exp02_lexicon/             sft_exp02.ipynb e eval_exp02.ipynb
  exp03_split_aug/           sft_exp03.ipynb e eval_exp03.ipynb
splits/                      IDs de treino e validação, por semente
data/dataset_augmentation_v3_seed<N>.json
```

## Como rodar

Os comandos abaixo partem da raiz do projeto.

Testes:

```bash
python -m pytest experiments/tests -q -p no:cacheprovider
```

Dry-run dos três experimentos, sem carregar o modelo:

```bash
cd experiments && python -m tupi_exp.dryrun
```

Regenerar a auditoria (só leitura sobre o dataset):

```bash
cd experiments && python -m tupi_exp.audit_orthography
```

Regenerar os splits, o v3 e o relatório da partição:

```bash
cd experiments && python -m tupi_exp.split_augment
```

**Treino:** abra `experiments/<experimento>/sft_expNN.ipynb` e use "Run All". A pasta de cada execução fica em `experiments/<experimento>/logs/`, com os *checkpoints* e o `test_dataset.json` daquela execução. O nome da pasta termina com a semente, como em `fit_..._dataset_sintetico_aumentado-SEED51`.

**Avaliação:** em `experiments/<experimento>/eval_expNN.ipynb`, aponte `run_dir` para a pasta da execução e use "Run All". O resultado sai em `run_dir/predictions_<experimento>-SEED<N>.csv`, com a coluna `Subconjunto` (`historico` ou `sintetico`). Com `use_lora = False`, o notebook gera a linha de base `predictions_base-SEED<N>.csv`. A semente do nome vem do sufixo `-SEED<N>` da pasta avaliada; sem esse sufixo, vale a `SEED` do `config.py`.

## Trocar a semente

A semente fica em uma única linha: `SEED`, em `tupi_exp/config.py`. Ela controla a divisão treino/validação e o treino, e qualquer número inteiro funciona.

Para trocar:

1. Mude `SEED` em `tupi_exp/config.py`.
2. Reinicie o kernel do notebook e use "Run All". O Python só lê o `config.py` uma vez por kernel.

Na primeira vez que uma semente é usada, o notebook gera sozinho os dados dela, em poucos segundos: `splits/seed<N>_*`, `data/dataset_augmentation_v3_seed<N>.json` e `experiments/reports/split_augment_seed<N>.md`. Nas próximas vezes, esses arquivos são reaproveitados. Se quiser conferir tudo antes de treinar, rode o dry-run:

```bash
cd experiments && python -m tupi_exp.dryrun
```

Dentro de uma semente, os experimentos são comparáveis frase a frase. Entre sementes não são, porque a validação muda. Para comparar com o baseline, ele precisa ter sido treinado com a mesma semente, e hoje só existe o da seed 13. Para treiná-lo com outra, mude `seed` em `notebooks/sft.ipynb`.

No `eval_exp02`, a `SEED` do `config.py` também precisa ser a do treino, porque o glossário depende dela. Se for outra, o notebook recusa a execução.

## Diferenças para os notebooks originais

Os notebooks foram criados como cópias de `notebooks/sft.ipynb` e `notebooks/eval_sft.ipynb`, sem as saídas salvas. A lista abaixo mostra todas as linhas alteradas.

### `exp01_hparams/sft_exp01.ipynb`

**Célula `b2231091`**

```diff
+sys.path.append("D:\\Dissertacao\\slm_tupi\\experiments")
+from tupi_exp.config import BASELINE_VAL_SHA256, SPLIT_SEED, check_lora_model, get_experiment
+from tupi_exp.splits import ids_hash
+
+experiment = get_experiment("exp01_hparams")
+hp = experiment.profile
+print(hp)
+
-    per_device_train_batch_size=1,
-    per_device_eval_batch_size=1,
+    per_device_train_batch_size=hp.per_device_train_batch_size,
+    per_device_eval_batch_size=hp.per_device_eval_batch_size,
-    num_train_epochs=3,
-    learning_rate=1e-5,
+    num_train_epochs=hp.num_train_epochs,
+    learning_rate=hp.learning_rate,
-    # Generated by script/build_datasets.py
-    training_dataset_path = r"D:\Dissertacao\slm_tupi\data\dataset_sintetico_aumentado.json",  # Gemini labels + synthetic pairs
+    training_dataset_path = experiment.dataset_path,
-    r=128,
+    r=hp.r,
-    seed=13
+    seed=hp.seed
```

**Célula `3b541bc3`**

```diff
-warmup_ratio = 0.5
-logging_steps = 1000
-evaluation_strategy="steps"  # eval loss on the 30% test split: monitoring only, never used to pick checkpoints or epochs
-save_strategy="epoch"
-eval_steps=1000
-max_grad_norm = 0.3
-MAX_LEN = 128
-weight_decay = 0.01
+warmup_ratio = hp.warmup_ratio
+logging_steps = hp.logging_steps
+evaluation_strategy=hp.evaluation_strategy  # eval loss on the 30% test split: monitoring only, never used to pick checkpoints or epochs
+save_strategy=hp.save_strategy
+eval_steps=hp.eval_steps
+max_grad_norm = hp.max_grad_norm
+MAX_LEN = experiment.max_seq_length
+weight_decay = hp.weight_decay
-    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",]
-    lora_alpha = 8
-    lora_dropout = 0
+    target_modules = list(hp.target_modules)
+    lora_alpha = hp.lora_alpha
+    lora_dropout = hp.lora_dropout
-            f"logs/peft_{r}_{src_lng[:2]}_{tgt_lng[:2]}/"
-            f"fit_{formatted_time}_{train_ratio}_{input_file_name}"
+            f"logs/{experiment.name}/peft_{r}_{src_lng[:2]}_{tgt_lng[:2]}/"
+            f"fit_{formatted_time}_{train_ratio}_{input_file_name}-SEED{train_seed}"
-            f"fit_{formatted_time}_{train_ratio}_{input_file_name}"
+            f"fit_{formatted_time}_{train_ratio}_{input_file_name}-SEED{train_seed}"
```

**Célula `75c796c5`**

```diff
-dataset = dataset.shuffle(seed=train_seed)
+dataset = dataset.shuffle(seed=SPLIT_SEED)
+
+# Same validation as the baseline in every experiment (P6)
+assert ids_hash(test_dataset["id"]) == BASELINE_VAL_SHA256, "validation differs from the baseline"
```

**Célula `d41f5a46`**

```diff
+check_lora_model(model, hp)
```

**Célula `30a11aef`**

```diff
+        gradient_accumulation_steps=hp.gradient_accumulation_steps,
-        lr_scheduler_type="cosine",
+        lr_scheduler_type=hp.lr_scheduler_type,
```

Células sem alteração: `cc53fead`, `bef846ff`, `dc785f2d`, `7d76bd4e`, `4e276d75`, `2fd859c0`.

### `exp01_hparams/eval_exp01.ipynb`

**Célula `dd80eb12`**

```diff
+
+sys.path.append("D:\\Dissertacao\\slm_tupi\\experiments")
+from tupi_exp.config import SUBCONJUNTO, seed_of_run
```

**Célula `9dbd3152`**

```diff
-run_dir = r"D:\Dissertacao\slm_tupi\notebooks\logs\peft_128_Tu_Po\fit_10_29_23_27_1.0_dataset_sintetico_aumentado"
-lora_path = os.path.join(run_dir, "checkpoint-1382")
+run_dir = r"D:\Dissertacao\slm_tupi\experiments\exp01_hparams\logs\exp01_hparams\peft_64_Tu_Po\fit_10_31_23_03_1.0_dataset_sintetico_aumentado-SEED42"
+run_seed = seed_of_run(run_dir)  # from the -SEED<N> suffix of the run folder
+lora_path = os.path.join(run_dir, "checkpoint-516")
```

**Célula `f0e0d8aa`**

```diff
+        "Subconjunto": SUBCONJUNTO[row["fonte"]],
```

**Célula `3f413811`**

```diff
-    os.path.join(run_dir, "predictions.csv" if use_lora else "predictions_base.csv"),
+    os.path.join(run_dir, f"predictions_exp01_hparams-SEED{run_seed}.csv" if use_lora else f"predictions_base-SEED{run_seed}.csv"),
```

Células sem alteração: nenhuma.

### `exp02_lexicon/sft_exp02.ipynb`

**Célula `b2231091`**

```diff
+sys.path.append("D:\\Dissertacao\\slm_tupi\\experiments")
+from tupi_exp.config import BASELINE_VAL_SHA256, SPLIT_SEED, check_lora_model, get_experiment
+from tupi_exp.splits import ids_hash
+from tupi_exp.lexicon import check_no_leak, load_lexicon, render_prompt
+
+experiment = get_experiment("exp02_lexicon")
+hp = experiment.profile
+print(hp)
+
-    per_device_train_batch_size=1,
-    per_device_eval_batch_size=1,
+    per_device_train_batch_size=hp.per_device_train_batch_size,
+    per_device_eval_batch_size=hp.per_device_eval_batch_size,
-    num_train_epochs=3,
-    learning_rate=1e-5,
+    num_train_epochs=hp.num_train_epochs,
+    learning_rate=hp.learning_rate,
-    # Generated by script/build_datasets.py
-    training_dataset_path = r"D:\Dissertacao\slm_tupi\data\dataset_sintetico_aumentado.json",  # Gemini labels + synthetic pairs
+    training_dataset_path = experiment.dataset_path,
-    r=128,
+    r=hp.r,
-    seed=13
+    seed=hp.seed
```

**Célula `3b541bc3`**

```diff
-warmup_ratio = 0.5
-logging_steps = 1000
-evaluation_strategy="steps"  # eval loss on the 30% test split: monitoring only, never used to pick checkpoints or epochs
-save_strategy="epoch"
-eval_steps=1000
-max_grad_norm = 0.3
-MAX_LEN = 128
-weight_decay = 0.01
+warmup_ratio = hp.warmup_ratio
+logging_steps = hp.logging_steps
+evaluation_strategy=hp.evaluation_strategy  # eval loss on the 30% test split: monitoring only, never used to pick checkpoints or epochs
+save_strategy=hp.save_strategy
+eval_steps=hp.eval_steps
+max_grad_norm = hp.max_grad_norm
+MAX_LEN = experiment.max_seq_length
+weight_decay = hp.weight_decay
-    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",]
-    lora_alpha = 8
-    lora_dropout = 0
+    target_modules = list(hp.target_modules)
+    lora_alpha = hp.lora_alpha
+    lora_dropout = hp.lora_dropout
-            f"logs/peft_{r}_{src_lng[:2]}_{tgt_lng[:2]}/"
-            f"fit_{formatted_time}_{train_ratio}_{input_file_name}"
+            f"logs/{experiment.name}/peft_{r}_{src_lng[:2]}_{tgt_lng[:2]}/"
+            f"fit_{formatted_time}_{train_ratio}_{input_file_name}-SEED{train_seed}"
-            f"fit_{formatted_time}_{train_ratio}_{input_file_name}"
+            f"fit_{formatted_time}_{train_ratio}_{input_file_name}-SEED{train_seed}"
```

**Célula `75c796c5`**

```diff
-dataset = dataset.shuffle(seed=train_seed)
+dataset = dataset.shuffle(seed=SPLIT_SEED)
+
+# Same validation as the baseline in every experiment (P6)
+assert ids_hash(test_dataset["id"]) == BASELINE_VAL_SHA256, "validation differs from the baseline"
+# Glossary built from the train split only (P4)
+lexicon = load_lexicon()
+check_no_leak(lexicon, [{"id": row["id"], "input": row["Tupi"]} for row in test_dataset])
+
-        "full_prompt": create_prompt(sample, src_lng=src_lng, tgt_lng=tgt_lng, eos_rep=0, mode="train", tokenizer=tokenizer)["full_prompt"]
+        "full_prompt": render_prompt(sample, lexicon, src_lng=src_lng, tgt_lng=tgt_lng, eos_rep=0, mode="train", tokenizer=tokenizer, example_id=sample["id"])["full_prompt"]
-        "full_prompt": create_prompt(sample, src_lng=src_lng, tgt_lng=tgt_lng, eos_rep=0, mode="train", tokenizer=tokenizer)["full_prompt"]
+        "full_prompt": render_prompt(sample, lexicon, src_lng=src_lng, tgt_lng=tgt_lng, eos_rep=0, mode="train", tokenizer=tokenizer, example_id=sample["id"])["full_prompt"]
```

**Célula `d41f5a46`**

```diff
+check_lora_model(model, hp)
```

**Célula `30a11aef`**

```diff
+        gradient_accumulation_steps=hp.gradient_accumulation_steps,
-        lr_scheduler_type="cosine",
+        lr_scheduler_type=hp.lr_scheduler_type,
```

Células sem alteração: `cc53fead`, `bef846ff`, `dc785f2d`, `7d76bd4e`, `4e276d75`, `2fd859c0`.

### `exp02_lexicon/eval_exp02.ipynb`

**Célula `dd80eb12`**

```diff
+
+sys.path.append("D:\\Dissertacao\\slm_tupi\\experiments")
+from tupi_exp.config import BASELINE_VAL_SHA256, SUBCONJUNTO, seed_of_run
+from tupi_exp.splits import ids_hash
+from tupi_exp.lexicon import load_lexicon, render_prompt
```

**Célula `9dbd3152`**

```diff
-run_dir = r"D:\Dissertacao\slm_tupi\notebooks\logs\peft_128_Tu_Po\fit_10_29_23_27_1.0_dataset_sintetico_aumentado"
-lora_path = os.path.join(run_dir, "checkpoint-1382")
+run_dir = r"logs\exp02_lexicon\peft_64_Tu_Po\fit_..."
+run_seed = seed_of_run(run_dir)  # from the -SEED<N> suffix of the run folder
+lora_path = os.path.join(run_dir, "checkpoint-516")
```

**Célula `f0e0d8aa`**

```diff
+# The glossary comes from the train split of the seed in config.py: the run must use that same seed (P4)
+assert ids_hash(df["id"]) == BASELINE_VAL_SHA256, "run_dir belongs to another seed: set SEED in config.py"
+
+lexicon = load_lexicon()
-    prompt = create_prompt(row, src_lng="Tupi", tgt_lng="Português", mode="inference", tokenizer=tokenizer)["full_prompt"]
+    prompt = render_prompt(row, lexicon, src_lng="Tupi", tgt_lng="Português", mode="inference", tokenizer=tokenizer, example_id=row["id"])["full_prompt"]
+        "Subconjunto": SUBCONJUNTO[row["fonte"]],
```

**Célula `3f413811`**

```diff
-    os.path.join(run_dir, "predictions.csv" if use_lora else "predictions_base.csv"),
+    os.path.join(run_dir, f"predictions_exp02_lexicon-SEED{run_seed}.csv" if use_lora else f"predictions_base-SEED{run_seed}.csv"),
```

Células sem alteração: nenhuma.

### `exp03_split_aug/sft_exp03.ipynb`

**Célula `b2231091`**

```diff
+sys.path.append("D:\\Dissertacao\\slm_tupi\\experiments")
+from tupi_exp.config import BASELINE_VAL_SHA256, SPLIT_SEED, check_lora_model, get_experiment
+from tupi_exp.splits import ids_hash
+from tupi_exp.splits import load_split
+
+experiment = get_experiment("exp03_split_aug")
+hp = experiment.profile
+print(hp)
+
-    per_device_train_batch_size=1,
-    per_device_eval_batch_size=1,
+    per_device_train_batch_size=hp.per_device_train_batch_size,
+    per_device_eval_batch_size=hp.per_device_eval_batch_size,
-    num_train_epochs=3,
-    learning_rate=1e-5,
+    num_train_epochs=hp.num_train_epochs,
+    learning_rate=hp.learning_rate,
-    # Generated by script/build_datasets.py
-    training_dataset_path = r"D:\Dissertacao\slm_tupi\data\dataset_sintetico_aumentado.json",  # Gemini labels + synthetic pairs
+    training_dataset_path = experiment.dataset_path,
-    r=128,
+    r=hp.r,
-    seed=13
+    seed=hp.seed
```

**Célula `3b541bc3`**

```diff
-warmup_ratio = 0.5
-logging_steps = 1000
-evaluation_strategy="steps"  # eval loss on the 30% test split: monitoring only, never used to pick checkpoints or epochs
-save_strategy="epoch"
-eval_steps=1000
-max_grad_norm = 0.3
-MAX_LEN = 128
-weight_decay = 0.01
+warmup_ratio = hp.warmup_ratio
+logging_steps = hp.logging_steps
+evaluation_strategy=hp.evaluation_strategy  # eval loss on the 30% test split: monitoring only, never used to pick checkpoints or epochs
+save_strategy=hp.save_strategy
+eval_steps=hp.eval_steps
+max_grad_norm = hp.max_grad_norm
+MAX_LEN = experiment.max_seq_length
+weight_decay = hp.weight_decay
-    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",]
-    lora_alpha = 8
-    lora_dropout = 0
+    target_modules = list(hp.target_modules)
+    lora_alpha = hp.lora_alpha
+    lora_dropout = hp.lora_dropout
-            f"logs/peft_{r}_{src_lng[:2]}_{tgt_lng[:2]}/"
-            f"fit_{formatted_time}_{train_ratio}_{input_file_name}"
+            f"logs/{experiment.name}/peft_{r}_{src_lng[:2]}_{tgt_lng[:2]}/"
+            f"fit_{formatted_time}_{train_ratio}_{input_file_name}-SEED{train_seed}"
-            f"fit_{formatted_time}_{train_ratio}_{input_file_name}"
+            f"fit_{formatted_time}_{train_ratio}_{input_file_name}-SEED{train_seed}"
```

**Célula `75c796c5`**

```diff
-dataset = dataset.shuffle(seed=train_seed)
+# v3 only adds train rows: the split comes from the saved ids, not from the row positions
+val_ids = set(load_split()[1])
+dataset = dataset.shuffle(seed=SPLIT_SEED)
-    lambda x, idx: {
-        "split": "train" if idx < int(0.7 * len(dataset)) else "test"
-    },
-    with_indices=True
+    lambda x: {
+        "split": "test" if x["id"] in val_ids else "train"
+    }
+
+# Same validation as the baseline in every experiment (P6)
+assert ids_hash(test_dataset["id"]) == BASELINE_VAL_SHA256, "validation differs from the baseline"
```

**Célula `d41f5a46`**

```diff
+check_lora_model(model, hp)
```

**Célula `30a11aef`**

```diff
+        gradient_accumulation_steps=hp.gradient_accumulation_steps,
-        lr_scheduler_type="cosine",
+        lr_scheduler_type=hp.lr_scheduler_type,
```

Células sem alteração: `cc53fead`, `bef846ff`, `dc785f2d`, `7d76bd4e`, `4e276d75`, `2fd859c0`.

### `exp03_split_aug/eval_exp03.ipynb`

**Célula `dd80eb12`**

```diff
+
+sys.path.append("D:\\Dissertacao\\slm_tupi\\experiments")
+from tupi_exp.config import SUBCONJUNTO, seed_of_run
```

**Célula `9dbd3152`**

```diff
-run_dir = r"D:\Dissertacao\slm_tupi\notebooks\logs\peft_128_Tu_Po\fit_10_29_23_27_1.0_dataset_sintetico_aumentado"
-lora_path = os.path.join(run_dir, "checkpoint-1382")
+run_dir = r"logs\exp03_split_aug\peft_64_Tu_Po\fit_..."
+run_seed = seed_of_run(run_dir)  # from the -SEED<N> suffix of the run folder
+lora_path = os.path.join(run_dir, "checkpoint-528")
```

**Célula `f0e0d8aa`**

```diff
+        "Subconjunto": SUBCONJUNTO[row["fonte"]],
```

**Célula `3f413811`**

```diff
-    os.path.join(run_dir, "predictions.csv" if use_lora else "predictions_base.csv"),
+    os.path.join(run_dir, f"predictions_exp03_split_aug-SEED{run_seed}.csv" if use_lora else f"predictions_base-SEED{run_seed}.csv"),
```

Células sem alteração: nenhuma.

