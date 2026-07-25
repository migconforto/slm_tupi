# Dados

| Arquivo | Descrição |
|---|---|
| `dataset_augmentated.json` | Conjunto de treino Tupi→Português (aumentado). Colunas `input` (Tupi) e `output` (Português). |
| `val_dataset.json` | Conjunto de validação (gerado pelo `sft.ipynb`). |
| `train_dataset.json` | Conjunto de treino efetivo (gerado pelo `sft.ipynb`). |

## Formato

Cada registro é um objeto JSON com, no mínimo:

```json
{ "input": "kunhãmukuetá îkó ka'ape", "output": "moças vivem na mata" }
```
