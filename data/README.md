# Dados

Esta pasta guarda os **datasets** do projeto. O conteúdo **não é versionado**
(ver `.gitignore`) — apenas cópias locais.

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

## Origem dos caminhos

No experimento original os dados estavam em:

```
C:\Users\ ... \dataset_augmentated.json
```

Ao migrar para esta estrutura, coloque o arquivo em `data/` e atualize a variável
`training_dataset_path` no notebook/script para um caminho relativo, por exemplo
`data/dataset_augmentated.json`.
