# Dados

| Arquivo | Descrição |
|---|---|
| `dataset_augmentated.json` | Conjunto de treino Tupi→Português (aumentado) sintético. Colunas `input` (Tupi) e `output` (Português). |
| `gemini-output_dataset.json` | Conjunto de frases históricas com as referências padrão-ouro e traduções do Gemini. |

## Formato

Cada registro é um objeto JSON com, no mínimo:

```json
{ "input": "kunhãmukuetá îkó ka'ape", "output": "moças vivem na mata" }
```