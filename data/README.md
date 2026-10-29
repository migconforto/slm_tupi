# Dados

| Arquivo | Descrição |
|---|---|
| `dataset_sintetico_aumentado.json` | **Conjunto de treino do SFT.** Frases históricas com a tradução do Gemini como rótulo (`fonte = "gemini"`) + pares sintéticos gerados por *templates* (`fonte = "sintetico"`). Gerado por `script/build_datasets.py`. |
| `dataset_historico.json` | Frases históricas com a tradução humana (padrão-ouro). É a referência da avaliação nas frases históricas. Gerado por `script/build_datasets.py`. |
| `gemini-output_dataset.json` | Fonte bruta das frases históricas: referência padrão-ouro (`output`) e tradução do Gemini (`output_gemini`). |
| `dataset_augmented.json` | Fonte bruta dos pares sintéticos (registros sem `instruction`). Nos registros históricos, o `output` é a tradução humana, por isso ele não é usado diretamente no treino. |

## Formato

Cada registro é um objeto JSON com, no mínimo:

```json
{ "id": "sint_0001", "input": "ixé kunhã", "output": "Eu sou mulher", "fonte": "sintetico" }
```

O `id` das frases históricas (`hist_XXXX`) é o mesmo nos dois arquivos gerados, o que liga o rótulo do Gemini à tradução humana.

## Geração e divisão

```bash
python script/build_datasets.py
```

O script remove pares vazios e frases em Tupi repetidas (comparadas após normalização de caixa, pontuação e apóstrofos), inclusive entre as partes histórica e sintética.

A divisão 70/30 em treino e teste é feita em `notebooks/sft.ipynb` (semente 13) sobre `dataset_sintetico_aumentado.json` e salva na pasta da execução (`train_dataset.json` e `test_dataset.json`, ao lado dos *checkpoints*). No teste, a coluna `Referência` traz a tradução humana nas frases históricas e o rótulo do *template* nas sintéticas.
