# CHANGES — Novos experimentos de SFT

Relatório de execução do `SPEC_experimentos_sft.md`. Quando o SPEC e o código divergem, vale o código (regra de precedência do SPEC, §1) e a divergência fica registrada na seção 5.

## 1. Respostas da T0

### 1.1 Como o split é feito e como reproduzi-lo

Em `notebooks/sft.ipynb`, célula `75c796c5`:

1. `pd.read_json(data/dataset_sintetico_aumentado.json)` e `pre_process` (`utils/utils_train.py`), que remove vazios e duplicatas (0 linhas removidas neste arquivo).
2. `Dataset.from_pandas` e `dataset.shuffle(seed=train_seed)`.
3. Índice `< int(0.7 * len(dataset))` vai para treino; o resto, para teste.

`train_seed` vem de `args.seed = 13` (células `b2231091` e `3b541bc3`). O resultado é **1.382 treino / 593 teste**.

Reproduzi o split fora do notebook. Ele é idêntico à execução real `notebooks/logs/peft_128_Tu_Po/fit_10_29_23_27_1.0_dataset_sintetico_aumentado/`: mesmos IDs, na mesma ordem, nos dois arquivos (`train_dataset.json` e `test_dataset.json`).

Hash SHA-256 da lista ordenada de IDs de teste (baseline):
`86c9431b2bf3d5265f30bf166dab5b9834dfab2bbb80d05b8882f8b022a4e5f3`

### 1.2 Como distinguir históricos de sintéticos

Há forma confiável, sem heurística. Cada linha de `data/dataset_sintetico_aumentado.json` tem o campo `fonte`, gravado por `script/build_datasets.py`:

| `fonte` | Prefixo do `id` | Linhas | Conteúdo |
|---|---|---:|---|
| `gemini` | `hist_` | 824 | Frases históricas; `output` é a tradução do Gemini |
| `sintetico` | `sint_` | 1.151 | Pares de *template* |

A tradução humana das históricas fica em `data/dataset_historico.json` (829 linhas), ligada pelo mesmo `id`.

### 1.3 Hiperparâmetros reais × SPEC

Valores lidos de `notebooks/sft.ipynb` (células `b2231091`, `3b541bc3`, `d41f5a46`, `30a11aef`).

| Parâmetro | SPEC ("Atual") | Código | Observação |
|---|---|---|---|
| `r` | 128 efetivo (argumento 256) | **128** | Sem ambiguidade: argumento e modelo instanciado coincidem (239.468.544 parâmetros treináveis, conferido na saída da célula `d41f5a46`) |
| `lora_alpha` | 8 | 8 | |
| `lora_dropout` | 0 | 0 | |
| `learning_rate` | 1e-5 | 1e-5 | |
| `lr_scheduler_type` | cosine | cosine | |
| `warmup_ratio` | 0,5 | 0,5 | |
| `weight_decay` | 0,01 | 0,01 | |
| `max_grad_norm` | 0,3 | 0,3 | |
| `num_train_epochs` | 3 (notebook: 1) | **3** no argumento | A execução `fit_10_29_23_27` rodou com 1 época (`trainer_state.json`: 1.382 passos); o notebook foi alterado para 3 depois dela |
| `per_device_train_batch_size` | 1 | 1 | |
| `gradient_accumulation_steps` | 1 | não definido (padrão 1) | |
| `max_seq_length` | — | 128 (`MAX_LEN`) | |
| Precisão / quantização | fp32 / nenhuma | fp32 / nenhuma | `load_in_4bit = False`; `fp16 = bf16 = False` |
| `seed` | 42 | **13** | Uma única variável controla split e treino |
| Avaliação no treino | — | `evaluation_strategy="steps"`, `eval_steps=1000` | `logging_steps=1000`; `save_strategy="epoch"` |
| Perda | — | só na resposta | `is_train_response_only=True`, `DataCollatorForCompletionOnlyLM` |

### 1.4 Código de avaliação e normalização das métricas

- **Geração das predições:** `notebooks/eval_sft.ipynb` e `script/EVAL_SFT.py`. Saída em `predictions.csv`, com as colunas `Tupi`, `fonte`, `Português_esperado`, `Português_gemini` e `Português_predito`.
- **Cálculo das métricas:** **não existe código** no repositório nem no histórico do git. A pasta `D:\Dissertacao\métricas\` só tem a planilha `avaliacao_traducoes-before_after-notas-SEED_42.xlsx` (valores fixos, sem fórmulas) e `tempo - SEED_13.txt`.
- **Normalização:** como não há código, a regra foi inferida das colunas `*_Normalizado*` da planilha. Esta expressão reproduz 1.207 de 1.207 valores de `tupi`, 1.203 de 1.203 de `Predicao` e 1.206 de 1.207 de `Referencia`:

```python
" ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
```

Ela troca apóstrofos e hífens por espaço (`ka'ape` vira `ka ape`), ou seja, parte palavras na oclusiva glotal. A única exceção é a linha com `Referencia = "estou na praia"` e `Referencia_Normalizada = "estou prep_y praia"`, um marcador de *template* que ficou na planilha.

### 1.5 Dicionário, vocabulário ou listas do gerador sintético

Não existem. A busca por `dicion*`, `glossar*`, `vocab*`, `lexic*`, `template*`, `gerador*` e `generator*` não encontrou nada na árvore de arquivos nem em todo o histórico do git. O gerador de frases sintéticas não está no repositório, só o resultado dele. Por isso, a prioridade 1 da R2.3 não se aplica.

### 1.6 A seed altera o split, o arquivo de dados ou a proporção?

Só altera **quais** linhas vão para cada lado. Não altera a proporção nem o arquivo.

| Arquivo | Linhas | Seed | Treino | Teste |
|---|---:|---:|---:|---:|
| `data/dataset_sintetico_aumentado.json` | 1.975 | 13 | 1.382 | 593 |
| `data/dataset_sintetico_aumentado.json` | 1.975 | 42 | 1.382 | 593 |
| Base antiga (commit `c94fc1e`) | 2.012 | 42 | 1.408 | 604 |

Os 1.382 passos vêm do arquivo: `int(0.7 * 1975) = 1382`. A diferença para 1.408 é a base menor, não a seed. Com seed 13 e seed 42 sobre o mesmo arquivo, só 188 das 593 frases de teste coincidem.

**Consequência para comparações.** A planilha SEED_42 usa a divisão antiga, que continua no git em `c94fc1e:data/val_dataset.json` (604 de 604 frases batem com a planilha). Das 604 frases dela:

| Situação no pipeline atual | Frases |
|---|---:|
| No **treino** do split atual | 380 |
| No teste do split atual | 146 |
| Fora de `dataset_sintetico_aumentado.json` | 78 |

Portanto, as notas da planilha SEED_42 não são comparáveis, frase a frase, com nenhum experimento novo. Reportado sem corrigir, como pede a T0.

### 1.7 Dados medidos na execução `fit_10_29_23_27`

- 1 época, 1.382 passos, `train_runtime` de 21.370 s (5 h 56 min).
- `eval_loss = 1,997` no passo 1000, a primeira perda de validação registrada pelo pipeline.
- Uma passada de avaliação levou **2.713 s (45 min)**, a 0,219 amostras/s sobre 593 frases. O SPEC estimava cerca de 40 min (R1.4).

## 3. Auditoria de ortografia

Relatório completo em `experiments/reports/orthography_audit.md`, gerado por `python -m tupi_exp.audit_orthography` sem alterar o dataset.

| Achado | Históricos (824) | Sintéticos (1.151) |
|---|---:|---:|
| Textos fora de NFC | 0 | 0 |
| Apóstrofo fora do padrão `'` | 33 | 0 |
| Linhas misturando tipos de apóstrofo | 2 | 0 |
| Espaços duplos, nas pontas ou outros brancos | 0 | 0 |
| Letras de grafia histórica (`ä ö ü ç à è ì ò ù` e similares) | 281 | 0 |
| Maiúscula depois da primeira palavra | 68 | 0 |
| Tupi idêntico ao `output` (Gemini copiou a palavra) | 8 | 0 |
| Tupi quase idêntico ao `output` | 5 | 0 |
| Tupi que parece Português | 1 | 0 |
| Vazios, duplicatas exatas, mesmo Tupi com referências diferentes | 0 | 0 |

Foram encontrados 138 grupos de grafias variantes, como `rupi`/`rupí`, `a'e`/`aé` e `catú`/`catù`/`catu`. O ruído ortográfico está quase todo nas frases históricas, o que é coerente com a hipótese H3.

A linha citada no SPEC (`Irei andando Padre meu:`) não está em `dataset_sintetico_aumentado.json`: ela só existe em `data/dataset_augmented.json` e não tem tradução do Gemini.

**Decisões:**

- **Aplicadas em `normalize_tupi`:** NFC, apóstrofo único, colapso de espaços e `strip`. Essa função só é usada em buscas e comparações. O texto de treino não é reescrito, porque a matriz da §6 não prevê a normalização como fator.
- **`normalize_tupi(safe=False)`:** levanta `NotImplementedError`, porque as regras agressivas ficam só como proposta.
- **Propostas, não aplicadas:** mapear a grafia histórica para a de Navarro, unificar hífens, unir variantes e remover a pontuação final. Exemplos na seção 9 do relatório.
- **Métricas:** `normalize_tupi` não substitui a normalização das métricas, que não existe em código (ver 1.4).

## 4. Estatísticas do v3 e do glossário

### 4.1 `data/dataset_augmentation_v3.json` (G4)

Gerado por `python -m tupi_exp.split_augment`. Relatório com 10 exemplos antes/depois em `experiments/reports/split_augment.md`.

| Medida | Valor |
|---|---:|
| Frases históricas no treino | 574 |
| Limiar de "longa" (percentil 75 do treino) | mais de 4 palavras |
| Frases longas | 81 |
| Frases particionadas | 15 |
| Novos pares | 30 |
| Descartados por vazamento ou duplicata | 0 |
| Treino: v2 → v3 | 1.382 → 1.412 (+2,2%) |

- A validação continua igual: mesmo hash do baseline (`86c9431b...`) e interseção zero com o treino aumentado, por asserção e por teste.
- Das 81 frases longas, só 25 têm o mesmo número de segmentos nos dois lados, e 15 cumprem também o mínimo de 2 palavras por segmento em Tupi. A regra conservadora gera pouco dado: o `exp03` pode ser pequeno demais para mostrar efeito.
- Para testar mais volume, existem `--percentile` e `--contiguous-windows`. Mudar o percentil para 50 (mais de 3 palavras) daria 18 frases particionadas.
- O alinhamento é só pela contagem de segmentos, sem verificação de sentido. Vale olhar os 10 exemplos do relatório.
- Os novos pares usam o `output` de v2, ou seja, a tradução do Gemini, e não a tradução humana. Assim o exp03 não muda o tipo de rótulo do treino (P3) nem usa as referências de avaliação.

### 4.2 Glossário do `exp02` (G2)

**Fonte (R2.3):** prioridade 2, derivada só do treino. Não existe dicionário no repositório (ver 1.5).

| Medida | Valor |
|---|---:|
| Entradas | 273 (142 de um token, 131 de dois) |
| Cópias do Tupi removidas (Gemini repetiu a palavra) | 5 |
| Máximo de entradas usadas em um prompt | 3 (limite: 8) |

| Conjunto | Exemplos | Com ao menos 1 entrada | Cobertura |
|---|---:|---:|---:|
| Treino, históricos (com leave-one-out) | 574 | 52 | 9% |
| Treino, sintéticos (com leave-one-out) | 808 | 348 | 43% |
| Validação, históricos | 250 | 20 | **8%** |
| Validação, sintéticos | 343 | 151 | 44% |

**Limitação importante.** O glossário cobre pouco justamente o grupo que a H2 quer melhorar: só 20 das 250 frases históricas de validação recebem alguma entrada. Considerei o léxico utilizável, e por isso não parei na prioridade 3 da R2.3, mas com essa cobertura o `exp02` testa a H2 de forma fraca. A busca sem diferenciar maiúsculas subiria de 20 para 25 frases, então a caixa não é o problema principal, e mantive a igualdade exata da R2.4. O ganho real viria de um dicionário curado (prioridade 1 da R2.3), que hoje não existe no repositório.

**Comprimento (R2.7), com o tokenizador do Qwen:**

| Prompt | Maior no treino | Maior na validação |
|---|---:|---:|
| Atual | 107 | 111 |
| Com glossário | 142 | 140 |

- `max_seq_length` do `exp02`: **144**, contra 128 nos outros experimentos. Nenhum exemplo é truncado.
- **Impacto estimado:** como o notebook usa `padding="max_length"`, toda sequência passa a ter 144 tokens. Isso dá cerca de 12,5% a mais de memória de ativações e de tempo por passo em relação ao `exp01`.

**Decisões de formato:**

- Sem nenhuma entrada, o prompt omite só a seção de glossário e mantém `Texto em Tupi:`, para que todos os exemplos tenham a mesma estrutura.
- A instrução atual não foi copiada: `render_prompt` chama o próprio `create_prompt` com o texto ampliado, então o prefixo é sempre igual ao do baseline.
- Traduções do glossário perdem só a pontuação final (`,;:.`).
