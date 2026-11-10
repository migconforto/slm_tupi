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

## 2. Arquivos

**Criados:**

- `experiments/README.md` e `experiments/CHANGES.md`
- `experiments/tupi_exp/`: `__init__.py`, `config.py`, `normalize.py`, `audit_orthography.py`, `splits.py`, `split_augment.py`, `lexicon.py`, `dryrun.py`
- `experiments/tests/`: `conftest.py`, `test_config.py`, `test_normalize.py`, `test_audit.py`, `test_split_augment.py`, `test_lexicon.py`, `test_dryrun.py`, `test_notebooks.py`
- `experiments/reports/`: `orthography_audit.md`, `split_augment_seed13.md`, `split_augment_seed42.md`
- `experiments/exp01_hparams/`: `sft_exp01.ipynb`, `eval_exp01.ipynb`
- `experiments/exp02_lexicon/`: `sft_exp02.ipynb`, `eval_exp02.ipynb`
- `experiments/exp03_split_aug/`: `sft_exp03.ipynb`, `eval_exp03.ipynb`
- `splits/seed13_*_ids.json` e `splits/seed42_*_ids.json`
- `data/dataset_augmentation_v3_seed13.json` e `data/dataset_augmentation_v3_seed42.json`

**Modificados:** nenhum arquivo existente.

**Não tocados, conferidos por SHA-256 no início e no fim:** `notebooks/sft.ipynb`, `notebooks/eval_sft.ipynb`, `data/dataset_sintetico_aumentado.json`, `data/dataset_historico.json`, `data/dataset_augmented.json` e `data/gemini-output_dataset.json`.

Os commits estão no branch `experimentos-sft`, um por tarefa. Nada foi enviado ao repositório remoto.

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

### 4.1 `data/dataset_augmentation_v3_seed13.json` (G4)

Números da seed 13; os da seed 42 estão na seção 7. Gerado por `python -m tupi_exp.split_augment`. Relatório com 10 exemplos antes/depois em `experiments/reports/split_augment_seed13.md`.

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

Números da seed 13; os da seed 42 estão na seção 7.

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

- `max_seq_length` do `exp02`: **152**, contra 128 nos outros experimentos, para caber o maior prompt de qualquer semente (149 tokens, ver seção 7). Nenhum exemplo é truncado.
- **Impacto estimado:** como o notebook usa `padding="max_length"`, toda sequência passa a ter 152 tokens. Isso dá cerca de 19% a mais de memória de ativações e de tempo por passo em relação ao `exp01`.

**Decisões de formato:**

- Sem nenhuma entrada, o prompt omite só a seção de glossário e mantém `Texto em Tupi:`, para que todos os exemplos tenham a mesma estrutura.
- A instrução atual não foi copiada: `render_prompt` chama o próprio `create_prompt` com o texto ampliado, então o prefixo é sempre igual ao do baseline.
- Traduções do glossário perdem só a pontuação final (`,;:.`).

## 5. Premissas e divergências

### 5.1 Divergências entre o SPEC e o código

Em todas valeu a regra de precedência: seguir o código.

| Tema | SPEC | Código e decisão |
|---|---|---|
| Nome do dataset base | `dataset_augmentation_v2.json`, `dataset_sintetico_aumentado_v2.json` | "v2" é `data/dataset_sintetico_aumentado.json`, o único arquivo com esse conteúdo |
| Nome do dataset novo | `dataset_augmentation_v3.json` e `dataset_sintetico_aumentado_v2.json` (§7) | Usei `data/dataset_augmentation_v3_seed<N>.json`: o nome da R4.6 e da T3, com a semente, porque cada semente tem o seu |
| Split | Seed 42, 1.408 / 604 | Seed 13, 1.382 / 593. Arquivos `splits/seed13_*`, não `seed42_*` |
| Validação | 604 exemplos | 593 exemplos, iguais aos da execução real do baseline (ver 1.1 e 1.6) |
| `seed` do `hp_v2` | 42 | Era 13 na entrega inicial. Depois, o usuário escolheu 42 para treino e split (ver seção 7) |
| `r` | "128 efetivo, argumento 256" | Não há mais ambiguidade: argumento e modelo usam 128. `check_lora_model` confere isso no notebook (R1.3) |
| Passos do `hp_v2` | `ceil(n/8) * 3`, 528 | O Trainer arredonda para baixo por época: **516** no `exp01` e no `exp02`, **528** no `exp03`. A fórmula do SPEC daria 519 e 531 |
| Planilha de saída | Gerada pelo notebook de treino | `sft.ipynb` não gera planilha. Criei cópias de `eval_sft.ipynb` por experimento, que gravam `predictions_<experimento>.csv` com a coluna `Subconjunto` |
| Métricas (BLEU, ChrF++ etc.) | "Código de avaliação existente" | Esse código não existe no repositório (ver 1.4). Os notebooks geram as predições, e as métricas seguem fora do projeto |
| Estrutura de arquivos | §7 | Acréscimos: `tupi_exp/splits.py`, `reports/split_augment_seed<N>.md`, `eval_expNN.ipynb` e o campo `segmento_fim` no v3, necessário para as janelas contíguas |

### 5.2 Decisões para você revisar

1. **Avaliação durante o treino (R1.4).** Mantive `eval_steps=1000` e `logging_steps=1000`, como pede o SPEC, mas com o `hp_v2` o treino tem só 516 passos. **Do jeito que está, nenhuma avaliação e nenhum log de perda acontecem durante o treino**, e o `exp01` terminaria sem curva nenhuma. Cada passada de avaliação custa de 40 a 45 min neste hardware (0,245 amostras/s sobre 604 na execução antiga; 0,219 amostras/s sobre 593 na última, ver 1.7). Por exemplo, `eval_steps=172` daria uma avaliação por época, somando cerca de 2 h 15 min ao treino. Para mudar, edite `eval_steps` e `logging_steps` em `HP_V2`, em `tupi_exp/config.py`.
2. **Glossário com pouca cobertura (R2.3).** Só 8% das frases históricas de validação recebem alguma entrada (ver 4.2). Um dicionário curado seria necessário para testar a H2 de verdade.
3. **Partição com pouco volume.** O `exp03` só acrescenta 30 pares (ver 4.1).
4. **Comparação com a planilha SEED_42.** Não é válida frase a frase, porque 380 das 604 frases estão no treino atual (ver 1.6). A comparação correta é com a execução `fit_10_29_23_27`, mas ela ainda não tem `predictions.csv`: é preciso rodar `notebooks/eval_sft.ipynb` para gerá-lo. Ela também usou 1 época, enquanto o `hp_v2` usa 3; para comparar com o mesmo número de épocas, o baseline teria de ser retreinado.

### 5.3 Premissas

- **Texto de treino:** nenhuma normalização reescreve o texto; `normalize_tupi` só serve para busca e comparação.
- **Frase "longa":** estritamente acima do percentil 75 (mais de 4 palavras), a opção mais conservadora.
- **Rótulos do v3 e do glossário:** vêm do `output` de v2. Nas frases históricas, é a tradução do Gemini, nunca a humana, para não misturar as referências de avaliação no treino.
- **Monitoramento:** a perda durante o treino usa os rótulos de treino, e a avaliação final usa a coluna `Referência`.
- **Tempo de treino estimado:** a 13,5 s por exemplo (medido no baseline), cerca de 15,5 h para o `exp01`, 18,5 h para o `exp02` e 16 h para o `exp03`. A execução real do `exp01` com seed 13 levou 14 h 29 min. O `r` menor pode reduzir esses tempos. A acumulação de gradiente não reduz, porque cada exemplo continua passando pelo modelo.
- **Precisão:** treino em fp32 e avaliação em 4 bits NF4, como no pipeline atual.
- **Combinações:** glossário + partição não foi preparada. Para criá-la, basta um novo item em `EXPERIMENTS`, em `tupi_exp/config.py`, além do notebook correspondente.
- **Dependências:** nenhuma nova. Os testes usam o `pytest` já instalado, que não consta no `requirements.txt`.

### 5.4 Verificação final (T7)

| Verificação | Resultado |
|---|---|
| Testes unitários | 92 passaram |
| Dry-runs dos 3 experimentos | Passaram |
| Hash de validação nos 3 experimentos | Igual ao baseline (`86c9431b...`) |
| Interseção validação × treino, no v3 e no glossário | Zero |
| Exemplos truncados no `exp02` | Zero (`max_seq_length=152`, maior prompt com 149 tokens) |
| Smoke test dos 3 `sft_expNN.ipynb` | Passou |
| Smoke test dos 3 `eval_expNN.ipynb` | Passou |
| `sft.ipynb` e dataset v2 byte a byte iguais | Sim, por SHA-256 |
| Treino completo iniciado | Nenhum |

**Limites do smoke test:** usei um modelo Qwen2 minúsculo e aleatório, 1 exemplo de treino, 1 de teste e `max_steps=2`, injetados só em memória, sem alterar os notebooks. Ele valida o código, não a qualidade. O Qwen real em fp32 **não** foi carregado, para não estourar a memória com o aplicativo aberto. O Qwen real só foi testado em 4 bits, sem adaptador, em uma frase do `eval_exp02`.

## 6. Comandos

Testes, a partir da raiz do projeto:

```bash
python -m pytest experiments/tests -q -p no:cacheprovider
```

Dry-run dos três experimentos:

```bash
cd experiments && python -m tupi_exp.dryrun
```

Dry-run de um experimento:

```bash
cd experiments && python -m tupi_exp.dryrun --experiment exp02_lexicon
```

Auditoria de ortografia:

```bash
cd experiments && python -m tupi_exp.audit_orthography
```

Splits, v3 e relatório da partição:

```bash
cd experiments && python -m tupi_exp.split_augment
```

**Iniciar cada treino:** abra o notebook e use "Run All".

| Experimento | Treino | Avaliação |
|---|---|---|
| `exp01_hparams` | `experiments/exp01_hparams/sft_exp01.ipynb` | `experiments/exp01_hparams/eval_exp01.ipynb` |
| `exp02_lexicon` | `experiments/exp02_lexicon/sft_exp02.ipynb` | `experiments/exp02_lexicon/eval_exp02.ipynb` |
| `exp03_split_aug` | `experiments/exp03_split_aug/sft_exp03.ipynb` | `experiments/exp03_split_aug/eval_exp03.ipynb` |

Na avaliação, ajuste antes o `run_dir` para a pasta `logs/<experimento>/peft_64_Tu_Po/fit_...` criada pelo treino.

## 7. Troca de semente: seed 42

O `exp01` parou com `AssertionError: validation differs from the baseline` depois da troca de `SPLIT_SEED` para 42, porque o hash esperado era fixo na validação da seed 13. Por decisão do usuário, a semente passou a valer para o split e para o treino, e cada semente agora tem dados próprios.

**O que mudou:**

- **Configuração:** `SEED`, em `tupi_exp/config.py`, é a única linha a alterar. `SPLIT_SEED` e o `seed` dos perfis vêm dela.
- **Verificação (P6):** `BASELINE_VAL_SHA256` agora é o hash da validação salva da semente atual. Dentro de uma semente, os três experimentos continuam obrigados a usar a mesma validação. O hash da execução real da seed 13 ficou em `BASELINE_RUNS_SHA256`.
- **Dados por semente:** `splits/seed<N>_*`, `data/dataset_augmentation_v3_seed<N>.json` e `experiments/reports/split_augment_seed<N>.md`. O arquivo da seed 13 é idêntico, byte a byte, ao antigo `data/dataset_augmentation_v3.json`.
- **Vazamento evitado:** reaproveitar o v3 da seed 13 na seed 42 colocaria no treino 6 pares vindos de frases de validação. O dry-run agora recusa pares derivados de frases de validação.
- **`max_seq_length` do `exp02`:** passou de 144 para **152**. Na seed 42 o maior prompt tem 149 tokens, e 3 exemplos seriam truncados. O valor de 149 é o pior caso para qualquer semente: é o limite calculado com a base inteira, e 26 sementes testadas não passaram disso.
- **`eval_exp02`:** recusa uma execução de outra semente, porque o glossário depende do treino da semente.
- **Testes:** removi `test_copies_carry_no_stale_outputs`, que falhava assim que um notebook era executado.

**Comparação entre as seeds:**

| Medida | Seed 13 | Seed 42 |
|---|---:|---:|
| Treino / validação | 1.382 / 593 | 1.382 / 593 |
| Históricos / sintéticos na validação | 250 / 343 | 241 / 352 |
| Hash da validação | `86c9431b...` | `9a3dd19d...` |
| Frases de validação em comum entre as duas | 188 | 188 |
| Frases históricas longas no treino | 81 | 83 |
| Frases particionadas | 15 | 15 |
| Novos pares no v3 | 30 | 31 |
| Treino do `exp03` | 1.412 | 1.413 |
| Entradas do glossário | 273 | 284 |
| Validação histórica com glossário | 20 de 250 (8%) | 20 de 241 (8%) |
| Validação sintética com glossário | 151 de 343 (44%) | 204 de 352 (58%) |
| Maior prompt do `exp02` | 142 tokens | 149 tokens |
| Execução de baseline | `fit_10_29_23_27` | Não existe |
| Execução do `exp01` | `fit_10_30_23_04...-SEED13` | Não existe |

**Consequências:**

- Resultados da seed 42 não são comparáveis, frase a frase, com os da seed 13: 405 das 593 frases de validação da seed 42 estavam no treino da seed 13.
- A seed 42 ainda não tem baseline. Para comparar, é preciso treinar `notebooks/sft.ipynb` com `seed=42`, o que levaria cerca de 18 h com 3 épocas.
- A seed 42 não recupera a planilha SEED_42: com a base atual, só 165 das 604 frases dela caem na validação, e 361 ficam no treino.

**Verificação:** 95 testes passaram com `SEED=42`. Em uma cópia temporária, testes e dry-runs passaram também com `SEED=13` e com uma semente nova (7), depois de gerar os dados dela. O smoke test dos seis notebooks passou na seed 42, com o mesmo modelo minúsculo da seção 5.4.

### 7.1 Qualquer semente, sem passo manual

Com `SEED = 51`, o `exp01` parou na segunda célula com `FileNotFoundError: splits\seed51_val_ids.json not found`, porque uma semente nova exigia rodar `split_augment` antes. Agora os dados de uma semente nova são gerados na primeira vez que ela é usada.

- **`split_augment.generate(seed)`:** é o corpo do antigo `main()`, sem mudança de lógica. O comando de linha continua funcionando igual.
- **`config.ensure_seed_data()`:** gera o split, o v3 e o relatório quando falta algum arquivo da semente. É chamada por `get_experiment()` e pela primeira leitura de `BASELINE_VAL_SHA256`, que passou a ser calculado sob demanda (PEP 562). Assim, o valor já está correto quando o notebook o importa.
- **Notebooks:** nenhuma alteração.
- **`max_seq_length` do `exp02`:** calculei o pior caso exato, testando, para cada frase, todas as combinações possíveis de entradas do glossário. O resultado é 149 tokens, então 152 vale para qualquer semente com a base atual.
- **Testes:** um teste novo copia o projeto para uma pasta temporária, troca `SEED` para 7 e confere que os três experimentos iniciam e que os dados aparecem. O teste do nome da planilha passou a aceitar sufixos como `-SEED42`.

**Verificação:** 97 testes passaram, o dry-run passou com a seed 51, e o smoke test dos seis notebooks passou com a seed 51, usando modelo minúsculo e um exemplo.

### 7.2 Semente nos nomes da pasta e da planilha

A pedido do usuário, a semente entra sozinha nos nomes, seguindo a convenção que ele já usava à mão (`-SEED42`).

- **Treino (`sft_expNN`):** a pasta da execução termina em `-SEED<N>`, por exemplo `fit_11_04_19_39_1.0_dataset_sintetico_aumentado-SEED51`. Mudou uma linha em cada notebook (as duas variantes de `output_dir`).
- **Avaliação (`eval_expNN`):** `run_seed = seed_of_run(run_dir)` lê a semente do sufixo da pasta avaliada, e a planilha sai como `predictions_<experimento>-SEED<N>.csv` ou `predictions_base-SEED<N>.csv`. Avaliar uma execução antiga não troca o rótulo, mesmo que a `SEED` do `config.py` já seja outra. Sem o sufixo, vale a `SEED` do `config.py`, com um aviso impresso.
- **`eval_exp01.ipynb`:** o nome da planilha, que tinha `-SEED42` escrito à mão, passou a usar `run_seed`. O `run_dir` escolhido pelo usuário e as saídas salvas ficaram intactos.

**Verificação:** 98 testes passaram, e o smoke test dos seis notebooks gerou pastas `-SEED51` e as planilhas `predictions_exp01_hparams-SEED51.csv`, `predictions_exp02_lexicon-SEED51.csv` e `predictions_exp03_split_aug-SEED51.csv`.
