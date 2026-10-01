# Relatório Final do Projeto

**Título:** Tradução Automática de Tupi Antigo para Português Brasileiro com *Small Language Models* e *Fine-Tuning* Eficiente (LoRA/PEFT)

**Autor:** Miguel Conforto

**Curso/Programa:** Mestrado — PUC

**Repositório:** migconforto/slm_tupi

**Data:** julho de 2026 (atualizado em outubro de 2026 com os resultados da avaliação)

> **Nota de conciliação (remover na versão final).** Os números de dados e de treinamento
> das Seções 3 e 5.1 (2.012 pares; 1.408 exemplos de treino e 604 de validação; 4.224
> passos; `checkpoint-4224`; posto LoRA `r` = 256) diferem dos reportados na dissertação
> (802 + 1.200 = 2.002 pares; 1.401 de treino e 601 de teste; 4.203 passos; `r` = 128).
> Em particular, os ≈ 239,5 M de parâmetros treináveis informados na Seção 3.2
> correspondem exatamente a `r` = 128 no Qwen2.5-3B (1.870.848 × `r` = 239.468.544), e
> não a `r` = 256 (≈ 479 M). As métricas da Seção 5 seguem a dissertação (conjunto de
> teste de 601 sentenças). Confirmar quais valores correspondem à execução descrita e
> uniformizar o texto.

---

## Resumo

Línguas de baixo recurso (*Low-Resource Languages*, LRLs) permanecem sub-representadas
nos sistemas atuais de tradução automática, o que limita seu acesso a tecnologias de
linguagem. Este projeto investiga a viabilidade de adaptar um *Small Language Model*
(SLM) de 3 bilhões de parâmetros (**Qwen2.5-3B-Instruct**) para a tradução do
**Tupi Antigo** para o **Português brasileiro**, um par extremamente escasso em dados
paralelos. Adotou-se *fine-tuning* supervisionado (SFT) com adaptação de baixo posto
(**LoRA/PEFT**) sobre um conjunto de dados aumentado de cerca de 2 mil pares de
sentenças: pares históricos, cujas traduções-alvo foram geradas por um modelo-professor
(Google Gemini), e pares sintéticos, gerados a partir de *templates* gramaticais
(destilação de conhecimento baseada em rótulos). O treinamento foi conduzido em hardware
de baixo consumo (NVIDIA RTX 2060, 6 GB), demonstrando que a *perda* de treino cai de
forma consistente (de ≈ 2,71 para ≈ 0,66 ao longo de três épocas, com perda média final
de 1,27). Na avaliação sobre o conjunto de teste, o ajuste elevou o ChrF++ médio de 12,5
para 28,8 e reduziu o TER médio de 435,9% para 118,6%, com diferenças estatisticamente
significativas nas cinco métricas, e eliminou a verborragia do modelo original. O ganho,
porém, concentra-se nas estruturas sintéticas: nas 247 sentenças históricas avaliadas
também para o professor, o modelo ajustado obtém ChrF++ de 12,6, contra 41,2 do
professor, e passa a produzir alucinações fluentes. O trabalho contribui com um
*pipeline* reprodutível de SFT e avaliação, delimita o alcance da abordagem e aponta a
qualidade da supervisão destilada e a avaliação humana por especialistas como próximos
passos.

**Palavras-chave:** tradução automática; línguas de baixo recurso; Tupi; LoRA; PEFT;
*Small Language Models*; destilação de conhecimento.

---

## 1. Introdução

### 1.1 Contexto e motivação

Os avanços recentes em *Large Language Models* (LLMs) e em Tradução Automática Neural
(NMT) melhoraram substancialmente a qualidade de tradução para idiomas de alto recurso.
Contudo, persiste uma disparidade acentuada para línguas de baixo recurso, tanto por
falta de dados paralelos quanto por sub-representação nos corpora de pré-treinamento. O
**Tupi Antigo** (Tupi clássico), língua histórica de grande importância cultural e
linguística no Brasil, é um caso extremo: praticamente não há sistemas de tradução
automática dedicados e os recursos digitais são fragmentados.

### 1.2 Problema

Modelos grandes o suficiente para traduzir bem costumam ser inviáveis em cenários com
restrição de recursos (privacidade, custo, hardware limitado). A questão central deste
projeto é: **é possível ajustar um modelo pequeno (3B), em hardware de consumo, para
produzir traduções úteis de Tupi Antigo para Português?**

### 1.3 Objetivos

**Objetivo geral:** avaliar a adaptação de um SLM para a tradução Tupi → Português por
meio de *fine-tuning* eficiente.

**Objetivos específicos:**

1. Construir um *pipeline* reprodutível de pré-processamento, *fine-tuning* (SFT com
   LoRA) e inferência para o par Tupi → Português.
2. Treinar o Qwen2.5-3B-Instruct sobre um conjunto aumentado de pares Tupi/Português.
3. Analisar o comportamento do treinamento (curva de perda, custo computacional) em
   hardware limitado.
4. Avaliar quantitativa e qualitativamente as traduções geradas — antes e depois do
   ajuste e em comparação com um modelo-professor — por meio de métricas automáticas
   (BLEU, ChrF++, TER, ROUGE-L e Jaccard), teste estatístico pareado e análise de
   exemplos.

---

## 2. Fundamentação teórica e trabalhos relacionados

O projeto se baseia no artigo *"Are Small Language Models the Silver Bullet to
Low-Resource Languages Machine Translation?"* (LoResMT 2026), que mostra que a
**destilação de conhecimento** a partir de modelos-professores fortes, usando
predominantemente dados monolíngues da língua-alvo, pode elevar a qualidade de tradução
de SLMs a ponto de igualar ou superar sistemas muito maiores.

Conceitos-chave utilizados:

- **Destilação de conhecimento baseada em rótulos (*sequence-level*):** um
  modelo-professor de grande porte gera traduções que passam a ser o alvo de treino do
  modelo-aluno, que aprende com as saídas textuais do professor e não com suas
  distribuições internas de probabilidade [6]; é a variante adotada por Song et al. [1]
  e neste projeto.
- **Supervised Fine-Tuning (SFT):** ajuste do modelo a pares (entrada, saída) formatados
  como diálogo instruído, usando o *template* de *chat* do próprio modelo.
- **LoRA (Low-Rank Adaptation):** técnica de *fine-tuning* paramétrico-eficiente que
  congela os pesos originais e treina apenas matrizes de baixo posto injetadas nas
  camadas de atenção e MLP, reduzindo drasticamente memória e custo.
- **PEFT (Parameter-Efficient Fine-Tuning):** família de métodos, aqui representada pela
  biblioteca `peft` da Hugging Face combinada ao `SFTTrainer` da biblioteca `trl`.

---

## 3. Metodologia

### 3.1 Dados

O conjunto de treino é o `dataset_augmentated.json`, com **2.012 pares** de sentenças,
onde a coluna `input` contém o texto em Tupi e `output`, a tradução em Português.

O pré-processamento (`utils_train.pre_process`) calcula o comprimento das entradas e
mantém a estrutura tabular; filtros opcionais para "alucinações" estão previstos,
mas desativados nesta configuração. Após embaralhamento (semente 42), o conjunto foi
dividido em **70% treino (1.408 exemplos)** e **30% validação (604 exemplos)**.

**Composição da base e rótulos do professor.** A base é **híbrida**, formada por dois
subconjuntos complementares:

- **Dados históricos (802 pares):** extraídos de gramáticas, dicionários, cartas e textos
  anotados, sobretudo dos textos *"A vida em Caraguatatuba"* e *"Os ensinamentos do Padre
  Cícero sobre como cuidar da terra"* [18] (traduzidos por Navarro), das *Cartas dos
  Índios Camarões* [16] e da *"Narração que faz um sertanejo a um seu amigo de uma viagem
  que fez pelo sertão"* [17]. A maior parte das fontes não traz traduções alinhadas
  sentença a sentença, o que exigiu curadoria manual (interpretação, segmentação e
  alinhamento); a ortografia foi normalizada com base no dicionário de Navarro [14].
- **Dados sintéticos (1.200 pares):** gerados por regras a partir de um léxico
  (substantivos, verbos, pronomes, advérbios e marcadores gramaticais) e de *templates*
  sintáticos inspirados no *Método moderno de Tupi Antigo* [15] (frases declarativas,
  negativas, subordinadas, interrogativas e plurais), com validação semiautomática e
  limite de 100 frases por *template*.

Nos pares históricos, a tradução-alvo do treinamento foi gerada pelo modelo-professor
**Google Gemini** [7] em modo *zero-shot* (sem ajuste ao Tupi Antigo), com um *prompt*
estruturado que o instrui a atuar como sistema de tradução e a retornar apenas a
tradução, em JSON, seguida de curadoria semiautomática (respostas vazias ou de formato
inválido foram removidas). Nos pares sintéticos, a tradução é a definida pelo próprio
*template*. Os rótulos do professor cobrem, portanto, apenas a parte histórica da base
(cerca de 40% dos pares). Na avaliação, a referência é a tradução humana (*padrão-ouro*)
nos pares históricos e o *rótulo padrão* do *template* nos sintéticos.

Cada exemplo é convertido em um *prompt* instruído com o *template* de *chat* do modelo:

```
<|im_start|>system
Você é um assistente de IA muito útil para traduções.<|im_end|>
<|im_start|>user
Traduza o seguinte texto em Tupi para Português. Não inclua informações
adicionais ou conteúdo irrelevante.

kunhãmukuetá îkó ka'ape<|im_end|>
<|im_start|>assistant
moças vivem na mata<|im_end|>
```

### 3.2 Modelo e configuração de treinamento

| Item | Valor |
|---|---|
| Modelo base | `Qwen/Qwen2.5-3B-Instruct` (decoder-only, 36 camadas, hidden 2048) |
| Método | LoRA (PEFT) |
| Posto LoRA (`r`) | 256 (experimento principal); variantes com `r` ∈ {8, 16, 128} |
| `lora_alpha` / `dropout` | 8 / 0 |
| Módulos-alvo | `q,k,v,o_proj`, `gate,up,down_proj` |
| Parâmetros treináveis | ≈ 239,5 M (7,20% de 3,33 B) |
| Épocas | 3 |
| *Learning rate* | 1e-5 |
| *Scheduler* | cosseno, `warmup_ratio` = 0,5 |
| *Batch* (treino/aval.) | 1 / 1 |
| `max_seq_length` | 128 |
| `weight_decay` / `max_grad_norm` | 0,01 / 0,3 |
| Precisão | fp32 (fp16/bf16 desabilitados) |
| Semente | 42 |
| *Framework* | Hugging Face `transformers` + `peft` + `trl` (`SFTTrainer`), *logging* TensorBoard |

### 3.3 Ambiente

- **GPU:** NVIDIA GeForce RTX 2060, 6 GB VRAM.
- **Software:** PyTorch 2.5.1 + CUDA 12.1; Python 3.10; Windows.

Por ser um modelo de 3B em fp32, o consumo de memória excedeu a VRAM física (pico
reportado de ≈ 17,7 GB), o que implicou uso de memória compartilhada/*offload* — principal
fator do tempo de treino elevado.

### 3.4 Avaliação

A inferência (`EVAL_SFT.py`) carrega o modelo base com o adaptador LoRA
(`checkpoint-4224`) e gera traduções do conjunto de validação, com quantização em 4-bit
(NF4) para caber em memória. Parâmetros de geração: `temperature` = 0,1, `top_p` = 0,9,
`max_new_tokens` = 50, amostragem ativada. As saídas são exportadas para
`predictions.csv`.

### 3.5 Métricas e protocolo de avaliação

A avaliação combina **métricas automáticas** e **análise qualitativa**. O modelo-aluno
foi avaliado **antes** do ajuste (modelo original em modo *zero-shot*, recebendo a
sentença em Tupi acompanhada de uma instrução de tradução) e **depois** dele, sobre o
conjunto de teste (30% da base, separado antes do treinamento), e comparado ao
modelo-professor, avaliado nas sentenças históricas. Foram usadas cinco métricas:

| Métrica | O que mede | Escala |
|---|---|---|
| BLEU [8] | precisão de *n*-gramas de palavras | 0–1 (↑) |
| ChrF++ [9] | *n*-gramas de caracteres e de palavras; mais tolerante à variação morfológica | 0–100 (↑) |
| TER [10] | esforço de edição para transformar a predição na referência; pode exceder 100% | % (↓) |
| ROUGE-L [11] | maior subsequência comum | 0–1 (↑) |
| Jaccard [12] | sobreposição do vocabulário | 0–1 (↑) |

Antes do cálculo, predições e referências foram normalizadas (caixa baixa e remoção de
pontuação e de caracteres especiais). As diferenças pareadas por sentença foram testadas
com o teste de postos sinalizados de Wilcoxon [13]. O professor foi avaliado com a mesma
normalização e a mesma implementação das métricas; como seu conjunto de avaliação
(sentenças históricas) não coincide com o conjunto de teste do aluno, que também contém
sentenças sintéticas, a comparação direta entre os dois é feita sobre as **247 sentenças
históricas comuns** às duas avaliações, casadas pela sentença-fonte. A análise
qualitativa examinou preservação semântica, coerência gramatical, omissões e
alucinações, adequação contextual e fluidez em português.

---

## 4. Desenvolvimento e implementação

O *pipeline* foi organizado em módulos reutilizáveis:

- **Preparação de dados e *prompts*** (`utils/utils_train.py`): `pre_process`,
  `create_prompt` (template Qwen/ChatML) e `create_prompt_gemma` (template Alpaca), além
  de utilitários de contagem de parâmetros treináveis.
- **Treinamento** (`script/SFT.py`): montagem do *dataset*
  tokenizado, configuração LoRA, `SFTConfig`/`SFTTrainer` e laço de treino com avaliação
  periódica e salvamento por época.
- **Inferência** (`script/EVAL_SFT.py`): quatro estratégias — *pipeline* local da Hugging
  Face.

---

## 5. Resultados e discussão

### 5.1 Dinâmica do treinamento

A perda de treino apresentou queda consistente ao longo dos 4.224 passos (3 épocas):

| Passo | Época | *Loss* de treino | *Learning rate* |
|---:|---:|---:|---:|
| 1000 | 0,71 | 2,706 | 4,73e-6 |
| 2000 | 1,42 | 1,069 | 9,47e-6 |
| 3000 | 2,13 | 0,796 | 6,24e-6 |
| 4000 | 2,84 | 0,657 | 2,75e-7 |
| — | 3,00 | **1,274** (média) | — |

A trajetória decrescente indica que o modelo aprendeu efetivamente o mapeamento
Tupi → Português a partir do conjunto aumentado, sem sinais de divergência. O *warmup*
longo (50%) combinado ao *scheduler* cosseno explica o comportamento do *learning rate*.
Essa queda da perda de treino, contudo, não implica generalização: a avaliação das
Seções 5.3 a 5.5 mostra que o aprendizado se concentra nas estruturas recorrentes dos
dados sintéticos.

### 5.2 Custo computacional

- **Tempo total de treino:** ≈ 66.255 s (≈ 18 h 24 min).
- **Velocidade:** ≈ 0,064 passos/s (limitada pelo *offload* de memória).
- **Pico de memória reservada:** ≈ 17,7 GB (contra 6 GB de VRAM física).

O gargalo principal foi a memória: rodar um modelo de 3B em fp32 numa GPU de 6 GB obriga
o uso de memória do sistema, penalizando a velocidade.

### 5.3 Avaliação quantitativa: antes e depois do *fine-tuning*

A tabela abaixo compara as métricas médias do modelo-aluno antes e depois do ajuste, sobre
o conjunto de teste (n = 601). As colunas *melhoraram* e *pioraram* contam as sentenças
cuja métrica melhorou ou piorou após o ajuste (as demais empataram). Para o TER (↓),
menor é melhor.

| Métrica | Antes do FT | Depois do FT | Variação | Melhoraram | Pioraram |
|---|---:|---:|---:|---:|---:|
| BLEU (0–1) | 0,006 | 0,120 | +0,114 | 310 | 54 |
| ChrF++ | 12,47 | 28,83 | +16,37 | 423 | 174 |
| TER ↓ | 435,93 | 118,60 | −317,34 | 488 | 61 |
| ROUGE-L (0–1) | 0,062 | 0,257 | +0,195 | 334 | 79 |
| Jaccard (0–1) | 0,029 | 0,189 | +0,160 | 303 | 61 |

O ajuste fino melhorou as cinco métricas. O ChrF++ médio passou de 12,47 para 28,83
(+131%) e o TER médio caiu de 435,93% para 118,60% (−72,8%); BLEU, ROUGE-L e Jaccard,
próximos de zero no modelo original, passaram a 0,120, 0,257 e 0,189. Na comparação
pareada por sentença, 423 das 601 sentenças (70,4%) melhoraram em ChrF++ e 488 (81,2%)
em TER, enquanto 174 (29,0%) pioraram em ChrF++; o teste de Wilcoxon indica diferenças
significativas em todas as métricas (*p* < 10⁻¹²). A melhoria é, portanto, majoritária,
embora haja um subconjunto não desprezível de sentenças com piora (Seção 5.6).

**Distribuição das pontuações.** A tabela a seguir traz média, mediana e desvio-padrão
(DP) de cada métrica nos dois cenários.

| Métrica | Antes: média | Antes: mediana | Antes: DP | Depois: média | Depois: mediana | Depois: DP |
|---|---:|---:|---:|---:|---:|---:|
| BLEU (0–1) | 0,006 | 0,000 | 0,012 | 0,120 | 0,024 | 0,230 |
| ChrF++ | 12,47 | 12,14 | 5,88 | 28,83 | 17,08 | 25,79 |
| TER ↓ | 435,93 | 287,50 | 490,57 | 118,60 | 100,00 | 88,41 |
| ROUGE-L (0–1) | 0,062 | 0,053 | 0,079 | 0,257 | 0,167 | 0,295 |
| Jaccard (0–1) | 0,029 | 0,000 | 0,047 | 0,189 | 0,083 | 0,259 |

Antes do ajuste, o desempenho era baixo em todas as métricas: apenas 2 sentenças
atingiam ChrF++ ≥ 50, e o TER médio indicava que as predições exigiriam, em média, mais
de quatro vezes o tamanho da referência em operações de edição. Depois do ajuste, a
distribuição é assimétrica: a mediana do ChrF++ (17,1) é bem inferior à média (28,8);
119 sentenças (19,8%) atingem ChrF++ ≥ 50 e 16 (2,7%) são idênticas à referência, mas
329 (54,7%) permanecem abaixo de 20.

**Comportamento formal das predições.** O ajuste também mudou a forma das respostas:

| Aspecto | Antes do FT | Depois do FT |
|---|---:|---:|
| Comprimento médio das predições (palavras; referências: 5,2) | 16,2 | 6,7 |
| Predições com quebras de linha e conteúdo adicional | 85,1% | nenhuma |
| Predições com expressões metadiscursivas (“aqui está sua tradução”) | 145 | nenhuma |
| Predições com marcadores espúrios (“###”) | 139 | nenhuma |
| Predições vazias | 4 | nenhuma |

Sem ajuste, o modelo frequentemente não se limitava a traduzir: comentava, explicava ou
respondia em tom conversacional, comportamento incompatível com a instrução fornecida, e
repetia respostas genéricas desvinculadas da fonte, como “você está certo / Você está
correto” (12 predições). Isso indica que, diante de uma língua desconhecida, o modelo
recorre a respostas frequentes de seu treinamento original, isto é, alucina. Após o
ajuste, as respostas ficaram concisas e aderentes ao formato: o ajuste ensinou ao modelo
não apenas a traduzir melhor, mas também a seguir o formato esperado.

### 5.4 Aluno ajustado versus professor

O professor (Google Gemini), sem qualquer ajuste ao Tupi Antigo, foi avaliado nas 802
sentenças históricas da base: ChrF++ médio de 41,60 (mediana 33,51), TER de 78,29%, BLEU
de 0,114, ROUGE-L de 0,395 e Jaccard de 0,331. Suas predições têm em média 4,2 palavras
(as referências, 4,0), sem verborragia ou comentários metadiscursivos. Esses valores não
são diretamente comparáveis aos do aluno, cujo conjunto de teste inclui sentenças
sintéticas, nas quais ele pontua alto; a pequena vantagem do aluno em BLEU entre os
conjuntos completos (0,120 contra 0,114) decorre das traduções idênticas à referência
concentradas nesse material.

A comparação adequada é a pareada, sobre as 247 sentenças históricas comuns às duas
avaliações. “Melhor” indica em quantas sentenças cada modelo obteve a melhor pontuação
(para o TER, o menor valor); as demais são empates.

| Métrica | Professor | Aluno ajustado | Diferença (prof. − aluno) | Professor melhor | Aluno melhor | *p* (Wilcoxon) |
|---|---:|---:|---:|---:|---:|---:|
| BLEU (0–1) | 0,106 | 0,012 | +0,094 | 169 | 7 | < 10⁻²⁷ |
| ChrF++ | 41,20 | 12,58 | +28,63 | 216 | 29 | < 10⁻³⁴ |
| TER ↓ | 79,12 | 157,29 | −78,17 | 192 | 12 | < 10⁻³⁰ |
| ROUGE-L (0–1) | 0,381 | 0,070 | +0,311 | 170 | 7 | < 10⁻²⁸ |
| Jaccard (0–1) | 0,318 | 0,035 | +0,283 | 169 | 6 | < 10⁻²⁸ |

O desempenho do professor se mantém estável no subconjunto (ChrF++ de 41,20), enquanto o
do aluno cai para 12,58. O professor obtém o melhor ChrF++ em 216 das 247 sentenças
(87,4%) e o melhor TER em 192 (77,7%), com 27 traduções idênticas à referência contra
apenas uma do aluno, e as diferenças são significativas em todas as métricas. No material
histórico, portanto, o aluno ajustado recupera pouco menos de um terço do ChrF++ do
professor.

Os 29 casos em que o aluno supera o professor em ChrF++ são, em sua maioria, entradas
lexicais isoladas ou sentenças muito curtas (2,8 palavras em média, contra 4,1 no
subconjunto), nas quais ambos erram e a diferença se resume a poucos caracteres; em dois
deles (*apiar* e *ygapukuîtaba*), o professor apenas repetiu a palavra em Tupi, sem
traduzi-la. Convém notar ainda que o professor está longe de ser um teto elevado: cerca
de um terço de suas traduções tem ChrF++ inferior a 20, o que faz da qualidade da
supervisão destilada um gargalo da abordagem.

### 5.5 Sentenças sintéticas versus históricas

O ganho não é homogêneo entre os tipos de sentença. Separando-se heuristicamente as
sentenças cuja estrutura corresponde aos *templates* sintéticos das de origem histórica,
o ChrF++ médio do aluno ajustado é de **56,6** nas primeiras e de **23,5** nas demais.
Como essa separação é aproximada, o valor obtido nas 247 sentenças comprovadamente
históricas (12,6) é a estimativa mais direta do desempenho sobre o material histórico.
Em qualquer das duas leituras, o aluno aprendeu com eficácia os padrões composicionais
controlados dos dados sintéticos, mas seu desempenho nas construções históricas,
ortograficamente variáveis e semanticamente densas, permanece muito inferior.

### 5.6 Análise qualitativa

Exemplos do conjunto de validação (Tupi → referência em Português):

| Tupi | Referência |
|---|---|
| `Agoacem apyába cetà;` | Achei muitos índios. |
| `abá guyra'i oîpsyky ka'ape` | o homem capturou o passarinho na mata |
| `Oiecuáb Apyàba cetá,` | apareceram muitos índios. |
| `Çupí na xe anga recé rüã.` | e não por minha alma. |

As traduções geradas foram exportadas para `predictions.csv` para inspeção manual; as
métricas correspondentes estão nas Seções 5.3 a 5.5. A tabela a seguir reúne exemplos
representativos dos padrões recorrentes no conjunto de teste, com as predições antes e
depois do ajuste (respostas longas do modelo original abreviadas com [...]):

| Padrão | Fonte (Tupi) | Referência | Antes do FT | Depois do FT |
|---|---|---|---|---|
| Melhoria clara | `îagûara osó ka'a rupi 'ybá 'oka` | a onça foi pela mata para colher frutas | as águias estão voando [...] | a onça foi pela mata para colher frutas. |
| Melhoria clara | `abá kapibara oîpsyky paranãpe` | o homem capturou a capivara no mar | um abacaxi de pé na praia [...] | o homem capturou a capivara no mar. |
| Erro lexical | `abá ogûatá xe irunamo 'yembyîeîa rupi îagûara suí osykyîébo` | o homem caminhou comigo pela praia tendo medo da onça | você está certo / Você está correto. | o homem caminhou comigo pela praia tendo medo do lobo marinho |
| Piora (interrogativa curta) | `umãba'epe?` | qual? | Qualquer coisa? | de onde? |
| Piora (palavra isolada) | `sarigûeîa` | gambá | pomba [...] | pacaíba. |
| Alucinação fluente | `Xe py'a xe 'anga eîar nde mba'eramo taûîé.` | Toma logo meu coração e minha alma como coisas tuas. | Os índios estão se reunindo na aldeia [...] | Aquele homem foi para a casa do seu pai. |

- **Melhoria clara.** O padrão mais frequente de melhoria ocorre em sentenças de
  estrutura composicional regular, típicas dos dados sintéticos: o modelo sem ajuste
  produzia conteúdo sem relação com a fonte (“as águias estão voando”, “um abacaxi de pé
  na praia”), acompanhado de explicações espúrias, e o modelo ajustado reproduz a
  referência. Esse padrão responde pela concentração de pontuações altas na cauda
  superior da distribuição.
- **Melhoria parcial com erro lexical.** O modelo ajustado captura a estrutura sintática,
  mas erra itens lexicais: a estrutura “o homem caminhou comigo pela praia tendo medo de
  X” é reproduzida, porém “onça” (*îagûara*) vira “lobo marinho”. Substituições dentro da
  mesma categoria semântica (frequentemente nomes de animais do léxico sintético)
  sugerem que o modelo aprendeu os *templates* estruturais com mais solidez do que as
  correspondências lexicais individuais.
- **Piora após o ajuste.** As quedas de ChrF++ concentram-se em interrogativas e itens
  lexicais muito curtos: para *umãba'epe?* (“qual?”), a predição anterior ao ajuste
  (“Qualquer coisa?”) compartilhava caracteres com a referência e a posterior (“de
  onde?”) não compartilha nenhum; *sarigûeîa* (“gambá”) foi traduzida como “pacaíba”.
  Nesses casos, a piora medida é em parte um artefato da métrica, pois em sentenças muito
  curtas qualquer divergência lexical zera a sobreposição.
- **Alucinação fluente.** O padrão de erro mais relevante do modelo ajustado difere
  qualitativamente do observado antes do ajuste: sentenças históricas mais complexas
  passam a receber traduções gramaticalmente fluentes, mas sem relação semântica com a
  fonte (última linha da tabela). Observou-se ainda a repetição de predições idênticas
  para fontes distintas (por exemplo, “Aquele homem foi para a casa.” e “Não sei como
  aquele homem.”), o que indica que, diante de construções fora da distribuição aprendida,
  o modelo regride a padrões frequentes dos dados de treinamento. Esse tipo de alucinação
  é potencialmente mais problemático do que a verborragia do modelo original, pois
  produz saídas plausíveis à primeira vista, cuja incorreção só é detectável por quem
  conhece a língua-fonte.
- **Omissões e respostas vazias.** Antes do ajuste, quatro sentenças não receberam
  predição; após o ajuste não há respostas vazias, mas ocorrem predições truncadas que
  omitem constituintes da fonte (por exemplo, “Aquele homem está procurando.”, sem o
  objeto).

### 5.7 Discussão

Considerados em conjunto, os resultados indicam que a destilação supervisionada cumpriu
seu objetivo central: transferir a um modelo compacto, treinável em *hardware* doméstico,
padrões da tarefa de tradução Tupi Antigo → Português a partir dos rótulos do professor e
de pares sintéticos. A melhoria é consistente entre métricas de naturezas distintas,
estatisticamente significativa e acompanhada de uma mudança qualitativa de comportamento:
da verborragia alucinatória do modelo original para respostas concisas e aderentes ao
formato da tarefa. Os resultados sugerem, contudo, uma leitura mais cautelosa:

- **Ganho concentrado nas estruturas sintéticas.** Os *templates* oferecem muitas
  instâncias de poucas estruturas, o que favorece a memorização de padrões, enquanto os
  textos históricos apresentam alta diversidade estrutural com pouquíssimos exemplos por
  padrão. Observa-se, assim, uma generalização limitada à vizinhança da distribuição de
  treinamento, esperada sob escassez extrema de dados. Como a divisão entre treino e
  teste é aleatória, as sentenças sintéticas do teste compartilham *templates* com as do
  treino, o que favorece as pontuações altas nesse material.
- **Direção das falhas.** A substituição de erros evidentes (verborragia, metadiscurso)
  por erros fluentes e plausíveis desloca o problema da detecção automática para a
  validação por especialistas, o que é sensível em aplicações de preservação e estudo de
  línguas históricas. As métricas automáticas capturam a melhoria relativa entre os
  cenários, mas não atestam a adequação absoluta das traduções para uso filológico ou
  pedagógico.
- **Qualidade da supervisão.** O professor, ainda que superior ao aluno, traduz com baixa
  fidelidade cerca de um terço das sentenças, e um aluno dificilmente supera a qualidade
  dos rótulos que recebe.
- **Valores absolutos.** Três fatores estruturais deprimem as métricas: a existência de
  uma única referência por sentença, que penaliza traduções alternativas válidas; a
  morfologia aglutinante do Tupi Antigo e a variação ortográfica das fontes, que reduzem a
  sobreposição lexical mesmo em traduções corretas; e o comprimento reduzido das
  referências (5,2 palavras em média), que torna as métricas instáveis em nível de
  sentença. A comparação entre cenários, calculada nas mesmas condições, permanece válida,
  mas os valores absolutos não devem ser lidos como medida de qualidade linguística.

### 5.8 Limitações

- **Volume de dados reduzido** 2.012 pares.
- **Restrição de hardware**, que limitou *batch size*, comprimento de sequência e
  velocidade.
- **Ruído na referência.** O conjunto de referência combina traduções históricas, por
  vezes interpretativas, e rótulos sintéticos definidos pelos *templates*, o que introduz
  ruído na avaliação; a separação entre sentenças sintéticas e históricas no conjunto de
  teste é heurística.
- **Escopo restrito.** A avaliação limita-se à direção Tupi Antigo → Português, a um único
  modelo-aluno e a um único modelo-professor.
- **Sem avaliação humana.** Não houve avaliação sistemática por especialistas na língua,
  que seria o padrão de referência adequado, o que é relevante dado o padrão de
  alucinações fluentes.
- **Uma configuração, uma execução.** Reporta-se uma única configuração de hiperparâmetros
  e uma única execução (semente 42), sem ablações que isolem o efeito dos rótulos do
  professor, dos dados sintéticos e do LoRA, nem comparação com *full fine-tuning*. Além
  disso, a inferência do *pipeline* (Seção 3.4) usa amostragem (`temperature` = 0,1) e
  quantização de 4 bits, de modo que pequenas variações entre execuções são possíveis; o
  efeito dessas escolhas não foi medido.
- **Conjuntos de avaliação distintos.** O professor foi avaliado nas 802 sentenças
  históricas e o aluno no conjunto de teste (601 sentenças); por isso, a comparação direta
  entre eles restringe-se às 247 sentenças comuns.

---

## 6. Conclusão e trabalhos futuros

Este projeto investigou se um SLM de 3 bilhões de parâmetros pode ser adaptado, em
hardware de consumo, à tradução de Tupi Antigo para Português por meio de destilação de
conhecimento baseada em rótulos e *fine-tuning* eficiente (SFT com LoRA). Em resposta à
pergunta central (Seção 1.2), a resposta é **sim, com ressalvas**: o ajuste é viável
nessas condições — ainda que lento (≈ 18 h 24 min, com memória acima da VRAM física) — e
produz melhoria substancial e estatisticamente significativa nas cinco métricas (ChrF++ de
12,5 para 28,8; TER de 435,9% para 118,6%), além de eliminar a verborragia e os
comentários metadiscursivos do modelo original. Contudo, o ganho concentra-se nas
estruturas sintéticas: nas 247 sentenças históricas comuns à avaliação do professor, o
aluno ajustado recupera pouco menos de um terço do ChrF++ do Gemini (12,6 contra 41,2) e
tende a produzir alucinações fluentes. Assim, no regime de dados estudado, a destilação
baseada em rótulos é eficaz para transferir padrões estruturais recorrentes, mas
insuficiente para induzir generalização robusta sobre a diversidade linguística real da
língua histórica.

Quanto aos objetivos específicos: (1) o *pipeline* reprodutível de pré-processamento, SFT
com LoRA e inferência foi construído (Seção 4); (2) o Qwen2.5-3B-Instruct foi treinado
sobre o conjunto aumentado de pares Tupi/Português (Seções 3 e 5.1); (3) o comportamento
do treinamento e o custo computacional foram caracterizados (Seções 5.1 e 5.2); e (4) as
traduções foram avaliadas quantitativa e qualitativamente (Seções 5.3 a 5.6).

**Trabalhos futuros:**

1. Comparar sistematicamente os postos LoRA (`r` ∈ {8, 16, 128, 256}) e o *full
   fine-tuning* — o resumo da versão publicada de Song et al. [1] aponta que o ajuste de
   todos os parâmetros supera o LoRA —, separando quanto da limitação de generalização
   decorre da capacidade dos adaptadores e quanto decorre da escassez de dados.
2. Ampliar e curar o corpus Tupi → Português, incluindo verificação por falantes/estudiosos
   e mais exemplos históricos alinhados, já que a lacuna entre aluno e professor
   concentra-se nessas construções.
3. Avaliar a tradução bidirecional (Português → Tupi) e o uso do módulo de RAG com
   dicionário/gramática.
4. Explorar *few-shot prompting* com dicas lexicais extraídas de dicionários de Tupi
   Antigo, tanto na inferência do aluno quanto na geração de rótulos pelo professor, o que
   elevaria a qualidade da própria supervisão destilada.
5. Acrescentar uma etapa de aprendizado por reforço após o SFT, com recompensa que
   penalize traduções sem relação semântica com a fonte (o que exige um sinal confiável
   apesar das limitações das métricas automáticas).
6. Testar outros modelos-professores e modelos-alunos e criar um tokenizador específico
   para o Tupi Antigo.
7. Realizar avaliação humana sistemática por especialistas em Tupi Antigo, necessária para
   aferir a qualidade linguística das traduções além das métricas automáticas.

---

## Referências

1. Song, Y.; Li, L.; Lothritz, C.; Ezzini, S.; Sleem, L.; Gentile, N.; State, R.;
   Bissyandé, T. F.; Klein, J. *Are Small Language Models the Silver Bullet to
   Low-Resource Languages Machine Translation?* Proceedings for the Ninth Workshop on
   Technologies for Machine Translation of Low Resource Languages (LoResMT 2026),
   Rabat, Morocco: ACL, 2026. Disponível em: https://aclanthology.org/2026.loresmt-1.1/

2. Hu, E. J. et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022.

3. Qwen Team. *Qwen2.5 Technical Report.* 2024.

4. NLLB Team et al. *No Language Left Behind: Scaling Human-Centered Machine Translation.*
   2022 (benchmark FLORES-200).

5. Hugging Face. Documentação das bibliotecas `transformers`, `peft`, `trl` e `datasets`.

6. Hinton, G.; Vinyals, O.; Dean, J. *Distilling the Knowledge in a Neural Network.*
   arXiv:1503.02531, 2015.

7. Gemini Team, Google. *Gemini: A Family of Highly Capable Multimodal Models.*
   arXiv:2312.11805, 2023.

8. Papineni, K.; Roukos, S.; Ward, T.; Zhu, W.-J. *BLEU: a Method for Automatic
   Evaluation of Machine Translation.* Proceedings of the 40th Annual Meeting of the ACL,
   2002, p. 311–318.

9. Popović, M. *chrF++: words helping character n-grams.* Proceedings of the Second
   Conference on Machine Translation (WMT), 2017, p. 612–618.

10. Snover, M.; Dorr, B.; Schwartz, R.; Micciulla, L.; Makhoul, J. *A Study of Translation
    Edit Rate with Targeted Human Annotation.* Proceedings of AMTA, 2006, p. 223–231.

11. Lin, C.-Y. *ROUGE: A Package for Automatic Evaluation of Summaries.* Text
    Summarization Branches Out (ACL Workshop), 2004, p. 74–81.

12. Jaccard, P. *The Distribution of the Flora in the Alpine Zone.* New Phytologist,
    v. 11, n. 2, 1912, p. 37–50.

13. Wilcoxon, F. *Individual Comparisons by Ranking Methods.* Biometrics Bulletin, v. 1,
    n. 6, 1945, p. 80–83.

14. Navarro, E. A. *Dicionário de Tupi Antigo: a língua indígena clássica do Brasil.*
    São Paulo: Global, 2013.

15. Navarro, E. A. *Método moderno de Tupi Antigo: a língua do Brasil dos primeiros
    séculos.* 3. ed. São Paulo: Global, 2008.

16. Navarro, E. A. *Transcrição e tradução integral anotada das cartas dos índios
    Camarões, escritas em 1645 em Tupi Antigo.* Boletim do Museu Paraense Emílio Goeldi.
    Ciências Humanas, v. 17, n. 3, e20210034, 2022.

17. Navarro, E. A. *Um texto anônimo, em língua geral amazônica, do século XVIII.*
    Revista USP, n. 90, p. 181–192, 2011.

18. Barbosa, A. L. *Curso de Tupi Antigo: gramática, exercícios, textos.* Rio de Janeiro:
    Livraria São José, 1956.

---
