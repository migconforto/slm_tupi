# Auditoria de ortografia

Arquivo auditado (somente leitura): `dataset_sintetico_aumentado.json`

SHA-256: `90bb2bae4abe898451c72a4f1d9b8b6f28fd500ffe586b407590e52504fcd72a`

Linhas: 1975 (824 históricas, 1151 sintéticas). No máximo 20 exemplos por categoria.

Nas históricas, o `output` é a tradução do Gemini; a tradução humana vem de `data/dataset_historico.json`.

## 1. Resumo

| Categoria | Históricos | Sintéticos |
|---|---:|---:|
| Fora de NFC (Tupi) | 0 | 0 |
| Fora de NFC (Português) | 0 | 0 |
| Apóstrofo fora do padrão `'` | 33 | 0 |
| Mais de um tipo de apóstrofo na linha | 2 | 0 |
| Espaços duplos | 0 | 0 |
| Espaços nas pontas | 0 | 0 |
| Outros brancos | 0 | 0 |
| Maiúscula depois da primeira palavra | 68 | 0 |
| Tupi idêntico ao `output` | 8 | 0 |
| Tupi quase idêntico ao `output` (similaridade ≥ 0.8) | 5 | 0 |
| Tupi idêntico à tradução humana | 0 | 0 |
| Tupi quase idêntico à tradução humana (similaridade ≥ 0.8) | 5 | 0 |
| Tupi que parece Português | 1 | 0 |
| Vazios | 0 | 0 |
| Duplicatas exatas | 0 | 0 |
| Mesmo Tupi com referências diferentes | 0 | 0 |
| Letras de grafia histórica (äëïöüÄËÏÖÜàèìòùÀÈÌÒÙçÇ) | 281 | 0 |
| Grupos de grafias variantes (conjunto todo) | 138 | |

## 2. Inventário de caracteres

### Históricos: lado Tupi

Letras e dígitos ASCII: 10759 ocorrências.

| Caractere | Codepoint | Nome | Frequência |
|---|---|---|---:|
| (espaço) | U+0020 | SPACE | 1651 |
| `é` | U+00E9 | LATIN SMALL LETTER E WITH ACUTE | 424 |
| `,` | U+002C | COMMA | 287 |
| `á` | U+00E1 | LATIN SMALL LETTER A WITH ACUTE | 251 |
| `í` | U+00ED | LATIN SMALL LETTER I WITH ACUTE | 219 |
| `.` | U+002E | FULL STOP | 150 |
| `ç` | U+00E7 | LATIN SMALL LETTER C WITH CEDILLA | 150 |
| `ó` | U+00F3 | LATIN SMALL LETTER O WITH ACUTE | 139 |
| `'` | U+0027 | APOSTROPHE | 123 |
| `î` | U+00EE | LATIN SMALL LETTER I WITH CIRCUMFLEX | 123 |
| `ã` | U+00E3 | LATIN SMALL LETTER A WITH TILDE | 110 |
| `ú` | U+00FA | LATIN SMALL LETTER U WITH ACUTE | 106 |
| `à` | U+00E0 | LATIN SMALL LETTER A WITH GRAVE | 98 |
| `û` | U+00FB | LATIN SMALL LETTER U WITH CIRCUMFLEX | 92 |
| `ý` | U+00FD | LATIN SMALL LETTER Y WITH ACUTE | 75 |
| `ò` | U+00F2 | LATIN SMALL LETTER O WITH GRAVE | 61 |
| `õ` | U+00F5 | LATIN SMALL LETTER O WITH TILDE | 61 |
| `è` | U+00E8 | LATIN SMALL LETTER E WITH GRAVE | 56 |
| `ẽ` | U+1EBD | LATIN SMALL LETTER E WITH TILDE | 51 |
| `ù` | U+00F9 | LATIN SMALL LETTER U WITH GRAVE | 49 |
| `?` | U+003F | QUESTION MARK | 47 |
| `;` | U+003B | SEMICOLON | 43 |
| `’` | U+2019 | RIGHT SINGLE QUOTATION MARK | 39 |
| `-` | U+002D | HYPHEN-MINUS | 24 |
| `:` | U+003A | COLON | 22 |
| `Ç` | U+00C7 | LATIN CAPITAL LETTER C WITH CEDILLA | 20 |
| `ŷ` | U+0177 | LATIN SMALL LETTER Y WITH CIRCUMFLEX | 13 |
| `!` | U+0021 | EXCLAMATION MARK | 7 |
| `ũ` | U+0169 | LATIN SMALL LETTER U WITH TILDE | 7 |
| `ì` | U+00EC | LATIN SMALL LETTER I WITH GRAVE | 2 |
| `ĩ` | U+0129 | LATIN SMALL LETTER I WITH TILDE | 2 |
| `ỹ` | U+1EF9 | LATIN SMALL LETTER Y WITH TILDE | 2 |
| `‘` | U+2018 | LEFT SINGLE QUOTATION MARK | 2 |
| `(` | U+0028 | LEFT PARENTHESIS | 1 |
| `)` | U+0029 | RIGHT PARENTHESIS | 1 |
| `Í` | U+00CD | LATIN CAPITAL LETTER I WITH ACUTE | 1 |
| `Î` | U+00CE | LATIN CAPITAL LETTER I WITH CIRCUMFLEX | 1 |

### Históricos: lado Português

Letras e dígitos ASCII: 14811 ocorrências.

| Caractere | Codepoint | Nome | Frequência |
|---|---|---|---:|
| (espaço) | U+0020 | SPACE | 2610 |
| `,` | U+002C | COMMA | 291 |
| `.` | U+002E | FULL STOP | 150 |
| `ã` | U+00E3 | LATIN SMALL LETTER A WITH TILDE | 148 |
| `é` | U+00E9 | LATIN SMALL LETTER E WITH ACUTE | 98 |
| `á` | U+00E1 | LATIN SMALL LETTER A WITH ACUTE | 87 |
| `ê` | U+00EA | LATIN SMALL LETTER E WITH CIRCUMFLEX | 54 |
| `ç` | U+00E7 | LATIN SMALL LETTER C WITH CEDILLA | 51 |
| `?` | U+003F | QUESTION MARK | 48 |
| `;` | U+003B | SEMICOLON | 43 |
| `-` | U+002D | HYPHEN-MINUS | 32 |
| `:` | U+003A | COLON | 24 |
| `ó` | U+00F3 | LATIN SMALL LETTER O WITH ACUTE | 19 |
| `í` | U+00ED | LATIN SMALL LETTER I WITH ACUTE | 17 |
| `à` | U+00E0 | LATIN SMALL LETTER A WITH GRAVE | 11 |
| `!` | U+0021 | EXCLAMATION MARK | 7 |
| `â` | U+00E2 | LATIN SMALL LETTER A WITH CIRCUMFLEX | 4 |
| `î` | U+00EE | LATIN SMALL LETTER I WITH CIRCUMFLEX | 3 |
| `ô` | U+00F4 | LATIN SMALL LETTER O WITH CIRCUMFLEX | 3 |
| `ú` | U+00FA | LATIN SMALL LETTER U WITH ACUTE | 3 |
| `"` | U+0022 | QUOTATION MARK | 2 |
| `(` | U+0028 | LEFT PARENTHESIS | 2 |
| `)` | U+0029 | RIGHT PARENTHESIS | 2 |
| `'` | U+0027 | APOSTROPHE | 1 |
| `À` | U+00C0 | LATIN CAPITAL LETTER A WITH GRAVE | 1 |
| `É` | U+00C9 | LATIN CAPITAL LETTER E WITH ACUTE | 1 |
| `õ` | U+00F5 | LATIN SMALL LETTER O WITH TILDE | 1 |

### Sintéticos: lado Tupi

Letras e dígitos ASCII: 22164 ocorrências.

| Caractere | Codepoint | Nome | Frequência |
|---|---|---|---:|
| (espaço) | U+0020 | SPACE | 3985 |
| `á` | U+00E1 | LATIN SMALL LETTER A WITH ACUTE | 1072 |
| `î` | U+00EE | LATIN SMALL LETTER I WITH CIRCUMFLEX | 916 |
| `'` | U+0027 | APOSTROPHE | 762 |
| `û` | U+00FB | LATIN SMALL LETTER U WITH CIRCUMFLEX | 685 |
| `é` | U+00E9 | LATIN SMALL LETTER E WITH ACUTE | 471 |
| `ó` | U+00F3 | LATIN SMALL LETTER O WITH ACUTE | 466 |
| `ã` | U+00E3 | LATIN SMALL LETTER A WITH TILDE | 276 |
| `ý` | U+00FD | LATIN SMALL LETTER Y WITH ACUTE | 108 |
| `í` | U+00ED | LATIN SMALL LETTER I WITH ACUTE | 107 |
| `ú` | U+00FA | LATIN SMALL LETTER U WITH ACUTE | 80 |
| `õ` | U+00F5 | LATIN SMALL LETTER O WITH TILDE | 68 |
| `-` | U+002D | HYPHEN-MINUS | 56 |
| `,` | U+002C | COMMA | 48 |
| `?` | U+003F | QUESTION MARK | 44 |
| `.` | U+002E | FULL STOP | 12 |
| `!` | U+0021 | EXCLAMATION MARK | 8 |

### Sintéticos: lado Português

Letras e dígitos ASCII: 29938 ocorrências.

| Caractere | Codepoint | Nome | Frequência |
|---|---|---|---:|
| (espaço) | U+0020 | SPACE | 5937 |
| `á` | U+00E1 | LATIN SMALL LETTER A WITH ACUTE | 258 |
| `ç` | U+00E7 | LATIN SMALL LETTER C WITH CEDILLA | 214 |
| `ã` | U+00E3 | LATIN SMALL LETTER A WITH TILDE | 112 |
| `?` | U+003F | QUESTION MARK | 44 |
| `-` | U+002D | HYPHEN-MINUS | 40 |
| `ê` | U+00EA | LATIN SMALL LETTER E WITH CIRCUMFLEX | 36 |
| `,` | U+002C | COMMA | 28 |
| `é` | U+00E9 | LATIN SMALL LETTER E WITH ACUTE | 28 |
| `ó` | U+00F3 | LATIN SMALL LETTER O WITH ACUTE | 16 |
| `!` | U+0021 | EXCLAMATION MARK | 8 |
| `í` | U+00ED | LATIN SMALL LETTER I WITH ACUTE | 7 |
| `õ` | U+00F5 | LATIN SMALL LETTER O WITH TILDE | 7 |
| `/` | U+002F | SOLIDUS | 6 |

## 3. Formas Unicode

| Grupo | Lado | Textos fora de NFC | Caracteres combinantes |
|---|---|---:|---:|
| Históricos | Tupi | 0 | 0 |
| Históricos | Português | 0 | 0 |
| Sintéticos | Tupi | 0 | 0 |
| Sintéticos | Português | 0 | 0 |

Combinantes presentes e exemplos fora de NFC:

- Nenhum.

## 4. Apóstrofo (oclusiva glotal) e hífens

Lado Tupi.

| Grupo | Sinal | Codepoint | Ocorrências | Linhas |
|---|---|---|---:|---:|
| Históricos | `'` | U+0027 | 123 | 85 |
| Históricos | `’` | U+2019 | 39 | 31 |
| Históricos | `‘` | U+2018 | 2 | 2 |
| Históricos | `-` | U+002D | 24 | 19 |
| Sintéticos | `'` | U+0027 | 762 | 529 |
| Sintéticos | `-` | U+002D | 56 | 56 |

Históricos: 33 linhas com apóstrofo fora do padrão; 2 misturam tipos.

- `hist_0004`: "Amõ itáybaté ‘árybo ixé agûapyk, 'ypa'ũporanga repîaka." → 'Sobre algum rochedo eu me sento, para ver a ilha bonita.'
- `hist_0025`: 'Apŷaba i membyra pysykiré, a’e sygûasu ybõû.' → 'O homem pegou o filho dela, e a tamanduá-bandeira o feriu.'
- `hist_0026`: 'Sembîara nda sygûasu ruã erimba’e.' → 'A presa não era a tamanduá-bandeira, era o homem.'
- `hist_0027`: "A’ete, a'e abá sy." → 'Na verdade, ela era a mãe de alguém.'
- `hist_0030`: 'A’ete pŷaba o sy oîuká nhẽ.' → 'Na verdade, o homem matou a sua mãe.'
- `hist_0031`: 'Aîmondó benhẽ xe nhe’enga' → 'Envio logo a minha fala'
- `hist_0032`: 'opamenhẽ peẽme, xe ra’yretá.' → 'a todos vocês, meus filhos.'
- `hist_0035`: 'N’i katuî nhẽ kó nhandé retama pupé nhẽ;' → 'Não é nada bom dentro de nossa terra;'
- `hist_0038`: 'Kó xe nhe’enga aîmondó benhẽ' → 'Esta minha fala eu envio logo'
- `hist_0043`: 'peẽme opamenhẽ ma’e monhangagûama resé.' → 'para fazer todas as coisas para vocês.'
- `hist_0045`: 'ixébe ma’eaíba pe remimonhangûera resé kó marana,' → 'por causa do mal que vocês fizeram nesta guerra,'
- `hist_0046`: 'pe raûsubare’yma.' → 'a falta de amor de vocês.'
- `hist_0050`: 'Ma’ẽtepe é mokõî ygûã maramonhangápe?' → 'Por que existem dois motivos de guerra?'
- `hist_0051`: 'Oré resé omaramonhãba’e, apŷabaíba pitikoara,' → 'Quem guerreia contra nós, o homem mau pitiguara,'
- `hist_0052`: 'kanhẽmbabi Serinha’ẽme.' → 'perdeu-se em Serinhaém.'
- `hist_0056`: 'Apŷabaíba irũnamo o a’o resé omaramonhãba’epûera,' → 'O homem mau junto com o que guerreou por causa de sua roupa,'
- `hist_0058`: 'N’onheme’engi îeí ãgûa supé quartel.' → 'Não se deu logo o quartel para ele.'
- `hist_0064`: 'opabenhẽ mba’eaíba' → 'todo o mal'
- `hist_0066`: 'a’ereme, koba’e suí pe pysyrõagûama resé.' → 'então, para a salvação de vocês disso.'
- `hist_0072`: 'apŷabaíba, oîopa’ũme, oîoirũnamo' → 'o homem mau, entre si, uns com os outros'

Sintéticos: 0 linhas com apóstrofo fora do padrão; 0 misturam tipos.


## 5. Espaços, pontuação final e maiúsculas

Lado Tupi.

### Históricos

- Espaços duplos: 0; nas pontas: 0; outros brancos: 0.
- Começam com maiúscula: 507 de 824.
- Maiúscula depois da primeira palavra (possíveis nomes próprios): 68.
- Pontuação final: `(nenhuma)` 367, `,` 205, `.` 144, `?` 45, `;` 41, `:` 15, `!` 7.

- `hist_0000`: 'Ereîkuápe Karagûatatyba?' → 'Sabe por onde fica o Caraguatatiba?'
- `hist_0001`: "A'epe, asé e'ikatu Paranã pu rendupa" → 'Ali, costumamos ouvir o som do mar'
- `hist_0002`: 'Xe gûatáreme Paranã koty' → 'Em minha caminhada para o lado do mar'
- `hist_0015`: "A'e ri, ama'ẽ ybaka resé Tupãberaba repîápotá." → 'Depois disso, olho para o céu para ver o relâmpago de Deus.'
- `hist_0016`: 'Pytúnybo, Îasýobágûasú sẽmi.' → 'À noite, a Lua grande sai.'
- `hist_0052`: 'kanhẽmbabi Serinha’ẽme.' → 'perdeu-se em Serinhaém.'
- `hist_0062`: 'Marãnamo? Opamenhẽ pe rubetéramo' → 'Por quê? Sendo todos o pai de vocês'
- `hist_0084`: 'Okanhẽ capitão André de Souza Bioby' → 'Perdeu-se o capitão André de Souza Bioby'
- `hist_0128`: 'aèbé Apyabetà oçopár' → 'Também muita gente se juntou'
- `hist_0138`: 'Mocõi Caraíba reté, Umambaé xe moeté' → 'Dois capitães verdadeiros, Umambaé me honrou'
- `hist_0141`: 'aé xe Tutýra teité oinongucár cetà miapè' → 'Aquele meu tio mesmo mandou pôr muito pão'
- `hist_0161`: 'Çupí Sam Braz xe pycyrõ.' → 'Certamente Sam Braz me salvou.'
- `hist_0170`: 'Ixé Tutýra tuibaé,' → 'Meu tio é velho,'
- `hist_0172`: 'Sta Cruz çupí oromonhang,' → 'Certamente fizemos a Santa Cruz,'
- `hist_0173`: 'Padre Nosso abé oroçaang.' → 'Também ensaiamos o Padre Nosso.'
- `hist_0187`: 'Toicò xe irùmo Tupã eté.' → 'Que eles fiquem junto comigo como se fossem Deus.'
- `hist_0193`: 'Coritéi Camutápe acýc.' → 'Logo cheguei a Camutá.'
- `hist_0199`: 'Acepiác coyté Araticù;' → 'Vi então o Araticu;'
- `hist_0200`: 'Açó çapyà Pai robaké' → 'Fui de repente à presença do Padre'
- `hist_0202`: 'Opauán eí Apyabetà;' → 'Toda a gente disse "acabou";'

### Sintéticos

- Espaços duplos: 0; nas pontas: 0; outros brancos: 0.
- Começam com maiúscula: 35 de 1151.
- Maiúscula depois da primeira palavra (possíveis nomes próprios): 0.
- Pontuação final: `(nenhuma)` 1103, `?` 36, `.` 12.


## 6. Comprimento em palavras

| Grupo | Lado | Mín | P25 | Mediana | P75 | P90 | Máx | Média |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Históricos | Tupi | 1 | 1 | 3 | 4 | 5 | 9 | 3.0 |
| Históricos | Português | 1 | 2 | 4 | 6 | 7 | 13 | 4.2 |
| Sintéticos | Tupi | 2 | 3 | 4 | 5 | 8 | 9 | 4.5 |
| Sintéticos | Português | 2 | 4 | 6 | 8 | 10 | 12 | 6.2 |

## 7. Linhas suspeitas

Nada foi corrigido: as linhas abaixo são só para decisão (R3.4).

### Tupi idêntico ao `output`: 8

- `hist_0594`: 'takuba' → 'takuba'
- `hist_0627`: 'apakuî' → 'apakuî'
- `hist_0635`: 'akasanga' → 'akasanga'
- `hist_0668`: 'tupiara' → 'tupiara'
- `hist_0696`: 'apiar' → 'apiar'
- `hist_0732`: 'ygapukuîtaba' → 'ygapukuîtaba'
- `hist_0742`: "tepîaka'uba" → "tepîaka'uba"
- `hist_0774`: 'Pavuna' → 'Pavuna'

### Tupi quase idêntico ao `output` (similaridade ≥ 0.8): 5

- `hist_0084`: 'Okanhẽ capitão André de Souza Bioby' → 'Perdeu-se o capitão André de Souza Bioby'
- `hist_0375`: 'Ia que yso mesmo quizeste,' → 'Já que isso mesmo quiseste,'
- `hist_0680`: "mandi'oka" → 'mandioca'
- `hist_0739`: 'sabîá' → 'sabiá'
- `hist_0746`: 'tamandûá' → 'tamanduá'

### Tupi idêntico à tradução humana: 0

- Nenhuma.

### Tupi quase idêntico à tradução humana (similaridade ≥ 0.8): 5

- `hist_0322`: 'Cavados de primavéra abé' → humana: 'Cávados de primavera também,'
- `hist_0374`: 'Lá me dyse pa o Leste,' → humana: 'lá me disse. - Para o Leste,'
- `hist_0375`: 'Ia que yso mesmo quizeste,' → humana: 'já que isso mesmo quiseste'
- `hist_0680`: "mandi'oka" → humana: 'mandioca'
- `hist_0746`: 'tamandûá' → humana: 'tamanduá'

### Tupi que parece Português: 1

- `hist_0084`: 'Okanhẽ capitão André de Souza Bioby' → 'Perdeu-se o capitão André de Souza Bioby'

### Vazios: 0

- Nenhuma.

### Duplicatas exatas: 0

- Nenhuma.

### Mesmo Tupi com referências diferentes: 0

- Nenhuma.

### Letras de grafia histórica (äëïöüÄËÏÖÜàèìòùÀÈÌÒÙçÇ): 281

- `hist_0127`: 'Ixé copixápe catú aimõgatyrõ ygaruçú' → 'Eu na roça bem limpo o navio'
- `hist_0128`: 'aèbé Apyabetà oçopár' → 'Também muita gente se juntou'
- `hist_0129`: 'aé abé coytè oiporacár' → 'Ele também já mandou trabalhar'
- `hist_0130`: 'Mojú cotý auatàuatá' → 'Para o lado do Moju andei muito'
- `hist_0131`: 'Xe tutýra còpe apytá' → 'Fiquei na roça de meu tio'
- `hist_0132`: 'Oporandùb aé ixébo' → 'Ele perguntou para mim'
- `hist_0133`: 'Maiabépe nde pyà indébo?' → 'Como está o teu coração para ti?'
- `hist_0134`: 'Umamepe catú ereçó?' → 'Para onde, afinal, tu vais?'
- `hist_0135`: 'Abápe abé nde irùmo oçó?' → 'Com quem também tu vais?'
- `hist_0136`: 'Iepuxàpe' → 'Para o Iepuxá'
- `hist_0137`: 'Tapyýietà recoçàpe' → 'Para a habitação de muitos tapuias'
- `hist_0139`: 'Ixe irúnamo oço potár' → 'Ele quer ir junto comigo'
- `hist_0141`: 'aé xe Tutýra teité oinongucár cetà miapè' → 'Aquele meu tio mesmo mandou pôr muito pão'
- `hist_0143`: 'eí ixébo: tiambaèú' → 'Disse para mim: vamos comer'
- `hist_0144`: 'Co mirì aíra tembiú' → 'Este pouco de comida'
- `hist_0145`: 'Mojù çupí i poreauçub' → 'Certamente o Moju é de se amar'
- `hist_0146`: 'Aipò recè aé noçauçùb' → 'Por isso ele não o amava'
- `hist_0148`: 'Çupí catú aú ára iabé' → 'Certamente é bom comer nesta hora'
- `hist_0149`: 'Arecò tres taraýra' → 'Tenho três filhos'
- `hist_0150`: 'Cobè catú eté, aé ixupé' → 'Certamente isto é bom, disse eu a ele'

## 8. Variantes ortográficas prováveis

138 grupos com mais de uma grafia, sem contar diferenças de caixa. Os 20 mais frequentes:

- `rupi`: `rupi` (317), `rupí` (17)
- `aba`: `abá` (313), `abà` (1)
- `oka`: `oka` (104), `'oka` (100)
- `ae`: `a'e` (70), `aé` (40)
- `irunamo`: `irunamo` (100), `irúnamo` (3), `irũnamo` (2)
- `iko`: `îkó` (100), `ikó` (1)
- `parana`: `paranã` (71), `paranà` (1)
- `ybytyra`: `ybytýra` (68), `ybytyra` (2)
- `guyrai`: `guyra'i` (61), `gûyra'i` (1)
- `pupe`: `pupé` (47), `pupè` (3)
- `catu`: `catú` (27), `catù` (17), `catu` (1)
- `amo`: `amõ` (31), `amò` (9), `amó` (2)
- `ixe`: `ixé` (28), `ixe` (5)
- `cupi`: `çupí` (26), `çupi` (2)
- `abe`: `abé` (25), `abè` (2)
- `nde`: `nde` (24), `ndé` (1)
- `oiabab`: `oîabab` (24), `oiabàb` (1)
- `pai`: `paí` (17), `pai` (7), `paì` (1)
- `mbae`: `mbaé` (19), `mba'e` (4)
- `y`: `'y` (21), `y` (2)

## 9. Normalizações

**Aplicadas** em `normalize_tupi`, só para busca e comparação (o texto de treino não é reescrito): NFC, unificação dos apóstrofos para `'`, colapso de espaços e `strip`.

**Propostas, não aplicadas** (exigem decisão ou revisão de especialista):

1. Mapear as letras de grafia histórica para a grafia de Navarro (281 linhas). Exemplo: 'Ixé copixápe catú aimõgatyrõ ygaruçú'.
2. Unificar o uso de hífen interno (75 linhas). Exemplo: 'xe suí, cristão-ramo pe rekó pupé.'.
3. Unir as variantes da seção 8 em uma única grafia. Algumas podem ser palavras distintas, por isso a lista pede revisão.
4. Remover a pontuação final do lado Tupi.

