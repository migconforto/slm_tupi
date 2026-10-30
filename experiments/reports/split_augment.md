# Partição de frases históricas longas (`exp03_split_aug`)

Saída: `data/dataset_augmentation_v3.json`. Janelas contíguas: desligadas.

"Longa" significa mais de 4 palavras em Tupi: percentil 75 das 574 frases históricas do treino.

| Medida | Valor |
|---|---:|
| Frases históricas longas no treino | 81 |
| Frases particionadas | 15 |
| Segmentos candidatos | 30 |
| Descartados: iguais a uma frase de validação | 0 |
| Descartados: iguais a um par do treino | 0 |
| Descartados: repetidos entre os novos pares | 0 |
| Novos pares | 30 |
| Treino de v2 | 1382 |
| Treino aumentado | 1412 |

SHA-256 dos IDs de validação: `86c9431b2bf3d5265f30bf166dab5b9834dfab2bbb80d05b8882f8b022a4e5f3`

Interseção entre os Tupi normalizados de validação e do treino aumentado: 0 (conferido por asserção).

## 10 exemplos antes/depois

**`hist_0004`:** "Amõ itáybaté ‘árybo ixé agûapyk, 'ypa'ũporanga repîaka." → 'Sobre algum rochedo eu me sento, para ver a ilha bonita.'

- 'Amõ itáybaté ‘árybo ixé agûapyk,' → 'Sobre algum rochedo eu me sento,'
- "'ypa'ũporanga repîaka." → 'para ver a ilha bonita.'

**`hist_0006`:** "Xe ambyasýramo, a'ekatu tembi'ué gûaba:" → 'Quando estou triste, o alimento é bom:'

- 'Xe ambyasýramo,' → 'Quando estou triste,'
- "a'ekatu tembi'ué gûaba:" → 'o alimento é bom:'

**`hist_0015`:** "A'e ri, ama'ẽ ybaka resé Tupãberaba repîápotá." → 'Depois disso, olho para o céu para ver o relâmpago de Deus.'

- "A'e ri," → 'Depois disso,'
- "ama'ẽ ybaka resé Tupãberaba repîápotá." → 'olho para o céu para ver o relâmpago de Deus.'

**`hist_0025`:** 'Apŷaba i membyra pysykiré, a’e sygûasu ybõû.' → 'O homem pegou o filho dela, e a tamanduá-bandeira o feriu.'

- 'Apŷaba i membyra pysykiré,' → 'O homem pegou o filho dela,'
- 'a’e sygûasu ybõû.' → 'e a tamanduá-bandeira o feriu.'

**`hist_0051`:** 'Oré resé omaramonhãba’e, apŷabaíba pitikoara,' → 'Quem guerreia contra nós, o homem mau pitiguara,'

- 'Oré resé omaramonhãba’e,' → 'Quem guerreia contra nós,'
- 'apŷabaíba pitikoara,' → 'o homem mau pitiguara,'

**`hist_0065`:** 'pe remimonhanga resé, opabẽ culpa seîtykite' → 'por causa do que vocês fizeram, toda a culpa será lançada fora'

- 'pe remimonhanga resé,' → 'por causa do que vocês fizeram,'
- 'opabẽ culpa seîtykite' → 'toda a culpa será lançada fora'

**`hist_0079`:** 'Kûesé orogûar benhẽ 4 tubixaba, opabenhẽ i nhanama' → 'Ontem trouxemos logo 4 chefes, toda a gente deles'

- 'Kûesé orogûar benhẽ 4 tubixaba,' → 'Ontem trouxemos logo 4 chefes,'
- 'opabenhẽ i nhanama' → 'toda a gente deles'

**`hist_0103`:** 'Emonã pe rekóreme, xe rorybeténe' → 'Neste modo de vida de vocês, eu ficarei muito feliz'

- 'Emonã pe rekóreme,' → 'Neste modo de vida de vocês,'
- 'xe rorybeténe' → 'eu ficarei muito feliz'

**`hist_0150`:** 'Cobè catú eté, aé ixupé' → 'Certamente isto é bom, disse eu a ele'

- 'Cobè catú eté,' → 'Certamente isto é bom,'
- 'aé ixupé' → 'disse eu a ele'

**`hist_0169`:** 'Apàc coyté; xe abé ambyacý.' → 'Acordei já; eu também me entristeci.'

- 'Apàc coyté;' → 'Acordei já;'
- 'xe abé ambyacý.' → 'eu também me entristeci.'
