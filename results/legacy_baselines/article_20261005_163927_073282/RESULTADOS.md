# Comparação de sistemas — evidência fornecida

Dataset: hover. Protocolo: alt5-claim-evidence-v3-local.
Claims duplicadas, grupos de origem e grafos de evidência repetidos ficam no mesmo conjunto.
Entidades podem aparecer em conjuntos diferentes; esta avaliação admite entidades conhecidas.
Divisão interna do export convertido; não equivale ao teste oficial do benchmark.

| Model                    | Accuracy        | Macro-F1        | Macro-F1 95% CI   |    AUC |
|:-------------------------|:----------------|:----------------|:------------------|-------:|
| majority_baseline        | 0.5811 ± 0.0000 | 0.3675 ± 0.0000 | [0.3617, 0.3735]  | 0.5    |
| text_tfidf               | 0.4903 ± 0.0000 | 0.4820 ± 0.0000 | [0.4674, 0.4958]  | 0.476  |
| graph_pair_transformer   | 0.5811 ± 0.0000 | 0.3675 ± 0.0000 | [0.3617, 0.3735]  | 0.5643 |
| matching_nli             | 0.6181 ± 0.0000 | 0.6166 ± 0.0000 | [0.6017, 0.6311]  | 0.6712 |
| relational_gnn           | 0.6120 ± 0.0059 | 0.5971 ± 0.0060 | [0.5851, 0.6094]  | 0.6418 |
| claim_evidence_attention | 0.6448 ± 0.0009 | 0.6208 ± 0.0115 | [0.6073, 0.6339]  | 0.6861 |
| local_factcheck_pipeline | 0.6106 ± 0.0000 | 0.6100 ± 0.0000 | [0.5947, 0.6244]  | 0.6598 |

Média ± desvio amostral entre sementes. IC95%: bootstrap por grupos completos da média
do Macro-F1 entre sementes fixas. As sementes não são novas observações de teste.
paired_comparisons.csv contém diferenças pareadas com intervalos pontuais, sem correção
por comparações múltiplas; não fornece uma declaração automática de superioridade.
Matrizes de confusão são separadas por semente.

## Cobertura e sobreposição

| evaluation_role   |   label |    n |
|:------------------|--------:|-----:|
| test              |       0 | 1835 |
| test              |       1 | 2546 |
| train             |       0 | 6081 |
| train             |       1 | 8751 |
| validation        |       0 | 1111 |
| validation        |       1 | 1528 |

|               |   input |   eligible |   coverage |
|:--------------|--------:|-----------:|-----------:|
| SUPPORTED     |   12930 |      12825 |   0.991879 |
| NOT_SUPPORTED |    9057 |       9027 |   0.996688 |

Grupos: 20796; maior grupo: 7.
Grafos repetidos entre conjuntos: 0.
Claims duplicadas entre conjuntos: 0.
Entidades textuais compartilhadas entre conjuntos (permitidas): 35437.

## Escopo da comparação

text_tfidf usa claim textual + triplas verbalizadas; graph_pair_transformer usa os dois
grafos serializados; matching_nli e relational_gnn usam os dois grafos e NLI congelado;
claim_evidence_attention usa claim textual e tokens do grafo de evidência.
local_factcheck_pipeline extrai condições, recupera top-k evidências, aplica NLI
congelado e usa regressão logística ajustada no treino; C e o limiar são escolhidos
na validação. A auditoria e as métricas usam o mesmo escore global.
Os sistemas compartilham exemplos e split, mas diferem em entradas, truncamento e supervisão
prévia. A comparação não isola causalmente o efeito da arquitetura.
Embeddings congelados são compartilháveis; vocabulários, escalas e classificadores são
ajustados somente no treino. Features explícitas exact estão desativadas.
exact_match_diagnostic está em metrics.csv, fora da comparação aprendida.

A origem da conversão dos grafos precisa ser documentada no artigo. Este código não
certifica que a geração anterior dos grafos desconhecia os rótulos. A seleção de modelos
com resultados anteriores destes dados também deve ser declarada como exploração prévia.
