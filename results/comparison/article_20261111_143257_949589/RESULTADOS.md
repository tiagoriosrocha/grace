# Comparação de sistemas — evidência fornecida

Dataset: factkg. Protocolo: alt5-claim-evidence-v3-local.
Claims duplicadas, grupos de origem e grafos de evidência repetidos ficam no mesmo conjunto.
Entidades podem aparecer em conjuntos diferentes; esta avaliação admite entidades conhecidas.
Divisão interna do export convertido; não equivale ao teste oficial do benchmark.

| Model                    | Accuracy        | Macro-F1        | Macro-F1 95% CI   |    AUC |
|:-------------------------|:----------------|:----------------|:------------------|-------:|
| majority_baseline        | 0.5046 ± 0.0000 | 0.3354 ± 0.0000 | [0.3293, 0.3411]  | 0.5    |
| text_tfidf               | 0.8299 ± 0.0000 | 0.8298 ± 0.0000 | [0.8210, 0.8386]  | 0.9078 |
| graph_pair_transformer   | 0.9750 ± 0.0034 | 0.9750 ± 0.0034 | [0.9721, 0.9777]  | 0.9917 |
| matching_nli             | 0.7694 ± 0.0000 | 0.7694 ± 0.0000 | [0.7602, 0.7790]  | 0.8512 |
| kgat_style               | 0.9240 ± 0.0049 | 0.9239 ± 0.0049 | [0.9185, 0.9294]  | 0.9746 |
| gmn_style                | 0.9294 ± 0.0094 | 0.9293 ± 0.0094 | [0.9249, 0.9338]  | 0.978  |
| relational_gnn           | 0.9586 ± 0.0019 | 0.9585 ± 0.0019 | [0.9550, 0.9622]  | 0.9874 |
| claim_evidence_attention | 0.6319 ± 0.0856 | 0.5715 ± 0.1298 | [0.5598, 0.5832]  | 0.8451 |
| local_factcheck_pipeline | 0.8073 ± 0.0000 | 0.8072 ± 0.0000 | [0.7980, 0.8167]  | 0.8924 |

Média ± desvio amostral entre sementes. IC95%: bootstrap por grupos completos da média
do Macro-F1 entre sementes fixas. As sementes não são novas observações de teste.
paired_comparisons.csv contém diferenças pareadas com intervalos pontuais, sem correção
por comparações múltiplas; não fornece uma declaração automática de superioridade.
Matrizes de confusão são separadas por semente.

## Cobertura e sobreposição

| evaluation_role   |   label |     n |
|:------------------|--------:|------:|
| test              |       0 |  6300 |
| test              |       1 |  6418 |
| train             |       0 | 21183 |
| train             |       1 | 22009 |
| validation        |       0 |  3785 |
| validation        |       1 |  4058 |

|          |   input |   eligible |   coverage |
|:---------|--------:|-----------:|-----------:|
| SUPPORTS |   35109 |      32485 |   0.925261 |
| REFUTES  |   32902 |      31268 |   0.950337 |

Grupos: 39362; maior grupo: 19.
Grafos repetidos entre conjuntos: 0.
Claims duplicadas entre conjuntos: 0.
Entidades textuais compartilhadas entre conjuntos (permitidas): 508044.

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
