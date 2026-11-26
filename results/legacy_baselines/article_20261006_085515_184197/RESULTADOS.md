# Comparação de sistemas — evidência fornecida

Dataset: feverous. Protocolo: alt5-claim-evidence-v3-local.
Claims duplicadas, grupos de origem e grafos de evidência repetidos ficam no mesmo conjunto.
Entidades podem aparecer em conjuntos diferentes; esta avaliação admite entidades conhecidas.
Divisão interna do export convertido; não equivale ao teste oficial do benchmark.

| Model                    | Accuracy        | Macro-F1        | Macro-F1 95% CI   |    AUC |
|:-------------------------|:----------------|:----------------|:------------------|-------:|
| majority_baseline        | 0.4954 ± 0.0000 | 0.3313 ± 0.0000 | [0.3226, 0.3398]  | 0.5    |
| text_tfidf               | 0.6373 ± 0.0000 | 0.6373 ± 0.0000 | [0.6185, 0.6552]  | 0.689  |
| graph_pair_transformer   | 0.7967 ± 0.0103 | 0.7965 ± 0.0104 | [0.7828, 0.8095]  | 0.8748 |
| matching_nli             | 0.8159 ± 0.0000 | 0.8159 ± 0.0000 | [0.8020, 0.8309]  | 0.8878 |
| relational_gnn           | 0.7952 ± 0.0072 | 0.7951 ± 0.0073 | [0.7818, 0.8080]  | 0.8726 |
| claim_evidence_attention | 0.6449 ± 0.0098 | 0.6336 ± 0.0105 | [0.6188, 0.6478]  | 0.7113 |
| local_factcheck_pipeline | 0.7981 ± 0.0000 | 0.7981 ± 0.0000 | [0.7828, 0.8144]  | 0.8788 |

Média ± desvio amostral entre sementes. IC95%: bootstrap por grupos completos da média
do Macro-F1 entre sementes fixas. As sementes não são novas observações de teste.
paired_comparisons.csv contém diferenças pareadas com intervalos pontuais, sem correção
por comparações múltiplas; não fornece uma declaração automática de superioridade.
Matrizes de confusão são separadas por semente.

## Cobertura e sobreposição

| evaluation_role   |   label |    n |
|:------------------|--------:|-----:|
| test              |       0 | 1305 |
| test              |       1 | 1281 |
| train             |       0 | 4373 |
| train             |       1 | 4412 |
| validation        |       0 |  760 |
| validation        |       1 |  791 |

|          |   input |   eligible |   coverage |
|:---------|--------:|-----------:|-----------:|
| SUPPORTS |    6494 |       6484 |   0.99846  |
| REFUTES  |    6495 |       6438 |   0.991224 |

Grupos: 12872; maior grupo: 3.
Grafos repetidos entre conjuntos: 0.
Claims duplicadas entre conjuntos: 0.
Entidades textuais compartilhadas entre conjuntos (permitidas): 9144.

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
