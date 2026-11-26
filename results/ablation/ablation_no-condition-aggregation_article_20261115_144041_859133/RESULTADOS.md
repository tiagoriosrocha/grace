# Comparação de sistemas — evidência fornecida

Dataset: feverous. Protocolo: alt5-claim-evidence-v3-local.
Claims duplicadas, grupos de origem e grafos de evidência repetidos ficam no mesmo conjunto.
Entidades podem aparecer em conjuntos diferentes; esta avaliação admite entidades conhecidas.
Divisão interna do export convertido; não equivale ao teste oficial do benchmark.

| Model          | Accuracy        | Macro-F1        | Macro-F1 95% CI   |    AUC |
|:---------------|:----------------|:----------------|:------------------|-------:|
| relational_gnn | 0.7945 ± 0.0097 | 0.7940 ± 0.0099 | [0.7806, 0.8075]  | 0.8795 |

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
