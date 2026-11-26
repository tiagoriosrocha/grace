# Comparação de sistemas — evidência fornecida

Dataset: fever. Protocolo: alt5-claim-evidence-v3-local.
Claims duplicadas, grupos de origem e grafos de evidência repetidos ficam no mesmo conjunto.
Entidades podem aparecer em conjuntos diferentes; esta avaliação admite entidades conhecidas.
Divisão interna do export convertido; não equivale ao teste oficial do benchmark.

| Model                    | Accuracy        | Macro-F1        | Macro-F1 95% CI   |    AUC |
|:-------------------------|:----------------|:----------------|:------------------|-------:|
| majority_baseline        | 0.5078 ± 0.0000 | 0.3368 ± 0.0000 | [0.3278, 0.3457]  | 0.5    |
| text_tfidf               | 0.5356 ± 0.0000 | 0.5351 ± 0.0000 | [0.5156, 0.5557]  | 0.5655 |
| graph_pair_transformer   | 0.7134 ± 0.1916 | 0.6593 ± 0.2853 | [0.6493, 0.6696]  | 0.7953 |
| matching_nli             | 0.7916 ± 0.0000 | 0.7916 ± 0.0000 | [0.7749, 0.8083]  | 0.8577 |
| relational_gnn           | 0.8000 ± 0.0063 | 0.7999 ± 0.0064 | [0.7856, 0.8143]  | 0.8716 |
| claim_evidence_attention | 0.7745 ± 0.0068 | 0.7742 ± 0.0068 | [0.7590, 0.7892]  | 0.8641 |
| local_factcheck_pipeline | 0.7933 ± 0.0000 | 0.7931 ± 0.0000 | [0.7770, 0.8100]  | 0.8588 |

Média ± desvio amostral entre sementes. IC95%: bootstrap por grupos completos da média
do Macro-F1 entre sementes fixas. As sementes não são novas observações de teste.
paired_comparisons.csv contém diferenças pareadas com intervalos pontuais, sem correção
por comparações múltiplas; não fornece uma declaração automática de superioridade.
Matrizes de confusão são separadas por semente.

## Cobertura e sobreposição

| evaluation_role   |   label |    n |
|:------------------|--------:|-----:|
| test              |       0 | 1206 |
| test              |       1 | 1169 |
| train             |       0 | 4012 |
| train             |       1 | 4003 |
| validation        |       0 |  689 |
| validation        |       1 |  740 |

|          |   input |   eligible |   coverage |
|:---------|--------:|-----------:|-----------:|
| SUPPORTS |    6485 |       5912 |   0.911642 |
| REFUTES  |    6482 |       5907 |   0.911293 |

Grupos: 10568; maior grupo: 10.
Grafos repetidos entre conjuntos: 0.
Claims duplicadas entre conjuntos: 0.
Entidades textuais compartilhadas entre conjuntos (permitidas): 10900.

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
