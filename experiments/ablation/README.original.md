# Alternativa 5 — verificação com evidência fornecida

Uma configuração por sistema, com os mesmos exemplos, split e sementes [7, 21, 42].

| Sistema | Entrada e treinamento |
|---|---|
| majority_baseline | Classe majoritária do treino |
| text_tfidf | Claim textual + triplas verbalizadas; TF-IDF e regressão logística |
| graph_pair_transformer | Dois grafos serializados; BERT ajustado no treino |
| matching_nli | Matching e NLI congelados; agregação por relevância e regressão logística |
| relational_gnn | Grafos direcionados, relações, matching e NLI; agregação aprendida |
| claim_evidence_attention | Tokens da claim e nós/arestas da evidência; atenção cruzada |
| local_factcheck_pipeline | Mesmo pipeline local do alt6: retrieval top-k, NLI congelado e agregador supervisionado |

São configurações escolhidas para uma comparação fixa; não há busca de variantes no
teste. Os sistemas têm entradas e supervisão prévia diferentes. Os resultados comparam
sistemas completos e não isolam causalmente o efeito da arquitetura.

## Protocolo

Entidades repetidas são permitidas entre treino, validação e teste. Compartilhar uma
entidade não implica compartilhar o rótulo. A tarefa admite verificar novas alegações
sobre entidades conhecidas; não é avaliação zero-shot de entidades.

Claims textuais duplicadas (normalizadas), grupos de origem informados e grafos de
evidência repetidos ficam no mesmo grupo, inclusive transitivamente. A assinatura da
evidência usa o conjunto de triplas sanitizadas e normalizadas. Entidades isoladas não
unem exemplos. A ordem dos arquivos/registros não altera o split.

O split usa uma semente fixa e GroupShuffleSplit: alvo 20% teste, 15% do restante para
validação (aproximadamente 68/12/20% dos exemplos). Grupos inteiros são preservados.
Antes do treino, o código exige ambas as classes em cada conjunto e verifica desvios
máximos de 5 pontos percentuais no tamanho e 10 pontos na proporção da classe positiva.
Uma divisão inadequada interrompe a execução; não há busca de sementes por desempenho.

Encoder e NLI são congelados. TF-IDF, escalas, categorias de relações/tipos e modelos
aprendidos são ajustados somente no treino. Checkpoints são escolhidos na validação;
o limiar de classificação é fixo em 0,5. Features explícitas de igualdade são zeradas.
O diagnóstico exact_match_diagnostic não entra na tabela dos seis sistemas.

Os dados são um export convertido; o split é interno e não equivale ao teste oficial.
A auditoria registra exclusões e cobertura por classe. A geração anterior dos grafos
precisa ter sua procedência documentada; este repositório não certifica que essa geração
desconhecia os rótulos. Resultados anteriores usados para escolher os modelos constituem
exploração prévia e devem ser declarados no artigo.

## Execução na RTX 5080 (Linux x86_64, Python 3.10–3.12)

```bash
./RODAR_5080.sh --dataset factkg
```

Na primeira execução, o script cria .venv, instala PyTorch 2.7.1 CUDA 12.8 e dependências,
verifica a GPU, executa testes e smoke, e então inicia o experimento solicitado.
É necessária internet para a primeira instalação e o download dos modelos.

Com o ambiente já instalado, para verificar só os dados e o split:

```bash
.venv/bin/python run.py --dataset factkg --audit-only --device cpu
```

Para verificar todos os caminhos de treino e relatório sem baixar pesos:

```bash
.venv/bin/python run.py --smoke --device cpu
```

O smoke usa HoVer sintético, embeddings hash, NLI zerado e BERT pequeno aleatório,
com uma época. Os arquivos se chamam smoke_table; não são resultados científicos.
O instalador RODAR_5080.sh exige CUDA mesmo quando se pede um smoke CPU; para CPU
em outro ambiente, instale dependências compatíveis e use run.py diretamente.

O dataset é selecionado pelo nome e os arquivos particionados são lidos da pasta
correspondente em `data/`: `factkg`, `fever`, `hover`, `feverous` ou `scifact`.
Cada pasta pode conter um ou vários arquivos `.jsonl` com o padrão `part_N`; eles
são processados na ordem numérica das partes. Por exemplo:

```bash
.venv/bin/python run.py --dataset fever --audit-only --device cpu
.venv/bin/python run.py --dataset hover --audit-only --device cpu
.venv/bin/python run.py --dataset feverous --audit-only --device cpu
.venv/bin/python run.py --dataset scifact --audit-only --device cpu
```

Os rótulos originais são aceitos por dataset e convertidos para a mesma classe
binária interna: `SUPPORTS/REFUTES` (FactKG, FEVER e FEVEROUS),
`SUPPORTED/NOT_SUPPORTED` (HoVer) e `SUPPORT/CONTRADICT` (SciFact).

Para executar apenas uma fração determinística do export, passe `--fraction` (ou
`--dataset-fraction`). A amostragem usa somente o hash estável do `id`, ocorre
antes do split agrupado e não consulta rótulos para selecionar exemplos:

```bash
./RODAR_5080.sh --dataset factkg --fraction 0.10
./RODAR_5080.sh --dataset hover --fraction 0.25
```

O valor `1.0` é o padrão. A auditoria registra `sample_fraction` e
`sampled_input`; cada execução continua gravando em uma pasta nova. Frações
muito pequenas podem não conter as duas classes ou grupos suficientes para o
split — nesse caso o protocolo interrompe explicitamente, sem inventar rótulos.

É possível executar somente um sistema ou uma seleção de sistemas. O sistema local
usa o mesmo protocolo do projeto alternativa-6: embeddings e NLI congelados, recuperação
sem rótulos e regressão logística ajustada no treino, com C e limiar escolhidos na
validação. Sua decisão, auditoria e métricas usam o mesmo escore:

```bash
./RODAR_5080.sh --dataset factkg --systems local_factcheck_pipeline
./RODAR_5080.sh --dataset factkg --systems text_tfidf,matching_nli
./RODAR_5080.sh --dataset factkg --system text_tfidf --system matching_nli
```

As opções são `majority_baseline`, `text_tfidf`, `graph_pair_transformer`,
`matching_nli`, `relational_gnn`, `claim_evidence_attention` e
`local_factcheck_pipeline`. Sem `--systems` ou `--system`, todos são executados.
Uma execução isolada ainda prepara as representações necessárias ao sistema escolhido,
mas não treina os demais modelos.

O protocolo atual do `relational_gnn` não usa o termo de discordância. Para testar
as ablações, use `--ablation full`, `--ablation no-null`,
`--ablation no-relational-context`, `--ablation no-nli` ou
`--ablation no-condition-aggregation`. Todas as variantes mantêm o disagreement
desativado; cada execução usa uma pasta de saída nova dentro de `results/`.

## Resultados e retomada

Cada execução usa uma pasta nova em results/. --output aceita somente uma pasta ainda
inexistente. Reusar diretórios/checkpoints de outro split é bloqueado. Esta versão não
oferece retomada de treinos interrompidos; uma nova execução inicia do começo.

Os caches do sistema `claim_evidence_attention` são gravados por substituição atômica e
validados na leitura. Se `claim_tokens` ou `x23_graphs.pt` estiverem truncados,
corrompidos ou pertencerem a outro split, o cache é descartado e reconstruído; os demais
artefatos da execução não são apagados.

- RESULTADOS.md: tabela e escopo experimental.
- article_table.csv/.tex e summary.csv: média, desvio amostral e IC95%.
- paired_comparisons.csv: diferenças pareadas de Macro-F1 com intervalos pontuais.
- metrics.csv e predictions.csv: métricas e predições por semente.
- local_decisions_<seed>.jsonl e AUDIT_LOCAL.html: trilha auditável do sétimo sistema.
- confusion_matrices.json: uma matriz por modelo e semente.
- audit.json, excluded.csv e splits.csv: cobertura, exclusões e divisão usada.
- manifest.json: estado, configuração, hashes dos dados/split/código e versões efetivas.
- models/ e arquivos .joblib: modelos ajustados; histórico e truncamento do Transformer.

IC95% usa bootstrap de grupos completos e calcula a média do Macro-F1 entre as sementes
fixas em cada réplica. Todos os sistemas usam os mesmos grupos sorteados. Isso estima
incerteza entre grupos do teste, não toda a variabilidade de novas execuções/splits.
Intervalos das comparações são pontuais, sem correção de multiplicidade.

O resultado é completo apenas quando manifest.json tem status=complete. Auditorias
isoladas têm status=audit_complete; métricas parciais não são tabelas finais.

## Reexecutar após a correção

Treine novamente os seis sistemas com o novo split e diretório novo. Não combine métricas
antigas com esta execução. Arquivos de pesos pré-treinados no cache Hugging Face podem
ser reutilizados; os artefatos ajustados nos exemplos são reconstruídos.
