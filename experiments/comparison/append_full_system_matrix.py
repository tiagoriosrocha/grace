from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.shared import Inches

from create_comparacao_docx import add_body, add_table, set_run_font


DOCX = Path("/Users/tiagoriosdarocha/Desktop/proposta-tese-alternativa-5/comparacao_arquitetural_sistemas.docx")
TEMP = DOCX.with_suffix(".tmp.docx")


def add_heading_local(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = 9
    p.paragraph_format.space_after = 4
    run = p.add_run(text)
    set_run_font(run, size=13, bold=True, color="000000")
    return p


def main():
    doc = Document(DOCX)
    section = doc.sections[-1]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11)
    section.page_height = Inches(8.5)
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)

    doc.add_page_break()
    add_heading_local(doc, "7 Detalhamento técnico por componente")
    add_body(doc, "A matriz abaixo mantém o mesmo formato comparativo do quadro de referência, mas amplia a cobertura para todos os sete sistemas. Os termos grafo da claim e grafo da evidência referem-se, respectivamente, à proposição que será verificada e ao conjunto de fatos fornecidos para verificá-la.")

    headers = [
        "Componente",
        "majority\nbaseline",
        "text\ntfidf",
        "graph_pair\ntransformer",
        "matching\nnli",
        "relational\ngnn",
        "claim_evidence\nattention",
        "local_factcheck\npipeline",
    ]
    widths = [Inches(1.42)] + [Inches(1.21)] * 7

    rows = [
        [
            "Entrada principal",
            "Rótulos do treino.",
            "Claim textual + evidência verbalizada.",
            "Claim graph e evidence graph serializados.",
            "Triplas factuais da claim + triplas factuais da evidência.",
            "Dois grafos direcionados, com nós, arestas e relações.",
            "Claim textual + tokens de nós/arestas do grafo da evidência.",
            "Claim graph decomposto em condições + evidence graph para recuperação.",
        ],
        [
            "Grafo da claim",
            "Não usado.",
            "Usado somente de forma indireta, após linearização textual.",
            "Serializado como sequência de triplas; topologia não é operada diretamente.",
            "Fonte das condições/triplas a serem comparadas e dos embeddings de sujeito, relação e objeto.",
            "Grafo direcionado contextualizado por propagação relacional; suas arestas factuais formam as condições.",
            "Não é usado como grafo pareado completo; a claim entra principalmente como sequência textual.",
            "Fonte das condições factuais condition_0000, condition_0001 etc.; arestas EntityType não são obrigações factuais.",
        ],
        [
            "Grafo da evidência",
            "Não usado.",
            "Linearizado como texto após a claim.",
            "Serializado como sequência de triplas junto com a claim graph.",
            "Fonte dos candidatos top-k que serão comparados com cada tripla da claim.",
            "Grafo direcionado completo, usado com relações e contexto estrutural, além de comparações locais.",
            "Convertido em tokens de nós e arestas; seleção limitada por similaridade com a claim.",
            "Fonte de triplas diretas, correspondências exatas e caminhos de dois saltos.",
        ],
        [
            "Tripla factual",
            "Não existe no fluxo do modelo.",
            "Vira texto: sujeito, relação e objeto aparecem como tokens.",
            "Vira uma sequência textual serializada.",
            "Representada explicitamente como sujeito, relação e objeto para matching e NLI.",
            "Representada por nós, aresta direcionada, tipo de relação e embedding da relação.",
            "Arestas do evidence graph viram tokens ordenados [sujeito, relação, objeto].",
            "Cada tripla factual da claim vira uma condição; cada evidência pode ser uma tripla ou um caminho.",
        ],
        [
            "Embeddings",
            "Nenhum.",
            "TF-IDF esparso de unigramas e bigramas; até 50.000 features.",
            "Tokens BERT dentro de uma sequência de até 512 tokens.",
            "Embeddings BERT de 768 dimensões para sujeito, relação e objeto.",
            "Embeddings BERT congelados para nós e relações, projetados para hidden 128.",
            "Embeddings BERT token-level; projeção de claim e grafo para dimensão 128.",
            "Embeddings BERT para similaridade semântica de sujeito, relação e objeto.",
        ],
        [
            "Encoder BERT",
            "Não usado.",
            "Não usado.",
            "BERT base uncased fine-tuned para classificação binária.",
            "BERT base uncased congelado; mean pooling e normalização L2.",
            "BERT base uncased congelado; somente as camadas GNN/cabeças são treinadas.",
            "BERT base uncased congelado e cacheado por claim; atenção e cabeça são treinadas.",
            "BERT base uncased congelado; usado na recuperação e em features semânticas.",
        ],
        [
            "NLI",
            "Não usado.",
            "Não usado.",
            "Não usado como módulo separado.",
            "cross-encoder/nli-MiniLM2-L6-H768 congelado; pares top-4; máximo 128 tokens.",
            "NLI congelado no vetor de sinais dos candidatos; máximo 128 tokens no bundle comum.",
            "Não usado na variante selecionada; as features NLI de referência são zeradas explicitamente.",
            "cross-encoder/nli-MiniLM2-L6-H768 congelado; máximo 256 tokens no pipeline local.",
        ],
        [
            "Recuperação e matching",
            "Não há recuperação.",
            "Não há recuperação; o classificador usa todo o texto concatenado.",
            "Não há recuperação explícita; a sequência completa é entregue ao Transformer, sujeita a truncamento.",
            "Para cada tripla da claim, seleciona top-4 evidências por média das similaridades de sujeito, relação e objeto.",
            "Usa candidatos top-k do bundle comum e, simultaneamente, mantém o contexto dos dois grafos para a GNN.",
            "Seleciona nós e arestas do evidence graph por similaridade com a representação média da claim, respeitando o limite de tokens.",
            "Une top-4 semântico e top-4 alinhado; a recuperação é independente dos rótulos.",
        ],
        [
            "Normalização de relações",
            "Não se aplica.",
            "A relação é apenas texto submetido ao tokenizador TF-IDF.",
            "A relação aparece no texto serializado e no tokenizador BERT.",
            "Usa a relação na forma original e vocabulário categórico limitado; não faz aliases/inversões explícitos nesta variante.",
            "Relações são categorias; até 256 relações frequentes no treino e demais como UNK, preservando embedding textual.",
            "Relação aparece no token da aresta; tipos de nó são categorizados, até 64 tipos frequentes.",
            "Usa aliases e relações inversas, por exemplo employer → works at e employs → works at.",
        ],
        [
            "Correspondência exata",
            "Não há correspondência factual explícita.",
            "Pode ocorrer apenas como coincidência lexical no TF-IDF.",
            "Pode influenciar tokens, mas não existe feature exata separada.",
            "Desativada no protocolo principal: use_exact_features=false.",
            "Desativada como feature explícita no protocolo principal; a rede aprende a partir dos sinais disponíveis.",
            "Não há regra de igualdade exata; a interação é aprendida pela atenção.",
            "Usada explicitamente para sujeito, relação, objeto e combinações de alinhamento nas features.",
        ],
        [
            "Caminhos de dois saltos",
            "Não procura.",
            "Não representa caminhos explicitamente.",
            "Não enumera caminhos; apenas serializa a sequência disponível.",
            "Não procura caminhos explicitamente.",
            "Pode propagar informação por duas camadas RGAT, mas não enumera nem registra um caminho como justificativa explícita.",
            "Arestas incidentes podem interagir pela atenção, mas não há busca explícita de caminhos.",
            "Procura explicitamente caminhos de dois saltos e os apresenta ao NLI como evidência composta.",
        ],
        [
            "Agregação",
            "Classe majoritária global.",
            "Agregação implícita no vetor TF-IDF e no logit linear.",
            "Pooling/representação interna do Transformer e cabeça de classificação.",
            "Features hierárquicas de matching, relevância e NLI por condição e globalmente.",
            "Matching, polaridade, atenção entre condições, alternativa nula e penalidade de consistência.",
            "Cross-attention claim–evidence, pooling médio/máximo e cabeça MLP.",
            "Features locais por candidato → máximo por condição → mínimo/média/máximo entre condições.",
        ],
        [
            "Modelo final treinável",
            "Nenhum; somente contagem de classes no treino.",
            "TfidfVectorizer + LogisticRegression balanceada.",
            "BERTForSequenceClassification com cabeça binária.",
            "StandardScaler + LogisticRegression balanceada; max_iter 1000.",
            "RGATConv relacional + camadas MLP de matching, polaridade e cobertura; hidden 128, 2 camadas, 4 heads.",
            "Modelo X23: projeções, MultiheadAttention e MLP; dimensão 128, 4 heads, dropout 0,2.",
            "StandardScaler + LogisticRegression balanceada; max_iter 2000, solver lbfgs.",
        ],
        [
            "Ajuste de C",
            "Não se aplica.",
            "Configuração fixa/default da regressão logística; não há seleção na validação.",
            "Não se aplica como C; learning rate e épocas são configurados.",
            "Configuração fixa/default; não há busca de C na validação.",
            "Não se aplica; hiperparâmetros neurais são configurados.",
            "Não se aplica; learning rate, épocas e dropout são configurados.",
            "Selecionado na validação entre 0,01, 0,1, 1 e 10.",
        ],
        [
            "Ajuste do limiar",
            "Não há limiar; a classe é fixa.",
            "0,5.",
            "0,5 sobre a probabilidade da classe positiva.",
            "0,5.",
            "0,5.",
            "0,5.",
            "Selecionado na validação entre 0,10 e 0,90, maximizando Macro-F1.",
        ],
        [
            "Treinamento BERT/NLI",
            "Não existe.",
            "Não existe modelo pré-treinado.",
            "BERT é fine-tuned no treino; não usa NLI separado.",
            "BERT e NLI congelados; somente o agregador logístico é ajustado.",
            "BERT e NLI congelados; RGAT e cabeças finais são ajustados.",
            "BERT congelado; projeções, atenção e MLP são ajustadas.",
            "BERT e NLI congelados; somente o agregador logístico é ajustado.",
        ],
        [
            "Explicabilidade",
            "Muito alta, mas trivial: explica pela classe majoritária.",
            "Parcial: coeficientes lexicais podem indicar termos associados à classe.",
            "Limitada: a sequência pode ser inspecionada, mas o logit não aponta claramente a aresta decisiva.",
            "Boa para diagnóstico: similaridade, relevância, NLI, suporte, conflito e margem por condição.",
            "Moderada: há sinais e pesos aprendidos, mas a decisão passa por várias camadas.",
            "Moderada: mapas de atenção permitem inspecionar tokens claim–evidência.",
            "Mais explícita: condições, candidatos, correspondências, NLI, caminhos, features e limiar ficam registrados.",
        ],
        [
            "Rastreabilidade",
            "Não rastreia evidências.",
            "Rastreia os textos de entrada, mas não uma tripla decisiva.",
            "Rastreia sequências, checkpoint, truncamento e predições; não candidatos individualizados.",
            "Rastreia features e sinais de matching/NLI; menos direto que o pipeline local para reconstruir uma decisão.",
            "Rastreia nós, arestas, relações, checkpoint e predições; a cadeia causal interna é complexa.",
            "Rastreia tokens, nós/arestas atendidos, atenção e intervenções de remoção.",
            "Rastreia cada condição, candidato, aresta, caminho, score NLI, decisão e limiar final.",
        ],
        [
            "Artefatos principais",
            "metrics.csv, predictions.csv e matrizes de confusão.",
            "text_tfidf_<seed>.joblib, métricas e predições.",
            "Checkpoint BERT, history.csv, truncation.json, métricas e predições.",
            "matching_nli_features.csv, matching_nli_<seed>.joblib, métricas e predições.",
            "Checkpoint GNN, métricas, predições e matrizes de confusão.",
            "X23_attention, X23_b_interventions, checkpoints e métricas.",
            "local_candidates.jsonl, local_decisions.jsonl, local_predictions.csv, claim_features.csv, selection.json, coefficients.json e AUDIT_LOCAL.html.",
        ],
        [
            "Natureza do score",
            "Classe constante; não é score factual.",
            "Score/probabilidade do classificador lexical.",
            "Score global do Transformer sobre a sequência serializada.",
            "Score global baseado em matching, relevância e NLI agregados.",
            "Logit aprendido a partir de propagação relacional e sinais locais.",
            "Logit aprendido a partir de interações de atenção claim–evidência.",
            "Score global de agregação de evidências; não é prova lógica calibrada.",
        ],
    ]

    add_table(doc, headers, rows, widths, font_size=6.25)
    add_body(doc, "Nota: a matriz foi inserida ao final do arquivo existente. O conteúdo anterior, incluindo edições manuais já presentes no documento, foi preservado.")

    doc.save(TEMP)
    TEMP.replace(DOCX)


if __name__ == "__main__":
    main()
