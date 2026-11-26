from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml.ns import qn
from docx.shared import Inches

from create_comparacao_docx import add_body, add_heading, add_table, configure_styles


TARGET = Path("/Users/tiagoriosdarocha/Desktop/proposta-tese-alternativa-5/comparacao_arquitetural_sistemas.docx")
TEMP = TARGET.with_suffix(".overview.tmp.docx")


def add_page_break(doc):
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)
    return p


def add_reference(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.15)
    p.paragraph_format.first_line_indent = Inches(-0.15)
    p.paragraph_format.space_after = Inches(0.035)
    run = p.add_run(text)
    run.font.name = "Aptos"
    run.font.size = run.font.size or None
    return p


def build_fragment():
    fragment = Document()
    configure_styles(fragment)
    section = fragment.sections[0]
    section.orientation = 1
    section.page_width = Inches(11)
    section.page_height = Inches(8.5)
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)

    add_page_break(fragment)
    add_heading(fragment, "Visão geral para orientação", 1)
    add_body(fragment, "Esta seção adapta o documento Visão geral dos experimentos FactKG e alinhamento de grafos para situar a comparação arquitetural da alternativa 5. Ela resume a motivação científica, define a unidade de análise, posiciona a proposta na literatura e registra as referências usadas para a discussão com a orientação.")

    add_heading(fragment, "Resumo", 1)
    add_body(fragment, "A linha de pesquisa investiga verificação factual com dois artefatos estruturados: um grafo da claim, que representa o que a afirmação diz, e um grafo da evidência, que contém fatos capazes de sustentá-la ou refutá-la. O foco metodológico é verificar se preservar cada condição factual da claim durante a recuperação, o alinhamento e a agregação melhora a decisão global SUPPORTS ou REFUTES.")
    add_body(fragment, "As alternativas anteriores sugerem um padrão promissor: em implementações relacionadas, variantes que mantêm a identidade das condições até a decisão superaram referências de pooling ou de agregação global. O melhor resultado disponível nessa linha é H2 da alternativa 3, com Macro-F1 médio de 0,9719 em um teste interno. Esse resultado não é diretamente comparável ao resultado da alternativa 5, porque os protocolos, subconjuntos e famílias de modelos diferem.")
    add_body(fragment, "A alternativa 5 organiza uma comparação mais ampla entre sete sistemas: baseline majoritário, TF-IDF textual, Transformer sobre grafos serializados, matching com NLI, GNN relacional, atenção claim–evidência e pipeline local de fact-checking. Na tabela de resultados do documento, o melhor desempenho foi do graph_pair_transformer, seguido da relational_gnn. O local_factcheck_pipeline foi substituído pelo resultado da execução local50, mantendo o mesmo split e as mesmas três sementes da execução principal.")
    add_body(fragment, "A conclusão científica ainda deve ser tratada como hipótese experimental, não como prova de superioridade arquitetural. Para uma afirmação causal, será necessário manter fixos exemplos, evidência, split, recuperação, orçamento e regra de seleção, comparando texto forte, grafo linearizado e modelos estruturados no mesmo teste.", bold_lead="A conclusão científica ainda deve ser tratada como hipótese experimental, não como prova de superioridade arquitetural.")

    summary_headers = ["Linha", "Pergunta ou papel", "Resultado e escopo"]
    summary_rows = [
        ["Baselines", "Quanto do rótulo pode ser previsto pela claim, pelo texto e por sinais simples?", "TextTFIDF atingiu Macro-F1 0,8748, mas em teste próprio de 7.013 itens; não é comparação direta com a alternativa 5."],
        ["Alternativa 0", "Pooling global, Coverage ou alinhamento aprendido?", "G1 Coverage atingiu Macro-F1 0,9703 em teste interno de 6.641 itens."],
        ["Alternativa 2", "Cobertura por condição ajuda além do pooling global?", "Coverage atingiu Macro-F1 0,9712 no mesmo conjunto elegível das alternativas seguintes."],
        ["Alternativa 3", "O que acrescentam RGAT, NLI, cobertura e atenção?", "H2 atingiu Macro-F1 0,9719; H1 atingiu 0,9663 e H0 0,9614, no protocolo da alternativa 3."],
        ["Alternativa 4", "Resumo por condição e relevância × polaridade ajudam em um modelo linear?", "G19-H-RP atingiu Macro-F1 0,7465; sem GNN e com ajuste único por variante."],
        ["Alternativa 5", "Como comparar texto, grafos, NLI, GNN, atenção e pipeline local?", "Na tabela atual: graph_pair_transformer 0,9572; relational_gnn 0,9442; local50 0,7712. A execução principal usa 3.288 exemplos de teste."],
    ]
    add_table(fragment, summary_headers, summary_rows, [Inches(1.35), Inches(3.05), Inches(5.50)], font_size=7.9)
    add_body(fragment, "Os números acima são descritivos. Eles ajudam a reconstruir a evolução da investigação, mas não formam um ranking conjunto, porque variam elegibilidade, divisão, grafos materializados, recuperação, encoder, treinamento e orçamento.")

    add_heading(fragment, "1 Verificação factual e o FactKG", 1)
    add_body(fragment, "Verificação factual é a tarefa de decidir se uma afirmação é sustentada ou refutada por uma fonte de evidência. Neste estudo, a entrada é uma frase em linguagem natural acompanhada por evidência estruturada em um grafo de conhecimento; a saída supervisionada é SUPPORTS ou REFUTES. O desafio envolve reconhecer entidades e relações escritas de formas diferentes, verificar cadeias de relações e evitar interpretar ausência de evidência como prova automática de falsidade.")
    add_body(fragment, "O FactKG original reúne cerca de 108 mil afirmações e cinco formas de raciocínio — one-hop, conjunction, existence, multi-hop e negation — com redações formais e coloquiais [L1]. Os experimentos desta proposta usam uma exportação local de 35.145 registros convertidos em pares de grafos, e não a avaliação oficial integral. A auditoria da exportação identificou 33.210 exemplos elegíveis no conjunto completo; a execução principal da alternativa 5 usou uma amostra determinística de 50%, com 17.601 registros amostrados, 16.612 elegíveis e 3.288 exemplos no teste.")
    add_body(fragment, "Na conversão local, o grafo da claim é extraído da afirmação e o grafo da evidência é materializado a partir do knowledge graph. A unidade de supervisão continua sendo o item completo: o FactKG fornece um rótulo global para a claim. As correspondências entre triplas da claim e triplas da evidência não possuem, nessas execuções, uma anotação local independente de verdade.")
    add_body(fragment, "Uma tripla da claim representa uma condição a verificar, por exemplo (Obama, born in, Hawaii). Uma tripla da evidência representa um fato candidato, por exemplo (Obama, born in, Honolulu). Uma evidência também pode ser uma cadeia de duas relações, como (3Arena, location, Dublin) seguida de (Dublin, subdivisionName, Leinster), quando a claim afirma que 3Arena está localizada em uma cidade de Leinster. Os nós e arestas de metadados ou de tipo podem fornecer contexto estrutural, mas não devem ser confundidos com condições factuais da decisão local.")
    add_body(fragment, "A sanitização remove tipos de nó de metadados como textsummary, textdocument e documentchunk. As arestas são direcionadas; a direção sujeito → relação → objeto é preservada. No pipeline local, arestas EntityType são excluídas da lista de condições factuais, enquanto modelos de grafo podem manter parte do contexto estrutural ao construir suas representações.")

    add_heading(fragment, "2 Pergunta científica e unidade de análise", 1)
    add_body(fragment, "A pergunta científica é: em verificação factual baseada em grafos, preservar explicitamente as condições da claim e alinhar cada condição a evidências candidatas melhora a classificação SUPPORTS ou REFUTES em comparação com agregações globais, texto linearizado e modelos sem cobertura explícita por condição?")
    add_body(fragment, "Formalmente, cada exemplo é um par de grafos dirigidos e um rótulo global: x = (G_C, G_E, y). G_C é o grafo da claim; G_E é o grafo da evidência; y ∈ {SUPPORTS, REFUTES}. Uma tripla sujeito–relação–objeto é a unidade local de obrigação em G_C. O sistema recupera uma ou mais triplas ou caminhos candidatos em G_E, compara os pares com similaridade e, em alguns modelos, aplica NLI. A decisão final pode usar pooling global, vetores locais, atenção, estatísticas hierárquicas ou features supervisionadas.")
    add_body(fragment, "A hipótese operacional é específica: preservar a identidade de cada condição, admitir uma alternativa de não correspondência e combinar relevância com polaridade pode melhorar a decisão. Isso não equivale a afirmar que qualquer GNN, atenção ou par de grafos produz ganho. Também não equivale a tratar pesos de atenção, escores NLI ou alinhamentos recuperados como rótulos locais verdadeiros, pois o dataset fornece apenas o rótulo global da claim.")
    add_body(fragment, "Na alternativa 5, a unidade de avaliação é a claim completa no split de teste. A unidade de análise interna varia por sistema: o text_tfidf opera sobre texto concatenado; o graph_pair_transformer opera sobre uma sequência serializada; matching_nli opera sobre pares de triplas; relational_gnn opera sobre dois grafos direcionados; claim_evidence_attention opera sobre tokens da claim e do grafo de evidência; e o local_factcheck_pipeline opera sobre condições, candidatos e caminhos antes da agregação global.")
    add_body(fragment, "A comparação adequada exige controlar a coorte de exemplos, a evidência, a divisão, os candidatos, os encoders, o treinamento e a regra de seleção. Sem esse controle, diferenças de desempenho descrevem sistemas completos, mas não identificam causalmente a contribuição de uma única arquitetura.")

    add_heading(fragment, "3 Literatura e posição da proposta", 1)
    add_body(fragment, "FactKG [L1] é a referência de tarefa e de benchmark, mas os resultados publicados usam protocolos próprios e não devem ser comparados diretamente aos números internos desta proposta. Opsahl [L4] reporta 93,49% de acurácia com BERT sobre subgrafo linearizado e 78,08% com QA-GNN em sua configuração; esse contraste reforça a necessidade de um baseline textual forte no mesmo teste antes de atribuir ganho à estrutura de grafo.")
    add_body(fragment, "SimGRAG [L7] e ClaimPKG [L8] usam estruturas derivadas da consulta ou da claim para orientar a recuperação de subgrafos. PGR [L9] explora raciocínio programático sobre o KG. Upadhyay, Erickson e McGuinness [L11] discutem a diferença entre grounding em evidência e memorização semântica. Portanto, a mera existência de G_C e G_E, ou a mera recuperação de um subgrafo, não estabelece novidade por si só.")
    add_body(fragment, "A posição da alternativa 5 é experimental e comparativa: colocar no mesmo protocolo um baseline lexical, um Transformer sobre grafos serializados, um matching com NLI, uma GNN relacional, atenção claim–evidência e um pipeline local rastreável. A contribuição potencial está em avaliar, de forma auditável, se a cobertura por condição e o alinhamento entre grafos agregam valor quando comparados a texto e a representações estruturadas sob uma coorte comum.")
    add_body(fragment, "A afirmação central ainda requer um experimento confirmatório. O desenho mais informativo é manter os mesmos IDs e a mesma evidência para claim-only, texto com evidência, grafo linearizado e modelos estruturados; usar a mesma recuperação quando a ablação exigir; separar desenvolvimento e confirmação; e medir também cobertura de candidatos, erros por tipo de raciocínio, fidelidade das explicações, calibração, abstenção e custo total.")

    add_heading(fragment, "Referências", 1)
    refs = [
        "[L1] Kim et al. FactKG Fact Verification via Reasoning on Knowledge Graphs. ACL 2023. https://aclanthology.org/2023.acl-long.895/",
        "[L2] Kim et al. KG-GPT A General Framework for Reasoning on Knowledge Graphs Using Large Language Models. Findings EMNLP 2023. https://aclanthology.org/2023.findings-emnlp.631/",
        "[L3] Momii, Takiguchi e Ariki. Rule-based Fact Verification Utilizing Knowledge Graphs. IJCKG 2023.",
        "[L4] Opsahl. Fact or Fiction Improving Fact Verification with Knowledge Graphs through Simplified Subgraph Retrievals. 2024.",
        "[L5] Gautam et al. FactGenius Combining Zero-Shot Prompting and Fuzzy Relation Mining to Improve Fact Verification with Knowledge Graphs. 2024.",
        "[L6] Hou et al. KG-EGV A Framework for Question Answering with Integrated Knowledge Graphs and Large Language Models. Electronics 13, 4835, 2024. https://doi.org/10.3390/electronics13234835",
        "[L7] Cai et al. SimGRAG Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation. Findings ACL 2025. https://aclanthology.org/2025.findings-acl.163/",
        "[L8] Pham et al. ClaimPKG Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM. Findings ACL 2025. https://aclanthology.org/2025.findings-acl.274/",
        "[L9] Hao e Wu. Fact Verification on Knowledge Graph via Programmatic Graph Reasoning. Findings EMNLP 2025.",
        "[L10] Liu et al. Rethinking Soft Compression in Retrieval-Augmented Generation A Query-Conditioned Selector Perspective. WWW 2026.",
        "[L11] Upadhyay, Erickson e McGuinness. Evidence Grounding vs Memorization Why Neural Semantics Matter for Knowledge Graph Fact Verification. FEVER 2026. https://aclanthology.org/2026.fever-1.3/",
    ]
    for ref in refs:
        add_reference(fragment, ref)

    add_body(fragment, "As referências internas dos experimentos e os arquivos de implementação permanecem listados nas seções técnicas já existentes do documento.")
    add_page_break(fragment)
    return fragment


def insert_before(anchor, elements):
    for element in elements:
        anchor.addprevious(element)


def main():
    target = Document(TARGET)

    # Remove only the overview block inserted by a previous attempt. The
    # existing introduction and the architectural matrix are kept intact.
    intro_end = None
    anchor = None
    for paragraph in target.paragraphs:
        if paragraph.text.strip().startswith("Conclusão de leitura:"):
            intro_end = paragraph._p
        if paragraph.text.strip() == "1 Matriz arquitetural direta":
            anchor = paragraph._p
            break
    if intro_end is None or anchor is None:
        raise RuntimeError("Não encontrei os limites da introdução existente.")
    node = intro_end.getnext()
    while node is not None and node is not anchor:
        next_node = node.getnext()
        node.getparent().remove(node)
        node = next_node

    fragment = build_fragment()

    anchor = None
    for paragraph in target.paragraphs:
        if paragraph.text.strip() == "1 Matriz arquitetural direta":
            anchor = paragraph._p
            break
    if anchor is None:
        raise RuntimeError("Não encontrei o início da matriz arquitetural existente.")

    body_elements = []
    for child in list(fragment._body._element):
        if child.tag == qn("w:sectPr"):
            continue
        body_elements.append(deepcopy(child))
    insert_before(anchor, body_elements)
    target.save(TEMP)
    TEMP.replace(TARGET)


if __name__ == "__main__":
    main()
