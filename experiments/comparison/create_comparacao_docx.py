from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = "/Users/tiagoriosdarocha/Desktop/proposta-tese-alternativa-5/comparacao_arquitetural_sistemas.docx"

BLUE = "17365D"
LIGHT_BLUE = "D9EAF7"
PALE_BLUE = "F4F8FC"
GRAY = "D9D9D9"
BLACK = "000000"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_borders(cell, color=GRAY, size="4"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_row_cant_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    cant_split.set(qn("w:val"), "true")
    tr_pr.append(cant_split)


def set_cell_width(cell, width):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(int(width.inches * 1440)))
    tc_w.set(qn("w:type"), "dxa")


def set_run_font(run, name="Aptos", size=9.0, bold=False, color=BLACK, italic=False):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def clear_paragraph(paragraph):
    for run in list(paragraph.runs):
        paragraph._p.remove(run._r)


def write_cell(cell, text, header=False, center=False, font_size=8.2):
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_margins(cell)
    set_cell_borders(cell)
    p = cell.paragraphs[0]
    clear_paragraph(p)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center or header else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.02
    lines = str(text).split("\n")
    for idx, line in enumerate(lines):
        if idx:
            p.add_run().add_break()
        run = p.add_run(line)
        set_run_font(run, size=8.4 if header else font_size, bold=header, color="FFFFFF" if header else BLACK)
    if header:
        set_cell_shading(cell, BLUE)


def add_table(doc, headers, rows, widths, font_size=8.2):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    header_row = table.rows[0]
    set_repeat_table_header(header_row)
    set_row_cant_split(header_row)
    for i, header in enumerate(headers):
        set_cell_width(header_row.cells[i], widths[i])
        write_cell(header_row.cells[i], header, header=True, center=True)
    for ridx, row in enumerate(rows):
        new_row = table.add_row()
        set_row_cant_split(new_row)
        cells = new_row.cells
        for i, value in enumerate(row):
            set_cell_width(cells[i], widths[i])
            write_cell(cells[i], value, font_size=font_size)
            if ridx % 2 == 1:
                set_cell_shading(cells[i], PALE_BLUE)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.space_before = Pt(9 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    set_run_font(run, size=13 if level == 1 else 10.5, bold=True, color=BLACK)
    return p


def add_body(doc, text, bold_lead=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.08
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        set_run_font(r, size=9.7, bold=True)
        r2 = p.add_run(text[len(bold_lead):])
        set_run_font(r2, size=9.7)
    else:
        r = p.add_run(text)
        set_run_font(r, size=9.7)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Inches(0.22)
    p.paragraph_format.line_spacing = 1.02
    r = p.add_run(text)
    set_run_font(r, size=9.2)
    return p


def configure_styles(doc):
    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    normal.font.size = Pt(9.7)
    for style_name in ("Title", "Heading 1", "Heading 2"):
        style = doc.styles[style_name]
        style.font.name = "Aptos Display" if style_name == "Title" else "Aptos"
        style._element.rPr.rFonts.set(qn("w:ascii"), style.font.name)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), style.font.name)
        style.font.color.rgb = RGBColor.from_string(BLACK)


def add_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Proposta de tese alternativa 5  |  Comparação arquitetural")
    set_run_font(r, size=8, color="666666")


def main():
    doc = Document()
    configure_styles(doc)
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11)
    section.page_height = Inches(8.5)
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)
    add_footer(section)

    title = doc.add_paragraph(style="Title")
    title.paragraph_format.space_after = Pt(4)
    r = title.add_run("Comparação arquitetural dos sistemas")
    set_run_font(r, name="Aptos Display", size=21, bold=True, color=BLACK)

    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(8)
    r = subtitle.add_run("FactKG  |  protocolo alt5 claim evidence v3 local  |  execução principal com local50")
    set_run_font(r, size=9.5, color="555555", italic=True)

    add_body(doc, "Este documento reúne, em uma única matriz, os sete sistemas comparados na alternativa 5. O foco é distinguir o que cada modelo recebe como entrada, como utiliza o grafo da claim e o grafo da evidência, quais modelos pré-treinados ou treináveis emprega, como produz a decisão e quão explicável e rastreável é o resultado.")
    add_body(doc, "Conclusão de leitura: os modelos variam de um baseline sem representação textual até verificadores estruturados com embeddings, NLI, recuperação de evidências, GNN ou atenção. A comparação de desempenho é entre sistemas completos; ela não isola causalmente uma única arquitetura.", bold_lead="Conclusão de leitura:")

    add_heading(doc, "1 Matriz arquitetural direta", 1)
    headers = ["Sistema", "Entrada e papel dos grafos", "Representação e modelos", "Treinamento e decisão", "Fluxo resumido"]
    widths = [Inches(1.32), Inches(2.12), Inches(2.24), Inches(2.18), Inches(2.04)]
    rows = [
        [
            "majority_baseline",
            "Recebe apenas os rótulos do treino.\nGrafo da claim: não usado.\nGrafo da evidência: não usado.",
            "Nenhum encoder, embedding ou modelo neural.",
            "Calcula a classe majoritária do treino.\nPrediz a mesma classe em todos os exemplos.\nNão produz evidência ou score semântico.",
            "rótulos do treino → classe majoritária → decisão fixa"
        ],
        [
            "text_tfidf",
            "Concatena claim textual e evidência verbalizada.\nOs grafos são linearizados como texto; sua topologia não é usada diretamente.",
            "TF-IDF de unigramas e bigramas, até 50.000 features.\nRegressão logística balanceada.",
            "Ajuste somente no treino.\nSaída: P(SUPPORTS).\nLimiar público fixo em 0,5.",
            "claim + evidência textual → TF-IDF → regressão logística → decisão"
        ],
        [
            "graph_pair_transformer",
            "Usa os dois grafos, mas serializados como sequências: claim graph + evidence graph.\nNão executa convolução sobre a topologia.",
            "BERT base uncased ajustado para classificação binária.\nSequência máxima de 512 tokens.",
            "Fine-tuning supervisionado no treino.\nCheckpoint escolhido na validação.\n3 épocas máximas, learning rate 2e-5.\nLimiar 0,5.",
            "grafo da claim serializado + grafo da evidência serializado → BERT fine-tuned → classificação"
        ],
        [
            "matching_nli",
            "Usa triplas factuais do grafo da claim e do grafo da evidência.\nA claim fornece as condições; a evidência fornece candidatos top-k.",
            "BERT base uncased congelado para embeddings de sujeito, relação e objeto.\nNLI MiniLM2 congelado.\nAgregação G19-H-RP.",
            "Top-4 candidatos por condição.\nNLI sobre cada par.\nFeatures hierárquicas → StandardScaler → LogisticRegression balanceada.\nLimiar 0,5.",
            "tripla da claim → top-4 triplas da evidência → NLI + relevância → features → classificação"
        ],
        [
            "relational_gnn",
            "Usa os dois grafos como grafos direcionados.\nNós representam entidades; arestas representam relações e preservam a direção.",
            "Embeddings BERT congelados nos nós e relações.\nRGATConv com 2 camadas, hidden 128, 4 heads e 16 relation bases.\nCabeças de matching, polaridade e consistência.",
            "Treinamento end-to-end dos módulos GNN e cabeça final; BERT permanece congelado.\nCheckpoint selecionado na validação.\nLimiar 0,5.",
            "grafos direcionados → propagação relacional RGAT → matching/polaridade → score global → decisão"
        ],
        [
            "claim_evidence_attention",
            "Usa claim textual como sequência de consulta e nós/arestas do grafo da evidência como tokens.\nA claim graph não é usada como grafo pareado completo.",
            "BERT congelado para tokens.\nProjeção para dimensão 128.\nCross-attention multi-head entre tokens da claim e tokens da evidência.",
            "Modelo X23 treinado com binary cross-entropy.\nAté 4 heads, dropout 0,2, até 20 épocas.\nCheckpoint selecionado na validação.\nLimiar 0,5.",
            "tokens da claim ↔ tokens do grafo da evidência → cross-attention → pooling → MLP → decisão"
        ],
        [
            "local_factcheck_pipeline",
            "Claim graph: decomposto em condições factuais.\nEvidence graph: fonte de triplas diretas e caminhos de dois saltos.\nArestas EntityType não são obrigações factuais.",
            "BERT congelado para similaridade.\nNLI MiniLM2 congelado, max 256 tokens no local.\nFeatures locais por condição.",
            "Top-4 semântico + top-4 alinhado, aliases, inversões e caminhos.\nStandardScaler + LogisticRegression.\nC e limiar escolhidos na validação.",
            "condições da claim → recuperação direta/caminhos → NLI → max por condição → min/média/máximo → decisão"
        ],
    ]
    add_table(doc, headers, rows, widths, font_size=7.7)

    add_heading(doc, "2 O que cada grafo significa", 1)
    graph_headers = ["Elemento", "Grafo da claim", "Grafo da evidência", "Uso nos sistemas"]
    graph_widths = [Inches(1.42), Inches(2.55), Inches(2.55), Inches(3.38)]
    graph_rows = [
        ["Origem", "A proposição que será verificada.", "Fatos fornecidos para verificar a proposição.", "A claim define as condições; a evidência fornece candidatos."],
        ["Tripla", "(sujeito, relação, objeto) afirmada pela claim.", "(sujeito, relação, objeto) observado no material de evidência.", "Cada tripla da claim pode ser comparada a várias triplas da evidência."],
        ["Direção", "sujeito → relação → objeto; representa o sentido da afirmação.", "sujeito → relação → objeto; preserva o sentido do fato observado.", "A inversão automática é evitada nos modelos de grafos."],
        ["Exemplo", "(Obama, born in, Hawaii)", "(Obama, born in, Honolulu) ou (Hawaii, contains, Honolulu)", "A primeira pode apoiar diretamente; a segunda pode compor um caminho; uma relação incompatível pode contradizer."],
        ["Arestas de tipo", "Podem fornecer contexto de entidade, mas não são condições factuais no pipeline local.", "Podem permanecer como contexto estrutural para modelos de grafo.", "A recuperação factual e o NLI concentram-se nas arestas factuais."],
        ["Rastreamento", "Permite registrar qual condição da claim foi analisada.", "Permite registrar qual aresta ou caminho foi recuperado.", "É mais explícito no matching_nli e no local pipeline do que no TF-IDF e no Transformer serializado."],
    ]
    add_table(doc, graph_headers, graph_rows, graph_widths, font_size=8.0)

    add_heading(doc, "3 Explicabilidade e rastreabilidade", 1)
    explain_headers = ["Sistema", "Explicabilidade", "Rastreabilidade da evidência", "Artefatos disponíveis", "Limitações"]
    explain_widths = [Inches(1.35), Inches(2.22), Inches(2.35), Inches(2.18), Inches(1.80)]
    explain_rows = [
        [
            "majority_baseline",
            "Totalmente simples: a explicação é a classe majoritária do treino.",
            "Nenhuma. Não inspeciona claim ou evidência.",
            "Métricas, predições e matriz de confusão.",
            "Não verifica fatos e não oferece justificativa por exemplo."
        ],
        [
            "text_tfidf",
            "Parcial: coeficientes lexicais da regressão podem indicar termos associados a cada classe.",
            "Fraca: sabe quais textos entraram, mas não seleciona uma evidência ou tripla específica.",
            "Modelo TF-IDF/logístico, métricas, predições e probabilidades.",
            "Coeficientes não provam causalidade nem validade factual."
        ],
        [
            "graph_pair_transformer",
            "Limitada: a entrada pode ser inspecionada, mas o logit do Transformer não identifica claramente a aresta decisiva.",
            "Moderada: claim e evidência serializadas são preservadas, mas não há trilha explícita de candidatos.",
            "Checkpoint, histórico de validação, truncation.json, predições e métricas.",
            "Atenção interna não deve ser tratada como prova ou explicação causal."
        ],
        [
            "matching_nli",
            "Boa para diagnóstico: mostra similaridade, relevância, NLI, suporte, conflito e margem por condição.",
            "Boa: permite relacionar condições da claim a triplas candidatas e aos escores NLI.",
            "matching_nli_features.csv, pares/candidatos, métricas, predições e modelos joblib.",
            "No protocolo principal, exact features ficam desativadas; o resultado ainda é um agregador estatístico."
        ],
        [
            "relational_gnn",
            "Moderada: há sinais de matching, polaridade, pesos locais e consistência, mas a decisão passa por várias camadas aprendidas.",
            "Moderada: nós, arestas e relações são preservados; o caminho causal da decisão é difícil de resumir.",
            "Checkpoint GNN, matrizes de confusão, predições e métricas.",
            "Pesos e ativações não equivalem automaticamente a uma justificativa factual."
        ],
        [
            "claim_evidence_attention",
            "Moderada: mapas de atenção permitem inspeção dos tokens claim–evidência.",
            "Moderada: registra tokens/nós/arestas atendidos, mas a atenção é evidência diagnóstica, não prova.",
            "X23_attention, intervenções de remoção, checkpoints, predições e métricas.",
            "Atenção pode ser difusa e não representa necessariamente importância causal."
        ],
        [
            "local_factcheck_pipeline",
            "Mais explícita: mostra condições, candidatos, correspondências, NLI, caminhos, features e limiar final.",
            "Muito boa: cada decisão pode apontar para uma ou mais triplas da evidência, inclusive caminhos de dois saltos.",
            "local_candidates.jsonl, local_decisions.jsonl, local_predictions.csv, claim_features.csv, selection.json, coefficients.json e auditoria HTML.",
            "A recuperação é explicável, mas o classificador final ainda é aprendido e o score não é uma prova lógica calibrada."
        ],
    ]
    add_table(doc, explain_headers, explain_rows, explain_widths, font_size=7.7)

    add_heading(doc, "4 Resultados usados na comparação", 1)
    add_body(doc, "Os seis sistemas da execução principal mantêm os valores da execução article_20261002_155519_046159. O local_factcheck_pipeline foi substituído pelo resultado da execução article_20261002_155519_046159_local50, que usa o mesmo split, as mesmas três sementes e 3.288 exemplos de teste.")
    metric_headers = ["Sistema", "Accuracy", "Macro-F1", "AUC", "Origem"]
    metric_widths = [Inches(3.10), Inches(1.10), Inches(1.10), Inches(1.00), Inches(3.60)]
    metric_rows = [
        ["graph_pair_transformer", "96,39%", "0,9572", "0,9900", "execução principal"],
        ["relational_gnn", "95,28%", "0,9442", "0,9860", "execução principal"],
        ["claim_evidence_attention", "87,85%", "0,8425", "0,9482", "execução principal"],
        ["text_tfidf", "81,36%", "0,7822", "0,8736", "execução principal"],
        ["local_factcheck_pipeline", "79,81%", "0,7712", "0,8725", "article_20261002_155519_046159_local50"],
        ["matching_nli", "77,07%", "0,7490", "0,8677", "execução principal"],
        ["majority_baseline", "70,71%", "0,4142", "0,5000", "execução principal"],
    ]
    add_table(doc, metric_headers, metric_rows, metric_widths, font_size=8.5)

    add_heading(doc, "5 Observações metodológicas", 1)
    add_bullet(doc, "Todos os sistemas usam o mesmo split agrupado e as mesmas classes, mas não recebem necessariamente a mesma representação de entrada.")
    add_bullet(doc, "O BERT de embeddings e o modelo NLI são congelados; o que muda entre os sistemas é o uso desses sinais e o modelo treinável posterior.")
    add_bullet(doc, "A comparação de sistemas completos não identifica isoladamente o efeito de uma arquitetura, porque entrada, recuperação, features e supervisão também mudam.")
    add_bullet(doc, "Os scores de matching e do pipeline local são scores de agregação; não devem ser interpretados automaticamente como probabilidades factuais perfeitamente calibradas.")
    add_bullet(doc, "A execução usa uma divisão interna do export FactKG e não equivale ao teste oficial do benchmark.")
    add_bullet(doc, "A procedência da geração anterior dos grafos precisa ser documentada separadamente; o repositório não certifica que essa geração desconhecia os rótulos.")

    add_heading(doc, "6 Arquivos de referência no projeto", 1)
    add_body(doc, "Implementações centrais: final_pipeline.py, baselines/runner.py, hover_exp/data.py, hover_exp/aggregation.py, hover_exp/model.py, hover_exp/x23.py, factcheck_pipeline/local_alt6.py e factcheck_pipeline/local_alt6_alignment.py.")
    add_body(doc, "Resultados: results/article_20261002_155519_046159/RESULTADOS.md e results/article_20261002_155519_046159_local50/RESULTADOS.md.")

    doc.save(OUT)


if __name__ == "__main__":
    main()
