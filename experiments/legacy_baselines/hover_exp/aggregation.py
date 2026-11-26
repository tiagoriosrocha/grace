"""Agregações auditáveis; relevância nunca recebe os escores NLI."""
import numpy as np

VARIANTS = ('G19', 'G19-H', 'G19-H-RP')


def signals_array(pair, smoke=False):
    s = pair['signals'].detach().cpu().numpy().astype(float)
    expected = (*pair['candidates'].shape, 9)
    if s.shape != expected or s.shape[0] == 0 or not np.isfinite(s).all():
        raise ValueError(f"Sinais inválidos em {pair.get('id')}: {s.shape}; esperado {expected}")
    if np.any(np.abs(s[..., :3]) > 1.0001):
        raise ValueError('Similaridade fora de [-1,1]')
    if np.any((s[..., 3:6] != 0) & (s[..., 3:6] != 1)):
        raise ValueError('Correspondências exatas devem ser binárias')
    nli = s[..., 6:9]
    if np.any(nli < 0) or np.any(nli > 1):
        raise ValueError('NLI fora de [0,1]')
    if not smoke and not np.allclose(nli.sum(-1), 1, atol=1e-5):
        raise ValueError('NLI ausente/não normalizado: zeros só permitidos no smoke')
    return s


def relevance(s):
    """Heurística fixa [0,1], não probabilidade calibrada.

    A relação deve ser semelhante e pelo menos um endpoint deve ancorar o
    fato. Não exige objeto idêntico: uma divergência pode ser contraevidência.
    Endpoints são comparados na posição original, nunca em sentido inverso.
    """
    cos = np.clip(s[..., :3], 0, 1)
    return cos[..., 1] * np.maximum(cos[..., 0], cos[..., 2])


def legacy(s):
    """As mesmas 29 features do G19 da alternativa 3, na mesma ordem.

    Apenas o caso vazio passa a ter estatísticas finitas explícitas.
    """
    result = {}
    for name, block in [('similarity', s[..., :3]), ('exact', s[..., 3:6]), ('nli_all', s[..., 6:9])]:
        x = block.ravel() if block.size else np.zeros(1)
        for stat, value in [('mean', x.mean()), ('min', x.min()), ('max', x.max()), ('fraction_above_half', (x > .5).mean())]:
            result[f'{name}_{stat}'] = float(value)
    c, e, n = (s[..., i] for i in (6, 7, 8))
    for name, block in [('entailment', e), ('contradiction', c), ('neutral', n), ('margin', e-c)]:
        x = block.ravel() if block.size else np.zeros(1)
        for stat, value in [('mean', x.mean()), ('min', x.min()), ('max', x.max())]:
            result[f'{name}_{stat}'] = float(value)
    result.update(entailment_dominant=float((e > np.maximum(c, n)).mean()) if e.size else 0.,
                  contradiction_dominant=float((c > np.maximum(e, n)).mean()) if c.size else 0.,
                  neutral_dominant=float((n > np.maximum(e, c)).mean()) if n.size else 0.,
                  condition_count=float(s.shape[0]), candidate_count=float(s.shape[1]))
    return result


def conditions(pair, s, weighted):
    """Uma linha por condição; índices referem-se a pairs.jsonl."""
    r = relevance(s)
    rows = []
    factual = pair['claim']['factual'].tolist()
    candidates = pair['candidates'].tolist()
    for i, claim_edge in enumerate(factual):
        raw = s[i, :, 6:9]
        channels = raw * r[i, :, None] if weighted else raw
        row = dict(claim_edge=int(claim_edge), candidate_edges=candidates[i],
                   relevance=r[i].tolist(), nli=raw.tolist(), channels=channels.tolist())
        if len(channels):
            strength = channels.max(0)
            best = s[i, :, :3].mean(-1).argmax()
            row.update(support=float(strength[1]), conflict=float(strength[0]),
                       neutral=float(channels[:, 2].mean()), best_match=float(s[i, best, :3].mean()),
                       best_relation_exact=float(s[i, best, 4]), best_relevance=float(r[i].max()),
                       support_edge=int(candidates[i][channels[:, 1].argmax()]),
                       conflict_edge=int(candidates[i][channels[:, 0].argmax()]))
        else:
            row.update(support=0., conflict=0., neutral=0., best_match=0., best_relation_exact=0.,
                       best_relevance=0., support_edge=None, conflict_edge=None)
        rows.append(row)
    return rows


def hierarchical(pair, embeddings, rows, threshold):
    """31 features inspiradas no notebook 19 + 5 resumos explícitos.

    H e H-RP têm nomes, dimensões e agregador idênticos: só muda se NLI é
    ponderado por relevância ANTES dos resumos locais. Min é uma feature,
    nunca uma regra de decisão obrigatória.
    """
    c, e = pair['claim'], pair['evidence']
    cn, en = len(c['node_text']), len(e['node_text'])
    if cn and en:
        matches = (embeddings[c['node_text']] @ embeddings[e['node_text']].T).max(1).values.cpu().numpy()
    else:
        matches = np.zeros(max(cn, 1))
    f = dict(node_claim_count=float(cn), node_evidence_count=float(en),
             node_match_mean=float(matches.mean()), node_match_min=float(matches.min()),
             node_match_max=float(matches.max()), node_match_fraction_above_threshold=float((matches >= threshold).mean()),
             node_claim_empty=float(cn == 0), node_evidence_empty=float(en == 0),
             claim_triple_count=float(len(c['factual'])), evidence_triple_count=float(len(e['factual'])))
    a = lambda key: np.array([row[key] for row in rows], dtype=float)
    m, support, conflict, neutral = (a(key) for key in ('best_match', 'support', 'conflict', 'neutral'))
    margin, confidence = support-conflict, np.maximum(support, conflict)
    f.update(triple_best_match_mean=float(m.mean()), triple_best_match_min=float(m.min()),
             triple_best_match_max=float(m.max()), triple_fraction_above_threshold=float((m >= threshold).mean()),
             best_relation_exact_fraction=float(a('best_relation_exact').mean()),
             claim_triple_empty=float(len(c['factual']) == 0), evidence_triple_empty=float(len(e['factual']) == 0),
             nli_support_mean=float(support.mean()), nli_support_min=float(support.min()), nli_support_max=float(support.max()),
             nli_contradiction_mean=float(conflict.mean()), nli_contradiction_max=float(conflict.max()),
             nli_neutral_mean=float(neutral.mean()), nli_margin_mean=float(margin.mean()),
             nli_margin_min=float(margin.min()), nli_margin_max=float(margin.max()),
             nli_support_dominant_fraction=float((margin > 0).mean()),
             nli_contradiction_dominant_fraction=float((margin < 0).mean()),
             nli_uncertain_fraction=float((confidence < .5).mean()),
             nli_confidence_mean=float(confidence.mean()), nli_confidence_min=float(confidence.min()))
    r, simultaneous = a('best_relevance'), np.minimum(support, conflict)
    f.update(relevance_mean=float(r.mean()), relevance_min=float(r.min()), relevance_max=float(r.max()),
             simultaneous_support_conflict_mean=float(simultaneous.mean()),
             simultaneous_support_conflict_max=float(simultaneous.max()))
    return f


def extract(pair, embeddings, variant, cfg):
    variant = {'matching_nli': 'G19-H-RP'}.get(variant, variant)
    s = signals_array(pair, cfg['smoke'])
    if variant == 'G19':
        return legacy(s), []
    if variant not in VARIANTS:
        raise ValueError(variant)
    rows = conditions(pair, s, weighted=variant == 'G19-H-RP')
    return hierarchical(pair, embeddings, rows, cfg['matching_threshold']), rows
