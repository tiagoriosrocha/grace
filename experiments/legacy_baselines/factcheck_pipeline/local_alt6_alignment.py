"""Label independent retrieval used by Alternative 6."""
import re
import unicodedata
import numpy as np


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().replace('_', ' ').split())


def relation(text):
    return norm(re.sub(r'([a-z])([A-Z])', r'\1 \2', str(text)))


ALIASES = {
    'birth place': 'born in', 'born in': 'born in',
    'death place': 'died in', 'died in': 'died in',
    'works at': 'works at', 'employer': 'works at',
    'manufacturer': 'manufactured by', 'manufactured by': 'manufactured by',
    'design company': 'designed by', 'designer': 'designed by',
    'designed by': 'designed by', 'founder': 'founded by', 'founded by': 'founded by',
    'foundation place': 'founded in', 'founded in': 'founded in',
}
INVERSES = {'manufactures': 'manufactured by', 'designed': 'designed by',
            'founded': 'founded by', 'employs': 'works at'}


def canonical(triple):
    source, rel, target = triple['source_text'], relation(triple['relation']), triple['target_text']
    if rel in INVERSES:
        source, target, rel = target, source, INVERSES[rel]
    return source, ALIASES.get(rel, rel), target


def factual(triple):
    return all(norm(triple.get(key, '')).replace(' ', '') != 'entitytype'
               for key in ('source_type', 'target_type'))


def sentence(parts):
    return f'{parts[0]} {parts[1]} {parts[2]}.'


def retrieve(condition, evidence, vectors, top_k=4, max_paths=4):
    query = canonical(condition)
    if any(value not in vectors for value in query):
        return []
    query_vectors = np.stack([vectors[x] for x in query])
    parts = [canonical(t) for t in evidence]
    valid = [i for i, t in enumerate(evidence) if factual(t) and all(x in vectors for x in parts[i])]
    if not valid:
        return []
    values = np.asarray([[vectors[x] for x in parts[i]] for i in valid])
    similarities = np.einsum('nkd,kd->nk', values, query_vectors)
    exact = np.asarray([[float(norm(a) == norm(b)) for a, b in zip(query, parts[i])] for i in valid])
    semantic = similarities.mean(1)
    aligned = semantic + exact[:, 0] + exact[:, 1] + .5 * exact[:, 2]
    selected = sorted(set(np.argsort(-semantic, kind='stable')[:top_k].tolist()) |
                      set(np.argsort(-aligned, kind='stable')[:top_k].tolist()),
                      key=lambda j: (-aligned[j], valid[j]))
    result = []
    for j in selected:
        i = valid[j]
        result.append(dict(edge_indices=[i], premise=sentence(parts[i]), hypothesis=sentence(query),
                           evidence_triples=[evidence[i]], canonical_triples=[list(parts[i])],
                           similarity=similarities[j].tolist(), exact=exact[j].tolist(),
                           retrieval_score=float(aligned[j]), path=False))
    outgoing = {}
    for i in valid:
        outgoing.setdefault(norm(parts[i][0]), []).append(i)
    paths = []
    for i in outgoing.get(norm(query[0]), []):
        for j in outgoing.get(norm(parts[i][2]), []):
            if i == j or norm(parts[j][2]) != norm(query[2]):
                continue
            paths.append((i, j))
            if len(paths) >= max_paths:
                break
        if len(paths) >= max_paths:
            break
    for i, j in paths[:max_paths]:
        result.append(dict(edge_indices=[i, j], premise=sentence(parts[i]) + ' ' + sentence(parts[j]),
                           hypothesis=sentence(query), evidence_triples=[evidence[i], evidence[j]],
                           canonical_triples=[list(parts[i]), list(parts[j])],
                           similarity=[1., 0., 1.], exact=[1., 0., 1.],
                           retrieval_score=0., path=True))
    return result
