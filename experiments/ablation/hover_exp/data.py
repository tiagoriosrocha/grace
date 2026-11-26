"""Datasets de verificação: contrato, auditoria, splits e representações congeladas."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .source_contract import triples, assign_roles, normalize_node_type, decode, EXCLUDED_NODE_TYPES

DATASETS = ('factkg', 'fever', 'hover', 'feverous', 'scifact')

# The exported records use the original benchmark label names.  Internally all
# datasets share the same binary convention: 0 is the negative class and 1 is
# the positive/supporting class.
DATASET_LABELS = {
    'factkg': {'REFUTES': 0, 'SUPPORTS': 1},
    'fever': {'REFUTES': 0, 'SUPPORTS': 1},
    'hover': {'NOT_SUPPORTED': 0, 'SUPPORTED': 1},
    'feverous': {'REFUTES': 0, 'SUPPORTS': 1},
    'scifact': {'CONTRADICT': 0, 'SUPPORT': 1},
}
LABELS = {label: value for labels in DATASET_LABELS.values() for label, value in labels.items()}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def file_hash(path):
    path = Path(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(path.glob('*.jsonl')):
            digest.update(child.name.encode('utf-8'))
            digest.update(file_hash(child).encode('ascii'))
        return digest.hexdigest()
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def norm(text):
    return ' '.join(str(text).casefold().replace('_', ' ').split())


def obligations(ts):
    # Type assertions remain context, but are not factual obligations.
    return [i for i, t in enumerate(ts)
            if normalize_node_type(t['target_type']) != 'entitytype'
            and normalize_node_type(t['source_type']) != 'entitytype']


def retained_nodes(value):
    """Keep isolated entity nodes too, as in notebooks 19 and 23."""
    result=[]
    for node in decode(value)['nodes']:
        kind=node.get('type',node.get('node_type',''))
        if normalize_node_type(kind) in EXCLUDED_NODE_TYPES:
            continue
        key=str(node.get('id',node.get('node_id','')))
        result.append(dict(id=key,text=str(node.get('text',node.get('name',node.get('label',key)))),
                           type=normalize_node_type(kind)))
    return result


def evidence_entity_keys(value):
    """Return stable entity names used to build the strict split groups.

    Local node IDs are intentionally ignored: the same DBpedia entity must be
    grouped even when two converted graphs assigned different local IDs.
    """
    result = set()
    for node in decode(value).get('nodes', []):
        kind = normalize_node_type(node.get('type', node.get('node_type', '')))
        if kind != 'entity':
            continue
        text = node.get('text', node.get('name', node.get('label', '')))
        key = norm(text)
        if key:
            result.add(key)
    return sorted(result)


def canonical_triples(ts):
    return sorted({(norm(t['source_text']), norm(t['relation']), norm(t['target_text'])) for t in ts})


def prepare(path, out, cfg):
    if cfg.get('entity_disjoint_split', False):
        raise ValueError('Este protocolo permite entidades compartilhadas; use entity_disjoint_split=false.')
    rows, excluded, seen = [], [], set()
    input_labels = {}
    stats = {'excluded_node_types': ['textsummary', 'textdocument', 'documentchunk']}
    for side in ('claim', 'evidence'):
        stats[side] = dict(graphs=0, nodes_before=0, nodes_after=0, nodes_removed=0,
                           edges_before=0, edges_after=0, edges_removed=0, empty_after=0)
    total = 0
    selected = 0
    sample_fraction = float(cfg.get('sample_fraction', 1.0))
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError('sample_fraction deve estar no intervalo (0, 1].')
    path = Path(path)
    def part_order(file):
        match = re.search(r'part_(\d+)', file.name)
        return (int(match.group(1)) if match else 10**9, file.name)
    input_files = sorted(path.glob('*.jsonl'), key=part_order) if path.is_dir() else [path]
    if not input_files:
        raise ValueError(f'Nenhum arquivo JSONL encontrado em {path}')
    line_no = 0
    for input_file in input_files:
      with open(input_file, encoding='utf-8') as stream:
        for local_line_no, line in enumerate(stream, 1):
            line_no += 1
            if not line.strip():
                continue
            total += 1
            r = json.loads(line)
            key = str(r.get('id', ''))
            if not key or key in seen:
                raise ValueError(f'ID ausente/duplicado em {input_file.name}, linha {local_line_no}: {key}')
            seen.add(key)
            # Subsample before graph preparation and before the grouped split.
            # The hash uses only the stable ID, never the label or split role.
            if not cfg.get('smoke', False) and sample_fraction < 1.0:
                digest = int(hashlib.sha256(key.encode('utf-8')).hexdigest()[:16], 16)
                if digest / float(2**64 - 1) >= sample_fraction:
                    continue
            selected += 1
            dataset = str(r.get('dataset', '')).casefold()
            allowed = DATASET_LABELS.get(cfg.get('dataset'), {})
            if dataset != cfg['dataset'] or r.get('label') not in allowed:
                raise ValueError(f'Dataset/rótulo inválido: {key}')
            input_labels[r['label']] = input_labels.get(r['label'], 0) + 1
            if bool(r.get('synthetic', False)) != bool(cfg['smoke']):
                raise ValueError('Dados sintéticos são permitidos somente em --smoke.')
            if r.get('split') not in ('train', 'validation', 'dev', 'test'):
                raise ValueError(f'Split inválido: {r.get("split")}')
            try:
                ct = triples(r['grafo_claim'], stats, 'claim')
                et = triples(r['grafo_evidencia'], stats, 'evidence')
                # Exact duplicate triples are redundant obligations, remove deterministically.
                def unique(ts):
                    return list({(t['source_id'], t['relation_key'], t['target_id']): t for t in ts}.values())
                ct, et = unique(ct), unique(et)
                if not ct or not obligations(ct):
                    raise ValueError('claim sem relações factuais após sanitização')
                if not et:
                    raise ValueError('evidência vazia após sanitização; excluída, não convertida em negativo')
            except (ValueError, KeyError, TypeError) as exc:
                excluded.append({'id': key, 'label': r['label'], 'reason': str(exc)})
                continue
            claim = r.get('claim', '')
            if not isinstance(claim, str) or not claim.strip():
                raise ValueError(f'Claim textual ausente: {key}')
            rows.append(dict(id=key, claim=claim, label=allowed[r['label']],
                             label_name=r['label'], dataset=dataset,
                             source_split=str(r.get('split', 'train')).casefold(),
                             source_group_id=str(r.get('group_id') or ''), evaluation_role='',
                             claim_triples=ct, evidence_triples=et, num_hops=r.get('num_hops'),
                             claim_nodes=retained_nodes(r['grafo_claim']),
                             evidence_nodes=retained_nodes(r['grafo_evidencia']),
                             claim_entity_keys=evidence_entity_keys(r['grafo_claim']),
                             evidence_entity_keys=evidence_entity_keys(r['grafo_evidencia'])))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('Nenhum par elegível')
    # Claims/source families and repeated evidence are indivisible groups.
    # Shared entities alone do not merge examples in this protocol.
    from .source_contract import group_claims
    initial = group_claims(df)
    parents = list(range(len(df)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    keys = {}
    for i, row in enumerate(rows):
        evidence_fingerprint = hashlib.sha256(
            json.dumps(canonical_triples(row['evidence_triples']), ensure_ascii=False).encode('utf-8')
        ).hexdigest()
        row['evidence_group_id'] = 'evidence_' + evidence_fingerprint
        group_keys = ['claim:' + initial[i], 'evidence:' + evidence_fingerprint]
        for key in group_keys:
            if key in keys:
                parents[find(i)] = find(keys[key])
            else:
                keys[key] = i
    members = {}
    for i, row in enumerate(rows):
        members.setdefault(find(i), []).append(row['id'])
    component_ids = {root: hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()
                     for root, ids in members.items()}
    df = pd.DataFrame(rows).sort_values('id').reset_index(drop=True)
    id_to_component = {row['id']: component_ids[find(i)] for i, row in enumerate(rows)}
    df['source_group_id'] = df.id.map(id_to_component)
    df = assign_roles(df, cfg)
    records = df.to_dict('records')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'pairs.jsonl').open('w', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    df[['id','group_id','evidence_group_id','evaluation_role','label','num_hops']].to_csv(out/'splits.csv', index=False)
    pd.DataFrame(excluded, columns=['id','label','reason']).to_csv(out/'excluded.csv', index=False)
    sizes = {side: [len(r[side + '_triples']) for r in records] for side in ('claim','evidence')}
    cross_role_evidence = int(df.groupby('evidence_group_id').evaluation_role.nunique().gt(1).sum())
    cross_role_entities = 0
    entity_roles = {}
    for row in records:
        for entity in set(row['evidence_entity_keys'] + row['claim_entity_keys']):
            entity_roles.setdefault(entity, set()).add(row['evaluation_role'])
    cross_role_entities = int(sum(len(roles) > 1 for roles in entity_roles.values()))
    cross_role_claims = int(df.assign(claim_key=df.claim.map(lambda s: ' '.join(s.casefold().split())))
                           .groupby('claim_key').evaluation_role.nunique().gt(1).sum())
    if cross_role_evidence or cross_role_claims or df.groupby('group_id').evaluation_role.nunique().gt(1).any():
        raise ValueError('Sobreposição proibida de claims, evidências ou grupos entre conjuntos.')
    report = dict(input=total, sampled_input=selected, sample_fraction=sample_fraction,
                  eligible=len(records), excluded=len(excluded), coverage=len(records)/total,
                  input_sha256=file_hash(path), sanitization=stats,
                  coverage_by_label={label: dict(input=n, eligible=int((df.label_name == label).sum()),
                                                coverage=float((df.label_name == label).sum()/n))
                                     for label,n in input_labels.items()},
                  roles=df.groupby(['evaluation_role','label']).size().rename('n').reset_index().to_dict('records'),
                  graph_edges={s: dict(max=max(v), median=float(np.median(v)), p95=float(np.percentile(v,95))) for s,v in sizes.items()},
                  split_groups=dict(mode='claim_and_evidence_disjoint_shared_entities',
                                    groups=int(df.group_id.nunique()),
                                    largest_group=int(df.groupby('group_id').size().max()),
                                    cross_role_claims=cross_role_claims,
                                    entity_overlap_allowed=True,
                                    distinct_evidence_groups=int(df.evidence_group_id.nunique()),
                                    cross_role_evidence_groups=cross_role_evidence,
                                    distinct_entities=len(entity_roles),
                                    cross_role_entities=cross_role_entities),
                  test_scope='Divisão interna agrupada do arquivo selecionado; não é o teste oficial do dataset.',
                  empty_evidence_policy='Excluir originais vazios e relatar cobertura; vazio sintético só em diagnóstico.')
    save_json(out/'audit.json', report)
    return records, report


def embed(texts, cfg, device):
    if cfg['smoke']:
        result = torch.zeros(len(texts), 32)
        for i, text in enumerate(texts):
            for token in re.findall(r'\w+', text.casefold()):
                h = hashlib.sha256(token.encode()).digest()
                result[i, int.from_bytes(h[:4], 'big') % 32] += 1 if h[4] % 2 else -1
        return F.normalize(result, dim=-1), {'backend':'hash_ONLY_SMOKE', 'scientific_result':False}
    from transformers import AutoModel, AutoTokenizer
    opts = dict(revision=cfg['embedding_revision'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['embedding_model'], **opts)
    model = AutoModel.from_pretrained(cfg['embedding_model'], use_safetensors=True, **opts).eval().to(device)
    pieces, truncated = [], 0
    with torch.inference_mode():
        for start in range(0, len(texts), cfg['embedding_batch_size']):
            batch = texts[start:start+cfg['embedding_batch_size']]
            truncated += sum(len(ids) > cfg['max_length'] for ids in tokenizer(batch, truncation=False)['input_ids'])
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=cfg['max_length'], return_tensors='pt').to(device)
            hidden = model(**inputs).last_hidden_state
            # Mean pooling over valid BERT tokens, followed by L2
            # normalization, matching the textual encoder used in the
            # previous project.
            mask = inputs.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pieces.append(F.normalize(pooled, dim=-1).float().cpu())
            if start % (cfg['embedding_batch_size'] * 50) == 0:
                print(f'Embeddings: {min(start+len(batch),len(texts))}/{len(texts)}', flush=True)
    metadata = dict(backend='transformer', model=cfg['embedding_model'],
                    resolved_revision=getattr(model.config,'_commit_hash',None),
                    texts=len(texts), truncated_texts=truncated, max_length=cfg['max_length'])
    del model, tokenizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return torch.cat(pieces), metadata


def nli_features(records, pairs, cfg, device):
    """Score each retrieved evidence/claim triple pair with frozen NLI."""
    rows, locations = [], []
    for record, pair in zip(records, pairs):
        for ci, candidates in enumerate(pair['candidates'].tolist()):
            claim_triple = record['claim_triples'][int(pair['claim_factual'][ci])]
            hypothesis = f"{claim_triple['source_text']} [REL] {claim_triple['relation']} [OBJ] {claim_triple['target_text']}"
            for position, evidence_index in enumerate(candidates):
                evidence_triple = record['evidence_triples'][int(evidence_index)]
                premise = f"{evidence_triple['source_text']} [REL] {evidence_triple['relation']} [OBJ] {evidence_triple['target_text']}"
                rows.append((premise, hypothesis))
    if not rows:
        return [torch.zeros_like(pair['signals'][..., :3]) for pair in pairs], {'backend':'none','scientific_result':False}
    if cfg['smoke']:
        return [torch.zeros_like(pair['signals'][..., :3]) for pair in pairs], {'backend':'zeros_ONLY_SMOKE','scientific_result':False}
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = cfg['nli_model']
    tokenizer = AutoTokenizer.from_pretrained(name, revision=cfg['nli_revision'])
    model = AutoModelForSequenceClassification.from_pretrained(name, revision=cfg['nli_revision'], use_safetensors=True).eval().to(device)
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    def find(label):
        return next((i for i, value in labels.items() if label in value), None)
    ids = [find('contradiction'), find('entailment'), find('neutral')]
    if any(index is None for index in ids):
        raise ValueError(f'Rótulos NLI incompatíveis: {labels}')
    resolved_revision=getattr(model.config,'_commit_hash',None)
    values=[]
    with torch.inference_mode():
        for start in range(0,len(rows),cfg['nli_batch_size']):
            batch=rows[start:start+cfg['nli_batch_size']]
            encoded=tokenizer([x[0] for x in batch],[x[1] for x in batch],padding=True,truncation=True,
                              max_length=cfg['nli_max_length'],return_tensors='pt').to(device)
            probs=model(**encoded).logits.softmax(-1)[:,ids].float().cpu()
            values.append(probs)
    values=torch.cat(values)
    outputs=[torch.zeros_like(pair['signals'][..., :3]) for pair in pairs]
    offset=0
    for pi,pair in enumerate(pairs):
        shape=pair['signals'][..., :3].shape
        n=shape[0]*shape[1]
        if n: outputs[pi]=values[offset:offset+n].reshape(shape)
        offset += n
    del model, tokenizer
    if device.type=='cuda': torch.cuda.empty_cache()
    return outputs, {'backend':'transformer','model':name,'labels':labels,'resolved_revision':resolved_revision,
                     'pairs':len(rows),'max_length':cfg['nli_max_length']}


def build_graph(ts, text_ids, rel_vocab):
    entities = {}
    for t in ts:
        entities.setdefault(t['source_id'], t['source_text'])
        entities.setdefault(t['target_id'], t['target_text'])
    node_map = {key:i for i,key in enumerate(entities)}
    src = torch.tensor([node_map[t['source_id']] for t in ts], dtype=torch.long)
    dst = torch.tensor([node_map[t['target_id']] for t in ts], dtype=torch.long)
    n = len(entities)
    structure = torch.stack([torch.bincount(src,minlength=n),torch.bincount(dst,minlength=n)],1).float().log1p()
    return dict(node_text=torch.tensor([text_ids[x] for x in entities.values()],dtype=torch.long),
                edge_text=torch.tensor([text_ids[t['relation']] for t in ts],dtype=torch.long),
                edge_type=torch.tensor([rel_vocab.get(t['relation_key'],0) for t in ts],dtype=torch.long),
                edge_index=torch.stack([src,dst]), structure=structure,
                factual=torch.tensor(obligations(ts),dtype=torch.long))


def candidate_features(ct, et, cg, eg, embeddings, top_k, use_exact=False):
    def parts(g):
        src,dst=g['edge_index']
        return embeddings[g['node_text'][src]], embeddings[g['edge_text']], embeddings[g['node_text'][dst]]
    cp,ep=parts(cg),parts(eg)
    cidx=cg['factual']
    eidx=eg['factual']
    # No factual evidence means only null. Type edges still contextualize the GNN.
    if not len(eidx):
        return torch.empty(len(cidx),0,dtype=torch.long),torch.empty(len(cidx),0,6)
    cosines=torch.stack([a[cidx] @ b[eidx].T for a,b in zip(cp,ep)],-1)
    rank=cosines.mean(-1)
    chosen=torch.argsort(rank,dim=-1,descending=True,stable=True)[:,:min(top_k,len(eidx))]
    candidates=eidx[chosen]
    sims=cosines.gather(1,chosen[:,:,None].expand(-1,-1,3))
    if use_exact:
        exact=torch.tensor([[[float(norm(ct[i][key])==norm(et[j][key])) for key in ('source_text','relation','target_text')]
                            for j in js] for i,js in zip(cidx.tolist(),candidates.tolist())])
    else:
        exact=torch.zeros((*candidates.shape,3),dtype=sims.dtype)
    return candidates,torch.cat([sims,exact],-1)


def features(records, cfg, device, out):
    texts=sorted({t[key] for r in records for side in ('claim_triples','evidence_triples')
                  for t in r[side] for key in ('source_text','relation','target_text')})
    ids={s:i for i,s in enumerate(texts)}
    from collections import Counter
    counts=Counter(t['relation_key'] for r in records if r['evaluation_role']=='train'
                   for side in ('claim_triples','evidence_triples') for t in r[side])
    relations=sorted(counts,key=lambda s:(-counts[s],s))[:cfg['relation_vocab_limit']]
    save_json(out/'relation_audit.json',dict(train_distinct=len(counts),categorical_kept=len(relations),
              tail_policy='Categoria UNK; embedding textual preservado para TODAS as relações.'))
    vocab={'__UNK__':0,**{s:i+1 for i,s in enumerate(relations)}}
    embeddings,info=embed(texts,cfg,device)
    pairs=[]
    for r in records:
        ct,et=r['claim_triples'],r['evidence_triples']
        cg,eg=build_graph(ct,ids,vocab),build_graph(et,ids,vocab)
        candidates,signals=candidate_features(ct,et,cg,eg,embeddings,cfg['top_k'],cfg.get('use_exact_features',False))
        pairs.append(dict(id=r['id'],group_id=r['group_id'],role=r['evaluation_role'],
                          label=int(r['label']),num_hops=r['num_hops'],
                          claim=cg,evidence=eg,candidates=candidates,signals=signals,
                          claim_factual=cg['factual'].tolist()))
    nli_values,nli_info=nli_features(records,pairs,cfg,device)
    for pair,values in zip(pairs,nli_values):
        pair['signals']=torch.cat([pair['signals'],values],-1)
    structure=torch.cat([p[s]['structure'] for p in pairs if p['role']=='train' for s in ('claim','evidence')])
    mean,std=structure.mean(0),structure.std(0,unbiased=False).clamp_min(1e-6)
    for p in pairs:
        for side in ('claim','evidence'):
            p[side]['structure']=(p[side]['structure']-mean)/std
    bundle=dict(embeddings=embeddings,pairs=pairs,relation_vocab=vocab,info=info,nli_info=nli_info)
    torch.save(bundle,out/'features.pt.tmp')
    (out/'features.pt.tmp').replace(out/'features.pt')
    save_json(out/'embedding_info.json',info)
    save_json(out/'nli_info.json',nli_info)
    save_json(out/'normalization.json',dict(fit='train',mean=mean.tolist(),std=std.tolist()))
    save_json(out/'relation_vocab.json',vocab)
    save_json(out/'text_index.json',texts)
    save_json(out/'features_complete.json',{'complete':True})
    return bundle
