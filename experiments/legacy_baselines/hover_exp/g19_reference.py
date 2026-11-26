"""Notebook 19 M1: oriented retrieval, claim-centric 31 features, balanced LR.

Ported to the shared HoVer split and frozen BERT. Original retrieval weights,
top-5, verbalization, feature order and LogisticRegression settings retained.
"""
import re
import gzip
import numpy as np
import torch
from .data import embed,save_json
from .nli import NLIScorer

SIMILARITY_FEATURES=[
 'node_claim_count','node_evidence_count','node_match_mean','node_match_min','node_match_max',
 'node_match_fraction_above_threshold','node_claim_empty','node_evidence_empty',
 'claim_triple_count','evidence_triple_count','triple_best_match_mean','triple_best_match_min',
 'triple_best_match_max','triple_fraction_above_threshold','best_relation_exact_fraction',
 'claim_triple_empty','evidence_triple_empty']
NLI_FEATURES=[
 'nli_support_mean','nli_support_min','nli_support_max','nli_contradiction_mean',
 'nli_contradiction_max','nli_neutral_mean','nli_margin_mean','nli_margin_min','nli_margin_max',
 'nli_support_dominant_fraction','nli_contradiction_dominant_fraction','nli_uncertain_fraction',
 'nli_confidence_mean','nli_confidence_min']
FEATURE_NAMES=SIMILARITY_FEATURES+NLI_FEATURES


def normalize_text(value):
    return re.sub(r'\s+',' ',str(value).strip()) or '[SEM TEXTO]'


def relation_phrase(value):
    text=str(value).strip().replace('::forward','').replace('::reverse','')
    text=re.sub(r'([a-z0-9])([A-Z])',r'\1 \2',text).replace('_',' ').replace('-',' ')
    return re.sub(r'\s+',' ',text).strip().lower() or 'related to'


def verbalize(t):
    return f"{normalize_text(t['source_text'])} {relation_phrase(t['relation_key'])} {normalize_text(t['target_text'])}."


def aggregate(node_sim,match,exact,probs,claim_count,evidence_count):
    """Exact 31 feature definitions from notebook 19; probabilities are C/E/N."""
    f=dict.fromkeys(FEATURE_NAMES,0.)
    nc,ne=node_sim.shape
    f.update(node_claim_count=float(nc),node_evidence_count=float(ne),node_claim_empty=float(nc==0),
             node_evidence_empty=float(ne==0),claim_triple_count=float(claim_count),
             evidence_triple_count=float(evidence_count),claim_triple_empty=float(claim_count==0),
             evidence_triple_empty=float(evidence_count==0))
    if nc and ne:
        best=node_sim.max(1)
        for stat in ('mean','min','max'):f['node_match_'+stat]=float(getattr(best,stat)())
        f['node_match_fraction_above_threshold']=float((best>=.5).mean())
    if not claim_count:return np.array([f[k] for k in FEATURE_NAMES],dtype=np.float32),[]
    if match.shape[1]:
        best=match[:,0]; rel=exact[:,0]
        support=probs[:,:,1].max(1); conflict=probs[:,:,0].max(1); neutral=probs[:,:,2].mean(1)
    else:
        best=rel=support=conflict=neutral=np.zeros(claim_count)
    margin=support-conflict; confidence=np.maximum(support,conflict)
    for stat in ('mean','min','max'):
        f['triple_best_match_'+stat]=float(getattr(best,stat)())
        f['nli_support_'+stat]=float(getattr(support,stat)())
        f['nli_margin_'+stat]=float(getattr(margin,stat)())
    f.update(triple_fraction_above_threshold=float((best>=.5).mean()),best_relation_exact_fraction=float(rel.mean()),
             nli_contradiction_mean=float(conflict.mean()),nli_contradiction_max=float(conflict.max()),
             nli_neutral_mean=float(neutral.mean()),nli_support_dominant_fraction=float((margin>0).mean()),
             nli_contradiction_dominant_fraction=float((margin<0).mean()),
             nli_uncertain_fraction=float((confidence<.5).mean()),nli_confidence_mean=float(confidence.mean()),
             nli_confidence_min=float(confidence.min()))
    local=[dict(best_match=float(best[i]),support=float(support[i]),contradiction=float(conflict[i]),
                neutral=float(neutral[i]),margin=float(margin[i])) for i in range(claim_count)]
    return np.array([f[k] for k in FEATURE_NAMES],dtype=np.float32),local


def prepare_reference(records,cfg,device,out):
    cache=out/'g19_reference.pt'
    if cache.exists():return torch.load(cache,map_location='cpu',weights_only=True)
    texts=set()
    for row in records:
        for side in ('claim','evidence'):
            texts.update(normalize_text(n['text']) for n in row[side+'_nodes'])
            for t in row[side+'_triples']:
                texts.update((normalize_text(t['source_text']),normalize_text(t['target_text']),
                              relation_phrase(t['relation_key']),verbalize(t)))
    texts=sorted(texts); ids={s:i for i,s in enumerate(texts)}
    ecfg={**cfg,'max_length':64}
    embeddings,info=embed(texts,ecfg,device)
    def matrix(strings):return embeddings[torch.tensor([ids[s] for s in strings],dtype=torch.long)]
    pending=[]; all_rows=[]
    for row in records:
        ct,et=row['claim_triples'],row['evidence_triples']
        def sides(ts):
            return [matrix([fn(t) for t in ts]) for fn in (verbalize,
                lambda t:normalize_text(t['source_text']),lambda t:normalize_text(t['target_text']),
                lambda t:relation_phrase(t['relation_key']))]
        cs,es=sides(ct),sides(et)
        score=sum(w*(a@b.T) for w,a,b in zip((.5,.2,.2,.1),cs,es))
        chosen=torch.argsort(score,descending=True,stable=True,dim=1)[:,:min(5,len(et))]
        match=score.gather(1,chosen).numpy()
        exact=np.array([[float(ct[i]['relation_key']==et[j]['relation_key']) for j in js]
                        for i,js in enumerate(chosen.tolist())],dtype=np.float32)
        nmat=matrix([normalize_text(n['text']) for n in row['claim_nodes']])@matrix(
            [normalize_text(n['text']) for n in row['evidence_nodes']]).T
        pairs=[(verbalize(et[j]),verbalize(ct[i])) for i,js in enumerate(chosen.tolist()) for j in js]
        pending.append((nmat.numpy(),match,exact,chosen.tolist(),len(pairs)))
        all_rows.extend(pairs)
    scorer=NLIScorer({**cfg,'nli_max_length':128},device)
    probs=scorer.score(all_rows).numpy(); nli_info=scorer.metadata.copy();scorer.close()
    values=[];offset=0
    with gzip.open(out/'G19_REF_alignments.jsonl.gz','wt',encoding='utf-8') as stream:
        import json
        for row,(node_sim,match,exact,candidates,count) in zip(records,pending):
            p=probs[offset:offset+count].reshape(*match.shape,3);offset+=count
            f,local=aggregate(node_sim,match,exact,p,len(row['claim_triples']),len(row['evidence_triples']))
            values.append(f)
            for i,item in enumerate(local):
                item.update(claim_index=i,claim_triple=row['claim_triples'][i],
                    candidates=[dict(evidence_index=j,triple=row['evidence_triples'][j],match=float(match[i,k]),
                                     nli=p[i,k].tolist()) for k,j in enumerate(candidates[i])])
            stream.write(json.dumps(dict(id=row['id'],conditions=local),ensure_ascii=False)+'\n')
    result=dict(features=torch.from_numpy(np.stack(values)),ids=[r['id'] for r in records],feature_names=FEATURE_NAMES)
    torch.save(result,cache.with_suffix('.pt.tmp'));cache.with_suffix('.pt.tmp').replace(cache)
    save_json(out/'G19_REF_protocol.json',dict(embedding=info,nli=nli_info,features=FEATURE_NAMES,top_k=5,
        retrieval_weights=[.5,.2,.2,.1],class_weight='balanced',C=1.,max_iter=4000,
        adaptations=['HoVer binary labels and shared grouped split','BERT instead of original MiniLM',
                     'shared sanitation, exact duplicate edges removed; all retained edges including type assertions']))
    return result
