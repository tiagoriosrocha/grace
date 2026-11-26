"""Frozen BERT token cache on disk and directed evidence graph tokens."""
import json
import os
import pickle
from collections import Counter
import numpy as np
import torch
from .data import embed,save_json


class ClaimStore:
    def __init__(self,path):
        self.path=path
        self.info=json.loads((path/'index.json').read_text())
        self.values=np.load(path/'hidden.npy',mmap_mode='r')

    def get(self,index):
        n=self.info['lengths'][index]
        # Copy only this example; mmap never materializes the whole cache.
        return torch.from_numpy(np.array(self.values[index,:n],dtype=np.float32))


def _remove_cache_files(*paths):
    """Remove only the explicitly named cache files after a failed load."""
    for path in paths:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _cache_ids(records):
    return [r['id'] for r in records]


def cache_claims(records,cfg,device,out):
    path=out/'claim_tokens';path.mkdir(exist_ok=True)
    index_path = path/'index.json'
    hidden_path = path/'hidden.npy'
    if index_path.exists():
        try:
            cached = ClaimStore(path)
            if cached.info.get('ids') != _cache_ids(records):
                raise ValueError('cache de claims pertence a outra execução')
            if len(cached.info.get('lengths', [])) != len(records):
                raise ValueError('cache de claims tem quantidade incompatível')
            if cached.values.ndim != 3 or cached.values.shape[0] != len(records):
                raise ValueError('cache de claims tem formato incompatível')
            if any(n < 0 or n > cached.values.shape[1] for n in cached.info['lengths']):
                raise ValueError('cache de claims tem comprimentos inválidos')
            return cached
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            # An interrupted mmap/index pair must be rebuilt together.
            _remove_cache_files(index_path, hidden_path)
            _remove_cache_files(path/'index.json.tmp', path/'hidden.npy.tmp')
    length=cfg['x23_claim_max_length'];batch_size=cfg['embedding_batch_size']
    lengths=[];labels=[];specials=[];truncated=0
    if cfg['smoke']:
        d=32
        mm=np.lib.format.open_memmap(path/'hidden.npy',mode='w+',dtype='float16',shape=(len(records),length,d))
        for i,row in enumerate(records):
            words=row['claim'].split()[:length] or ['[EMPTY]']
            values,_=embed(words,cfg,device)
            mm[i,:len(words)]=values.numpy();lengths.append(len(words));labels.append(words);specials.append([False]*len(words))
        backend='hash_ONLY_SMOKE';revision=None
    else:
        from transformers import AutoModel,AutoTokenizer
        opts=dict(revision=cfg['embedding_revision'])
        tok=AutoTokenizer.from_pretrained(cfg['embedding_model'],**opts)
        model=AutoModel.from_pretrained(cfg['embedding_model'],use_safetensors=True,**opts).eval().to(device)
        d=model.config.hidden_size
        mm=np.lib.format.open_memmap(path/'hidden.npy',mode='w+',dtype='float16',shape=(len(records),length,d))
        with torch.inference_mode():
            for start in range(0,len(records),batch_size):
                texts=[r['claim'] for r in records[start:start+batch_size]]
                truncated+=sum(len(x)>length for x in tok(texts,truncation=False)['input_ids'])
                inp=tok(texts,truncation=True,max_length=length,padding=True,return_tensors='pt').to(device)
                hidden=model(**inp).last_hidden_state.float().cpu().numpy()
                for j,n in enumerate(inp.attention_mask.sum(1).cpu().tolist()):
                    mm[start+j,:n]=hidden[j,:n];lengths.append(n)
                    ids=inp.input_ids[j,:n].cpu().tolist()
                    labels.append(tok.convert_ids_to_tokens(ids));specials.append([x in tok.all_special_ids for x in ids])
                print(f"BERT tokens: {min(start+batch_size,len(records))}/{len(records)}",flush=True)
        backend='transformer';revision=getattr(model.config,'_commit_hash',None)
        del model,tok
        if device.type=='cuda':torch.cuda.empty_cache()
    mm.flush();del mm
    save_json(path/'index.json',dict(ids=[r['id'] for r in records],lengths=lengths,tokens=labels,special=specials,
        dim=d,max_length=length,truncated_claims=truncated,backend=backend,resolved_revision=revision,
        storage='float16 mmap; encoder frozen and eval during cache creation'))
    return ClaimStore(path)


def select_graph(nodes,edges,embeddings,claim_mean,cap,use_edges):
    nnodes,nedges=len(nodes),len(edges) if use_edges else 0
    if nnodes+nedges<=cap:return list(range(nnodes)),list(range(nedges))
    claim_mean=torch.nn.functional.normalize(claim_mean,dim=0)
    ns=embeddings[nodes]@claim_mean
    if use_edges:
        nb=min(nnodes,cap//2);eb=min(nedges,cap-nb);nb=min(nnodes,cap-eb)
        es=torch.nn.functional.normalize(embeddings[edges].mean(1),dim=-1)@claim_mean
    else:nb=min(nnodes,cap);eb=0
    ni=sorted(torch.argsort(ns,descending=True,stable=True)[:nb].tolist())
    ei=sorted(torch.argsort(es,descending=True,stable=True)[:eb].tolist()) if eb else []
    return ni,ei


def prepare_x23(records,bundle,cfg,device,out,reference):
    claims=cache_claims(records,cfg,device,out)
    assert claims.info['ids']==reference['ids']==[r['id'] for r in records]
    cache=out/'x23_graphs.pt'
    temporary = cache.with_suffix('.pt.tmp')
    if cache.exists():
        try:
            cached = torch.load(cache,map_location='cpu',weights_only=True)
            if not isinstance(cached, dict) or not {'examples','embeddings','types'} <= set(cached):
                raise ValueError('cache x23 sem campos obrigatórios')
            examples = cached['examples']
            if not isinstance(examples, list) or not all(isinstance(x, dict) for x in examples):
                raise ValueError('cache x23 tem exemplos inválidos')
            cached_ids = [x.get('id') for x in examples]
            if cached_ids != _cache_ids(records):
                raise ValueError('cache x23 pertence a outra execução')
            if len(examples) != len(records):
                raise ValueError('cache x23 tem quantidade incompatível')
            return claims,cached
        except (RuntimeError, EOFError, OSError, ValueError, KeyError, TypeError,
                IndexError, pickle.UnpicklingError):
            # The final cache may have been produced by an older non-atomic
            # writer or copied while incomplete. Rebuild it safely below.
            _remove_cache_files(cache, temporary)
    texts=json.loads((out/'text_index.json').read_text())
    known=set(texts)
    missing=sorted({n['text'] for r in records for n in r['evidence_nodes']}-known)
    embeddings=bundle['embeddings']
    if missing:
        extra,_=embed(missing,cfg,device);embeddings=torch.cat([embeddings,extra]);texts+=missing
    ids={x:i for i,x in enumerate(texts)}
    counts=Counter(n['type'] for r in records if r['evaluation_role']=='train' for n in r['evidence_nodes'])
    types={'__UNK__':0,**{k:i+1 for i,k in enumerate(sorted(counts,key=lambda k:(-counts[k],k))[:64])}}
    examples=[];capped=0
    for idx,r in enumerate(records):
        nodes=torch.tensor([ids[n['text']] for n in r['evidence_nodes']],dtype=torch.long)
        edges=torch.tensor([[ids[t[k]] for k in ('source_text','relation','target_text')]
                            for t in r['evidence_triples']],dtype=torch.long).reshape(-1,3)
        cm=claims.get(idx).mean(0)
        selections={}
        for use_edges in (False,True):
            ni,ei=select_graph(nodes,edges,embeddings,cm,cfg['x23_max_graph_tokens'],use_edges)
            selections[str(use_edges)]=dict(nodes=torch.tensor(ni,dtype=torch.long),edges=torch.tensor(ei,dtype=torch.long))
        capped+=int(len(nodes)+len(edges)>cfg['x23_max_graph_tokens'])
        examples.append(dict(id=r['id'],index=idx,role=r['evaluation_role'],group_id=r['group_id'],
            label=int(r['label']),num_hops=r['num_hops'],nodes=nodes,edges=edges,
            types=torch.tensor([types.get(n['type'],0) for n in r['evidence_nodes']],dtype=torch.long),
            selections=selections,nli_features=reference['features'][idx]))
    result=dict(examples=examples,embeddings=embeddings,types=types)
    torch.save(result,temporary)
    temporary.replace(cache)
    save_json(out/'x23_data_audit.json',dict(node_types=types,node_type_fit='train',capped_examples=capped,
        graph_token_limit=cfg['x23_max_graph_tokens'],raw_claim=True,raw_evidence=False,
        query='BERT token-level',edge_token='ordered [source, relation, target]',reverse_edges=False))
    return claims,result
