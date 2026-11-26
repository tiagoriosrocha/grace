"""Training and 23-b token-removal diagnostics for the X23 family."""
import json
import math
import time
import numpy as np
import pandas as pd
import torch
from .data import save_json
from .model import move
from .experiment import seed_all,metrics
from .x23 import X23,VARIANTS


@torch.no_grad()
def probabilities(model,examples,store,device):
    model.eval()
    return np.array([float(model(move(x,device),store.get(x['index']).to(device)).sigmoid().cpu()) for x in examples])


def make_model(data,cfg,variant):
    train=torch.stack([x['nli_features'] for x in data['examples'] if x['role']=='train'])
    return X23(data['embeddings'],len(data['types']),cfg,variant,train.mean(0),train.std(0,unbiased=False).clamp_min(1e-6))


def train(model,examples,store,cfg,seed,device,folder):
    checkpoint=folder/f'{model.variant}_{seed}.pt';infofile=folder/f'{model.variant}_{seed}.json'
    if checkpoint.exists() and infofile.exists():return
    train_set=[x for x in examples if x['role']=='train'];val=[x for x in examples if x['role']=='validation']
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['x23_learning_rate'],weight_decay=cfg['weight_decay'])
    best=-1.;stale=0;history=[];started=time.time();state=None
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,cfg['x23_epochs']+1):
        model.train();total=0.
        order=np.random.default_rng(seed+1009*epoch).permutation(len(train_set))
        for start in range(0,len(order),cfg['batch_size']):
            chunk=order[start:start+cfg['batch_size']];optimizer.zero_grad(set_to_none=True)
            for idx in chunk:
                example=train_set[idx]
                logit=model(move(example,device),store.get(example['index']).to(device))
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,logit.new_tensor(float(example['label'])))
                if not torch.isfinite(loss):raise RuntimeError(f'Loss inválida: {model.variant}/{seed}')
                (loss/len(chunk)).backward();total+=float(loss.detach().cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        prob=probabilities(model,val,store,device);score=metrics([x['label'] for x in val],prob)['macro_f1']
        history.append(dict(epoch=epoch,train_loss=total/len(train_set),validation_macro_f1=score))
        pd.DataFrame(history).to_csv(folder/f'history_{model.variant}_{seed}.csv',index=False)
        print(f'{model.variant}/{seed} época={epoch} loss={total/len(train_set):.4f} val_F1={score:.4f}',flush=True)
        if score>best+1e-6:
            best=score;stale=0;best_epoch=epoch
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=cfg['patience']:break
    torch.save(state,checkpoint.with_suffix('.pt.tmp'));checkpoint.with_suffix('.pt.tmp').replace(checkpoint)
    save_json(infofile,dict(variant=model.variant,seed=seed,best_epoch=best_epoch,validation_macro_f1=best,
        seconds=time.time()-started,epochs_run=len(history),trainable_parameters=sum(p.numel() for p in model.parameters()),
        peak_cuda_allocated_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None))


def random_removal(meta,top,rng):
    """Same number and same node/edge composition as TOP."""
    chosen=[]
    for kind in ('node','edge'):
        n=sum(meta[i][0]==kind for i in top)
        pool=[i for i,m in enumerate(meta) if m[0]==kind]
        if n:chosen.extend(rng.choice(pool,size=n,replace=False).tolist())
    return chosen


@torch.no_grad()
def faithfulness(model,examples,store,records,cfg,seed,device,out):
    model.eval();rng=np.random.default_rng(seed)
    test=sorted((x for x in examples if x['role']=='test'),key=lambda x:x['id'])
    # Fixed random sample rather than first lexicographic IDs; independent of labels/predictions.
    indices=np.random.default_rng(cfg['split_seed']).permutation(len(test))[:cfg['diagnostic_examples']]
    rows=[];alignments=[];lookup={r['id']:r for r in records}
    for index in indices:
        ex=test[index];p=move(ex,device);q=store.get(ex['index']).to(device)
        logit,aux=model(p,q,True);original=float(logit.sigmoid().cpu());klass=int(original>=.5)
        n=len(aux['metadata'])
        if n<2:continue
        non_special=torch.tensor([not x for x in store.info['special'][ex['index']]],device=device)
        if not non_special.any():non_special=torch.ones(len(q),dtype=torch.bool,device=device)
        importance=aux['attention'][0,:,non_special,:].mean((0,1)).cpu().numpy()
        for label,count in [('top1',1),('top10pct',max(1,math.ceil(.1*n)))]:
            count=min(count,n-1);top=np.argsort(-importance,kind='stable')[:count].tolist()
            top_prob=float(model(p,q,drop_indices=top).sigmoid().cpu())
            random_probs=[]
            for _ in range(cfg['x23_random_repeats']):
                drop=random_removal(aux['metadata'],top,rng)
                random_probs.append(float(model(p,q,drop_indices=drop).sigmoid().cpu()))
            confidence=lambda v:v if klass else 1-v
            top_drop=confidence(original)-confidence(top_prob)
            random_drop=float(np.mean([confidence(original)-confidence(v) for v in random_probs]))
            r=lookup[ex['id']]
            descriptions=[dict(kind=kind,index=j,element=r['evidence_nodes'][j] if kind=='node'
                              else r['evidence_triples'][j]) for kind,j in (aux['metadata'][i] for i in top)]
            rows.append(dict(id=ex['id'],group_id=ex['group_id'],seed=seed,mode=label,label=ex['label'],
                predicted_class=klass,original_probability=original,top_probability=top_prob,
                random_probabilities=random_probs,drop_top=top_drop,drop_random=random_drop,
                fidelity_advantage=top_drop-random_drop,top_flip=int((top_prob>=.5)!=bool(klass)),
                random_flip_rate=float(np.mean([(v>=.5)!=bool(klass) for v in random_probs])),removed=descriptions))
        if len(alignments)<10:
            alignments.append(dict(id=ex['id'],claim=lookup[ex['id']]['claim'],query_tokens=store.info['tokens'][ex['index']],
                graph_tokens=[dict(kind=k,index=j,element=lookup[ex['id']]['evidence_nodes'][j] if k=='node'
                                   else lookup[ex['id']]['evidence_triples'][j]) for k,j in aux['metadata']],
                attention=aux['attention'][0].mean(0).cpu().tolist(),probability=original))
    summary=[]
    for mode in ('top1','top10pct'):
        subset=[r for r in rows if r['mode']==mode]
        if not subset:continue
        groups=sorted({r['group_id'] for r in subset})
        sums=np.array([sum(r['fidelity_advantage'] for r in subset if r['group_id']==g) for g in groups])
        counts=np.array([sum(r['group_id']==g for r in subset) for g in groups])
        draw=rng.integers(len(groups),size=(cfg['bootstrap_samples'],len(groups)))
        samples=sums[draw].sum(1)/counts[draw].sum(1)
        summary.append(dict(seed=seed,mode=mode,n=len(subset),groups=len(groups),
            mean_advantage=float(sums.sum()/counts.sum()),ci95=np.quantile(samples,[.025,.975]).tolist(),
            top_flip_rate=float(np.mean([r['top_flip'] for r in subset])),
            random_flip_rate=float(np.mean([r['random_flip_rate'] for r in subset]))))
    save_json(out/f'X23_b_interventions_{seed}.json',rows)
    save_json(out/f'X23_b_summary_{seed}.json',summary)
    save_json(out/f'X23_attention_{seed}.json',alignments)


def run_x23(data,store,records,cfg,device,out):
    result=[]
    for variant in VARIANTS:
        for seed in cfg['seeds']:
            seed_all(seed);model=make_model(data,cfg,variant).to(device)
            train(model,data['examples'],store,cfg,seed,device,out/'models')
            model.load_state_dict(torch.load(out/'models'/f'{variant}_{seed}.pt',map_location=device,weights_only=True))
            for role in ('validation','test'):
                examples=[x for x in data['examples'] if x['role']==role]
                for ex,prob in zip(examples,probabilities(model,examples,store,device)):
                    result.append(dict(id=ex['id'],group_id=ex['group_id'],num_hops=ex['num_hops'],label=ex['label'],
                        variant=variant,seed=seed,role=role,probability=float(prob),prediction=int(prob>=.5)))
            if variant=='X23-2':faithfulness(model,data['examples'],store,records,cfg,seed,device,out)
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
    return pd.DataFrame(result)
