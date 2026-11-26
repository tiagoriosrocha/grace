"""Treino, avaliação pareada, intervenções e relatório autocontido."""
import copy
import importlib.metadata
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from .data import save_json,build_graph,candidate_features
from .model import Verifier, move, VARIANTS


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def metrics(y,prob,threshold=.5):
    prob=np.asarray(prob)
    pred=prob>=threshold
    p,r,f,_=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
    result=dict(macro_f1=float(f.mean()),accuracy=float(accuracy_score(y,pred)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                brier=float(np.mean((np.asarray(y)-prob)**2)))
    for i,name in enumerate(('not_supported','supported')):
        result.update({f'{name}_precision':float(p[i]),f'{name}_recall':float(r[i]),f'{name}_f1':float(f[i])})
    result['auc']=float(roc_auc_score(y,prob)) if len(set(y))==2 else None
    return result


@torch.no_grad()
def predict(model,pairs,device):
    model.eval()
    values=[]
    for pair in pairs:
        values.append(float(model(move(pair,device)).sigmoid().cpu()))
    return np.asarray(values)


def train_one(bundle,cfg,variant,seed,device,out):
    checkpoint=out/f'{variant}_{seed}.pt'
    info_file=out/f'{variant}_{seed}.json'
    if checkpoint.exists() and info_file.exists():
        print(f'Reutilizando treino concluído: {variant}/{seed}',flush=True)
        return json.loads(info_file.read_text())
    seed_all(seed)
    model=Verifier(bundle['embeddings'],len(bundle['relation_vocab']),cfg,variant).to(device)
    # No variant-specific freezing: ablations must differ only in the
    # contextual encoder and the final aggregation rule.
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    train=[p for p in bundle['pairs'] if p['role']=='train']
    val=[p for p in bundle['pairs'] if p['role']=='validation']
    history=[]
    best=-1.
    stale=0
    best_state=None
    start=time.time()
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,cfg['epochs']+1):
        model.train()
        order=np.random.default_rng(seed+epoch*1009).permutation(len(train))
        total=0.
        for start_idx in range(0,len(order),cfg['batch_size']):
            indices=order[start_idx:start_idx+cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            # Sequential backward: at most one pair's activations on the GPU.
            loss_total=torch.zeros((),device=device)
            for i in indices:
                pair=move(train[int(i)],device)
                logit=model(pair)
                target=logit.new_tensor(float(pair['label']))
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,target)
                (loss/len(indices)).backward()
                loss_total+=loss.detach()
            torch.nn.utils.clip_grad_norm_(params,1.)
            optimizer.step()
            total+=float(loss_total.cpu())
        probabilities=predict(model,val,device)
        score=metrics([p['label'] for p in val],probabilities)['macro_f1']
        if not math.isfinite(total):
            raise RuntimeError(f'Loss não finita: {variant}/{seed}')
        history.append(dict(epoch=epoch,train_loss=total/len(train),validation_macro_f1=score))
        pd.DataFrame(history).to_csv(out/f'history_{variant}_{seed}.csv',index=False)
        print(f'{variant} seed={seed} época={epoch:02d} loss={total/len(train):.4f} val_F1={score:.4f}',flush=True)
        if score>best+1e-6:
            best=score
            best_epoch=epoch
            best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            stale=0
        else:
            stale+=1
        if stale>=cfg['patience']:
            break
    # Persist completion only after best state is durably stored.
    torch.save(best_state,checkpoint)
    info=dict(variant=variant,seed=seed,best_epoch=best_epoch,validation_macro_f1=best,
              epochs_run=len(history),seconds=time.time()-start,trainable_parameters=sum(p.numel() for p in params),
              peak_cuda_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None)
    save_json(info_file,info)
    del model,optimizer
    if device.type=='cuda':
        torch.cuda.empty_cache()
    return info


def paired_bootstrap(predictions,cfg):
    # Resample groups with all their examples; average metric differences over fixed seeds.
    base=predictions[(predictions.variant=='H2') & (predictions.seed==cfg['seeds'][0])].sort_values('id')
    groups=base.group_id.to_numpy()
    group_names=np.unique(groups)
    indices=[np.flatnonzero(groups==g) for g in group_names]
    y=base.label.to_numpy()
    rows=[]
    def macro_from_confusion(cm):
        tp=cm[..., [0,1],[0,1]]
        denom=cm.sum(-1)+cm.sum(-2)
        return np.divide(2*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom>0).mean(-1)
    rng=np.random.default_rng(cfg['split_seed'])
    # Same group draws for both comparisons.
    draws=rng.multinomial(len(group_names),np.full(len(group_names),1/len(group_names)),size=cfg['bootstrap_samples'])
    for baseline in ('H0','H1'):
        cms=[]
        points=[]
        for seed in cfg['seeds']:
            pair=[]
            for variant in ('H2',baseline):
                frame=predictions[(predictions.variant==variant)&(predictions.seed==seed)].sort_values('id')
                if frame.id.tolist()!=base.id.tolist() or frame.label.tolist()!=base.label.tolist():
                    raise ValueError('Predições não pareadas')
                pred=frame.prediction.to_numpy()
                pair.append(np.stack([confusion_matrix(y[idx],pred[idx],labels=[0,1]) for idx in indices]))
            cms.append(pair)
            points.append(macro_from_confusion(pair[0].sum(0))-macro_from_confusion(pair[1].sum(0)))
        # [seed,variant,group,true,pred] -> [bootstrap,seed,variant,true,pred]
        sampled=np.einsum('bg,svgij->bsvij',draws,np.asarray(cms))
        scores=macro_from_confusion(sampled)
        deltas=(scores[:,:,0]-scores[:,:,1]).mean(1)
        lo,hi=np.quantile(deltas,[.025,.975])
        rows.append(dict(comparison=f'H2 - {baseline}',delta_macro_f1=float(np.mean(points)),
                         ci95_low=float(lo),ci95_high=float(hi),bootstrap_groups=len(group_names),
                         seed_differences_std=float(np.std(points,ddof=1)) if len(points)>1 else 0.,
                         note='IC por grupos, condicional às sementes fixadas; não incorpora incerteza de seleção.'))
    return rows


def empty_evidence(pair):
    p=copy.deepcopy(pair)
    g=p['evidence']
    for key in ('node_text','edge_text','edge_type','factual'):
        g[key]=g[key][:0]
    g['edge_index']=g['edge_index'][:,:0]
    g['structure']=g['structure'][:0]
    p['candidates']=p['candidates'][:,:0]
    p['signals']=p['signals'][:,:0,:]
    return p



@torch.no_grad()
def diagnostics(model,pairs,device,cfg,records,bundle,out):
    """No invented gold labels. Changes measure sensitivity, not correctness."""
    result=[]
    by_id={r['id']:r for r in records}
    texts=json.loads((out/'text_index.json').read_text())
    text_ids={s:i for i,s in enumerate(texts)}
    normalization=json.loads((out/'normalization.json').read_text())
    mean=torch.tensor(normalization['mean']); std=torch.tensor(normalization['std'])
    rng=np.random.default_rng(cfg['split_seed'])
    from .nli import NLIScorer,marked_triple
    scorer=NLIScorer(cfg,device)
    for pair in pairs:
        row=by_id[pair['id']]
        for i,edges in enumerate(pair['candidates'].tolist()):
            hypothesis=marked_triple(row['claim_triples'][pair['claim_factual'][i]])
            for col,edge in enumerate(edges):
                scorer.cache[(marked_triple(row['evidence_triples'][edge]),hypothesis)]=pair['signals'][i,col,6:9].cpu()
    def remove_edge(pair,edge):
        row=by_id[pair['id']]
        ct=row['claim_triples']
        et=[t for j,t in enumerate(row['evidence_triples']) if j!=edge]
        if not et:
            return empty_evidence(pair)
        p=copy.deepcopy(pair)
        g=build_graph(et,text_ids,bundle['relation_vocab'])
        candidates,signals=candidate_features(ct,et,p['claim'],g,bundle['embeddings'],cfg['top_k'])
        g['structure']=(g['structure']-mean)/std
        # Cache by the actual premise/hypothesis, not mutable edge positions.
        # Surviving pairs keep their NLI; newly retrieved pairs are scored.
        queries=[(marked_triple(et[j]),marked_triple(ct[pair['claim_factual'][i]]))
                 for i,edges in enumerate(candidates.tolist()) for j in edges]
        values=scorer.score(queries).reshape(*signals.shape[:2],3)
        signals=torch.cat([signals,values],-1)
        p.update(evidence=g,candidates=candidates,signals=signals)
        return p
    for pair in sorted(pairs,key=lambda p:p['id'])[:cfg['diagnostic_examples']]:
        p=move(pair,device)
        original,aux=model(p,return_aux=True)
        empty=model(move(empty_evidence(pair),device))
        factual=pair['evidence']['factual'].tolist()
        top_probability=random_probability=None
        top_edge=random_edge=None
        if factual and pair['candidates'].numel():
            mass=torch.zeros(pair['evidence']['edge_index'].shape[1])
            mass.scatter_add_(0,pair['candidates'].flatten(),aux['weights'].cpu().flatten())
            top_edge=max(factual,key=lambda j:float(mass[j]))
            random_edge=int(rng.choice(factual))
            top_probability=float(model(move(remove_edge(pair,top_edge),device)).sigmoid().cpu())
            random_probability=float(model(move(remove_edge(pair,random_edge),device)).sigmoid().cpu())
        result.append(dict(id=pair['id'],label=pair['label'],original_probability=float(original.sigmoid().cpu()),
                           empty_evidence_probability=float(empty.sigmoid().cpu()),
                           drop_empty=float((original.sigmoid()-empty.sigmoid()).cpu()),
                           removed_top_edge=top_edge,removed_random_edge=random_edge,
                           top_removed_probability=top_probability,random_removed_probability=random_probability,
                           local_support=aux['local_support'].cpu().tolist(),
                           null_mass=aux['null_mass'].cpu().tolist(),
                           consistency=float(aux['consistency'].cpu())))
    # Save a few aligned examples with actual triples, not just indices.
    examples=[]
    for pair in sorted(pairs,key=lambda p:p['id'])[:10]:
        prob,aux=model(move(pair,device),return_aux=True)
        r=by_id[pair['id']]
        entries=[]
        for row,claim_idx in enumerate(pair['claim']['factual'].tolist()):
            weights=aux['weights'][row].cpu()
            candidates=pair['candidates'][row].tolist()
            entries.append(dict(claim_triple=r['claim_triples'][claim_idx],
                                latent_support=float(aux['local_support'][row].cpu()),
                                null_mass=float(aux['null_mass'][row].cpu()),
                                candidates=[dict(triple=r['evidence_triples'][j],weight=float(w)) for j,w in zip(candidates,weights)]))
        examples.append(dict(id=pair['id'],claim=r['claim'],gold=pair['label'],probability=float(prob.sigmoid().cpu()),conditions=entries))
    scorer.close()
    return result,examples


def evaluate(bundle,cfg,device,out,records):
    test=[p for p in bundle['pairs'] if p['role']=='test']
    predictions=[]
    validation_predictions=[]
    scores=[]
    for variant in VARIANTS:
        for seed in cfg['seeds']:
            model=Verifier(bundle['embeddings'],len(bundle['relation_vocab']),cfg,variant).to(device)
            model.load_state_dict(torch.load(out/'models'/f'{variant}_{seed}.pt',map_location=device,weights_only=True))
            validation=[p for p in bundle['pairs'] if p['role']=='validation']
            for p,prob in zip(validation,predict(model,validation,device)):
                validation_predictions.append(dict(id=p['id'],group_id=p['group_id'],num_hops=p['num_hops'],
                    label=p['label'],variant=variant,seed=seed,probability=float(prob),prediction=int(prob>=.5)))
            probs=predict(model,test,device)
            score=dict(variant=variant,seed=seed,**metrics([p['label'] for p in test],probs))
            scores.append(score)
            for p,prob in zip(test,probs):
                predictions.append(dict(id=p['id'],group_id=p['group_id'],num_hops=p['num_hops'],label=p['label'],
                                        variant=variant,seed=seed,probability=float(prob),prediction=int(prob>=.5)))
            if variant=='H2':
                diag,examples=diagnostics(model,test,device,cfg,records,bundle,out)
                save_json(out/f'diagnostics_{seed}.json',diag)
                save_json(out/f'alignments_{seed}.json',examples)
            del model
            if device.type=='cuda':
                torch.cuda.empty_cache()
    frame=pd.DataFrame(predictions)
    pd.DataFrame(validation_predictions).to_csv(out/'validation_predictions.csv',index=False)
    frame.to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'metrics.csv',index=False)
    comparisons=paired_bootstrap(frame,cfg)
    pd.DataFrame(comparisons).to_csv(out/'comparisons.csv',index=False)
    by_hops=[]
    for (variant,seed,hops),g in frame.groupby(['variant','seed','num_hops']):
        by_hops.append(dict(variant=variant,seed=seed,num_hops=hops,n=len(g),**metrics(g.label.to_numpy(),g.probability.to_numpy())))
    pd.DataFrame(by_hops).to_csv(out/'metrics_by_hops.csv',index=False)
    confusion=[]
    for (variant,seed),g in frame.groupby(['variant','seed']):
        cm=confusion_matrix(g.label,g.prediction,labels=[0,1])
        confusion.append(dict(variant=variant,seed=int(seed),matrix=cm.tolist()))
    save_json(out/'confusion_matrices.json',confusion)
    majority=int(np.mean([p['label'] for p in bundle['pairs'] if p['role']=='train'])>=.5)
    save_json(out/'majority_baseline.json',metrics([p['label'] for p in test],np.full(len(test),majority,dtype=float)))
    report(out,cfg,scores,comparisons)


def report(out,cfg,scores,comparisons):
    f=pd.DataFrame(scores)
    summary=f.groupby('variant',sort=False).agg(macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std'),
                                              accuracy_mean=('accuracy','mean'),balanced_accuracy_mean=('balanced_accuracy','mean'))
    summary=summary.fillna(0.)
    summary.to_csv(out/'summary.csv')
    lines=['# Resultado: experimento HoVer — alternativa 3','',
           '**TESTE SINTÉTICO: sem validade científica.**' if cfg['smoke'] else 'Divisão interna do export de treino HoVer; não é o teste oficial.', '',
           'Limiar fixo 0,5. Checkpoints escolhidos somente por macro-F1 de validação.', '',
           '| Modelo | Macro-F1 médio | DP entre sementes | Acurácia | Balanced accuracy |',
           '|---|---:|---:|---:|---:|']
    for name,row in summary.iterrows():
        lines.append(f'| {name} | {row.macro_f1_mean:.4f} | {row.macro_f1_std:.4f} | {row.accuracy_mean:.4f} | {row.balanced_accuracy_mean:.4f} |')
    lines+=['','## Comparações pareadas','']
    for c in comparisons:
        verdict='sinal favorável' if c['ci95_low']>0 else 'sinal desfavorável' if c['ci95_high']<0 else 'inconclusivo: intervalo inclui zero'
        lines.append(f"- {c['comparison']}: ΔF1={c['delta_macro_f1']:.4f}; IC95% [{c['ci95_low']:.4f}, {c['ci95_high']:.4f}]; {verdict}.")
    lines+=['','IC por grupos, condicional às sementes fixadas. Esta comparação avalia o pacote de mudanças; não isola causalmente cada componente.',
            '', '## Sensibilidade à evidência', '']
    for seed in cfg['seeds']:
        d=json.loads((out/f'diagnostics_{seed}.json').read_text())
        lines.append(f"- Semente {seed}: queda média de P(suporte) com evidência vazia = {np.mean([r['drop_empty'] for r in d]):.4f} ({len(d)} pares).")
        valid=[r for r in d if r['top_removed_probability'] is not None]
        if valid:
            top=np.mean([r['original_probability']-r['top_removed_probability'] for r in valid])
            random_drop=np.mean([r['original_probability']-r['random_removed_probability'] for r in valid])
            lines.append(f'  Deleção de uma aresta: queda top-atenção={top:.4f}; aleatória={random_drop:.4f}. Candidatas e graus recalculados após cada intervenção.')
    lines+=['','Remover toda a evidência é um diagnóstico extremo fora da distribuição, não prova de raciocínio ou de explicação fiel.',
            'As saídas locais são latentes: HoVer fornece rótulo global binário, sem supervisão de alinhamento/conflito local.',
            'Triplas são aproximações das condições da claim. Negação, tempo e quantificadores perdidos no export não são recuperados.',
            'Não há busca explícita de caminhos: duas camadas RGAT usam somente a direção original e fornecem contexto limitado. Resultados por 2/3/4 hops estão em metrics_by_hops.csv.',
            'H0 é matching + NLI sem contexto RGAT; H1 acrescenta RGAT; H2 acrescenta agregação Coverage.',
            '', '## Arquivos', '',
            '- predictions.csv: probabilidades por exemplo, modelo e semente.',
            '- comparisons.csv, metrics.csv, summary.csv, confusion_matrices.json: métricas e comparação.',
            '- alignments_*.json: exemplos de correspondências e massas nulas.',
            '- models/: checkpoints, histórico e tempo/memória de treinamento.',
            '- audit.json e excluded.csv: cobertura dos dados e exclusões.']
    (out/'RESULTADOS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Self-contained, responsive HTML table for opening without a notebook/server.
    import html
    content='\n'.join('<p>'+html.escape(line)+'</p>' for line in lines if line)
    table=summary.round(4).to_html()
    (out/'RESULTADOS.html').write_text('<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HoVer — resultados</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#183142}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #ccd}th{background:#edf4f7}p{line-height:1.5}</style><h1>HoVer — alternativa 3</h1>'+table+content+'</html>',encoding='utf-8')
