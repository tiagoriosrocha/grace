"""One protocol and one report for all statistical/neural variants."""
import html
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from .experiment import metrics
from .linear_runs import validation_threshold
from .data import save_json

COMPARISONS=[('H2','H0'),('H2','H1'),('G19_REF','G19'),('X23-2','X23-0'),('X23-1','X23-0'),
             ('X23-2','X23-1'),('X23-2','G19_REF'),('X23-NLI','X23-2'),('G19','H2')]


def grouped_comparisons(frame,cfg):
    # Statistical models have one fit; do not duplicate it as fake seed runs.
    reference=frame.sort_values(['variant','seed','id']).groupby(['variant','seed'],sort=False)
    base=next(iter(reference))[1].sort_values('id')
    groups=sorted(base.group_id.unique());gids=pd.Categorical(base.group_id,categories=groups).codes
    group_sizes=np.bincount(gids,minlength=len(groups))
    cms={}
    for (variant,seed),part in reference:
        part=part.sort_values('id')
        if part.id.tolist()!=base.id.tolist() or part.label.tolist()!=base.label.tolist() or part.group_id.tolist()!=base.group_id.tolist():
            raise ValueError(f'Avaliação não pareada: {variant}/{seed}')
        cm=np.zeros((len(groups),2,2),dtype=np.float64)
        np.add.at(cm,(gids,part.label.to_numpy(int),part.prediction.to_numpy(int)),1)
        cms.setdefault(variant,[]).append(cm)
    def f1(cm):
        tp=np.diagonal(cm,axis1=-2,axis2=-1);denom=cm.sum(-1)+cm.sum(-2)
        return np.divide(2*tp,denom,out=np.zeros_like(tp),where=denom>0).mean(-1)
    rng=np.random.default_rng(cfg['split_seed']);samples={v:[] for v in cms}
    for start in range(0,cfg['bootstrap_samples'],128):
        draws=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),
                              size=min(128,cfg['bootstrap_samples']-start))
        for variant,cm in cms.items():
            sampled=np.einsum('bg,sgij->bsij',draws,np.asarray(cm))
            samples[variant].extend(f1(sampled).mean(1).tolist())
    results=[]
    for a,b in COMPARISONS:
        if a not in cms or b not in cms:continue
        delta=np.array(samples[a])-np.array(samples[b]);lo,hi=np.quantile(delta,[.025,.975])
        results.append(dict(comparison=f'{a} - {b}',delta_macro_f1=float(f1(np.array(cms[a]).sum(1)).mean()-f1(np.array(cms[b]).sum(1)).mean()),
            ci95_low=float(lo),ci95_high=float(hi),bootstrap_groups=len(groups),
            note='IC por grupos, condicional aos modelos treinados; sem incerteza de seleção.'))
    return results


def consolidate(out,cfg,linear,x23):
    neural=pd.concat([pd.read_csv(out/'predictions.csv').assign(role='test'),
                      pd.read_csv(out/'validation_predictions.csv').assign(role='validation')],ignore_index=True)
    all_predictions=pd.concat([neural,linear,x23],ignore_index=True)
    if all_predictions.duplicated(['variant','seed','role','id']).any():raise ValueError('Predições duplicadas')
    test=all_predictions[all_predictions.role=='test'].copy()
    val=all_predictions[all_predictions.role=='validation'].copy()
    test.to_csv(out/'predictions.csv',index=False);val.to_csv(out/'validation_predictions.csv',index=False)
    scores=[];tuned=[];hops=[];confusions=[];thresholds=[]
    for (variant,seed),part in test.groupby(['variant','seed'],sort=False):
        y=part.label.to_numpy();prob=part.probability.to_numpy()
        scores.append(dict(variant=variant,seed=seed,threshold=.5,n=len(part),**metrics(y,prob)))
        validation=val[(val.variant==variant)&(val.seed==seed)]
        threshold=validation_threshold(validation.label.to_numpy(),validation.probability.to_numpy())
        thresholds.append(dict(variant=variant,seed=int(seed),threshold=threshold,source='validation'))
        tuned.append(dict(variant=variant,seed=seed,threshold=threshold,n=len(part),**metrics(y,prob,threshold)))
        for hop,chunk in part.groupby('num_hops'):
            hops.append(dict(variant=variant,seed=seed,num_hops=hop,n=len(chunk),**metrics(chunk.label.to_numpy(),chunk.probability.to_numpy())))
        confusions.append(dict(variant=variant,seed=int(seed),matrix=confusion_matrix(y,prob>=.5,labels=[0,1]).tolist()))
    pd.DataFrame(scores).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(tuned).to_csv(out/'metrics_validation_threshold.csv',index=False)
    pd.DataFrame(hops).to_csv(out/'metrics_by_hops.csv',index=False)
    save_json(out/'confusion_matrices.json',confusions);save_json(out/'thresholds.json',thresholds)
    comparisons=grouped_comparisons(test,cfg);pd.DataFrame(comparisons).to_csv(out/'comparisons.csv',index=False)
    f=pd.DataFrame(scores)
    summary=f.groupby('variant',sort=False).agg(runs=('seed','count'),macro_f1_mean=('macro_f1','mean'),
        macro_f1_std=('macro_f1','std'),accuracy_mean=('accuracy','mean'),balanced_accuracy_mean=('balanced_accuracy','mean'),auc_mean=('auc','mean'))
    summary.to_csv(out/'summary.csv')
    histories=[];training=[]
    for path in sorted((out/'models').glob('history_*.csv')):
        variant,seed=path.stem.removeprefix('history_').rsplit('_',1)
        histories.append(pd.read_csv(path).assign(variant=variant,seed=int(seed)))
        info=out/'models'/f'{variant}_{seed}.json'
        if info.exists():training.append(json.loads(info.read_text()))
    if histories:pd.concat(histories,ignore_index=True).to_csv(out/'training_history.csv',index=False)
    save_json(out/'training_runs.json',training)
    lines=['# HoVer: comparação de todas as variantes','',
        '**SMOKE SINTÉTICO: sem validade científica.**' if cfg['smoke'] else 'Divisão interna agrupada do export train HoVer; não é o teste oficial.','',
        'Resultado principal: limiar 0,5 para todos. Checkpoints escolhidos na validação. Resultado secundário com limiar ajustado na validação: metrics_validation_threshold.csv.',
        'G19 e G19_REF têm um ajuste determinístico; o desvio entre sementes não se aplica a eles.','',
        '| Modelo | Ajustes | Macro-F1 | DP | Acurácia | Balanced accuracy | AUC |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for variant,row in summary.iterrows():
        std='—' if pd.isna(row.macro_f1_std) else f'{row.macro_f1_std:.4f}'
        lines.append(f'| {variant} | {int(row.runs)} | {row.macro_f1_mean:.4f} | {std} | {row.accuracy_mean:.4f} | {row.balanced_accuracy_mean:.4f} | {row.auc_mean:.4f} |')
    lines+=['','## Comparações pareadas','']
    for c in comparisons:
        lines.append(f"- {c['comparison']}: ΔF1={c['delta_macro_f1']:.4f}, IC95% [{c['ci95_low']:.4f}, {c['ci95_high']:.4f}].")
    lines+=['','IC condicionado aos modelos treinados. Comparações exploratórias; sem correção para múltiplos testes.',
        '', '## Fidelidade X23-b','',
        'X23-b remove tokens após a projeção, sem reescolher candidatos. Compara TOP com RANDOM de mesma quantidade e composição nó/aresta, usando a probabilidade da classe originalmente predita.',
        'Remover um token de nó não remove a informação desse nó nos tokens de arestas incidentes. O diagnóstico mede sensibilidade da representação; atenção não é uma prova factual.','']
    for seed in cfg['seeds']:
        for r in json.loads((out/f'X23_b_summary_{seed}.json').read_text()):
            lines.append(f"- Semente {seed}, {r['mode']}, n={r['n']}: vantagem TOP−RANDOM={r['mean_advantage']:.4f}, IC95% {r['ci95']}.")
    lines+=['','## Interpretação e arquivos','',
        '- G19: baseline anterior de 29 características achatadas; G19_REF: matching e 31 características do notebook 19, adaptados a BERT/HoVer.',
        '- X23-0: pooling independente; X23-1: atenção a nós; X23-2: atenção a nós+arestas; X23-NLI: X23-2 + as 31 características G19_REF normalizadas no treino.',
        '- X23 usa claim textual; H/G19 usam grafos. Uma vantagem X23 não isola apenas o mecanismo de atenção.',
        '- X23-2 versus X23-0 muda interação e pooling; X23-2 versus X23-1 acrescenta arestas. Nenhum X23 usa RGAT ou busca explícita de caminhos.',
        '- Nenhuma variante adiciona arestas reversas. O token de aresta concatena sujeito, relação e objeto nessa ordem.',
        '- models/: checkpoints, históricos e modelos estatísticos. G19*_contributions.csv: contribuição exata por característica ao logit; não é causalidade.',
        '- G19_REF_alignments.jsonl.gz: candidatos e NLI por condição. X23_attention_*.json: atenção por token para exemplos.',
        '- training_history.csv e training_runs.json: históricos, épocas escolhidas e recursos, também fora da pasta models/.',
        '- X23_b_interventions_*.json: remoções e probabilidades, incluindo todas as repetições aleatórias.',
        '- Cache BERT em claim_tokens/hidden.npy: leitura por exemplo; não faz parte dos parâmetros treinados.']
    (out/'RESULTADOS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'RESULTADOS.html').write_text('<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HoVer</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:16px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}pre{white-space:pre-wrap}</style>'+summary.round(4).to_html(na_rep='—')+'<pre>'+html.escape('\n'.join(lines))+'</pre></html>',encoding='utf-8')
