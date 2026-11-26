"""Group bootstrap of the mean metric across fixed training seeds."""
from itertools import combinations

import numpy as np
import pandas as pd

from hover_exp.data import save_json
from hover_exp.experiment import metrics

MODEL_ORDER = ['majority_baseline', 'text_tfidf', 'graph_pair_transformer',
               'matching_nli', 'kgat_style', 'gmn_style', 'relational_gnn', 'claim_evidence_attention',
               'local_factcheck_pipeline']


def macro_f1_from_counts(counts):
    # Last dimension: TN, FP, FN, TP. Fixed two-class macro averaging.
    tn, fp, fn, tp = np.moveaxis(np.asarray(counts, dtype=float), -1, 0)
    den0, den1 = 2*tn+fp+fn, 2*tp+fp+fn
    f0 = np.divide(2*tn, den0, out=np.zeros_like(tn), where=den0 != 0)
    f1 = np.divide(2*tp, den1, out=np.zeros_like(tp), where=den1 != 0)
    return (f0+f1)/2


def group_bootstrap(predictions, seeds, samples=2000, random_state=9001):
    """Same sampled groups for every model/seed; no seed treated as a new subject."""
    models = [m for m in MODEL_ORDER if m in set(predictions.model)]
    base = predictions[(predictions.model == models[0]) & (predictions.seed == seeds[0])].sort_values('id')
    if base.empty or base.id.duplicated().any():
        raise ValueError('IDs ausentes/duplicados nas predições.')
    groups, gi = np.unique(base.group_id.to_numpy(), return_inverse=True)
    counts = np.zeros((len(models), len(seeds), len(groups), 4), dtype=np.int64)
    for mi, model in enumerate(models):
        for si, seed in enumerate(seeds):
            part = predictions[(predictions.model == model) & (predictions.seed == seed)].sort_values('id')
            if not part[['id','group_id','label']].reset_index(drop=True).equals(base[['id','group_id','label']].reset_index(drop=True)):
                raise ValueError(f'Predições não pareadas: {model}/{seed}')
            prob = part.probability.to_numpy()
            if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
                raise ValueError('Probabilidades inválidas.')
            cell = 2*part.label.to_numpy(dtype=int)+(prob >= .5).astype(int)
            np.add.at(counts[mi,si], (gi,cell), 1)
    rng = np.random.default_rng(random_state)
    values = np.empty((samples, len(models)))
    for b in range(samples):
        draw = rng.integers(len(groups), size=len(groups))
        values[b] = macro_f1_from_counts(counts[:,:,draw].sum(axis=2)).mean(axis=1)
    observed = macro_f1_from_counts(counts.sum(axis=2)).mean(axis=1)
    intervals = {m: np.quantile(values[:,i], [.025,.975]).tolist() for i,m in enumerate(models)}
    comparisons = []
    for a,b in combinations(range(len(models)), 2):
        low,high = np.quantile(values[:,a]-values[:,b], [.025,.975])
        comparisons.append(dict(model_a=models[a], model_b=models[b],
            macro_f1_difference=float(observed[a]-observed[b]), ci95_low=float(low), ci95_high=float(high)))
    return intervals, comparisons


def write_reports(out, cfg, frame, metric_rows, prediction_rows, audit, manifest):
    per_seed = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    per_seed.to_csv(out/'metrics.csv', index=False)
    predictions.to_csv(out/'predictions.csv', index=False)
    intervals, comparisons = group_bootstrap(predictions, cfg['seeds'], cfg['bootstrap_samples'], cfg['split_seed']+9001)
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv', index=False)
    rows = []
    matrices = {}
    selected_models = [model for model in MODEL_ORDER if model in set(predictions.model)]
    for model in selected_models:
        sub = per_seed[per_seed.model == model]
        if len(sub) != len(cfg['seeds']):
            raise ValueError(f'Comparação incompleta: {model}')
        row = dict(model=model, seeds=len(sub), test_examples=int((frame.role=='test').sum()))
        for name in ('accuracy','balanced_accuracy','macro_f1','auc','brier'):
            row[name+'_mean'] = float(sub[name].mean())
            row[name+'_std'] = float(sub[name].std(ddof=1)) if len(sub)>1 else 0.
        row['macro_f1_ci95_low'],row['macro_f1_ci95_high'] = intervals[model]
        rows.append(row)
        matrices[model] = {}
        for seed, part in predictions[predictions.model == model].groupby('seed'):
            cell = 2*part.label.to_numpy(dtype=int)+(part.probability.to_numpy()>=.5).astype(int)
            matrices[model][str(seed)] = np.bincount(cell,minlength=4).reshape(2,2).tolist()
    summary = pd.DataFrame(rows)
    summary.to_csv(out/'summary.csv',index=False)
    # A smoke run must never produce an artifact named article_table.
    stem = 'smoke_table' if cfg['smoke'] else 'article_table'
    summary.to_csv(out/f'{stem}.csv',index=False)
    table = pd.DataFrame({'Model': summary.model,
        'Accuracy': [f'{m:.4f} ± {s:.4f}' for m,s in zip(summary.accuracy_mean,summary.accuracy_std)],
        'Macro-F1': [f'{m:.4f} ± {s:.4f}' for m,s in zip(summary.macro_f1_mean,summary.macro_f1_std)],
        'Macro-F1 95% CI': [f'[{a:.4f}, {b:.4f}]' for a,b in zip(summary.macro_f1_ci95_low,summary.macro_f1_ci95_high)],
        'AUC': summary.auc_mean.map(lambda x:f'{x:.4f}')})
    (out/f'{stem}.tex').write_text(table.to_latex(index=False,escape=True),encoding='utf-8')
    save_json(out/'confusion_matrices.json',matrices)
    save_json(out/'audit.json',audit)
    manifest.update(status='complete', scientific_result=not cfg['smoke'], models=selected_models,
                    inference='cluster percentile bootstrap, mean Macro-F1 across fixed seeds; paired pointwise CIs, no multiplicity adjustment')
    save_json(out/'manifest.json',manifest)
    title = 'SMOKE SINTÉTICO — sem validade científica' if cfg['smoke'] else 'Comparação de sistemas — evidência fornecida'
    text = f'''# {title}

Dataset: {cfg['dataset']}. Protocolo: {cfg['protocol_version']}.
Claims duplicadas, grupos de origem e grafos de evidência repetidos ficam no mesmo conjunto.
Entidades podem aparecer em conjuntos diferentes; esta avaliação admite entidades conhecidas.
Divisão interna do export convertido; não equivale ao teste oficial do benchmark.

{table.to_markdown(index=False)}

Média ± desvio amostral entre sementes. IC95%: bootstrap por grupos completos da média
do Macro-F1 entre sementes fixas. As sementes não são novas observações de teste.
paired_comparisons.csv contém diferenças pareadas com intervalos pontuais, sem correção
por comparações múltiplas; não fornece uma declaração automática de superioridade.
Matrizes de confusão são separadas por semente.

## Cobertura e sobreposição

{pd.DataFrame(audit['roles']).to_markdown(index=False)}

{pd.DataFrame(audit['coverage_by_label']).T.to_markdown()}

Grupos: {audit['split_groups']['groups']}; maior grupo: {audit['split_groups']['largest_group']}.
Grafos repetidos entre conjuntos: {audit['split_groups']['cross_role_evidence_groups']}.
Claims duplicadas entre conjuntos: {audit['split_groups']['cross_role_claims']}.
Entidades textuais compartilhadas entre conjuntos (permitidas): {audit['split_groups']['cross_role_entities']}.

## Escopo da comparação

text_tfidf usa claim textual + triplas verbalizadas; graph_pair_transformer usa os dois
grafos serializados; matching_nli e relational_gnn usam os dois grafos e NLI congelado;
claim_evidence_attention usa claim textual e tokens do grafo de evidência.
local_factcheck_pipeline extrai condições, recupera top-k evidências, aplica NLI
congelado e usa regressão logística ajustada no treino; C e o limiar são escolhidos
na validação. A auditoria e as métricas usam o mesmo escore global.
Os sistemas compartilham exemplos e split, mas diferem em entradas, truncamento e supervisão
prévia. A comparação não isola causalmente o efeito da arquitetura.
Embeddings congelados são compartilháveis; vocabulários, escalas e classificadores são
ajustados somente no treino. Features explícitas exact estão desativadas.
exact_match_diagnostic está em metrics.csv, fora da comparação aprendida.

A origem da conversão dos grafos precisa ser documentada no artigo. Este código não
certifica que a geração anterior dos grafos desconhecia os rótulos. A seleção de modelos
com resultados anteriores destes dados também deve ser declarada como exploração prévia.
'''
    (out/'RESULTADOS.md').write_text(text,encoding='utf-8')
