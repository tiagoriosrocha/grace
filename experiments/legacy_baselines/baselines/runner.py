from __future__ import annotations

import json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from .data import feature_matrix, triple_key


def metrics(y, prob):
    pred = np.asarray(prob) >= .5
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {"macro_f1": float(f.mean()), "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "not_supported_f1": float(f[0]), "supported_f1": float(f[1]), "auc": float(roc_auc_score(y, prob)) if len(set(y)) == 2 else None}


def fit_sklearn(train, test, labels, seed, texts):
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=1), LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"))
    pipe.fit(texts[train], labels[train])
    prob = pipe.predict_proba(texts[test])[:, 1]
    return prob


def fit_features(train, test, seed, df):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
    x = feature_matrix(df)
    model.fit(x[train], df.label.to_numpy()[train])
    return model.predict_proba(x[test])[:, 1]


def exact_rule(test, df):
    out=[]
    for i in test:
        ck={triple_key(x) for x in df.iloc[i].claim_triples}; ek={triple_key(x) for x in df.iloc[i].evidence_triples}
        out.append(1.0 if ck <= ek else 0.0)
    return np.asarray(out)


def transformer(train, val, test, df, cfg, seed, mode, device, output_dir=None):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.manual_seed(seed); np.random.seed(seed)
    opts={"revision": cfg["transformer_revision"], "local_files_only": cfg["local_files_only"]}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
    if cfg.get('smoke',False):
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
        if output_dir is None: raise ValueError('Smoke requer output_dir')
        vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','person','city','organization','born','in','works','at','located','other','institution']
        vocab_path=output_dir/'vocab.txt'
        vocab_path.write_text('\n'.join(vocab)+'\n',encoding='utf-8')
        tok=BertTokenizerFast(vocab_file=str(vocab_path))
        model=BertForSequenceClassification(BertConfig(vocab_size=len(vocab),hidden_size=32,
                    num_hidden_layers=1,num_attention_heads=2,intermediate_size=64,num_labels=2)).to(device)
    else:
        tok=AutoTokenizer.from_pretrained(cfg["transformer_model"], **opts)
        model=AutoModelForSequenceClassification.from_pretrained(cfg["transformer_model"], num_labels=2, use_safetensors=True, **opts).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=cfg["transformer_learning_rate"])
    def text(ix):
        rows=df.iloc[ix]
        if mode == "claim": return rows.claim.tolist()
        if mode == "text": return [a+" [EVIDENCE] "+b for a,b in zip(rows.claim, rows.evidence)]
        if mode == "graph": return [a+" [GRAPH] "+b for a,b in zip(rows.claim, rows.graph_evidence)]
        return ["[CLAIM_GRAPH] "+a+" [EVIDENCE_GRAPH] "+b for a,b in zip(rows.graph_claim, rows.graph_evidence)]
    def batch(ix):
        return tok(text(ix), padding=True, truncation=True, max_length=cfg["transformer_max_length"], return_tensors="pt").to(device)
    def predict(ix):
        model.eval(); out=[]
        with torch.no_grad():
            for start in range(0,len(ix),cfg["transformer_batch_size"]):
                out.append(model(**batch(ix[start:start+cfg["transformer_batch_size"]])).logits.softmax(-1)[:,1].cpu().numpy())
        return np.concatenate(out)
    best=-1.; best_state=None; stale=0; rng=np.random.default_rng(seed)
    history=[]
    for _epoch in range(cfg["transformer_epochs"]):
        model.train(); order=rng.permutation(train)
        for start in range(0,len(order),cfg["transformer_batch_size"]):
            ix=order[start:start+cfg["transformer_batch_size"]]; opt.zero_grad(set_to_none=True)
            labels=torch.tensor(df.label.to_numpy()[ix],device=device)
            loss=model(**batch(ix), labels=labels).loss; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        score=metrics(df.label.to_numpy()[val],predict(val))["macro_f1"]
        history.append(dict(epoch=_epoch+1,validation_macro_f1=score))
        print(f'graph_pair_transformer/{seed} época={_epoch+1} val_F1={score:.4f}',flush=True)
        if score>best:
            best=score; stale=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale += 1
        if stale >= cfg["transformer_patience"]: break
    model.load_state_dict(best_state)
    if output_dir is not None:
        model.save_pretrained(output_dir, safe_serialization=True)
        tok.save_pretrained(output_dir)
        pd.DataFrame(history).to_csv(output_dir/'history.csv',index=False)
        (output_dir/'training.json').write_text(json.dumps(dict(seed=seed,validation_macro_f1=best,
            resolved_revision=getattr(model.config,'_commit_hash',None),smoke=cfg.get('smoke',False))),encoding='utf-8')
        truncation={}
        for name,indices in [('train',train),('validation',val),('test',test)]:
            lengths=[len(x) for x in tok(text(indices),truncation=False)['input_ids']]
            truncation[name]=dict(examples=len(lengths),truncated=sum(n>cfg['transformer_max_length'] for n in lengths))
        (output_dir/'truncation.json').write_text(json.dumps(truncation),encoding='utf-8')
    result=predict(test); del model, opt
    if device.type == "cuda": torch.cuda.empty_cache()
    return result, best


def run(df, cfg, out, skip_transformers=False):
    out=Path(out); out.mkdir(parents=True, exist_ok=True)
    train=np.flatnonzero(df.role.to_numpy()=="train"); val=np.flatnonzero(df.role.to_numpy()=="validation"); test=np.flatnonzero(df.role.to_numpy()=="test")
    rows=[]; costs=[]; prediction_rows=[]; y=df.label.to_numpy()
    names={"claim_tfidf":df.claim, "text_tfidf":df.claim+" [EVIDENCE] "+df.evidence, "graph_tfidf":df.claim+" [GRAPH] "+df.graph_evidence}
    majority=float(df.iloc[train].label.mean()>=.5)
    for seed in cfg["seeds"]:
        for name,prob in [("Majority",np.full(len(test),majority)),("ExactTriple",exact_rule(test,df))]:
            t=time.perf_counter(); m=metrics(y[test],prob); rows.append(dict(model=name,seed=seed,split="test",**m)); costs.append(dict(model=name,seed=seed,seconds=time.perf_counter()-t))
            prediction_rows.extend(dict(id=df.iloc[i].id,group_id=df.iloc[i].group_id,label=int(df.iloc[i].label),model=name,seed=seed,probability=float(prob[j])) for j,i in enumerate(test))
        for name,texts in names.items():
            t=time.perf_counter(); prob=fit_sklearn(train,test,y,seed,texts); public=name.title().replace('_',''); m=metrics(y[test],prob); rows.append(dict(model=public,seed=seed,split="test",**m)); costs.append(dict(model=name,seed=seed,seconds=time.perf_counter()-t)); prediction_rows.extend(dict(id=df.iloc[i].id,group_id=df.iloc[i].group_id,label=int(df.iloc[i].label),model=public,seed=seed,probability=float(prob[j])) for j,i in enumerate(test))
        t=time.perf_counter(); prob=fit_features(train,test,seed,df); m=metrics(y[test],prob); rows.append(dict(model="GraphFeatures",seed=seed,split="test",**m)); costs.append(dict(model="GraphFeatures",seed=seed,seconds=time.perf_counter()-t)); prediction_rows.extend(dict(id=df.iloc[i].id,group_id=df.iloc[i].group_id,label=int(df.iloc[i].label),model="GraphFeatures",seed=seed,probability=float(prob[j])) for j,i in enumerate(test))
        if not skip_transformers:
            device=torch.device(cfg["device"] if cfg["device"] == "cpu" or torch.cuda.is_available() else "cpu")
            for mode, name in (("claim","ClaimTransformer"),("text","TextTransformer"),("graph","GraphTransformer"),("graph_pair","GraphPairTransformer")):
                t=time.perf_counter(); prob,best=transformer(train,val,test,df,cfg,seed,mode,device); m=metrics(y[test],prob); rows.append(dict(model=name,seed=seed,split="test",validation_macro_f1=best,**m)); costs.append(dict(model=name,seed=seed,seconds=time.perf_counter()-t,device=str(device))); prediction_rows.extend(dict(id=df.iloc[i].id,group_id=df.iloc[i].group_id,label=int(df.iloc[i].label),model=name,seed=seed,probability=float(prob[j])) for j,i in enumerate(test))
    metrics_df=pd.DataFrame(rows); metrics_df.to_csv(out/"metrics.csv",index=False); pd.DataFrame(costs).to_csv(out/"costs.csv",index=False)
    summary=metrics_df.groupby("model",sort=False).agg(macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std"),accuracy_mean=("accuracy","mean"),balanced_accuracy_mean=("balanced_accuracy","mean")).fillna(0); summary.to_csv(out/"summary.csv")
    pd.DataFrame(prediction_rows).to_csv(out/"predictions.csv",index=False)
    lines=["# Baselines FactKG", "", "Divisão interna agrupada por claim. Os valores são condicionais à amostra/configuração da execução e não são o teste oficial do FactKG.", "", "## Resumo", "", summary.round(4).to_string(), "", "## Interpretação", "", "Os baselines supervisionados e simbólicos usam os mesmos IDs, rótulos e divisão. ExactTriple é uma heurística de cobertura e não deve ser interpretada como prova de refutação quando uma tripla não aparece. GraphFeatures resume sinais estruturais e de sobreposição; não executa inferência de caminhos.", "", "## Arquivos", "", "- `metrics.csv`: métricas por modelo e seed.", "- `summary.csv`: médias e desvio entre seeds.", "- `costs.csv`: tempo e dispositivo.", "- `audit.json`: cobertura, exclusões e sanitização.", "- `predictions.csv`: IDs do teste e rótulos usados na avaliação."]
    (out/"RESULTADOS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary
