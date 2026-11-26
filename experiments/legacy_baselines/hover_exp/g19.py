"""G19 legado: aproximação de 29 features; a reprodução está em g19_reference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score


def _features(pair):
    s=pair['signals'].detach().cpu().numpy()
    if s.shape[-1] != 9:
        raise ValueError('G19 requer exatamente 9 sinais por candidato')
    sim=s[:,:,:3]; exact=s[:,:,3:6]; nli=s[:,:,6:9]
    # Same global matching/NLI idea as notebook 19: aggregate candidate
    # statistics, never use raw claim/evidence text in the classifier.
    flat=lambda x: x.reshape(-1) if x.size else np.zeros(1)
    vals=[]
    for x in (sim,exact,nli):
        z=flat(x)
        vals += [float(z.mean()),float(z.min()),float(z.max()),float((z>0.5).mean())]
    entail=nli[:,:,1]; contra=nli[:,:,0]; neutral=nli[:,:,2]
    margin=entail-contra
    for x in (entail,contra,neutral,margin):
        z=flat(x); vals += [float(z.mean()),float(z.min()),float(z.max())]
    vals += [float((entail>np.maximum(contra,neutral)).mean()) if entail.size else 0.,
             float((contra>np.maximum(entail,neutral)).mean()) if entail.size else 0.,
             float((neutral>np.maximum(entail,contra)).mean()) if entail.size else 0.,
             float(sim.shape[0]),float(sim.shape[1])]
    return np.asarray(vals,dtype=np.float32)


def run_g19(bundle,cfg,out):
    train=[p for p in bundle['pairs'] if p['role']=='train']
    val=[p for p in bundle['pairs'] if p['role']=='validation']
    test=[p for p in bundle['pairs'] if p['role']=='test']
    X=np.stack([_features(p) for p in train]); y=np.asarray([p['label'] for p in train])
    model=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(C=1.0,max_iter=2000,class_weight='balanced',random_state=cfg['split_seed']))])
    model.fit(X,y)
    pv=model.predict_proba(np.stack([_features(p) for p in val]))[:,1]
    thresholds=np.linspace(.1,.9,161)
    threshold=max(thresholds,key=lambda t:f1_score([p['label'] for p in val],pv>=t,average='macro'))
    pt=model.predict_proba(np.stack([_features(p) for p in test]))[:,1]
    yt=np.asarray([p['label'] for p in test]); pred=(pt>=threshold).astype(int)
    result=dict(variant='G19',seed=cfg['split_seed'],threshold=float(threshold),
                macro_f1=float(f1_score(yt,pred,average='macro')),accuracy=float(accuracy_score(yt,pred)),
                balanced_accuracy=float(balanced_accuracy_score(yt,pred)),auc=float(roc_auc_score(yt,pt)),
                train_features=int(X.shape[1]),protocol='StandardScaler + LogisticRegression; notebook 19 baseline')
    Path(out).mkdir(parents=True,exist_ok=True)
    Path(out,'G19_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame({'id':[p['id'] for p in test],'label':yt,'probability':pt,'prediction':pred,'variant':'G19'}).to_csv(Path(out,'G19_predictions.csv'),index=False)
    return result
