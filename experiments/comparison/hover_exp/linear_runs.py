"""Statistical controls with persisted models, named features and contributions."""
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .g19 import _features
from .g19_reference import FEATURE_NAMES
from .data import save_json
from .experiment import metrics

LEGACY_NAMES=([f'{block}_{stat}' for block in ('all_similarities','all_exact','all_nli')
               for stat in ('mean','min','max','fraction_gt_05')]+
              [f'{block}_{stat}' for block in ('entailment','contradiction','neutral','margin')
               for stat in ('mean','min','max')]+
              ['support_dominant_fraction','contradiction_dominant_fraction','neutral_dominant_fraction',
               'claim_factual_count','candidate_count'])


def validation_threshold(y,prob):
    # Fixed grid; ties prefer the usual threshold, never consult test labels.
    candidates=np.linspace(.1,.9,161)
    return float(max(candidates,key=lambda t:(metrics(y,prob,t)['macro_f1'],-abs(t-.5))))


def run_linear(bundle,reference,cfg,out):
    rows=[]
    pairs=bundle['pairs'];roles=np.array([p['role'] for p in pairs]);y=np.array([p['label'] for p in pairs])
    matrices={'G19':np.stack([_features(p) for p in pairs]),'G19_REF':reference['features'].numpy()}
    for variant,X in matrices.items():
        if not np.isfinite(X).all():raise ValueError(f'Características inválidas: {variant}')
        names=LEGACY_NAMES if variant=='G19' else FEATURE_NAMES
        model=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(C=1.,max_iter=2000 if variant=='G19' else 4000,
                                   class_weight='balanced',random_state=cfg['split_seed']))])
        model.fit(X[roles=='train'],y[roles=='train'])
        probabilities=model.predict_proba(X)[:,list(model.classes_).index(1)]
        threshold=validation_threshold(y[roles=='validation'],probabilities[roles=='validation'])
        joblib.dump(dict(model=model,feature_names=names),out/'models'/f'{variant}.joblib')
        protocol_path=out/f'{variant}_protocol.json'
        protocol=json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
        protocol.update(feature_names=names,primary_threshold=.5,validation_threshold=threshold,
                        fit='train only',one_deterministic_fit=True)
        save_json(protocol_path,protocol)
        pd.DataFrame({'feature':names,'coefficient':model['clf'].coef_[0],
                      'train_mean':model['scale'].mean_,'train_scale':model['scale'].scale_}).to_csv(
                      out/f'{variant}_coefficients.csv',index=False)
        features_frame=pd.DataFrame(X,columns=names)
        features_frame.insert(0,'id',[p['id'] for p in pairs]);features_frame.insert(1,'role',roles)
        features_frame.to_csv(out/f'{variant}_features.csv',index=False)
        test=roles=='test';Z=model['scale'].transform(X[test])
        contributions=Z*model['clf'].coef_[0]
        explanations=pd.DataFrame(contributions,columns=names)
        explanations.insert(0,'id',[p['id'] for p in pairs if p['role']=='test'])
        explanations['intercept']=float(model['clf'].intercept_[0])
        explanations['logit']=contributions.sum(1)+model['clf'].intercept_[0]
        np.testing.assert_allclose(explanations.logit,model.decision_function(X[test]),atol=1e-8)
        explanations.to_csv(out/f'{variant}_contributions.csv',index=False)
        for pair,prob in zip(pairs,probabilities):
            if pair['role'] not in ('validation','test'):continue
            rows.append(dict(id=pair['id'],group_id=pair['group_id'],num_hops=pair['num_hops'],label=pair['label'],
                variant=variant,seed=cfg['split_seed'],role=pair['role'],probability=float(prob),prediction=int(prob>=.5)))
        save_json(out/f'{variant}_metrics.json',dict(variant=variant,seed=cfg['split_seed'],threshold=.5,
            validation_threshold=threshold,train_features=X.shape[1],**metrics(y[test],probabilities[test])))
        pd.DataFrame([r for r in rows if r['variant']==variant and r['role']=='test']).to_csv(
            out/f'{variant}_predictions.csv',index=False)
    return pd.DataFrame(rows)
