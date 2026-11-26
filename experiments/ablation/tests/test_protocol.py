from pathlib import Path

import json
import torch
import copy
import numpy as np
import pandas as pd
import pytest

from hover_exp.fixtures import write_fixture
from hover_exp.data import prepare
from hover_exp.data import DATASET_LABELS, DATASETS, features
from reporting import group_bootstrap, macro_f1_from_counts
from run_contract import start_run


def config():
    cfg=json.loads((Path(__file__).parents[1]/'config.json').read_text())
    cfg.update(dataset='hover',smoke=True,split_max_class_deviation=.35)
    return cfg


def fixture_rows(tmp_path):
    source=tmp_path/'fixture.jsonl'
    write_fixture(source,n=128)
    return [json.loads(line) for line in source.read_text().splitlines()]


def save_rows(path, rows):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def test_fixture_has_no_cross_role_evidence_groups(tmp_path):
    source = tmp_path / "fixture.jsonl"
    write_fixture(source, n=64)
    cfg = json.loads((Path(__file__).parents[1] / "config.json").read_text())
    cfg.update(dataset="hover", smoke=True, split_seed=20251002, entity_disjoint_split=False,
               split_max_class_deviation=.35)
    records, audit = prepare(source, tmp_path / "prepared", cfg)
    assert records
    assert audit["split_groups"]["cross_role_evidence_groups"] == 0
    assert audit["split_groups"]["cross_role_claims"] == 0
    assert audit["split_groups"]["cross_role_entities"] > 0
    assert audit["split_groups"]["entity_overlap_allowed"] is True


def test_repeated_evidence_claims_and_source_groups_stay_together(tmp_path):
    rows=fixture_rows(tmp_path)
    # Different claim, same evidence, opposite label.
    evidence_copy=copy.deepcopy(rows[0]); evidence_copy.update(id='copy_evidence',claim='Another claim',label='NOT_SUPPORTED')
    # Same normalized claim, different evidence.
    claim_copy=copy.deepcopy(rows[1]);claim_copy.update(id='copy_claim',claim='  '+rows[0]['claim'].upper()+'  ')
    rows[2]['group_id']='family';rows[3]['group_id']='family'
    rows += [evidence_copy,claim_copy]
    path=tmp_path/'input.jsonl';save_rows(path,rows)
    records,audit=prepare(path,tmp_path/'first',config())
    by_id={r['id']:r for r in records}
    assert len({by_id[k]['group_id'] for k in ('smoke_0','copy_evidence','copy_claim')})==1
    assert by_id['smoke_2']['group_id']==by_id['smoke_3']['group_id']
    save_rows(path,list(reversed(rows)))
    reordered,_=prepare(path,tmp_path/'second',config())
    assert {r['id']:(r['group_id'],r['evaluation_role']) for r in records} == {r['id']:(r['group_id'],r['evaluation_role']) for r in reordered}


def test_split_rejects_single_giant_evidence_group(tmp_path):
    rows=fixture_rows(tmp_path)
    for r in rows:r['grafo_evidencia']=rows[0]['grafo_evidencia']
    path=tmp_path/'input.jsonl';save_rows(path,rows)
    with pytest.raises(ValueError):prepare(path,tmp_path/'prepared',config())


def test_dataset_mismatch_rejected(tmp_path):
    path=tmp_path/'input.jsonl';save_rows(path,fixture_rows(tmp_path))
    cfg=config();cfg['dataset']='factkg'
    with pytest.raises(ValueError,match='Dataset/rótulo'):prepare(path,tmp_path/'out',cfg)


@pytest.mark.parametrize('dataset,labels', DATASET_LABELS.items())
def test_all_supported_dataset_labels_are_binary(dataset, labels):
    assert dataset in DATASETS
    assert set(labels.values()) == {0, 1}


def test_config_registers_all_dataset_directories():
    cfg = json.loads((Path(__file__).parents[1] / 'config.json').read_text())
    assert set(DATASETS) <= set(cfg['data_paths'])
    assert all(cfg['data_paths'][name] == f'data/{name}' for name in DATASETS)


def test_test_labels_never_change_features_and_test_relations_are_unknown(tmp_path):
    path=tmp_path/'input.jsonl';save_rows(path,fixture_rows(tmp_path))
    cfg=config();out=tmp_path/'prepared';records,_=prepare(path,out,cfg)
    changed=copy.deepcopy(records)
    for r in changed:
        if r['evaluation_role']=='test':
            r['label']=1-r['label']
            for t in r['evidence_triples']:t['relation_key']='ONLY_TEST_CATEGORY'
    first=features(records,cfg,torch.device('cpu'),out)
    out2=tmp_path/'changed';out2.mkdir()
    second=features(changed,cfg,torch.device('cpu'),out2)
    assert first['relation_vocab']==second['relation_vocab']
    assert 'ONLY_TEST_CATEGORY' not in second['relation_vocab']
    assert json.loads((out/'normalization.json').read_text())==json.loads((out2/'normalization.json').read_text())
    for p,q in zip(first['pairs'],second['pairs']):
        assert torch.equal(p['signals'],q['signals'])
        assert torch.count_nonzero(p['signals'][...,3:6])==0
        if q['role']=='test':assert torch.count_nonzero(q['evidence']['edge_type'])==0


def test_group_bootstrap_is_paired_and_checks_alignment():
    rows=[]
    for model in ('majority_baseline','text_tfidf'):
        for seed in (7,21):
            for i in range(8):
                rows.append(dict(model=model,seed=seed,id=str(i),group_id='one_cluster',label=i%2,probability=.8 if i%2 else .2))
    df=pd.DataFrame(rows)
    intervals,comparisons=group_bootstrap(df,[7,21],50)
    assert intervals['text_tfidf']==[1.,1.]
    assert comparisons[0]['ci95_low']==comparisons[0]['ci95_high']==0
    with pytest.raises(ValueError,match='pareadas'):group_bootstrap(df.iloc[:-1],[7,21],10)
    assert float(macro_f1_from_counts([4,0,0,4]))==1.
    assert float(macro_f1_from_counts([0,4,4,0]))==0.


def test_bootstrap_resamples_whole_dependent_groups():
    rows=[]
    # One entirely correct group, one entirely wrong group, balanced labels.
    # Resampling clusters must admit all-correct/all-wrong replicates.
    for seed in (7,21):
        for i in range(100):
            label=i%2
            pred=label if i<50 else 1-label
            rows.append(dict(model='text_tfidf',seed=seed,id=str(i),group_id=str(i//50),
                             label=label,probability=float(pred)))
    intervals,_=group_bootstrap(pd.DataFrame(rows),[7,21],200)
    assert intervals['text_tfidf']==[0.,1.]


def test_output_directory_cannot_reuse_checkpoints(tmp_path):
    path=tmp_path/'input.jsonl';save_rows(path,fixture_rows(tmp_path))
    out=tmp_path/'run';start_run(out,config(),path)
    with pytest.raises(FileExistsError):start_run(out,config(),path)
