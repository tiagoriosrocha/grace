"""Tests for factorial GNN ablations; run with the project's PyTorch environment."""
import pytest
import torch
from torch.nn import functional as F
from hover_exp.model import Verifier


def make_model(use_null=True, use_disagreement=True, use_relational_context=True,
               use_nli=True, use_condition_aggregation=True):
    torch.manual_seed(13)
    cfg = dict(dropout=0.0, hidden_dim=8, heads=2, relation_bases=1, layers=0,
               ablation_use_null=use_null, ablation_use_disagreement=use_disagreement)
    cfg.update(ablation_use_relational_context=use_relational_context,
               ablation_use_nli=use_nli,
               ablation_use_condition_aggregation=use_condition_aggregation)
    model = Verifier(torch.randn(20, 6), 2, cfg, 'relational_gnn')
    model.eval()
    return model


def make_pair(empty=False):
    def graph(nodes, edges, factual):
        return dict(node_text=torch.tensor(nodes), structure=torch.zeros(len(nodes), 2),
                    edge_index=torch.tensor(edges, dtype=torch.long),
                    edge_text=torch.arange(len(edges[0]), dtype=torch.long) + 10,
                    edge_type=torch.zeros(len(edges[0]), dtype=torch.long),
                    factual=torch.tensor(factual, dtype=torch.long))
    claim = graph([0,1,2], [[0,1],[1,2]], [0,1])
    evidence = graph([3,4,5,6], [[0,1,0,2],[1,3,2,3]], [0,1,2,3])
    candidates = torch.empty((2,0), dtype=torch.long) if empty else torch.tensor([[0,2],[1,3]])
    return dict(claim=claim, evidence=evidence, candidates=candidates,
                signals=torch.zeros((2,candidates.shape[1],9)))


@pytest.mark.parametrize('use_null,use_disagreement', [(True,True),(False,True),(True,False),(False,False)])
@pytest.mark.parametrize('empty', [False,True])
def test_four_variants_and_empty_candidates(use_null, use_disagreement, empty):
    model = make_model(use_null, use_disagreement)
    logit, aux = model(make_pair(empty), return_aux=True)
    assert torch.isfinite(logit)
    assert torch.isfinite(aux['condition_vectors']).all()
    assert torch.isfinite(aux['local_support']).all()
    if use_null:
        torch.testing.assert_close(aux['weights'].sum(-1) + aux['null_mass'], torch.ones(2))
    else:
        torch.testing.assert_close(aux['null_mass'], torch.zeros(2))
        if not empty:
            torch.testing.assert_close(aux['weights'].sum(-1), torch.ones(2))
        else:
            torch.testing.assert_close(aux['condition_vectors'], torch.zeros_like(aux['condition_vectors']))
            torch.testing.assert_close(aux['local_support'], torch.zeros(2))
    if empty:
        assert aux['weights'].shape == (2,0)
        assert aux['consistency'].item() == 0


def test_disagreement_adjusts_logit_only_when_enabled():
    model = make_model()
    pair = make_pair()
    model.consistency = lambda pair, weights: weights.new_tensor(0.5)
    with torch.no_grad():
        model.use_disagreement = True
        with_term = model(pair)
        model.use_disagreement = False
        without_term = model(pair)
    expected = F.softplus(model.consistency_penalty) * 0.5
    torch.testing.assert_close(without_term - with_term, expected)


def test_shared_entity_disagreement():
    pair = make_pair()
    compatible = torch.tensor([[0.8,0.2],[0.8,0.2]])
    inconsistent = torch.tensor([[0.8,0.2],[0.2,0.8]])
    torch.testing.assert_close(Verifier.consistency(pair, compatible), torch.tensor(0.0))
    torch.testing.assert_close(Verifier.consistency(pair, inconsistent), torch.tensor(0.18))


@pytest.mark.parametrize('key', ['ablation_use_relational_context',
                                 'ablation_use_nli',
                                 'ablation_use_condition_aggregation'])
def test_new_ablation_variants_produce_finite_outputs(key):
    cfg = dict(use_relational_context=True, use_nli=True, use_condition_aggregation=True)
    cfg[key] = False
    model = make_model(**cfg)
    logit, aux = model(make_pair(), return_aux=True)
    assert torch.isfinite(logit)
    assert torch.isfinite(aux['condition_vectors']).all()
