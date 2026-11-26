"""Três ablações do pipeline híbrido matching + NLI + grafo."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGATConv

VARIANTS=('H0','H1','H2')


def move(value, device):
    if isinstance(value,torch.Tensor):
        return value.to(device)
    if isinstance(value,dict):
        return {k:move(v,device) for k,v in value.items()}
    return value


class Verifier(nn.Module):
    def __init__(self, embeddings, relations, cfg, variant):
        super().__init__()
        if variant not in VARIANTS and variant != 'relational_gnn':
            raise ValueError(variant)
        self.variant=variant
        self.dropout=cfg['dropout']
        h=cfg['hidden_dim']
        d=embeddings.shape[1]
        # Shared frozen table, never part of the optimizer or checkpoint.
        self.register_buffer('text_embeddings',embeddings, persistent=False)
        self.node_project=nn.Linear(d+2,h)
        self.rel_category=nn.Embedding(relations,16)
        self.rel_project=nn.Linear(d+16,h)
        self.convs=nn.ModuleList([RGATConv(h,h//cfg['heads'],heads=cfg['heads'],num_relations=relations,
                                           num_bases=min(cfg['relation_bases'],relations),edge_dim=h,
                                           dropout=cfg['dropout']) for _ in range(cfg['layers'])])
        self.norms=nn.ModuleList([nn.LayerNorm(h) for _ in self.convs])
        self.triple=nn.Linear(3*h,h)
        # All variants receive the same local comparison and signal vector.
        # This makes H0/H1/H2 differ only in contextualization/aggregation.
        self.compare=nn.Sequential(nn.Linear(4*h+9,h),nn.GELU(),nn.Dropout(cfg['dropout']))
        self.head=nn.Sequential(nn.Linear(2*h,h),nn.GELU(),nn.Dropout(cfg['dropout']),nn.Linear(h,1))
        self.align=nn.Linear(4*h+9,1)
        nn.init.zeros_(self.align.weight)
        nn.init.zeros_(self.align.bias)
        if variant in ('H2','relational_gnn'):
            # Relevance (matching) and polarity (support/contradiction) are
            # learned in separate channels. Contradiction is not subtracted
            # from relevance and can therefore support NOT_SUPPORTED.
            self.coverage_align=nn.Sequential(nn.Linear(4*h+6,h//2),nn.GELU(),nn.Linear(h//2,1))
            self.polarity=nn.Sequential(nn.Linear(3,h//2),nn.GELU(),nn.Linear(h//2,1))
            nn.init.zeros_(self.coverage_align[-1].weight)
            nn.init.zeros_(self.coverage_align[-1].bias)
            self.null_score=nn.Sequential(nn.Linear(h+9,h//2),nn.GELU(),nn.Linear(h//2,1))
            # Keep a vector for every local comparison until the final decision.
            # The previous version reduced these vectors to a soft minimum too early.
            self.local_vector=nn.Sequential(nn.Linear(4*h+9,h),nn.GELU(),nn.Dropout(cfg['dropout']))
            self.local_support=nn.Sequential(nn.Linear(4*h+9,h),nn.GELU(),nn.Dropout(cfg['dropout']),nn.Linear(h,1))
            self.null_vector=nn.Sequential(nn.Linear(h+9,h),nn.GELU(),nn.Dropout(cfg['dropout']))
            self.condition_attention=nn.Sequential(nn.Linear(h+2,h//2),nn.GELU(),nn.Linear(h//2,1))
            self.coverage_head=nn.Sequential(nn.Linear(3*h+2,h),nn.GELU(),nn.Dropout(cfg['dropout']),nn.Linear(h,1))
            self.consistency_penalty=nn.Parameter(torch.tensor(-2.))

    def encode(self,g,contextual=True):
        h=self.node_project(torch.cat([self.text_embeddings[g['node_text']],g['structure']],-1))
        r=self.rel_project(torch.cat([self.text_embeddings[g['edge_text']],self.rel_category(g['edge_type'])],-1))
        if not len(h) or not contextual:
            return h,r
        # Keep only the direction present in the graph. Reversing an edge would
        # change the fact being checked (subject -> relation -> object).
        edges=g['edge_index']
        types=g['edge_type']
        attrs=r
        for conv,norm in zip(self.convs,self.norms):
            message=conv(h,edges,edge_type=types,edge_attr=attrs) if edges.shape[1] else torch.zeros_like(h)
            h=norm(h+F.dropout(F.gelu(message),p=self.dropout,training=self.training))
        return h,r

    def triples(self,h,r,g):
        src,dst=g['edge_index']
        return self.triple(torch.cat([h[src],r,h[dst]],-1))

    @staticmethod
    def pool(h, reference):
        return h.mean(0) if len(h) else reference.new_zeros(reference.shape[-1])

    @staticmethod
    def consistency(pair,weights):
        """Soft disagreement of mappings for repeated claim entities; not a proof."""
        cg,eg=pair['claim'],pair['evidence']
        if weights.shape[1]==0:
            return weights.new_zeros(())
        cond=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
        factual=cg['factual']
        expected=[]
        claim_ids=[]
        for endpoint in (0,1):
            target=eg['edge_index'][endpoint][pair['candidates']]
            dist=weights.new_zeros(len(factual),len(eg['node_text']))
            dist.scatter_add_(1,target,cond)
            expected.append(dist)
            claim_ids.append(cg['edge_index'][endpoint][factual])
        expected=torch.cat(expected)
        ids=torch.cat(claim_ids)
        sums=weights.new_zeros(len(cg['node_text']),len(eg['node_text']))
        sums.index_add_(0,ids,expected)
        count=torch.bincount(ids,minlength=len(cg['node_text'])).to(weights.dtype)
        means=sums/count.clamp_min(1)[:,None]
        mask=count[ids]>1
        diff=(expected-means[ids]).square().sum(-1)
        # Avoid conditional CPU synchronization on CUDA.
        return (diff*mask).sum()/mask.sum().clamp_min(1)

    def forward(self,pair,return_aux=False):
        contextual=self.variant in ('H1','H2','relational_gnn')
        c,cr=self.encode(pair['claim'],contextual=contextual)
        e,er=self.encode(pair['evidence'],contextual=contextual)
        ct=self.triples(c,cr,pair['claim'])[pair['claim']['factual']]
        et=self.triples(e,er,pair['evidence'])
        b=et[pair['candidates']]
        a=ct[:,None,:].expand_as(b)
        local=torch.cat([a,b,(a-b).abs(),a*b],-1)
        signals=pair['signals']
        if signals.shape[-1] != 9:
            raise ValueError(f'Esperados 9 sinais por candidato, recebido {signals.shape[-1]}')
        nli=signals[:,:,6:9]
        if self.variant in ('H0','H1'):
            if not b.shape[1]:
                aligned=ct.new_zeros(len(ct),ct.shape[-1])
                weights=ct.new_zeros((len(ct),0))
            else:
                inputs=torch.cat([local,signals],-1)
                weights=self.align(inputs).squeeze(-1).softmax(-1)
                aligned=(weights[:,:,None]*self.compare(inputs)).sum(1)
            result=self.head(torch.cat([aligned.mean(0),aligned.max(0).values])).squeeze(-1)
            return (result,{'nli':nli,'weights':weights}) if return_aux else result
        inputs=torch.cat([local,signals],-1)
        relevance_inputs=torch.cat([local,signals[:,:,:6]],-1)
        polarity=self.polarity(nli)
        scores=self.coverage_align(relevance_inputs).squeeze(-1)+polarity.squeeze(-1)
        absolute=signals.max(1).values if signals.shape[1] else ct.new_zeros(len(ct),9)
        null=self.null_score(torch.cat([ct,absolute],-1))
        weights=torch.cat([scores,null],-1).softmax(-1)
        compatibility=self.local_support(inputs).squeeze(-1).sigmoid()
        support=(weights[:,:-1]*compatibility).sum(-1)
        null_mass=weights[:,-1]
        # Preserve local comparison vectors, including an explicit vector for
        # the null alternative, and let the final set aggregator learn how the
        # claim conditions should interact. No mandatory minimum is applied.
        if b.shape[1]:
            candidate_vectors=self.local_vector(inputs)
            condition_vectors=(weights[:,:-1,None]*candidate_vectors).sum(1)
        else:
            candidate_vectors=inputs.new_zeros((len(ct),0,ct.shape[-1]))
            condition_vectors=inputs.new_zeros((len(ct),ct.shape[-1]))
        condition_vectors=condition_vectors+null_mass[:,None]*self.null_vector(torch.cat([ct,absolute],-1))
        condition_keys=torch.cat([condition_vectors,support[:,None],null_mass[:,None]],-1)
        condition_weights=self.condition_attention(condition_keys).squeeze(-1).softmax(0)
        mean_vector=condition_vectors.mean(0) if len(condition_vectors) else ct.new_zeros(ct.shape[-1])
        attended_vector=(condition_weights[:,None]*condition_vectors).sum(0) if len(condition_vectors) else ct.new_zeros(ct.shape[-1])
        max_vector=condition_vectors.max(0).values if len(condition_vectors) else ct.new_zeros(ct.shape[-1])
        global_vector=torch.cat([mean_vector,attended_vector,max_vector,support.mean()[None],null_mass.mean()[None]])
        consistency=self.consistency(pair,weights[:,:-1])
        result=self.coverage_head(global_vector).squeeze(-1)-F.softplus(self.consistency_penalty)*consistency
        aux=dict(local_support=support,null_mass=null_mass,weights=weights[:,:-1],
                 condition_weights=condition_weights,condition_vectors=condition_vectors,
                 local_vectors=candidate_vectors,consistency=consistency,
                 local_compatibility=compatibility)
        return (result,aux) if return_aux else result
