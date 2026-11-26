"""Notebook 23-a family; frozen cached encoder, no graph convolution."""
import torch
from torch import nn

VARIANTS=('X23-0','X23-1','X23-2','X23-NLI')


class X23(nn.Module):
    def __init__(self,embeddings,ntypes,cfg,variant,nli_mean=None,nli_std=None):
        super().__init__()
        if variant not in VARIANTS and variant != 'claim_evidence_attention':raise ValueError(variant)
        self.variant=variant;self.use_edges=variant!='X23-1'
        d=embeddings.shape[1];h=cfg['x23_dim'];drop=cfg['x23_dropout']
        self.register_buffer('embeddings',embeddings,persistent=False)
        self.node_projection=nn.Linear(d,h)
        self.edge_projection=nn.Sequential(nn.Linear(3*d,h),nn.GELU(),nn.Linear(h,h))
        self.kind_embedding=nn.Embedding(2,h);self.type_embedding=nn.Embedding(ntypes,h)
        self.graph_norm=nn.LayerNorm(h);self.dropout=nn.Dropout(drop)
        self.claim_projection=nn.Linear(d,h)
        if variant!='X23-0':
            self.attention=nn.MultiheadAttention(h,cfg['x23_heads'],dropout=drop,batch_first=True)
            self.cross_norm=nn.LayerNorm(h);self.ff_norm=nn.LayerNorm(h)
            self.ff=nn.Sequential(nn.Linear(h,2*h),nn.GELU(),nn.Dropout(drop),nn.Linear(2*h,h))
        self.register_buffer('nli_mean',torch.zeros(31) if nli_mean is None else nli_mean)
        self.register_buffer('nli_std',torch.ones(31) if nli_std is None else nli_std)
        self.classifier=nn.Sequential(nn.Linear(2*h+(31 if variant=='X23-NLI' else 0),h),
                                     nn.GELU(),nn.Dropout(drop),nn.Linear(h,1))

    def graph_tokens(self,example):
        sel=example['selections'][str(self.use_edges)]
        ni,ei=sel['nodes'],sel['edges']
        nodes=self.node_projection(self.embeddings[example['nodes'][ni]])
        nodes=nodes+self.kind_embedding.weight[0]+self.type_embedding(example['types'][ni])
        tokens=[self.dropout(self.graph_norm(nodes))]
        meta=[('node',int(i)) for i in ni.tolist()]
        if self.use_edges:
            edges=self.edge_projection(self.embeddings[example['edges'][ei]].flatten(1))
            edges=edges+self.kind_embedding.weight[1]
            tokens.append(self.dropout(self.graph_norm(edges)))
            meta += [('edge',int(i)) for i in ei.tolist()]
        return torch.cat(tokens),meta

    def forward(self,example,claim_tokens,return_aux=False,drop_indices=None):
        claim=self.claim_projection(claim_tokens)
        graph,meta=self.graph_tokens(example)
        keep=torch.ones(len(graph),dtype=torch.bool,device=graph.device)
        if drop_indices is not None:keep[drop_indices]=False
        graph=graph[keep]
        if not len(graph):graph=claim.new_zeros(1,claim.shape[-1])
        if self.variant=='X23-0':
            pooled=torch.cat([claim.mean(0),graph.mean(0)])
            attention=None
        else:
            cross,attention=self.attention(claim[None],graph[None],graph[None],
                need_weights=return_aux,average_attn_weights=False)
            h=self.cross_norm(claim+self.dropout(cross[0]))
            h=self.ff_norm(h+self.dropout(self.ff(h)))
            pooled=torch.cat([h.mean(0),h.max(0).values])
        if self.variant=='X23-NLI':
            pooled=torch.cat([pooled,(example['nli_features']-self.nli_mean)/self.nli_std])
        logit=self.classifier(pooled).squeeze(-1)
        return (logit,dict(attention=attention,metadata=meta,kept=keep)) if return_aux else logit
