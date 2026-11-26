"""Frozen NLI scorer shared by reference G19 and controlled interventions."""
import torch


class NLIScorer:
    def __init__(self,cfg,device):
        self.cfg,self.device=cfg,device
        self.cache={}
        self.model=None
        self.metadata={'backend':'zeros_ONLY_SMOKE' if cfg['smoke'] else 'transformer',
                       'model':cfg['nli_model'],'order':['contradiction','entailment','neutral']}

    def score(self,rows):
        rows=[tuple(x) for x in rows]
        missing=list(dict.fromkeys(x for x in rows if x not in self.cache))
        if self.cfg['smoke']:
            self.cache.update({x:torch.zeros(3) for x in missing})
        elif missing:
            if self.model is None:
                from transformers import AutoTokenizer,AutoModelForSequenceClassification
                opts=dict(revision=self.cfg['nli_revision'])
                self.tokenizer=AutoTokenizer.from_pretrained(self.cfg['nli_model'],**opts)
                self.model=AutoModelForSequenceClassification.from_pretrained(
                    self.cfg['nli_model'],use_safetensors=True,**opts).eval().to(self.device)
                labels={int(k):str(v).lower() for k,v in self.model.config.id2label.items()}
                self.ids=[next((i for i,v in labels.items() if name in v),None)
                          for name in self.metadata['order']]
                if None in self.ids or len(set(self.ids))!=3:
                    raise ValueError(f'Rótulos NLI incompatíveis: {labels}')
                self.metadata['resolved_revision']=getattr(self.model.config,'_commit_hash',None)
            with torch.inference_mode():
                for start in range(0,len(missing),self.cfg['nli_batch_size']):
                    batch=missing[start:start+self.cfg['nli_batch_size']]
                    tokens=self.tokenizer([p for p,h in batch],[h for p,h in batch],padding=True,
                        truncation=True,max_length=self.cfg['nli_max_length'],return_tensors='pt').to(self.device)
                    values=self.model(**tokens).logits.softmax(-1)[:,self.ids].float().cpu()
                    self.cache.update(zip(batch,values))
        return torch.stack([self.cache[x] for x in rows]) if rows else torch.empty(0,3)

    def close(self):
        self.model=None
        if hasattr(self,'tokenizer'): del self.tokenizer
        if self.device.type=='cuda': torch.cuda.empty_cache()


def marked_triple(t):
    return f"{t['source_text']} [REL] {t['relation']} [OBJ] {t['target_text']}"
