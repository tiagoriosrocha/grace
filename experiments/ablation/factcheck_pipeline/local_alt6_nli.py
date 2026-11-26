"""Frozen NLI scorer shared by the alt6-compatible local system."""
import numpy as np


class FrozenNLI:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        self.metadata = dict(model=cfg['nli_model'], revision=cfg['nli_revision'],
                             order=['contradiction', 'entailment', 'neutral'], pairs=0,
                             truncated_pairs=0, backend='synthetic_ONLY_SMOKE' if cfg['smoke'] else 'frozen_transformer')
        if not cfg['smoke']:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            options = dict(revision=cfg['nli_revision'])
            self.tokenizer = AutoTokenizer.from_pretrained(cfg['nli_model'], **options)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                cfg['nli_model'], use_safetensors=True, attn_implementation='eager', **options).eval().to(device)
            self.model.requires_grad_(False)
            labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
            self.ids = [next((i for i, value in labels.items() if name in value), None)
                        for name in self.metadata['order']]
            if None in self.ids or len(set(self.ids)) != 3:
                raise ValueError(f'Classe NLI ambígua: {labels}')
            self.metadata['resolved_revision'] = getattr(self.model.config, '_commit_hash', None)

    def score(self, candidates):
        import torch
        if not candidates:
            return np.empty((0, 3))
        self.metadata['pairs'] += len(candidates)
        if self.cfg['smoke']:
            return np.asarray([[.05, .9, .05] if all(c['exact']) else [.1, .1, .8]
                               for c in candidates])
        results = []
        with torch.inference_mode():
            for start in range(0, len(candidates), self.cfg['nli_batch_size']):
                batch = candidates[start:start + self.cfg['nli_batch_size']]
                premises = [c['premise'] for c in batch]
                hypotheses = [c['hypothesis'] for c in batch]
                lengths = self.tokenizer(premises, hypotheses, truncation=False, verbose=False)['input_ids']
                self.metadata['truncated_pairs'] += sum(len(x) > self.cfg['nli_max_length'] for x in lengths)
                tokens = self.tokenizer(premises, hypotheses, padding=True, truncation=True,
                                        max_length=self.cfg['nli_max_length'], return_tensors='pt').to(self.device)
                results.append(self.model(**tokens).logits.softmax(-1)[:, self.ids].float().cpu().numpy())
        return np.concatenate(results)
