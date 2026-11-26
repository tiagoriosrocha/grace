"""Dados fictícios pequenos, exclusivamente para testes offline."""
import json


def graph(triples):
    names=list(dict.fromkeys(x for s,r,o in triples for x in (s,o)))
    ids={name:i for i,name in enumerate(names)}
    return dict(nodes=[dict(id=i,text=name,type='Entity') for i,name in enumerate(names)],
                edges=[dict(source=ids[s],target=ids[o],type=r) for s,r,o in triples])


def write_fixture(path,n=64):
    rows=[]
    for i in range(n):
        supported=i%2==0
        person=f'person {i}'
        # Shared entities exercise the intended known-entity protocol.
        city=f'city {i%7}'
        org=f'organization {i%9}'
        claim=[(person,'born in',city),(person,'works at',org)]
        evidence=[(person,'born in',city),(person,'works at',org if supported else f'other institution {i}'),
                  (org,'located in',city)]
        rows.append(dict(id=f'smoke_{i}',dataset='hover',claim=f'{person} born in {city} works at {org}',
                         label='SUPPORTED' if supported else 'NOT_SUPPORTED',split='train',synthetic=True,
                         num_hops=2+i%3,grafo_claim=graph(claim),grafo_evidencia=graph(evidence)))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
