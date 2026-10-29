"""Local dense embeddings + TF-IDF, fused with reciprocal ranks."""
import hashlib
import json
import os
import sqlite3
from collections import defaultdict
import httpx
import numpy as np
from .providers import local_config


def root_id(doc):
    return doc.get('document_id',doc['id'])

async def embedding_identity(client, url, model):
    response=await client.get(url+'/api/tags',timeout=2)
    response.raise_for_status()
    entry=next((m for m in response.json().get('models',[]) if m['name']==model),None)
    if not entry:
        raise ValueError('Embedding model is not installed')
    return model+':'+entry['digest']

async def vectors(client,url,model,texts,identity,path):
    hashes=[hashlib.sha256((identity+'\n'+text).encode()).hexdigest() for text in texts]
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS embeddings (fingerprint TEXT PRIMARY KEY, vector TEXT NOT NULL)')
        existing={row[0]:json.loads(row[1]) for row in db.execute('SELECT fingerprint,vector FROM embeddings')}
    missing=list(dict.fromkeys(i for i,key in enumerate(hashes) if key not in existing))
    for start in range(0,len(missing),24):
        batch=missing[start:start+24]
        response=await client.post(url+'/api/embed',json={'model':model,'input':[texts[i] for i in batch],'truncate':True,'keep_alive':'10m'})
        response.raise_for_status()
        output=response.json()['embeddings']
        if len(output)!=len(batch):
            raise ValueError('Embedding response size mismatch')
        with sqlite3.connect(path,timeout=20) as db:
            for index,vector in zip(batch,output):
                if not vector or not all(isinstance(n,(float,int)) for n in vector):
                    raise ValueError('Invalid embedding vector')
                existing[hashes[index]]=vector
                db.execute('INSERT OR REPLACE INTO embeddings VALUES (?,?)',(hashes[index],json.dumps(vector)))
    array=np.asarray([existing[key] for key in hashes],dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError('Non-finite embeddings')
    return array/np.maximum(np.linalg.norm(array,axis=1,keepdims=True),1e-8)

async def search(library, query, k=5, document_id=None, overview=False, mode='hybrid'):
    retriever=library.retriever()
    scopes=document_id if isinstance(document_id,list) else [document_id] if document_id else []
    if len(scopes)>1:
        # Interleave results so neither document crowds out the other.
        groups=[await search(library,query,k=2,document_id=root,overview=overview,mode=mode) for root in scopes]
        return [group[i] for i in range(2) for group in groups if len(group)>i]
    lexical=retriever.search(query,k=max(20,k),document_id=document_id,overview=overview)
    if overview:
        return [{**doc,'retrieval_mode':'overview'} for doc in lexical[:k]]
    if mode=='keyword' or os.getenv('SEMANTIC_SEARCH','on').lower()=='off':
        return [{**doc,'retrieval_mode':'keyword'} for doc in lexical[:k]]
    candidates=[doc for doc in retriever.documents if not scopes or root_id(doc) in scopes]
    if not candidates:
        return []
    try:
        url,_=local_config(); model=os.getenv('EMBEDDING_MODEL','all-minilm:22m')
        if 'cloud' in model.lower():
            raise ValueError('Only local embeddings are supported')
        async with httpx.AsyncClient(timeout=45,trust_env=False) as client:
            identity=await embedding_identity(client,url,model)
            # Inputs remain within this small model's context; full passages are retained for citations.
            texts=[(doc['title']+'\n'+doc['text'])[:1500] for doc in candidates]+[query[:1500]]
            embedded=await vectors(client,url,model,texts,identity,library.path)
        similarities=embedded[:-1]@embedded[-1]
        floor=float(os.getenv('SEMANTIC_MIN_SIMILARITY','0.32'))
        semantic=sorted([(doc,float(score)) for doc,score in zip(candidates,similarities) if score>=floor],key=lambda pair:-pair[1])[:20]
        fused=defaultdict(float)
        known={d['id']:d for d in lexical}
        for rank,doc in enumerate(lexical,1): fused[doc['id']]+=1/(60+rank)
        similarity={}
        for rank,(doc,score) in enumerate(semantic,1):
            fused[doc['id']]+=1/(60+rank); known[doc['id']]=doc; similarity[doc['id']]=score
        ordered=sorted(fused,key=lambda key:(-fused[key],key))[:k]
        return [{**known[key],'score':round(fused[key],5),'semantic_similarity':round(similarity[key],4) if key in similarity else None,'retrieval_mode':'hybrid','embedding_model':model} for key in ordered]
    except (httpx.HTTPError,ValueError,KeyError,TypeError,sqlite3.Error):
        return [{**doc,'retrieval_mode':'keyword_fallback','retrieval_warning':'Local semantic search unavailable; keyword results shown.'} for doc in lexical[:k]]
