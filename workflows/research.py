"""Conversations, traceable source pages, feedback, and reproducible retrieval evaluation."""
import io
import asyncio
import json
import re
import sqlite3
import statistics
import time
import uuid
from datetime import datetime, timezone
from flask import jsonify, request, send_file
from .retrieval import ROOT, tokens
from .text import clean_text
from .hybrid import search as hybrid_search

class ResearchStore:
    def __init__(self, database):
        self.database = database
        with sqlite3.connect(database) as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, title TEXT, document_id TEXT, created TEXT);
                CREATE TABLE IF NOT EXISTS conversation_runs (run_id TEXT PRIMARY KEY, conversation_id TEXT);
                CREATE INDEX IF NOT EXISTS conversation_lookup ON conversation_runs(conversation_id);
                CREATE TABLE IF NOT EXISTS feedback (run_id TEXT PRIMARY KEY, helpful INTEGER, supported INTEGER, note TEXT);
                CREATE TABLE IF NOT EXISTS evaluations (id TEXT PRIMARY KEY, created TEXT, result TEXT);
                CREATE TABLE IF NOT EXISTS evaluation_sets (id TEXT PRIMARY KEY, name TEXT, dataset TEXT, created TEXT);
            ''')

    def create(self, title, document_id):
        conversation_id = uuid.uuid4().hex
        with sqlite3.connect(self.database) as db:
            db.execute('INSERT INTO conversations VALUES (?,?,?,?)', (conversation_id, title[:80], json.dumps(document_id) if isinstance(document_id,list) else document_id, datetime.now(timezone.utc).isoformat()))
        return conversation_id

    def conversation(self, conversation_id):
        with sqlite3.connect(self.database) as db:
            row = db.execute('SELECT title,document_id FROM conversations WHERE id=?', (conversation_id,)).fetchone()
            if not row:
                return None
            messages = db.execute('SELECT r.result FROM runs r JOIN conversation_runs c ON c.run_id=r.id WHERE c.conversation_id=? ORDER BY r.rowid DESC LIMIT 20', (conversation_id,)).fetchall()
        return {'id':conversation_id, 'title':row[0], 'document_id':json.loads(row[1]) if row[1] and row[1].startswith('[') else row[1], 'messages':[json.loads(r[0]) for r in reversed(messages)]}

    def link(self, run_id, conversation_id):
        with sqlite3.connect(self.database) as db:
            db.execute('INSERT INTO conversation_runs VALUES (?,?)', (run_id, conversation_id))

    def run(self, run_id):
        with sqlite3.connect(self.database) as db:
            row = db.execute('SELECT result FROM runs WHERE id=?', (run_id,)).fetchone()
        return json.loads(row[0]) if row else None


def source_details(library, source):
    document_id = source.get('document_id') or re.sub(r'-part-\d+$', '', source['id'])
    doc = next((d for d in library.documents() if d['id'] == document_id), None)
    if not doc:
        return None
    raw = doc['text']
    markers = list(re.finditer(r'\[Page (\d+)\]', raw)) if doc.get('metadata', {}).get('format') == 'pdf' else []
    pages = [(int(m[1]), clean_text(raw[m.start():markers[i+1].start() if i+1<len(markers) else len(raw)])) for i,m in enumerate(markers)] if markers else [(None, clean_text(raw))]
    page = source.get('page')
    inferred = page is None and bool(markers)
    if page is None:
        # Legacy runs lack page metadata. Recover the most relevant extracted page.
        terms = set(tokens(source['text']))
        page = max(pages, key=lambda p: len(terms & set(tokens(p[1]))))[0]
    page_text = next((text for number,text in pages if number==page), pages[0][1])
    source_words = re.findall(r'\S+', clean_text(source['text']))
    page_words = list(re.finditer(r'\S+', page_text))
    normalized = ' '.join(m[0] for m in page_words)
    phrase = ' '.join(source_words)
    location = normalized.find(phrase)
    # Sentence boundaries can change whitespace. Highlight a short exact prefix if
    # an old character-based passage cannot be matched in full.
    if location < 0:
        phrase = ' '.join(source_words[:30])
        location = normalized.find(phrase)
    highlight = None
    if location >= 0 and phrase:
        start_word = normalized[:location].count(' ')
        end_word = min(start_word + len(phrase.split()) - 1, len(page_words)-1)
        highlight = [page_words[start_word].start(), page_words[end_word].end()]
    return {'document_id':document_id, 'title':doc['title'], 'page':page, 'page_inferred':inferred,
            'page_text':page_text, 'passage':source['text'], 'highlight':highlight,
            'pdf_url':f'/api/documents/{document_id}/pdf' if doc.get('metadata', {}).get('has_original') else None,
            'format':doc.get('metadata',{}).get('format','text'), 'ocr_page':page in doc.get('metadata',{}).get('ocr_pages',[]), 'ocr_confidence':doc.get('metadata',{}).get('ocr_confidence',{}).get(str(page))}


def evaluate_retrieval(library, normalizer, dataset=None):
    dataset=dataset or json.loads((ROOT/'data/evaluation.json').read_text())
    retriever=library.retriever()
    async def run():
        await hybrid_search(library,dataset['questions'][0]['query'],k=3)
        normalizer.normalize(dataset['questions'][0]['query'])
        rows=[]
        for case in dataset['questions']:
            row={**case}
            for mode in ('original','hybrid','seq2seq'):
                start=time.perf_counter()
                normalized=normalizer.normalize(case['query']) if mode=='seq2seq' else {'query':case['query'],'mode':'original'}
                hits=await hybrid_search(library,normalized['query'],k=3,mode='keyword' if mode=='original' else 'hybrid')
                roots=list(dict.fromkeys(doc.get('document_id',doc['id']) for doc in hits))
                expected=set(case['expected_documents'])
                rank=next((i+1 for i,root in enumerate(roots) if root in expected),None)
                row[mode]={'query':normalized['query'],'normalization':normalized['mode'],'retrieved':roots,
                           'recall_at_3':len(expected & set(roots))/len(expected) if expected else None,
                           'reciprocal_rank':1/rank if rank else 0,'no_hit_correct':not hits if not expected else None,
                           'retrieval_mode':hits[0].get('retrieval_mode') if hits else 'no_hits','ms':round((time.perf_counter()-start)*1000,3)}
            rows.append(row)
        return rows
    rows=asyncio.run(run());metrics={}
    mean=lambda values:statistics.mean(values) if values else None
    for mode in ('original','hybrid','seq2seq'):
        answered=[row[mode] for row in rows if row['expected_documents']]
        negative=[row[mode] for row in rows if not row['expected_documents']]
        latency=sorted(row[mode]['ms'] for row in rows)
        metrics[mode]={'recall_at_3':mean([r['recall_at_3'] for r in answered]),'mrr':mean([r['reciprocal_rank'] for r in answered]),
                       'no_hit_accuracy':mean([r['no_hit_correct'] for r in negative]),'median_ms':statistics.median(latency),
                       'p95_ms':latency[int(.95*(len(latency)-1))], 'hybrid_queries':sum(r[mode]['retrieval_mode']=='hybrid' for r in rows)}
    return {'dataset':dataset['name'],'scope':dataset['scope'],'questions':len(rows),'corpus_passages':len(retriever.documents),
            'metrics':metrics,'cases':rows,'model_loaded':normalizer.model is not None,
            'limitations':'Retrieval-only evaluation. No automatic answer faithfulness score. Timings use a warm document index and include embedding requests for hybrid search, but exclude answer generation, app HTTP and MCP. No-hit accuracy is not full answer abstention. Review labels when the library changes.'}


def register_research(app, database, library, normalizer):
    store = ResearchStore(database)

    def load_dataset(identifier):
        if identifier=='builtin':return json.loads((ROOT/'data/evaluation.json').read_text())
        if not isinstance(identifier,str):raise ValueError('Choose an evaluation set')
        with sqlite3.connect(database) as db:
            row=db.execute('SELECT dataset FROM evaluation_sets WHERE id=?',(identifier,)).fetchone()
        return json.loads(row[0]) if row else None

    @app.get('/api/evaluation-sets')
    def evaluation_sets():
        with sqlite3.connect(database) as db:
            rows=db.execute('SELECT id,name,dataset FROM evaluation_sets ORDER BY rowid DESC LIMIT 100').fetchall()
        return jsonify(sets=[{'id':'builtin','name':'Educational fixtures (50 questions)','questions':50}]+[{'id':i,'name':name,'questions':len(json.loads(data)['questions'])} for i,name,data in rows])

    @app.get('/api/evaluation-sets/<identifier>')
    def get_evaluation_set(identifier):
        dataset=load_dataset(identifier)
        return (jsonify(dataset),200) if dataset else (jsonify(error='Evaluation set not found'),404)

    @app.post('/api/evaluation-sets')
    def save_evaluation_set():
        body=request.get_json(silent=True)
        if not isinstance(body,dict) or not isinstance(body.get('name'),str) or not 1<=len(body['name'].strip())<=80:
            raise ValueError('Name the evaluation set (1–80 characters)')
        cases=body.get('questions')
        if not isinstance(cases,list) or not 1<=len(cases)<=100:raise ValueError('Add 1–100 labeled questions')
        known={d['id'] for d in library.documents()};normalized=[];seen=set()
        for case in cases:
            if not isinstance(case,dict) or not isinstance(case.get('query'),str) or not 1<=len(case['query'].strip())<=1000:
                raise ValueError('Each question must contain 1–1000 characters')
            expected=case.get('expected_documents')
            if not isinstance(expected,list) or len(expected)>8 or any(not isinstance(i,str) or i not in known for i in expected):
                raise ValueError('Label questions with existing documents, or leave labels empty for an unanswerable question')
            query=case['query'].strip()
            if query.casefold() in seen:raise ValueError('Remove duplicate questions before saving')
            seen.add(query.casefold());normalized.append({'query':query,'expected_documents':list(dict.fromkeys(expected))})
        identifier=uuid.uuid4().hex
        dataset={'name':body['name'].strip(),'scope':'User-authored questions and document-level relevance labels. Results depend on label quality and the current library.','questions':normalized}
        with sqlite3.connect(database) as db:
            db.execute('INSERT INTO evaluation_sets VALUES (?,?,?,?)',(identifier,dataset['name'],json.dumps(dataset),datetime.now(timezone.utc).isoformat()))
        return jsonify(id=identifier,status='saved'),201

    @app.get('/api/conversations/<conversation_id>')
    def conversation(conversation_id):
        result = store.conversation(conversation_id)
        return (jsonify(result),200) if result else (jsonify(error='Conversation not found'),404)

    @app.get('/api/workflows/<run_id>/sources/<int:index>')
    def source(run_id, index):
        run = store.run(run_id)
        if not run or not 1 <= index <= len(run.get('sources',[])):
            return jsonify(error='Source not found'),404
        result = source_details(library, run['sources'][index-1])
        return (jsonify(result),200) if result else (jsonify(error='Document not found'),404)

    @app.get('/api/documents/<document_id>/pdf')
    def original_pdf(document_id):
        content = library.original_pdf(document_id)
        if content is None:
            return jsonify(error='Original PDF unavailable. Attach the original or upload it again.'),404
        response = send_file(io.BytesIO(content), mimetype='application/pdf', as_attachment=False, download_name=document_id+'.pdf')
        response.headers['Cache-Control']='no-store'
        return response

    @app.post('/api/workflows/<run_id>/feedback')
    def feedback(run_id):
        if not store.run(run_id):
            return jsonify(error='Run not found'),404
        body = request.get_json(silent=True)
        if not isinstance(body,dict) or type(body.get('helpful')) is not bool or type(body.get('supported')) is not bool:
            raise ValueError('Rate whether the answer was helpful and supported by its sources')
        note = body.get('note','')
        if not isinstance(note,str) or len(note)>1000:
            raise ValueError('Feedback notes allow up to 1000 characters')
        with sqlite3.connect(database) as db:
            db.execute('INSERT OR REPLACE INTO feedback VALUES (?,?,?,?)',(run_id,int(body['helpful']),int(body['supported']),note))
        return jsonify(status='saved')

    @app.get('/api/evaluations')
    def evaluations():
        with sqlite3.connect(database) as db:
            latest=db.execute('SELECT result FROM evaluations ORDER BY rowid DESC LIMIT 1').fetchone()
            rated=db.execute('SELECT COUNT(*),AVG(helpful),AVG(supported) FROM feedback').fetchone()
        return jsonify(latest=json.loads(latest[0]) if latest else None, feedback={'rated_answers':rated[0], 'helpful_rate':rated[1], 'supported_rate':rated[2]})

    @app.post('/api/evaluations')
    def run_evaluation():
        if not request.is_json:
            raise ValueError('Send a JSON request')
        body=request.get_json(silent=True)
        if not isinstance(body,dict):raise ValueError('Provide evaluation options')
        selected=body.get('dataset_id','builtin')
        dataset=load_dataset(selected)
        if dataset is None:return jsonify(error='Evaluation set not found'),404
        result=evaluate_retrieval(library,normalizer,dataset)
        result['dataset_id']=selected
        result.update(id=uuid.uuid4().hex,created=datetime.now(timezone.utc).isoformat())
        with sqlite3.connect(database) as db:
            db.execute('INSERT INTO evaluations VALUES (?,?,?)',(result['id'],result['created'],json.dumps(result)))
        return jsonify(result)

    return store
