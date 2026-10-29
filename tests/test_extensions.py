import asyncio,io,json,threading
import httpx
import pytest
from workflows.app import create_app
from workflows.library import Library
from workflows.nlp import Normalizer
from workflows.providers import GroundedAnswer,local_answer
from workflows.hybrid import search
from workflows.progress import model_preview

@pytest.fixture
def client(tmp_path):
    return create_app(tmp_path/'runs.db',Normalizer(tmp_path/'none.pt')).test_client()


def test_comparison_balances_documents_and_enforces_scope(client,monkeypatch):
    seen=[]
    async def answer(query,sources,context=None):
        seen.append(sources)
        return GroundedAnswer('MCP exposes tools, while reliability guidance explains retries. [1] [2]',True)
    monkeypatch.setattr('workflows.providers.local_answer',answer)
    result=client.post('/api/workflows',json={'query':'Compare these documents','provider':'local','document_ids':['mcp','reliability']})
    assert result.status_code==200,result.get_json()
    body=result.get_json();assert body['evidence_status']=='model_supported'
    assert {d['document_id'] for d in seen[0]}=={'mcp','reliability'}
    assert body['document_id']==['mcp','reliability']
    follow=client.post('/api/workflows',json={'query':'What differs?','provider':'local','conversation_id':body['conversation_id']})
    assert follow.status_code==200
    assert client.post('/api/workflows',json={'query':'Compare','document_ids':['mcp','mcp']}).status_code==400


def test_semantic_search_finds_different_wording_and_caches(tmp_path,monkeypatch):
    monkeypatch.setenv('SEMANTIC_SEARCH','on')
    library=Library(tmp_path/'library.sqlite')
    root=library.add('Resilience guide','Network failures are retried with exponential delays to recover automatically.')
    original=httpx.AsyncClient;embedded=[]
    def handler(request):
        if request.url.path=='/api/tags':return httpx.Response(200,json={'models':[{'name':'all-minilm:22m','digest':'test-digest'}]})
        texts=json.loads(request.content)['input'];embedded.extend(texts)
        return httpx.Response(200,json={'embeddings':[[1.,0.] if 'Network failures' in text or 'temporary service hiccups' in text else [0.,1.] for text in texts]})
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))
    async def run():
        assert not library.retriever().search('temporary service hiccups')
        hits=await search(library,'temporary service hiccups')
        assert hits[0]['document_id']==root and hits[0]['retrieval_mode']=='hybrid'
        count=len(embedded);await search(library,'temporary service hiccups');assert len(embedded)==count
    asyncio.run(run())


def test_unsupported_model_answer_is_not_presented_as_fact(monkeypatch):
    original=httpx.AsyncClient
    def handler(request):
        return httpx.Response(200,json={'message':{'content':json.dumps({'answer':'Made up cost 100 dollars.','supported':False,'missing_evidence':'No electricity costs are reported.','sources':[]})}})
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))
    answer=asyncio.run(local_answer('What was the electricity bill?',[{'id':'x','title':'Paper','text':'The model trained on GPUs.'}]))
    assert answer.supported is False and '100 dollars' not in answer and 'not provide enough evidence' in answer


def test_streamed_job_emits_draft_and_final(client,monkeypatch):
    async def answer(query,sources,context=None):
        model_preview('{"answer":"This is a streamed draft')
        await asyncio.sleep(.03)
        return GroundedAnswer('MCP exposes standardized tools. [1]',True)
    monkeypatch.setattr('workflows.providers.local_answer',answer)
    started=client.post('/api/workflow-jobs',json={'query':'mcp tools','provider':'local'})
    assert started.status_code==202
    events=client.get('/api/workflow-jobs/'+started.get_json()['job_id']+'/events').data.decode()
    assert '"type": "draft"' in events and '"type": "result"' in events and '"type": "done"' in events
    assert 'streamed draft' in events


def test_job_cancel_stops_generation_and_saves_no_draft(client,monkeypatch):
    entered=threading.Event()
    async def answer(*args):
        entered.set();await asyncio.sleep(60)
    monkeypatch.setattr('workflows.providers.local_answer',answer)
    started=client.post('/api/workflow-jobs',json={'query':'mcp tools','provider':'local'}).get_json()
    assert entered.wait(3)
    assert client.delete('/api/workflow-jobs/'+started['job_id']).status_code==200
    events=client.get('/api/workflow-jobs/'+started['job_id']+'/events').data.decode()
    assert '"status": "cancelled"' in events
    run=client.get('/api/workflows').get_json()['runs'][0]
    assert run['status']=='cancelled'


def test_custom_evaluation_set(client):
    body={'name':'My documents','questions':[{'query':'How are tools exposed?','expected_documents':['mcp']},{'query':'What did the author eat?','expected_documents':[]}]}
    result=client.post('/api/evaluation-sets',json=body)
    assert result.status_code==201
    selected=result.get_json()['id']
    report=client.post('/api/evaluations',json={'dataset_id':selected}).get_json()
    assert report['questions']==2 and report['dataset']=='My documents' and 'hybrid' in report['metrics']
    assert client.get('/api/evaluation-sets/'+selected).get_json()['questions']==body['questions']
    assert client.post('/api/evaluation-sets',json={'name':'Bad','questions':[{'query':'test','expected_documents':['missing']}]}).status_code==400


def test_local_ocr_recovers_image_only_pdf():
    from PIL import Image,ImageDraw,ImageFont
    from workflows.ingestion import extract
    from workflows.ocr import status
    if not status()['available']:pytest.skip('OCR dependencies unavailable')
    image=Image.new('RGB',(1400,900),'white');draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',48)
    draw.text((80,150),'QUARTERLY RESEARCH REVIEW',font=font,fill='black')
    draw.text((80,270),'MEETING THURSDAY AT NOON',font=font,fill='black')
    buffer=io.BytesIO();image.save(buffer,format='PDF',resolution=120)
    text,metadata=extract(buffer.getvalue(),'.pdf',True)
    assert 'THURSDAY' in text and metadata['ocr_pages']==[1]
    assert metadata['ocr_confidence']['1']>0

@pytest.mark.parametrize('provider,module,client_name',[('openai','openai','AsyncOpenAI'),('claude','anthropic','AsyncAnthropic')])
def test_cloud_streaming_adapters_emit_answer_drafts(monkeypatch,provider,module,client_name):
    import sys
    from types import SimpleNamespace
    from workflows import providers
    from workflows.progress import progress_scope
    output='{"answer":"Grounded explanation","supported":true,"missing_evidence":""}'
    async def chunks():
        for text in [output[:20],output[20:]]:
            yield SimpleNamespace(type='response.output_text.delta',delta=text) if provider=='openai' else text
    class ClaudeStream:
        text_stream=None
        async def __aenter__(self):self.text_stream=chunks();return self
        async def __aexit__(self,*args):pass
    class Client:
        def __init__(self,**kwargs):
            self.responses=self;self.messages=self
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def create(self,**kwargs):
            assert kwargs['stream'] is True and kwargs['store'] is False
            return chunks()
        def stream(self,**kwargs):return ClaudeStream()
    monkeypatch.setitem(sys.modules,module,SimpleNamespace(**{client_name:Client}))
    monkeypatch.setattr(providers,'provider_config',lambda p:SimpleNamespace(configured=True,model='test-model',api_key='test-only'))
    events=[]
    with progress_scope(events.append):
        result=asyncio.run(providers.complete(provider,'Question and evidence'))
    assert result==output
    assert events[-1]=={'type':'draft','data':{'text':'Grounded explanation'}}
