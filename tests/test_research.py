import io
import sqlite3
import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject,NameObject,DecodedStreamObject
from workflows.app import create_app
from workflows.library import Library
from workflows.nlp import Normalizer
from workflows.research import source_details


def make_pdf():
    writer=PdfWriter()
    for text in ['The riverstone design uses a cached index for document lookup.', 'The uniquezephyr experiment reduced document lookup latency significantly.']:
        page=writer.add_blank_page(width=612,height=792)
        font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
        page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
        stream=DecodedStreamObject();stream.set_data(f'BT /F1 12 Tf 40 700 Td ({text}) Tj ET'.encode())
        page[NameObject('/Contents')]=writer._add_object(stream)
    output=io.BytesIO();writer.write(output);return output.getvalue()

@pytest.fixture
def client(tmp_path):
    return create_app(tmp_path/'runs.db',Normalizer(tmp_path/'none.pt')).test_client()


def test_pdf_pages_original_and_highlight(client):
    response=client.post('/api/documents',data={'file':(io.BytesIO(make_pdf()),'experiment.pdf')},content_type='multipart/form-data')
    assert response.status_code==201
    root=response.get_json()['id']
    run=client.post('/api/workflows',json={'query':'uniquezephyr','document_id':root}).get_json()
    assert run['sources'][0]['page']==2
    details=client.get(f"/api/workflows/{run['id']}/sources/1").get_json()
    assert details['page']==2 and not details['page_inferred']
    assert details['highlight'] and 'uniquezephyr' in details['page_text'][slice(*details['highlight'])]
    original=client.get(details['pdf_url'])
    assert original.status_code==200 and original.mimetype=='application/pdf'
    assert original.data==make_pdf()
    assert client.get(f"/api/workflows/{run['id']}/sources/999").status_code==404
    assert client.get('/api/documents/missing/pdf').status_code==404


def test_legacy_source_is_recovered_without_guessing_new_page(tmp_path):
    library=Library(tmp_path/'library.sqlite')
    root=library.add('Legacy PDF','[Page 1]\nThe first section introduces network tools and schema validation.\n\n[Page 2]\nThe uniquezephyr experiment reports significantly lower retrieval latency.',{'format':'pdf'})
    source={'id':root+'-part-7','text':'The uniquezephyr experiment reports significantly lower retrieval latency.'}
    result=source_details(library,source)
    assert result['page']==2 and result['page_inferred'] and result['pdf_url'] is None


def test_conversation_retains_context_and_validates_scope(client,monkeypatch):
    seen=[]
    async def answer(query,sources,context=None):
        seen.append((query,context,sources))
        return 'MCP lets an assistant call standardized tools with defined inputs. [1]'
    monkeypatch.setattr('workflows.providers.local_answer',answer)
    first=client.post('/api/workflows',json={'query':'Explain MCP tools','provider':'local','document_id':'mcp'}).get_json()
    assert first['status']=='completed'
    second=client.post('/api/workflows',json={'query':'Explain that more simply','provider':'local','conversation_id':first['conversation_id']}).get_json()
    assert second['status']=='completed' and second['document_id']=='mcp'
    assert seen[1][1] and 'MCP' in seen[1][1][0]['query']
    conversation=client.get('/api/conversations/'+first['conversation_id']).get_json()
    assert [m['id'] for m in conversation['messages']]==[first['id'],second['id']]
    assert any(s['step']=='conversation_context' for s in second['trace'])
    assert client.post('/api/workflows',json={'query':'Other question','conversation_id':first['conversation_id'],'document_id':'flask'}).status_code==400
    assert client.post('/api/workflows',json={'query':'Question','conversation_id':'missing'}).status_code==400


def test_evaluation_and_feedback_are_honest_and_persistent(client):
    initial=client.get('/api/evaluations').get_json()
    assert initial['latest'] is None and initial['feedback']['rated_answers']==0
    result=client.post('/api/evaluations',json={}).get_json()
    assert result['questions']==50
    assert 0<=result['metrics']['original']['recall_at_3']<=1
    assert result['metrics']['original']['recall_at_3']==result['metrics']['seq2seq']['recall_at_3']
    assert 'Retrieval-only' in result['limitations']
    assert client.get('/api/evaluations').get_json()['latest']['id']==result['id']
    run=client.post('/api/workflows',json={'query':'mcp tools'}).get_json()
    endpoint='/api/workflows/'+run['id']+'/feedback'
    assert client.post(endpoint,json={'helpful':True,'supported':False,'note':'Needs a better citation.'}).status_code==200
    feedback=client.get('/api/evaluations').get_json()['feedback']
    assert feedback['rated_answers']==1 and feedback['helpful_rate']==1 and feedback['supported_rate']==0
    assert client.post(endpoint,json={'helpful':'yes','supported':True}).status_code==400


def test_original_attachment_requires_same_text(client,tmp_path):
    from workflows.ingestion import extract
    data=make_pdf(); text,metadata=extract(data,'.pdf')
    library=Library(tmp_path/'library.sqlite')
    root=library.add('Old upload',text,metadata)
    response=client.post('/api/documents/'+root+'/original',data={'file':(io.BytesIO(data),'same.pdf')},content_type='multipart/form-data')
    assert response.status_code==200
    assert client.get('/api/documents/'+root+'/pdf').data==data
    other=library.add('Different upload','[Page 1]\nThis is an entirely unrelated document with different text.',{'format':'pdf'})
    response=client.post('/api/documents/'+other+'/original',data={'file':(io.BytesIO(data),'wrong.pdf')},content_type='multipart/form-data')
    assert response.status_code==400
