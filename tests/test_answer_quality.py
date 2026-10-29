import asyncio
import pytest
from workflows.engine import execute
from workflows.library import Library
from workflows.nlp import Normalizer
from workflows.text import clean_text, passages


def test_readable_sentences_and_ligatures():
    text = 'Abstract\nWe introduce a signiﬁcantly faster document retrieval system.\nIt combines multiple query representations to improve search.\n1 Introduction\nA long inter-\nnational collaboration developed the system.'
    cleaned = clean_text(text)
    assert 'significantly' in cleaned and 'international' in cleaned
    chunks = passages(text, target=160)
    assert any(chunk.startswith('Abstract') for chunk in chunks)
    assert all(not chunk.startswith('ers,') for chunk in chunks)
    assert any('faster document retrieval system.' in chunk for chunk in chunks)


def test_overview_and_document_scope(tmp_path):
    library = Library(tmp_path/'library.sqlite')
    title = 'Copper Harbor research paper'
    root = library.add(title, 'Authors\nauthor@example.com\n\nAbstract\nWe introduce a new retrieval system that combines complementary document signals. The system improves search accuracy without changing the underlying documents. Experiments demonstrate lower latency than repeated full-corpus scans.\n1 Introduction\nEarlier systems scanned every document for each incoming question.\n2 Details\n' + 'Attention to implementation details is helpful. '*100)
    other = library.add('Unrelated notes', 'This document discusses irrelevant search details in another project. ' * 30)
    retriever = library.retriever()
    hits = retriever.search('Can you explain what Copper Harbor paper is about?', overview=True)
    assert hits[0]['section'] == 'abstract'
    assert all(d['document_id'] == root for d in hits)
    scoped = retriever.search('retrieval', document_id=other)
    assert all(d['document_id'] == other for d in scoped)
    result = asyncio.run(execute('What is this paper about?', Normalizer(tmp_path/'none.pt'), library_path=library.path, document_id=root))
    assert 'new retrieval system' in result['answer']
    assert result['answer_kind'] == 'extractive_summary'
    assert 'not an AI paraphrase' in result['answer_notice']


def test_local_generation_uses_retrieved_evidence(tmp_path, monkeypatch):
    library = Library(tmp_path/'library.sqlite')
    root = library.add('Quartz project', 'The Quartz project improves document retrieval using a cached inverted index.')
    async def answer(query, sources):
        assert query == 'Explain Quartz project'
        assert sources and all(d['document_id'] == root for d in sources)
        return 'Quartz helps find documents by keeping an index ready for searches. [1]'
    monkeypatch.setattr('workflows.providers.local_answer', answer)
    result = asyncio.run(execute('Explain Quartz project', Normalizer(tmp_path/'none.pt'), provider='local', library_path=library.path, document_id=root))
    assert result['provider'] == 'local'
    assert result['answer_kind'] == 'local_explanation'
    assert result['sources'][0]['citation'] == '1'
    assert any(s['step'] == 'local_model' for s in result['trace'])


def test_scoped_tools_over_mcp(tmp_path):
    from workflows.tools import tool_connection
    library = Library(tmp_path/'library.sqlite')
    root = library.add('Quartz overview', 'Abstract\nThe Quartz search system uses an index to retrieve documents quickly.\n1 Introduction\nIt replaces repeated scans with cached lookup tables.')
    async def run():
        async with tool_connection('mcp', library.path, root, True) as tools:
            hits = await tools.call('search_documents', {'query':'overview'})
            assert hits[0]['section'] == 'abstract'
            assert all(d['document_id'] == root for d in hits)
    asyncio.run(run())


def test_local_model_rejects_remote_endpoint(monkeypatch):
    from workflows.providers import local_config
    monkeypatch.setenv('OLLAMA_BASE_URL', 'https://example.com')
    with pytest.raises(ValueError, match='loopback'):
        local_config()


def test_structured_local_answer_and_sources(monkeypatch):
    import json
    import httpx
    from workflows.providers import local_answer
    original = httpx.AsyncClient
    def handler(request):
        payload = json.loads(request.content)
        assert payload['stream'] is False
        assert set(payload['format']['required']) == {'answer', 'explanation', 'details', 'sources','supported','missing_evidence'}
        content = {'answer': 'The project makes document search faster.', 'explanation': 'It keeps an index ready so searches avoid repeated scans.', 'details': 'The supplied notes describe cached lookup tables.', 'sources': [1]}
        return httpx.Response(200, json={'done_reason':'stop', 'message': {'content':json.dumps(content)}})
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    result = asyncio.run(local_answer('What is this project?', [{'title':'Project', 'text':'The project uses a cached inverted index.'}]))
    assert result.count('\n\n') == 3
    assert result.endswith('Sources: [1]')
    assert 'index ready' in result


def test_short_followup_has_no_forced_long_sections(monkeypatch):
    import json,httpx
    from workflows.providers import local_answer
    original=httpx.AsyncClient
    def handler(request):
        payload=json.loads(request.content)
        assert set(payload['format']['properties'])=={'answer','sources','supported','missing_evidence'}
        assert '2-3 short sentences' in payload['messages'][0]['content']
        return httpx.Response(200,json={'message':{'content':json.dumps({'answer':'The system finds the words that matter, so it can understand text faster.','sources':[1]})}})
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))
    result=asyncio.run(local_answer('Explain that in simpler words.',[{'title':'Model','text':'Attention relates different positions of a sequence.'}],[{'query':'Explain attention','answer':'It relates sequence positions.'}]))
    assert result.endswith('Sources: [1]') and 'understand text faster' in result
