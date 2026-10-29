import asyncio
import json
import pytest
import torch
from workflows.app import create_app
from workflows.engine import execute, retry, WorkflowError
from workflows.nlp import Normalizer, Seq2Seq
from workflows.retrieval import Retriever
from workflows.tools import LocalTools, tool_connection

@pytest.fixture
def normalizer(tmp_path):
    return Normalizer(tmp_path / "missing.pt")

def test_api_run_and_persistence(tmp_path, normalizer):
    app = create_app(tmp_path / "runs.db", normalizer)
    client = app.test_client()
    result = client.post("/api/workflows", json={"query": "mcp tool integration"})
    assert result.status_code == 200
    body = result.get_json()
    assert body["status"] == "completed" and body["sources"][0]["id"] == "mcp"
    assert [t["step"] for t in body["trace"]] == ["normalize", "search_documents", "get_document", "retrieval_quality"]
    assert client.get('/api/workflows/' + body['id']).get_json() == body
    assert client.get('/api/workflows/missing').status_code == 404
    for payload in [None, [], {"query": 3}, {"query": " "}, {"query": "x"*1001}, {"query": "mcp", "provider": "bad"}]:
        assert client.post("/api/workflows", json=payload).status_code == 400
    assert client.post("/api/normalize", json={"query": "hello"}).get_json()["mode"] == "identity"
    assert client.post("/api/workflows", data="x"*17000).status_code == 413

def test_failure_persisted(tmp_path, normalizer, monkeypatch):
    async def fail(*args):
        raise ConnectionError("private detail")
    monkeypatch.setattr("workflows.providers.complete", fail)
    client = create_app(tmp_path / "runs.db", normalizer).test_client()
    response = client.post("/api/workflows", json={"query": "mcp", "provider": "claude"})
    assert response.status_code == 502
    body = response.get_json()
    assert "private detail" not in json.dumps(body)
    assert client.get('/api/workflows/' + body['id']).get_json()["status"] == "failed"

def test_retrieval_equivalence():
    retriever = Retriever()
    for query in ["mcp tools", "model model flask", "unknownword", ""]:
        assert retriever.search(query) == retriever.search(query, indexed=False)
    assert retriever.search("xyzzy") == []

def test_tool_allowlist():
    async def run():
        tools = LocalTools()
        for name, args in [("exec", {}), ("get_document", {"document_id": "../../etc/passwd"}), ("search_documents", {"query": "mcp", "extra": True})]:
            with pytest.raises(ValueError):
                await tools.call(name, args)
    asyncio.run(run())

def test_retry_transient_only():
    async def run():
        calls = 0
        async def flaky():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise TimeoutError()
            return "ok"
        trace = []
        assert await retry(flaky, trace, "test") == "ok"
        assert len(trace) == 3
        async def invalid():
            raise ValueError("invalid")
        trace = []
        with pytest.raises(ValueError):
            await retry(invalid, trace, "bad")
        assert len(trace) == 1
    asyncio.run(run())

def test_agent_loop_and_provider_fallback(normalizer, monkeypatch):
    actions = iter([json.dumps({"tool": "search_documents", "arguments": {"query": "mcp"}}), json.dumps({"answer": "MCP exposes tools [mcp]"})])
    async def complete(provider, history):
        if provider == "claude":
            raise ValueError("not configured")
        return next(actions)
    monkeypatch.setattr("workflows.providers.complete", complete)
    result = asyncio.run(execute("mcp", normalizer, "auto"))
    assert result["provider"] == "openai"
    assert result["sources"]
    assert any(t["step"] == "fallback" for t in result["trace"])

def test_agent_step_limit(normalizer, monkeypatch):
    async def malformed(*args):
        return "not json"
    monkeypatch.setattr("workflows.providers.complete", malformed)
    with pytest.raises(WorkflowError):
        asyncio.run(execute("mcp", normalizer, "openai"))

def test_checkpoint_round_trip(tmp_path):
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>", "hello"]
    model = Seq2Seq(len(vocab))
    path = tmp_path / "model.pt"
    torch.save({"vocab": vocab, "hidden": 64, "state_dict": model.state_dict()}, path)
    nlp = Normalizer(path)
    assert nlp.model is not None
    assert isinstance(nlp.normalize("hello")["query"], str)
    assert nlp.normalize("unseen")["mode"] == "identity"
    output = model(torch.tensor([[4, 2]]), torch.tensor([[1, 4]]))
    assert output.shape == (1, 2, len(vocab))

def test_real_mcp_transport(normalizer):
    pytest.importorskip("mcp")
    async def run():
        async with tool_connection("mcp") as tools:
            hits = await tools.call("search_documents", {"query": "mcp"})
            assert hits[0]["id"] == "mcp"
            doc = await tools.call("get_document", {"document_id": "mcp"})
            assert doc["id"] == "mcp"
    asyncio.run(run())

def test_document_import_search_history_and_restart(tmp_path, normalizer):
    database = tmp_path / 'runs.db'
    client = create_app(database, normalizer).test_client()
    assert client.post('/api/documents', json={'title': ' ', 'text': 'bad'}).status_code == 400
    assert client.post('/api/documents', json={'title': 'Test', 'text': 'x' * 2000001}).status_code == 400
    imported = client.post('/api/documents', json={'title': 'Orion launch notes', 'text': 'The Orion launch readiness review is on Thursday. The owner is Mira.'})
    assert imported.status_code == 201
    doc_id = imported.get_json()['id']
    listing = client.get('/api/documents').get_json()['documents']
    assert len(listing) == 8 and listing[-1]['id'] == doc_id
    run = client.post('/api/workflows', json={'query': 'Orion readiness'}).get_json()
    assert run['status'] == 'completed'
    assert run['sources'][0]['id'] == doc_id
    assert 'Thursday' in run['answer'] and run['elapsed_ms'] >= 0
    history = client.get('/api/workflows').get_json()['runs']
    assert history[0]['query'] == 'Orion readiness' and history[0]['id'] == run['id']
    restarted = create_app(database, normalizer).test_client()
    assert len(restarted.get('/api/documents').get_json()['documents']) == 8
    assert restarted.get('/api/workflows/' + run['id']).get_json() == run
    html = restarted.get('/').data
    assert b'Add your own knowledge' in html
    assert restarted.get('/static/workspace.js').status_code == 200


def test_large_document_chunking_and_mcp_visibility(tmp_path):
    pytest.importorskip('mcp')
    from workflows.library import Library
    path = tmp_path / 'library.sqlite'
    library = Library(path)
    assert library.retriever().search('quartznebula') == []
    doc_id = library.add('Observatory notes', 'Astronomy observations. ' * 150 + 'quartznebula appears at midnight.')
    local_hits = library.retriever().search('quartznebula')
    assert local_hits and local_hits[0]['id'].startswith(doc_id)
    assert len(local_hits[0]['text']) <= 1800
    async def run():
        async with tool_connection('mcp', path) as tools:
            hits = await tools.call('search_documents', {'query': 'quartznebula'})
            assert [(d['id'],d['text']) for d in hits] == [(d['id'],d['text']) for d in local_hits]
            doc = await tools.call('get_document', {'document_id': hits[0]['id']})
            assert 'midnight' in doc['text']
    asyncio.run(run())


def test_workflow_history_pagination_is_stable(tmp_path, normalizer):
    import sqlite3
    database = tmp_path / 'runs.db'
    client = create_app(database, normalizer).test_client()
    with sqlite3.connect(database) as db:
        for i in range(12):
            db.execute('INSERT INTO runs VALUES (?,?,?)', (str(i), '2026-10-28T12:00:00+00:00', json.dumps({'query':f'Question {i}', 'status':'completed', 'provider':'demo'})))
    first = client.get('/api/workflows').get_json()
    assert len(first['runs']) == 5 and first['total'] == 12 and first['has_more']
    assert [r['id'] for r in first['runs']] == ['11','10','9','8','7']
    # New results inserted between pages must not duplicate or skip older rows.
    with sqlite3.connect(database) as db:
        db.execute('INSERT INTO runs VALUES (?,?,?)', ('new','2026-10-28',json.dumps({'query':'New question'})))
    second = client.get('/api/workflows?before=' + str(first['next_cursor'])).get_json()
    assert [r['id'] for r in second['runs']] == ['6','5','4','3','2']
    last = client.get('/api/workflows?before=' + str(second['next_cursor'])).get_json()
    assert [r['id'] for r in last['runs']] == ['1','0']
    assert not last['has_more'] and last['next_cursor'] is None
    for query in ['limit=0', 'limit=51', 'limit=bad', 'before=-2', 'before=bad']:
        assert client.get('/api/workflows?' + query).status_code == 400
