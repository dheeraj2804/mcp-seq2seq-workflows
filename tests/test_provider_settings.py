import asyncio
import json
import os
import re
from types import SimpleNamespace
import pytest
from workflows.app import create_app
from workflows.nlp import Normalizer
from workflows.settings import ProviderSettings, ProviderConfig, provider_scope, provider_config

KEY = 'sk-test-only-credential-not-a-real-key'

@pytest.fixture
def setup(tmp_path, monkeypatch):
    for name in ('OPENAI_API_KEY','OPENAI_MODEL','ANTHROPIC_API_KEY','ANTHROPIC_MODEL'):
        monkeypatch.delenv(name, raising=False)
    app = create_app(tmp_path/'runs.db', Normalizer(tmp_path/'none.pt'))
    client = app.test_client()
    token = re.search(rb'name="settings-token" content="([^"]+)"', client.get('/').data).group(1).decode()
    return app, client, {'X-Settings-Token': token}, tmp_path


def test_save_masking_disconnect_and_restart(setup):
    app, client, headers, path = setup
    result = client.post('/api/settings/providers/openai', json={'api_key': KEY, 'model':'test-text-model'}, headers=headers)
    assert result.status_code == 200
    assert KEY not in result.get_data(as_text=True)
    public = client.get('/api/settings/providers')
    assert public.headers['Cache-Control'] == 'no-store'
    assert public.get_json()['providers']['openai']['configured']
    assert 'api_key' not in public.get_data(as_text=True)
    assert os.getenv('OPENAI_API_KEY') is None
    # Empty password means preserve the saved key when editing the model.
    edited = client.post('/api/settings/providers/openai', json={'api_key':'', 'model':'other-model'}, headers=headers)
    assert edited.status_code == 200
    assert app.extensions['provider_settings'].snapshot()['openai'].api_key == KEY
    client.delete('/api/settings/providers/openai', headers=headers)
    assert not client.get('/api/settings/providers').get_json()['providers']['openai']['has_key']
    assert not create_app(path/'runs.db', Normalizer(path/'none.pt')).extensions['provider_settings'].snapshot()['openai'].configured
    for file in path.glob('*.db'):
        assert KEY.encode() not in file.read_bytes()


def test_settings_csrf_and_validation(setup):
    _, client, headers, _ = setup
    payload = {'api_key':KEY, 'model':'test-model'}
    assert client.post('/api/settings/providers/openai', json=payload).status_code == 403
    assert client.post('/api/settings/providers/openai', json=payload, headers={**headers, 'Origin':'https://untrusted.example'}).status_code == 403
    for provider, body in [('bad',payload), ('openai',{}), ('openai',{'api_key':'short','model':'text'}), ('openai',{'api_key':KEY,'model':'bad model'}), ('openai',{'api_key':KEY,'model':''})]:
        response = client.post('/api/settings/providers/' + provider, json=body, headers=headers)
        assert response.status_code == 400
        assert KEY not in response.get_data(as_text=True)


def test_connection_and_models_use_draft_without_saving(setup, monkeypatch):
    _, client, headers, _ = setup
    async def models(provider, config):
        assert provider == 'claude' and config.api_key == KEY
        return ['available-text-model']
    async def test(provider, config):
        assert config.api_key == KEY and config.model == 'available-text-model'
    monkeypatch.setattr('workflows.app.list_provider_models', models)
    monkeypatch.setattr('workflows.app.test_provider_connection', test)
    result = client.post('/api/settings/providers/claude/models', json={'api_key':KEY}, headers=headers)
    assert result.get_json()['models'] == ['available-text-model']
    response = client.post('/api/settings/providers/claude/test', json={'api_key':KEY,'model':'available-text-model'}, headers=headers)
    assert response.get_json()['status'] == 'verified'
    assert not client.get('/api/settings/providers').get_json()['providers']['claude']['has_key']
    class Rejected(Exception):
        status_code = 401
    async def fail(*args):
        raise Rejected('private request header ' + KEY)
    monkeypatch.setattr('workflows.app.test_provider_connection', fail)
    error = client.post('/api/settings/providers/claude/test', json={'api_key':KEY,'model':'available-text-model'}, headers=headers)
    assert error.status_code == 502 and 'rejected' in error.get_json()['error']
    assert KEY not in error.get_data(as_text=True)


def test_workflow_uses_saved_key_without_exporting_it(setup, monkeypatch):
    _, client, headers, _ = setup
    client.post('/api/settings/providers/openai', json={'api_key':KEY,'model':'test-model'}, headers=headers)
    calls = 0
    async def complete(provider, history):
        nonlocal calls
        assert provider_config(provider).api_key == KEY
        calls += 1
        return json.dumps({'tool':'search_documents','arguments':{'query':'mcp'}}) if calls == 1 else json.dumps({'answer':'MCP exposes tools [mcp]'})
    monkeypatch.setattr('workflows.providers.complete', complete)
    response = client.post('/api/workflows', json={'query':'mcp','provider':'openai'})
    assert response.status_code == 200
    assert KEY not in response.get_data(as_text=True)
    assert KEY not in client.get('/api/workflows/'+response.get_json()['id']).get_data(as_text=True)


def test_contexts_and_environment_fallback_are_isolated(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'environment-key')
    monkeypatch.setenv('OPENAI_MODEL', 'environment-model')
    settings = ProviderSettings()
    assert settings.snapshot()['openai'].api_key == 'environment-key'
    settings.disconnect('openai')
    assert not settings.snapshot()['openai'].configured
    assert ProviderSettings().snapshot()['openai'].configured
    async def task(key):
        with provider_scope({'openai':ProviderConfig(key,'model')}):
            await asyncio.sleep(0)
            return provider_config('openai').api_key
    async def run():
        return await asyncio.gather(task('first'),task('second'))
    assert asyncio.run(run()) == ['first','second']
    assert 'private-key' not in repr(ProviderConfig('private-key','model'))
