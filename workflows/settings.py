"""Write-only, process-memory credentials. Never serialize keys to app responses."""
import os
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

ENV = {'openai': ('OPENAI_API_KEY', 'OPENAI_MODEL'), 'claude': ('ANTHROPIC_API_KEY', 'ANTHROPIC_MODEL')}

@dataclass(frozen=True)
class ProviderConfig:
    api_key: str = field(repr=False)
    model: str = ''
    source: str = 'environment'

    @property
    def configured(self):
        return bool(self.api_key and self.model)


def environment_config(provider):
    key, model = ENV[provider]
    return ProviderConfig(os.getenv(key, ''), os.getenv(model, ''))


class ProviderSettings:
    def __init__(self):
        self._lock = threading.Lock()
        self._overrides = {}

    def snapshot(self):
        with self._lock:
            return {p: self._overrides.get(p, environment_config(p)) for p in ENV}

    def public(self):
        return {p: {'configured': c.configured, 'has_key': bool(c.api_key), 'model': c.model, 'source': c.source} for p, c in self.snapshot().items()}

    def candidate(self, provider, key=None, model=None, require_model=True):
        if provider not in ENV:
            raise ValueError('Choose OpenAI or Claude')
        current = self.snapshot()[provider]
        if key is not None and (not isinstance(key, str) or (key and (not 16 <= len(key.strip()) <= 1024 or re.search(r'\s', key.strip())))):
            raise ValueError('Enter a valid API key without spaces')
        key = key.strip() if isinstance(key, str) and key.strip() else current.api_key
        if not key:
            raise ValueError('Enter an API key first')
        model = current.model if model is None else model
        if not isinstance(model, str) or (model and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}', model)):
            raise ValueError('Enter a valid model ID')
        if model.startswith('sk-'):
            raise ValueError('Enter the API key in the key field, not the model field')
        if require_model and not model:
            raise ValueError('Choose a model or enter its ID')
        return ProviderConfig(key, model, 'session')

    def save(self, provider, config):
        with self._lock:
            self._overrides[provider] = config

    def disconnect(self, provider):
        if provider not in ENV:
            raise ValueError('Choose OpenAI or Claude')
        with self._lock:
            self._overrides[provider] = ProviderConfig('', '', 'session')


_active_configs = ContextVar('provider_configs', default=None)

@contextmanager
def provider_scope(configs):
    token = _active_configs.set(configs)
    try:
        yield
    finally:
        _active_configs.reset(token)


def provider_config(provider):
    configs = _active_configs.get()
    return configs[provider] if configs is not None else environment_config(provider)
