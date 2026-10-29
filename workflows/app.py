import asyncio
import json
import os
import sqlite3
import uuid
import time
import secrets
from urllib.parse import urlsplit
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, Request, jsonify, request, render_template
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv
from .engine import execute, WorkflowError
from .providers import local_status, list_provider_models, test_provider_connection, connection_error
from .settings import ProviderSettings, provider_scope
from .research import register_research
from .jobs import register_jobs
from .ocr import status as ocr_status
from .nlp import Normalizer
from .retrieval import ROOT
from .library import Library
from .ingestion import MAX_UPLOAD_BYTES, MAX_TEXT_CHARS, extract_upload

class BoundedRequest(Request):
    @property
    def max_content_length(self):
        return MAX_UPLOAD_BYTES + 1024 * 1024 if self.path.startswith("/api/documents") else 256*1024 if self.path.startswith("/api/evaluation-sets") else 16 * 1024


def create_app(db_path=None, normalizer=None):
    load_dotenv(ROOT / ".env")
    app = Flask(__name__)
    app.request_class = BoundedRequest
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    provider_settings = ProviderSettings()
    settings_token = secrets.token_urlsafe(32)
    app.extensions['provider_settings'] = provider_settings
    database = str(db_path or ROOT / "artifacts/runs.sqlite")
    library = Library(Path(database).parent / "library.sqlite")
    (ROOT / "artifacts").mkdir(exist_ok=True)
    nlp = normalizer or Normalizer()
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created TEXT, result TEXT)")

    research = register_research(app, database, library, nlp)

    def text_input():
        if request.content_length and request.content_length > app.config["MAX_CONTENT_LENGTH"]:
            raise RequestEntityTooLarge()
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("query"), str) or not 0 < len(body["query"].strip()) <= 1000:
            raise ValueError("query must be a nonempty string of at most 1000 characters")
        return body

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="File uploads allow 20 MB per file (21 MB request limit); workflow requests allow 16 KiB"), 413

    @app.errorhandler(ValueError)
    def invalid(exc):
        return jsonify(error=str(exc)), 400

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_loaded=nlp.model is not None, transport=os.getenv("TOOL_TRANSPORT", "local"), providers={p: c.configured for p, c in provider_settings.snapshot().items()}, local_model=local_status(), ocr=ocr_status())

    @app.get("/")
    def index():
        return render_template("index.html", settings_token=settings_token)

    @app.before_request
    def protect_provider_settings():
        if urlsplit('//' + request.host).hostname not in {'localhost', '127.0.0.1', '::1'}:
            return jsonify(error='This application is available only through localhost.'), 403
        if request.path.startswith('/api/settings/providers') and request.method != 'GET':
            supplied = request.headers.get('X-Settings-Token', '')
            origin = request.headers.get('Origin')
            if not secrets.compare_digest(supplied.encode(), settings_token.encode()) or (origin and urlsplit(origin).netloc != request.host):
                return jsonify(error='Refresh this page before changing provider settings.'), 403

    @app.after_request
    def settings_cache_control(response):
        if request.path.startswith('/api/settings') or request.path == '/':
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/api/settings/providers')
    def get_provider_settings():
        return jsonify(providers=provider_settings.public(), storage='server_memory', expires='app_restart')

    @app.post('/api/settings/providers/<provider>')
    def save_provider_settings(provider):
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ValueError('Provide an API key and model ID')
        config = provider_settings.candidate(provider, body.get('api_key'), body.get('model'))
        provider_settings.save(provider, config)
        return jsonify(status='saved', providers=provider_settings.public())

    @app.delete('/api/settings/providers/<provider>')
    def disconnect_provider(provider):
        provider_settings.disconnect(provider)
        return jsonify(status='disconnected', providers=provider_settings.public())

    @app.post('/api/settings/providers/<provider>/<action>')
    def provider_action(provider, action):
        if action not in ('models', 'test'):
            return jsonify(error='Unknown provider action'), 404
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ValueError('Provide provider settings')
        config = provider_settings.candidate(provider, body.get('api_key'), body.get('model'), require_model=action == 'test')
        try:
            if action == 'models':
                models = asyncio.run(list_provider_models(provider, config))
                return jsonify(models=models)
            asyncio.run(test_provider_connection(provider, config))
            return jsonify(status='verified', message='Connection verified for the selected model. Test settings were not saved automatically.')
        except Exception as exc:
            return jsonify(error=connection_error(exc)), 502

    @app.get("/api/documents")
    def documents():
        docs = library.documents()
        return jsonify(documents=[{"id": d["id"], "title": d["title"], "characters": len(d["text"]), "metadata": d.get("metadata", {"format": "sample"}), "kind": "imported" if d["id"].startswith("doc-") else "sample"} for d in docs], limits={"upload_bytes": MAX_UPLOAD_BYTES, "text_characters": MAX_TEXT_CHARS, "pdf_pages": 500})

    @app.post("/api/documents")
    def add_document():
        if request.content_length and request.content_length > request.max_content_length:
            raise RequestEntityTooLarge()
        if request.mimetype == "multipart/form-data":
            upload = request.files.get("file")
            if upload is None or not upload.filename:
                raise ValueError("Choose a file to upload")
            filename = Path(upload.filename.replace('\\', '/')).name[:255]
            title = request.form.get("title", "").strip() or Path(filename).stem[:160]
            if len(title) > 160:
                raise ValueError("Title must contain 1–160 characters")
            data = upload.stream.read(MAX_UPLOAD_BYTES + 1)
            text, metadata = extract_upload(data, filename, request.form.get('ocr') == 'true')
            metadata["filename"] = filename
            doc_id = library.add(title, text, metadata, pdf_bytes=data if metadata["format"] == "pdf" else None)
            return jsonify(id=doc_id, status="imported", characters=len(text), metadata=metadata), 201
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ValueError("Provide a document title and text")
        doc_id = library.add(body.get("title"), body.get("text"))
        return jsonify(id=doc_id, status="imported", characters=len(body["text"].strip()), metadata={"format": "text", "warnings": []}), 201

    @app.post('/api/documents/<document_id>/original')
    def attach_original(document_id):
        upload = request.files.get('file')
        if upload is None or not upload.filename or Path(upload.filename).suffix.lower() != '.pdf':
            raise ValueError('Select the original PDF file')
        data = upload.stream.read(MAX_UPLOAD_BYTES + 1)
        text, metadata = extract_upload(data, upload.filename)
        library.attach_pdf(document_id, data, text)
        return jsonify(status='attached')

    @app.get("/api/workflows")
    def recent_runs():
        try:
            limit = int(request.args.get("limit", "5"))
            before = int(request.args["before"]) if "before" in request.args else None
        except ValueError:
            raise ValueError("History limit and cursor must be integers")
        if not 1 <= limit <= 50 or (before is not None and before < 1):
            raise ValueError("History limit must be 1–50 and the cursor must be positive")
        with sqlite3.connect(database) as db:
            total = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            rows = db.execute("SELECT rowid,id,created,result FROM runs WHERE (? IS NULL OR rowid < ?) ORDER BY rowid DESC LIMIT ?", (before, before, limit + 1)).fetchall()
        has_more = len(rows) > limit
        rows = rows[:limit]
        return jsonify(runs=[{"id": i, "created": created, "query": (d := json.loads(result)).get("query", "Earlier workflow"), "status": d.get("status"), "provider": d.get("provider", "unknown"), "elapsed_ms": d.get("elapsed_ms")} for _, i, created, result in rows], total=total, has_more=has_more, next_cursor=rows[-1][0] if rows and has_more else None)

    @app.post("/api/normalize")
    def normalize():
        return jsonify(nlp.normalize(text_input()["query"]))

    def perform_workflow(body, trace=None, job=None):
        provider = body.get("provider", "demo")
        if provider not in ("demo", "openai", "claude", "auto", "local"):
            raise ValueError("Unsupported provider")
        document_id = body.get("document_id")
        selected=body.get('document_ids')
        known={d['id'] for d in library.documents()}
        if selected is not None:
            if not isinstance(selected,list) or len(selected)!=2 or any(not isinstance(item,str) or item not in known for item in selected) or len(set(selected))!=2:
                raise ValueError('Choose two different documents for comparison')
            document_id=selected
        if selected is None and document_id is not None and (not isinstance(document_id, str) or document_id not in known):
            raise ValueError("Choose an existing document from the library")
        conversation_id = body.get('conversation_id')
        context = []
        if conversation_id is not None:
            if not isinstance(conversation_id,str) or not (conversation := research.conversation(conversation_id)):
                raise ValueError('Conversation not found; start a new conversation')
            if ('document_id' in body or 'document_ids' in body) and document_id != conversation['document_id']:
                raise ValueError('Start a new conversation to change the document scope')
            document_id = conversation['document_id']
            completed = [m for m in conversation['messages'] if m.get('status') == 'completed']
            context = [{'query':conversation['title'], 'answer':''}] + [{'query':m['query'][:300], 'answer':m.get('answer','')[:600]} for m in completed[-2:]] if completed else []
        else:
            conversation_id = research.create(body['query'], document_id)
        run_id = uuid.uuid4().hex
        started = time.perf_counter()
        trace = trace if trace is not None else []
        async def run():
            if job: job.bind(asyncio.get_running_loop(),asyncio.current_task())
            with provider_scope(provider_settings.snapshot()):
                return await asyncio.wait_for(execute(body["query"], nlp, provider, os.getenv("TOOL_TRANSPORT", "local"), trace, library.path, document_id, context), 210 if provider == "local" else 90)
        status = 200
        try:
            result = asyncio.run(run())
            result["status"] = "completed"
        except asyncio.CancelledError:
            trace.append({'step':'cancelled','status':'cancelled'})
            result={'status':'cancelled','error':'Workflow cancelled by you. No draft answer was saved.','trace':trace}
            status=499
        except Exception as exc:
            app.logger.warning("Workflow %s failed: %s", run_id, type(exc).__name__)
            # Never expose SDK exceptions (which can contain request data or headers).
            result = {"status": "failed", "error": "Workflow failed; check provider configuration and server logs", "error_type": type(exc).__name__, "trace": trace}
            status = 502
        result.setdefault("provider", provider)
        result["id"] = run_id
        result["query"] = body["query"]
        result["document_id"] = document_id
        result["conversation_id"] = conversation_id
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        with sqlite3.connect(database) as db:
            db.execute("INSERT INTO runs VALUES (?, ?, ?)", (run_id, datetime.now(timezone.utc).isoformat(), json.dumps(result)))
        research.link(run_id, conversation_id)
        return result, status

    @app.post('/api/workflows')
    def workflow():
        result,status=perform_workflow(text_input())
        return jsonify(result),status

    register_jobs(app,perform_workflow)

    @app.get("/api/workflows/<run_id>")
    def get_run(run_id):
        with sqlite3.connect(database) as db:
            row = db.execute("SELECT result FROM runs WHERE id = ?", (run_id,)).fetchone()
        return (jsonify(json.loads(row[0])), 200) if row else (jsonify(error="Run not found"), 404)

    return app

if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")))
