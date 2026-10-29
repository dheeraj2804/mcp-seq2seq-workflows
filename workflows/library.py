"""Local SQLite document library shared by Flask and MCP subprocesses."""
import json
import sqlite3
import uuid
import re
from functools import lru_cache
from pathlib import Path
from .retrieval import ROOT, Retriever
from .ingestion import MAX_TEXT_CHARS
from .text import passages, clean_text

DEFAULT_DB = ROOT / "artifacts/library.sqlite"

class Library:
    def __init__(self, path=None):
        self.path = str(path or DEFAULT_DB)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, title TEXT NOT NULL, text TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP)")

            db.execute("CREATE TABLE IF NOT EXISTS document_files (document_id TEXT PRIMARY KEY, content BLOB NOT NULL)")
            columns = {row[1] for row in db.execute("PRAGMA table_info(documents)")}
            if "metadata" not in columns:
                db.execute("ALTER TABLE documents ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")

    def add(self, title, text, metadata=None, pdf_bytes=None):
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 160:
            raise ValueError("Title must contain 1–160 characters")
        if not isinstance(text, str) or not 20 <= len(text.strip()) <= MAX_TEXT_CHARS:
            raise ValueError("Document text must contain 20–2,000,000 characters")
        doc_id = "doc-" + uuid.uuid4().hex[:12]
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO documents (id,title,text,metadata) VALUES (?,?,?,?)", (doc_id, title.strip(), text.strip(), json.dumps(metadata or {"format": "text"})))
            if pdf_bytes is not None:
                db.execute("INSERT INTO document_files VALUES (?,?)", (doc_id, pdf_bytes))
        return doc_id

    def documents(self):
        original = json.loads((ROOT / "data/documents.json").read_text())
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT id,title,text,metadata FROM documents ORDER BY rowid").fetchall()
            files = {row[0] for row in db.execute("SELECT document_id FROM document_files")}
        return original + [{"id": row[0], "title": row[1], "text": row[2], "metadata": {**json.loads(row[3]), "has_original": row[0] in files}} for row in rows]

    def original_pdf(self, document_id):
        with sqlite3.connect(self.path) as db:
            row = db.execute('SELECT content FROM document_files WHERE document_id=?', (document_id,)).fetchone()
        return row[0] if row else None

    def attach_pdf(self, document_id, data, extracted_text):
        doc = next((d for d in self.documents() if d['id'] == document_id), None)
        if not doc or doc.get('metadata', {}).get('format') != 'pdf':
            raise ValueError('Choose an existing PDF document')
        if clean_text(doc['text']) != clean_text(extracted_text):
            raise ValueError('This PDF does not match the indexed document. Upload it as a new document instead.')
        with sqlite3.connect(self.path) as db:
            db.execute('INSERT OR REPLACE INTO document_files VALUES (?,?)', (document_id, data))

    def retriever(self):
        with sqlite3.connect(self.path) as db:
            revision = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        return indexed_library(self.path, revision)

@lru_cache(maxsize=2)
def indexed_library(path, revision):
    # A new insertion changes the cache key; existing runs retain a stable snapshot.
    documents = []
    for doc in Library(path).documents():
        raw = doc['text']
        markers = list(re.finditer(r'\[Page (\d+)\]', raw)) if doc.get('metadata', {}).get('format') == 'pdf' else []
        page_groups = [(int(marker[1]), raw[marker.start():markers[i+1].start() if i+1 < len(markers) else len(raw)]) for i, marker in enumerate(markers)] if markers else [(None, raw)]
        chunks = [(page, chunk) for page, text in page_groups for chunk in passages(text)]
        for index, (page, text) in enumerate(chunks):
            chunk_id = doc['id'] if len(chunks) == 1 else f"{doc['id']}-part-{index+1}"
            documents.append({**doc, 'id': chunk_id, 'document_id': doc['id'], 'page': page,
                              'title': doc['title'], 'text': text, 'position': index,
                              'section': 'abstract' if text.startswith('Abstract') else 'body'})
    return Retriever(documents)
