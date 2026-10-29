"""Deterministic TF-IDF retrieval, with an equivalent scan baseline."""
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOPWORDS = set("a an the is are was were be been to of in on at for from and or with does do how what when where who why me about this that it can could you please explain paper document summary summarize overview".split())

def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())

class Retriever:
    def __init__(self, documents=None):
        self.documents = documents if documents is not None else json.loads((ROOT / "data/documents.json").read_text())
        self.by_id = {d["id"]: d for d in self.documents}
        self.counts = {d["id"]: Counter(tokens(d["title"] + " " + d["text"])) for d in self.documents}
        df = Counter(t for c in self.counts.values() for t in c)
        self.idf = {t: math.log((1 + len(self.documents)) / (1 + n)) + 1 for t, n in df.items()}
        self.index = defaultdict(list)
        for doc_id, counts in self.counts.items():
            for t, count in counts.items():
                self.index[t].append((doc_id, count * self.idf[t]))

    def search(self, query, k=3, indexed=True, document_id=None, overview=False):
        scores = defaultdict(float)
        for term, count in Counter(t for t in tokens(query) if t not in STOPWORDS).items():
            if indexed:
                postings = self.index.get(term, [])
            else:
                postings = [(i, c[term] * self.idf.get(term, 0)) for i, c in self.counts.items() if term in c]
            for doc_id, weight in postings:
                scores[doc_id] += count * weight
        if document_id:
            scores = {i: value for i, value in scores.items() if self.by_id[i].get('document_id', i) == document_id}
        if overview:
            # Resolve a document before selecting its abstract and opening passages.
            roots = defaultdict(list)
            for doc in self.documents:
                root = doc.get('document_id', doc['id'])
                if not document_id or root == document_id:
                    roots[root].append(doc)
            if not roots:
                return []
            query_terms = set(tokens(query)) - STOPWORDS
            def root_score(root):
                title_terms = set(tokens(roots[root][0]['title'])) - STOPWORDS
                title_overlap = len(query_terms & title_terms)
                best = max((scores.get(d['id'], 0) for d in roots[root]), default=0)
                return (title_overlap, best)
            chosen = document_id or max(roots, key=root_score)
            if not document_id and not any(root_score(chosen)):
                return []
            readable = [d for d in roots[chosen] if '@' not in d['text']] or roots[chosen]
            ordered = sorted(readable, key=lambda d: (d.get('section') != 'abstract', d.get('position', 0)))
            return [{**d, 'score': round(scores.get(d['id'], 0), 4)} for d in ordered[:max(k, 4)]]
        ranked = sorted(scores, key=lambda i: (-scores[i], i))[:k]
        return [{**self.by_id[i], "score": round(scores[i], 4)} for i in ranked]
