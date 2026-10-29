"""Readable text and sentence boundaries for retrieval and extractive answers."""
import re
import unicodedata


def clean_text(text):
    text = unicodedata.normalize('NFKC', text).replace('\u00ad', '')
    # Undo a PDF line-wrap hyphen; retain compounds such as English-to-German.
    text = re.sub(r'([a-z])-\n([a-z]+)', lambda m: m[1] + ('-' if m[2] in {'to', 'of', 'the', 'and'} else '') + m[2], text)
    text = re.sub(r'(?m)^\s*\d{1,3}\s*$', '', text)  # isolated page numbers
    text = re.sub(r'(?m)^(Abstract|\d+(?:\.\d+)*\s+[A-Z][^\n]{2,90})\s*$', r'\n\n\1\n\n', text)
    text = re.sub(r'(\[Page \d+\])', r'\n\n\1\n\n', text)
    blocks = [' '.join(block.split()) for block in re.split(r'\n\s*\n', text)]
    return '\n\n'.join(block for block in blocks if block)


def sentences(text):
    text = re.sub(r'\[Page \d+\]', '', text)
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z“"(])', text) if s.strip()]


def passages(text, target=1500):
    """Group paragraphs/sentences, cutting only very long sentences at word boundaries."""
    output, current = [], ''
    for block in clean_text(text).split('\n\n'):
        # Abstract and section headings stay with the following prose.
        if re.fullmatch(r'Abstract|\d+(?:\.\d+)*\s+.{2,90}', block):
            if current:
                output.append(current)
            current = block
            continue
        units = sentences(block) if len(block) > target else [block]
        for unit in units:
            if len(unit) > target:
                words, split, size = unit.split(), [], 0
                for word in words:
                    if size + len(word) > target and split:
                        if current:
                            output.append(current); current = ''
                        output.append(' '.join(split)); split, size = [], 0
                    split.append(word); size += len(word) + 1
                unit = ' '.join(split)
            if len(current) + len(unit) + 2 > target and current:
                output.append(current); current = ''
            current = (current + '\n\n' + unit).strip()
    if current:
        output.append(current)
    return output


def overview_question(query):
    return bool(re.search(r'\b(summar(?:y|ize|ise)|overview|main (?:idea|contribution)|paper.*about|document.*about|what.*(?:paper|document).*|explain.*(?:paper|document))\b', query, re.I))


def extractive_answer(hits, query, overview=False):
    from .retrieval import tokens, STOPWORDS
    query_terms = set(tokens(query)) - STOPWORDS
    selected, seen = [], set()
    # Abstract-first for overview; query-relevant sentences for specific questions.
    candidates = []
    for rank, hit in enumerate(hits):
        body = re.sub(r'^(?:Abstract|\d+(?:\.\d+)*\s+[^\n]+)\s*\n+', '', hit['text'])
        for index, sentence in enumerate(sentences(body)):
            sentence = ' '.join(sentence.split())
            if not (30 if overview else 12) <= len(sentence) <= 650 or not sentence[0].isupper():
                continue
            if '@' in sentence or len(re.findall(r'[=∑√]|\bFigure\b|\bTable\b', sentence)) > 1:
                continue
            key = re.sub(r'\W', '', sentence.lower())
            if key in seen:
                continue
            seen.add(key)
            score = len(query_terms & set(tokens(sentence)))
            candidates.append((rank, index, score, sentence, hit['id']))
    if not overview:
        candidates.sort(key=lambda c: (-c[2], c[0], c[1]))
    length = 0
    for _, _, _, sentence, source in candidates:
        if length + len(sentence) > 1900:
            continue
        selected.append(f'• {sentence} [{source}]')
        length += len(sentence)
        if len(selected) == (5 if overview else 4):
            break
    return '\n\n'.join(selected) if selected else 'The retrieved passages do not contain enough readable prose to produce a useful summary. Try a more specific question or use a configured AI provider.'
