"""Bounded file text extraction; executed in a separate process for uploads."""
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARS = 2_000_000
MAX_PDF_PAGES = 500
SUPPORTED = {'.pdf', '.docx', '.txt', '.md', '.markdown'}


def extract(data, suffix, enable_ocr=False):
    if suffix not in SUPPORTED:
        raise ValueError('Choose a PDF, DOCX, TXT, or Markdown file. Legacy .doc files must be saved as .docx first.')
    if not data:
        raise ValueError('The selected file is empty.')
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError('File exceeds the 20 MB upload limit.')
    metadata = {'format': suffix.lstrip('.'), 'bytes': len(data), 'warnings': []}
    if suffix == '.pdf':
        from pypdf import PdfReader
        if not data.startswith(b'%PDF-'):
            raise ValueError('This file is not a valid PDF.')
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValueError('Password-protected PDFs are not supported. Upload an unlocked copy.')
        metadata['pages'] = len(reader.pages)
        if metadata['pages'] > MAX_PDF_PAGES:
            raise ValueError('PDF exceeds 500 pages. Split it into smaller files.')
        parts, length, empty = [], 0, 0
        metadata['ocr_pages']=[]; metadata['ocr_confidence']={}
        for index, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or '').strip()
            if enable_ocr and len(''.join(c for c in text if c.isalpha()))<30:
                from .ocr import recognize_page,MAX_OCR_PAGES
                if len(metadata['ocr_pages'])>=MAX_OCR_PAGES:
                    raise ValueError('OCR supports up to 20 scanned pages per upload. Split this PDF.')
                text,confidence=recognize_page(data,index-1)
                metadata['ocr_pages'].append(index);metadata['ocr_confidence'][str(index)]=confidence
            if not text:
                empty += 1
                continue
            length += len(text) + 24
            if length > MAX_TEXT_CHARS:
                raise ValueError('Extracted text exceeds 2 million characters. Split the document.')
            parts.append(f'[Page {index}]\n{text}')
        if metadata['ocr_pages']:
            metadata['warnings'].append('OCR was used. Recognition can misread words, numbers and equations; verify the original PDF.')
            if any(value<70 for value in metadata['ocr_confidence'].values()):metadata['warnings'].append('Some OCR pages have low recognition confidence. A clearer scan may help.')
        if not parts:
            raise ValueError('OCR could not recognize readable text. Try a clearer English scan.' if enable_ocr else 'No selectable text found in this PDF. Scanned or image-only PDFs need OCR first.')
        if empty:
            metadata['warnings'].append(f'{empty} page(s) contained no extractable text; scanned pages may need OCR.')
        text = '\n\n'.join(parts)
    elif suffix == '.docx':
        from defusedxml import ElementTree
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > 50 * 1024 * 1024:
                raise ValueError('DOCX expands beyond 50 MB. Split or simplify the document.')
            if 'word/document.xml' not in archive.namelist():
                raise ValueError('This file is not a valid Word DOCX document.')
            root = ElementTree.fromstring(archive.read('word/document.xml'))
            ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
            paragraphs = []
            length = 0
            for paragraph in root.iter(ns + 'p'):
                value = ''.join((el.text or '') if el.tag == ns+'t' else '\t' if el.tag == ns+'tab' else '\n' if el.tag in (ns+'br', ns+'cr') else '' for el in paragraph.iter()).strip()
                if value:
                    length += len(value) + 1
                    if length > MAX_TEXT_CHARS:
                        raise ValueError('Extracted text exceeds 2 million characters. Split the document.')
                    paragraphs.append(value)
            text = '\n'.join(paragraphs)
        metadata['warnings'].append('Word body text and table text imported. Images, headers, footnotes, and layout are not indexed.')
    else:
        try:
            text = data.decode('utf-8-sig')
        except UnicodeDecodeError:
            try:
                text = data.decode('utf-16') if data.startswith((b'\xff\xfe', b'\xfe\xff')) else None
            except UnicodeDecodeError:
                text = None
            if text is None:
                raise ValueError('Text files must use UTF-8 or UTF-16 encoding.')
        if '\x00' in text:
            raise ValueError('This appears to be a binary file, not a text document.')
    text = text.strip()
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError('Extracted text exceeds 2 million characters. Split the document.')
    if len(text) < 20:
        raise ValueError('The document contains fewer than 20 readable characters.')
    return text, metadata


def extract_upload(data, filename, enable_ocr=False):
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED or len(data) > MAX_UPLOAD_BYTES or not data:
        return extract(data, suffix, enable_ocr)  # Reuse user-facing validation messages.
    try:
        process = subprocess.run([sys.executable, '-m', 'workflows.ingestion', suffix] + (['--ocr'] if enable_ocr else []), input=data, capture_output=True, timeout=180 if enable_ocr else 60)
    except subprocess.TimeoutExpired:
        raise ValueError('Extraction exceeded its runtime limit (180 seconds with OCR, 60 without). Split or simplify the document.')
    if process.returncode:
        raise ValueError('The document could not be processed within the extraction limits. Try a smaller or repaired copy.')
    try:
        result = json.loads(process.stdout)
    except (ValueError, UnicodeError):
        raise ValueError('The document could not be read. Try a repaired copy.')
    if 'error' in result:
        raise ValueError(result['error'])
    return result['text'], result['metadata']


if __name__ == '__main__':
    # Limits apply only to this disposable parser process, not the Flask/model server.
    try:
        import resource
        budget=120 if '--ocr' in sys.argv else 45
        resource.setrlimit(resource.RLIMIT_CPU, (budget, budget))
    except (ImportError, ValueError, OSError):
        pass
    try:
        text, metadata = extract(sys.stdin.buffer.read(MAX_UPLOAD_BYTES + 1), sys.argv[1], '--ocr' in sys.argv)
        output = {'text': text, 'metadata': metadata}
    except ValueError as exc:
        output = {'error': str(exc)}
    except Exception:
        output = {'error': 'Unable to read this document. It may be damaged or use an unsupported format.'}
    sys.stdout.write(json.dumps(output))
