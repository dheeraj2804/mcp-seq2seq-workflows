"""Optional local English OCR with per-page runtime and pixel bounds."""
import csv
import io
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

MAX_OCR_PAGES=20

def status():
    import importlib.util
    executable=shutil.which('tesseract')
    if not executable and Path('/opt/homebrew/bin/tesseract').exists():executable='/opt/homebrew/bin/tesseract'
    return {'available':bool(executable and importlib.util.find_spec('pypdfium2')), 'engine':'Tesseract', 'language':'English', 'max_pages':MAX_OCR_PAGES}

def recognize_page(data,index):
    import pypdfium2 as pdfium
    executable=shutil.which('tesseract') or '/opt/homebrew/bin/tesseract'
    if not status()['available']:
        raise ValueError('OCR needs Tesseract and pypdfium2 installed on this computer.')
    document=pdfium.PdfDocument(data)
    page=document[index]
    width,height=page.get_size();scale=min(2.2,math.sqrt(8_000_000/max(1,width*height)))
    bitmap=page.render(scale=scale)
    with tempfile.TemporaryDirectory(prefix='workflow-ocr-') as directory:
        path=Path(directory)/'page.png'
        bitmap.to_pil().save(path)
        try:
            result=subprocess.run([executable,str(path),'stdout','-l','eng','--psm','3','tsv'],capture_output=True,text=True,timeout=20)
        except subprocess.TimeoutExpired:
            raise ValueError('OCR exceeded 20 seconds on a page. Try a clearer or smaller scan.')
        if result.returncode:
            raise ValueError('Tesseract could not recognize this page. Check that the English language data is installed.')
    bitmap.close();page.close();document.close()
    lines={};confidence=[]
    for row in csv.DictReader(io.StringIO(result.stdout),delimiter='\t'):
        word=(row.get('text') or '').strip()
        try:score=float(row.get('conf',-1))
        except ValueError:continue
        if not word or score<0:continue
        key=(row['block_num'],row['par_num'],row['line_num'])
        lines.setdefault(key,[]).append(word);confidence.append(score)
    text='\n'.join(' '.join(words) for words in lines.values())
    return text,round(sum(confidence)/len(confidence),1) if confidence else 0
