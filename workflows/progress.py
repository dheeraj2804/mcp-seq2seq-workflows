"""Per-run progress hooks and safe previews of structured model output."""
import json
import re
from contextlib import contextmanager
from contextvars import ContextVar

_emitter=ContextVar('workflow_emitter',default=None)

@contextmanager
def progress_scope(callback):
    token=_emitter.set(callback)
    try: yield
    finally: _emitter.reset(token)

def emit(event):
    callback=_emitter.get()
    if callback: callback(event)

def streaming(): return _emitter.get() is not None

class EventTrace(list):
    def append(self,item):
        super().append(item)
        emit({'type':'step','data':item})


def partial_string(text,key):
    match=re.search(r'"'+re.escape(key)+r'"\s*:\s*"',text)
    if not match: return ''
    source=text[match.end():]; output=''; index=0
    escapes={'n':'\n','r':'\r','t':'\t','"':'"','\\':'\\','/':'/','b':'','f':''}
    while index<len(source):
        char=source[index]
        if char=='"': break
        if char=='\\':
            if index+1>=len(source): break
            code=source[index+1]
            if code=='u':
                if index+6>len(source): break
                try: output+=chr(int(source[index+2:index+6],16))
                except ValueError: break
                index+=6;continue
            output+=escapes.get(code,code);index+=2;continue
        output+=char;index+=1
    return output

def model_preview(text):
    value='\n\n'.join(value for key in ('answer','explanation','details') if (value:=partial_string(text,key)))
    if value: emit({'type':'draft','data':{'text':value}})
