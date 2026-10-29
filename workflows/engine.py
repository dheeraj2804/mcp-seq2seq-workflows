import asyncio
import json
import time
import re
from . import providers
from .tools import tool_connection
from .text import overview_question, extractive_answer

class WorkflowError(RuntimeError):
    pass

def transient(exc):
    status = getattr(exc, "status_code", None)
    return isinstance(exc, (TimeoutError, ConnectionError)) or status in (408, 429, 500, 502, 503, 504) or type(exc).__name__ in ("APIConnectionError", "APITimeoutError")

async def retry(operation, trace, label, attempts=3, timeout=30):
    for attempt in range(attempts):
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(operation(), timeout=timeout)
            trace.append({"step": label, "attempt": attempt + 1, "status": "ok", "ms": round((time.perf_counter()-start)*1000, 2)})
            return result
        except Exception as exc:
            trace.append({"step": label, "attempt": attempt + 1, "status": "error", "error": type(exc).__name__})
            if not transient(exc) or attempt == attempts - 1:
                raise
            await asyncio.sleep(0.2 * 2**attempt)

async def execute(query, normalizer, provider="demo", transport="local", trace=None, library_path=None, document_id=None, context=None):
    trace = trace if trace is not None else []
    context = context or []
    retrieval_query = query
    if context:
        retrieval_query = context[0]['query'] + ' ' + (context[-1]['query'] if len(context)>1 else '') + ' ' + query
        trace.append({'step':'conversation_context', 'turns':len(context), 'status':'ok'})
    normalized = normalizer.normalize(retrieval_query)
    trace.append({"step": "normalize", **normalized})
    evidence = {}
    comparison=isinstance(document_id,list)
    overview = comparison or overview_question(query) or bool(context and overview_question(context[0]['query']) and re.search(r'simpl|explain|summar|that|it', query, re.I))
    async with tool_connection(transport, library_path, document_id, overview) as tools:
        async def call(name, arguments):
            result = await retry(lambda: tools.call(name, arguments), trace, name)
            docs = result if isinstance(result, list) else [result]
            for doc in docs:
                evidence[doc["id"]] = {**evidence.get(doc["id"], {}), **doc}
            return result

        if provider in ("demo", "local"):
            hits = await call("search_documents", {"query": normalized["query"]})
            if not hits and normalized["query"] != query:
                hits = await call("search_documents", {"query": query})
                trace.append({"step": "original_query_fallback", "status": "ok"})
            if hits:
                doc = await call("get_document", {"document_id": hits[0]["id"]})
                if comparison and {d.get('document_id',d['id']) for d in hits} != set(document_id):
                    return {'answer':'There is not enough evidence from both selected documents to compare them.','evidence_status':'insufficient','provider':provider,'normalization':normalized,'sources':hits,'trace':trace}
                trace.append({'step':'retrieval_quality','mode':hits[0].get('retrieval_mode','keyword'),'status':'ok'})
                if provider == 'local':
                    trace.append({'step': 'synthesize', 'status': 'started'})
                    answer = await retry(lambda: providers.local_answer(query, hits, context) if context else providers.local_answer(query, hits), trace, 'local_model', attempts=1, timeout=180)
                    return {'answer': answer, 'evidence_status':'insufficient' if getattr(answer,'supported',None) is False else 'model_supported' if getattr(answer,'supported',None) is True else 'unverified', 'answer_kind': 'local_explanation', 'answer_notice': 'Written by a local language model using the retrieved evidence. Documents stay on this Mac. Check the citations for accuracy.', 'provider': 'local', 'normalization': normalized, 'sources': [{**hit, 'citation': str(i+1)} for i, hit in enumerate(hits[:4])], 'trace': trace}
                answer = extractive_answer(hits, query, overview)
            else:
                answer = "No matching evidence was found in the knowledge base."
            return {"answer": answer, "evidence_status": "extractive" if hits else "insufficient", "answer_kind": "extractive_summary", "answer_notice": "Key sentences selected from your documents, not an AI paraphrase. For a plain-language explanation, choose a configured Claude or OpenAI provider.", "provider": provider, "normalization": normalized, "sources": list(evidence.values()), "trace": trace}

        if provider not in ("openai", "claude", "auto"):
            raise ValueError("provider must be demo, openai, claude, or auto")
        candidates = ["claude", "openai"] if provider == "auto" else [provider]
        history = json.dumps({"question": query, "normalized_query": normalized["query"], "intent": "document overview" if overview else "question answering", "selected_document": document_id, "conversation": context, "comparison":comparison})
        for turn in range(6):
            raw = None
            for candidate in candidates:
                try:
                    raw = await retry(lambda: providers.complete(candidate, history), trace, candidate)
                    active = candidate
                    break
                except Exception as exc:
                    if provider != "auto":
                        raise
                    trace.append({"step": "fallback", "provider": candidate, "error": type(exc).__name__})
            if raw is None:
                raise WorkflowError("All configured providers failed")
            try:
                action = json.loads(raw)
                if not isinstance(action, dict):
                    raise ValueError("Expected an object")
                if "answer" in action:
                    if not isinstance(action["answer"], str) or not evidence:
                        raise ValueError("Search evidence is required before an answer")
                    supported=action.get('supported')
                    if supported is False:
                        action['answer']='The selected documents do not provide enough evidence to answer this question. Try a more specific question or add a document that reports the requested information.'
                    elif supported is True:
                        if comparison and {d.get('document_id',d['id']) for d in evidence.values()} != set(document_id):
                            raise ValueError('Retrieve evidence from both selected documents before comparing them')
                        citations=re.findall(r'\[([^\]]+)\]',action['answer'])
                        if not citations or any(citation not in evidence for citation in citations):
                            raise ValueError('Cite the exact retrieved passage ids before answering')
                    return {"answer": action["answer"], "evidence_status":'insufficient' if supported is False else 'model_supported' if supported is True else 'unverified', "answer_kind": "ai_explanation", "answer_notice": "AI-written explanation grounded in the retrieved passages. Check the cited sources for technical details.", "provider": active, "normalization": normalized, "sources": list(evidence.values()), "trace": trace}
                result = await call(action["tool"], action["arguments"])
                history += "\nAgent: " + raw + "\nTool result: " + json.dumps(result)
            except (ValueError, KeyError, TypeError) as exc:
                trace.append({"step": "invalid_action", "error": str(exc)})
                history += "\nInvalid action. Use a valid allowed tool with its required arguments, or answer after searching."
        raise WorkflowError("Agent exceeded the six-step limit")
