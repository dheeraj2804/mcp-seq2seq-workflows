"""Allowlisted local tools and a real MCP stdio client adapter."""
import asyncio
import json
import sys
import os
from contextlib import asynccontextmanager
from .library import Library
from .hybrid import search as hybrid_search

SCHEMAS = [
    {"name": "search_documents", "description": "Search the project knowledge base.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}},
    {"name": "get_document", "description": "Read a document by its exact id.", "parameters": {"type": "object", "properties": {"document_id": {"type": "string"}}, "required": ["document_id"], "additionalProperties": False}},
]

class LocalTools:
    def __init__(self, library_path=None, document_id=None, overview=False):
        self.library = Library(library_path)
        self.document_id = document_id
        self.overview = overview

    @property
    def retriever(self):
        return self.library.retriever()

    async def call(self, name, arguments):
        expected = {"search_documents": "query", "get_document": "document_id"}
        if name not in expected or not isinstance(arguments, dict):
            raise ValueError("Unknown tool or invalid arguments")
        key = expected[name]
        if set(arguments) != {key} or not isinstance(arguments[key], str) or not 0 < len(arguments[key]) <= 2000:
            raise ValueError("Invalid tool arguments")
        if name == "search_documents":
            return await hybrid_search(self.library, arguments[key], k=5, document_id=self.document_id, overview=self.overview)
        if arguments[key] not in self.retriever.by_id:
            raise ValueError("Unknown document id")
        doc = self.retriever.by_id[arguments[key]]
        allowed = self.document_id if isinstance(self.document_id,list) else [self.document_id] if self.document_id else []
        if allowed and doc.get("document_id", doc["id"]) not in allowed:
            raise ValueError("Document is outside the selected scope")
        return doc

class MCPTools:
    def __init__(self, session):
        self.session = session

    async def call(self, name, arguments):
        # Apply the same allowlist and validation at the transport boundary.
        if name not in {s["name"] for s in SCHEMAS}:
            raise ValueError("Unknown tool")
        result = await self.session.call_tool(name, arguments)
        if result.isError:
            raise ValueError("MCP tool returned an error")
        if result.structuredContent is not None:
            payload = result.structuredContent
            # FastMCP wraps non-object structured return values in `result`.
            return payload["result"] if name == "search_documents" and "result" in payload else payload
        blocks = [c.text for c in result.content if c.type == "text"]
        payload = json.loads("".join(blocks))
        return payload["result"] if name == "search_documents" and isinstance(payload, dict) and "result" in payload else payload

@asynccontextmanager
async def tool_connection(transport="local", library_path=None, document_id=None, overview=False):
    if transport == "local":
        yield LocalTools(library_path, document_id, overview)
    elif transport == "mcp":
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        args = ["-m", "workflows.mcp_server"]
        if library_path:
            args += ["--library", str(library_path)]
        if document_id:
            for selected in (document_id if isinstance(document_id,list) else [document_id]):
                args += ["--document", selected]
        if overview:
            args += ["--overview"]
        params = StdioServerParameters(command=sys.executable, args=args,env={name:os.environ[name] for name in ('SEMANTIC_SEARCH','EMBEDDING_MODEL','OLLAMA_BASE_URL','SEMANTIC_MIN_SIMILARITY') if name in os.environ})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), 10)
                yield MCPTools(session)
    else:
        raise ValueError("TOOL_TRANSPORT must be local or mcp")
