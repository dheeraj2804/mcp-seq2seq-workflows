"""Run with python -m workflows.mcp_server; stdout is reserved for MCP."""
import argparse
from mcp.server.fastmcp import FastMCP
from .tools import LocalTools

server = FastMCP("seq2seq-knowledge")
parser = argparse.ArgumentParser()
parser.add_argument("--library")
parser.add_argument("--document", action="append")
parser.add_argument("--overview", action="store_true")
args = parser.parse_args()
scope = args.document if args.document and len(args.document)>1 else args.document[0] if args.document else None
tools = LocalTools(args.library, scope, args.overview)

@server.tool()
async def search_documents(query: str) -> list[dict]:
    """Search the project knowledge base."""
    return await tools.call("search_documents", {"query": query})

@server.tool()
async def get_document(document_id: str) -> dict:
    """Read one project document by id."""
    return await tools.call("get_document", {"document_id": document_id})

if __name__ == "__main__":
    server.run(transport="stdio")
