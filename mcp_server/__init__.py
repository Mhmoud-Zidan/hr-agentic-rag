"""MCP server package for the Northwind HR agent.

Deliberately NOT named `mcp/`: with `pythonpath = ["."]` a top-level `mcp`
directory shadows the installed MCP SDK, so `from mcp.server import ...` inside
this very module would import the repository folder instead of the library.
"""
