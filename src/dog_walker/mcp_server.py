"""MCP facade: the SAME toolbox, exposed to any MCP host.

This is the interop boundary and nothing more. Our own agent
dispatches the registry in-process (we own both ends of that wire);
this facade exists so OTHER hosts -- Claude Desktop, Claude Code,
anything speaking the Model Context Protocol -- can plug the
dog-walker tools in without custom glue. One registry, two doors.

The whole facade is the loop below: MCPServer reads each function's
signature, type hints, and docstring and derives the MCP tool
definition (name, description, input schema) -- the same information
our hand-written schemas carry. Transport is stdio: the host launches
this process and speaks JSON-RPC over stdin/stdout.

(SDK note: mcp 2.x renamed FastMCP to MCPServer -- the class every
tutorial still calls FastMCP.)

Run it:            uv run python -m dog_walker.mcp_server
Plug into Claude:  claude mcp add dog-walker -- uv --project \
                       /Users/bryan/dev/agentic_dog_walker run \
                       python -m dog_walker.mcp_server
"""

from mcp.server.mcpserver import MCPServer

from dog_walker.toolbox import REGISTRY

mcp = MCPServer("dog-walker")

for _name, (_fn, _schema) in REGISTRY.items():
    mcp.tool(name=_name)(_fn)

if __name__ == "__main__":
    mcp.run()
