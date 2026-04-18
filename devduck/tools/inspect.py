"""
🔍 DevDuck inspect tool — passthrough to strands-inspect.

This is a thin re-export so the DevDuck tool config can reference
`devduck.tools:inspect` instead of `strands_inspect:inspect_tool`.

All real logic lives in the `strands-inspect` package:
    https://github.com/cagataycali/strands-inspect

Capabilities (via inspect_tool):
    - scan: Deep-scan any pip package (modules, classes, functions, signatures)
    - call: Call any function by dotted path (e.g., "json.dumps")
    - inspect: Detailed view of a class/function/module
    - search: Fuzzy search across a package's API
    - generate: Generate working code examples
    - exec: Execute Python code, capture output + return value
    - create: Compile code into reusable registered functions
    - profile: Runtime profiling (memory timeline, CPU flamegraph)
    - graph / hotspots / unused / deps / connections: call-graph analysis

If strands-inspect is not installed, a stub tool is exposed that tells
the user how to install it.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from strands_inspect import inspect_tool as _inspect_tool

    # Re-export under the name DevDuck uses
    inspect = _inspect_tool

    logger.debug("strands-inspect loaded successfully")

except ImportError as e:
    logger.warning(f"strands-inspect not available: {e}")

    from strands import tool

    @tool
    def inspect(action: str = "help", **kwargs) -> dict:
        """🔍 Code inspection (STUB — strands-inspect not installed).

        Install with: pip install strands-inspect

        Once installed, this tool provides package scanning, function calling,
        code profiling, call graphs, and more.
        """
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        "🔍 strands-inspect is not installed.\n\n"
                        "Install with: `pip install strands-inspect`\n\n"
                        "Then restart DevDuck to use the full inspect_tool capabilities."
                    )
                }
            ],
        }


__all__ = ["inspect"]
