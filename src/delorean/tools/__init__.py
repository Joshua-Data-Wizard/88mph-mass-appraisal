"""
delorean.tools — the things the agent is allowed to do.

A "tool" is not a special kind of object. It is a plain Python function, plus a
description written well enough that a model can work out when to call it.

    from delorean import tools

    tools.schemas(["lookup_parcel"])        # what the model is shown
    tools.dispatch("lookup_parcel", {"parcel_id": "4-127-3891"})

`dispatch()` never raises. Whatever the model sends -- a mangled parcel number,
a parameter under the wrong name, a tool name it half remembered -- it comes
back as a JSON string the agent can read and act on. See `base.py` for why that
matters more than it sounds.

Section 6 defines its tool inline in the notebook, on purpose: watching a plain
function become something a model can call IS the lesson there. This package is
the hardened version of the same idea, and Section 6 finishes by showing the
difference between the two.
"""

from .base import (REGISTRY, Tool, ToolCall, attempted_tool_call, dispatch,
                   fail, ok, parse_tool_calls, resolve_tool, schemas,
                   strip_tool_calls, tool)

# Importing these modules is what registers their tools. Adding a new tool file
# means adding a line here.
from . import parcel   # noqa: F401,E402  (imported for its side effect)
from . import policy   # noqa: F401,E402
from . import query    # noqa: F401,E402  (Section 7)
from . import review   # noqa: F401,E402  (Section 7)

__all__ = [
    "REGISTRY", "Tool", "ToolCall", "attempted_tool_call", "dispatch", "fail",
    "ok", "parse_tool_calls", "resolve_tool", "schemas", "strip_tool_calls",
    "tool", "parcel", "policy", "query", "review",
]
