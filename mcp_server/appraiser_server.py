"""
appraiser_server.py — the Appraiser's office, exposed over MCP.

    uv run python mcp_server/appraiser_server.py

Section 10. One of THREE servers, because Hill Valley County has three
elected offices with three vendors and no integration between them. That is
not a simplification for the demo — it is the situation in most counties, and
it is the whole reason this section exists.

WHAT MCP ACTUALLY IS, IN ONE PARAGRAPH
--------------------------------------
A tool (Section 6) is a Python function *you* wrote, living inside *your*
program. An MCP server is a separate program, owned by somebody else, that
publishes a list of things it can do. Your agent asks it "what have you got?"
and it answers. You never write a line of code describing its tools, because
it describes them itself.

That is the difference between a charger built for one phone and a USB-C
socket.

READ THIS PART IF YOU ARE ADAPTING IT FOR YOUR OWN OFFICE
---------------------------------------------------------
**The docstring under each @mcp.tool() is not documentation. It is the prompt.**
It is the literal text a language model reads when deciding whether to call
that function. Write it for the model, name the units, say when NOT to use it.
A vague docstring here is indistinguishable from a broken tool.

Everything below is read-only. Nothing this server exposes can change a
record, and that is a deliberate property of the design rather than an
accident of the queries: the connection is opened read-only, so SQLite itself
refuses a write.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# Run straight from a clone, without installing the package first.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcp.server.fastmcp import FastMCP           # noqa: E402

from delorean.data import CAMA_DB                # noqa: E402
from delorean.tools.parcel import normalize_parcel_id   # noqa: E402

#: The name the agent sees when it asks who it is talking to.
mcp = FastMCP("hill-valley-appraiser")

MAX_ROWS = 25


def query(sql: str, params: tuple = ()) -> list[dict]:
    """Run one read-only query against the Appraiser's roll.

    Opened through a `file:...?mode=ro` URI, which is stronger than promising
    to behave: SQLite refuses a write on this handle even if somebody later
    adds one to this file by mistake.
    """
    con = sqlite3.connect(f"file:{CAMA_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


@mcp.tool()
def get_parcel(parcel_id: str) -> dict:
    """Look up one parcel on the Hill Valley County assessment roll.

    Returns the owner of record, the situs address, the neighborhood, the
    property class, the year built, the living area and LOT SIZE in square
    feet, and the current assessed value in dollars.

    This is the Appraiser's own record of the property itself. It does NOT
    know who legally owns the parcel today (ask the Clerk of Courts) and it
    does NOT know whether any tax is owed (ask the Tax Collector).

    Args:
        parcel_id: The parcel number, written NN-NNN-NNNN, e.g. 04-127-3891.
                   A missing leading zero is repaired automatically.
    """
    canonical = normalize_parcel_id(parcel_id)
    if canonical is None:
        return {"ok": False,
                "error": f"{parcel_id!r} is not a Hill Valley parcel number.",
                "hint": "Parcel numbers have nine digits, NN-NNN-NNNN, for "
                        "example 04-127-3891. To search by owner or address, "
                        "use find_parcels instead."}

    rows = query("SELECT parcel_id, situs_address, owner_name, neighborhood, "
                 "property_class, property_class_desc, year_built, "
                 "living_area_sqft, lot_size_sqft, assessed_value, "
                 "assessment_year FROM parcels WHERE parcel_id = ?",
                 (canonical,))
    if not rows:
        return {"ok": False,
                "error": f"No parcel {canonical} on the roll.",
                "hint": "Check the number, or use find_parcels to search by "
                        "owner name or street address."}
    return {"ok": True, "office": "Appraiser", "data": rows[0]}


@mcp.tool()
def find_parcels(search: str) -> dict:
    """Find parcels by owner name or street address when you have no number.

    Use this when a question names a person or a place rather than a parcel
    number — "the property at 1955 Hilldale Ct", "everything Biff Tannen
    owns". Matches on part of a name or address, case-insensitively.

    Args:
        search: Part of an owner name or a situs address, e.g. "Hilldale Ct"
                or "Tannen".
    """
    text = (search or "").strip()
    if len(text) < 3:
        return {"ok": False,
                "error": "Search text is too short.",
                "hint": "Give at least three characters of an owner name or "
                        "street address."}

    like = f"%{text}%"
    rows = query("SELECT parcel_id, situs_address, owner_name, neighborhood, "
                 "lot_size_sqft, assessed_value FROM parcels "
                 "WHERE owner_name LIKE ? OR situs_address LIKE ? "
                 "ORDER BY parcel_id LIMIT ?", (like, like, MAX_ROWS))
    if not rows:
        return {"ok": False,
                "error": f"Nothing on the roll matches {text!r}.",
                "hint": "Try a surname on its own, or just the street name."}
    return {"ok": True, "office": "Appraiser",
            "matches": len(rows), "data": rows}


if __name__ == "__main__":
    # stdio: the agent starts this file as a subprocess and talks to it over
    # its standard input and output. No port, no network, nothing listening.
    # For an office server you would swap this for a streamable HTTP
    # transport behind your own authentication.
    mcp.run()
