"""
clerk_server.py — the Clerk of Courts, exposed over MCP.

    uv run python mcp_server/clerk_server.py

Section 10. A different elected official, a different vendor, a different
database. The Appraiser cannot read this one and never could.

WHY THIS SERVER CARRIES THE SECOND DEMO
---------------------------------------
The Appraiser's office does not work from the Clerk's live records. It works
from a **feed** — a file of recorded sales sent over periodically and loaded
into the CAMA system. Every ratio study in Section 9 was run against that
feed, pulled on 2026-01-15.

Feeds go stale, and worse than that, feeds go stale *silently*. A deed gets
recorded wrong, somebody notices, the Clerk corrects the record — and your
copy never hears about it. You are not doing anything careless. You simply
have a photograph of a filing cabinet that has since been rearranged.

`corrections_since()` exists so an agent can ask the question a human never
gets around to asking: **is my copy still true?**

The answer, for this county, is no. One deed in Courthouse Square was
recorded as a $647,000 arm's-length Warranty Deed and later reclassified a
QUIT CLAIM, which under IAAO standards must be excluded from a ratio study
altogether. Section 9 studied it as a valid sale.

As with the other two servers: **the docstring under each @mcp.tool() is the
text the model reads.** It is prompt engineering wearing a lab coat.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcp.server.fastmcp import FastMCP           # noqa: E402

from delorean.data import CLERK_DB               # noqa: E402
from delorean.tools.parcel import normalize_parcel_id   # noqa: E402

mcp = FastMCP("hill-valley-clerk")

MAX_ROWS = 50


def query(sql: str, params: tuple = ()) -> list[dict]:
    """Run one read-only query against the Clerk's deed records."""
    con = sqlite3.connect(f"file:{CLERK_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


@mcp.tool()
def get_deeds(parcel_id: str) -> dict:
    """Look up every recorded deed for one parcel at the Clerk of Courts.

    Returns the instrument number, the recording date, the sale price, who
    sold it (grantor), who bought it (grantee), the deed type, and whether
    the Clerk qualified the transfer as an arm's length sale.

    This is the legal record of ownership. Use it to establish who owns a
    parcel today and how they came to own it — the Appraiser's roll carries
    an owner name too, but the deed is the authority.

    A deed type of QUIT CLAIM, or arms_length = 0, means the price is NOT
    evidence of market value and the sale must not be used in a ratio study.

    Args:
        parcel_id: The parcel number, written NN-NNN-NNNN, e.g. 04-127-3891.
    """
    canonical = normalize_parcel_id(parcel_id)
    if canonical is None:
        return {"ok": False,
                "error": f"{parcel_id!r} is not a Hill Valley parcel number.",
                "hint": "Parcel numbers have nine digits, NN-NNN-NNNN, for "
                        "example 04-127-3891."}

    rows = query("SELECT instrument_number, parcel_id, recorded_date, "
                 "sale_price, grantor, grantee, deed_type, arms_length, "
                 "book, page, corrected_on FROM sales WHERE parcel_id = ? "
                 "ORDER BY recorded_date DESC", (canonical,))
    if not rows:
        return {"ok": False,
                "error": f"No deed recorded for {canonical}.",
                "hint": "The Clerk has nothing on file for that parcel. That "
                        "is normal for a property that has not sold."}
    return {"ok": True, "office": "Clerk of Courts",
            "deeds_on_file": len(rows), "data": rows}


@mcp.tool()
def corrections_since(since_date: str) -> dict:
    """List deeds the Clerk has CORRECTED since a given date.

    Use this to check whether a copy of the sales file is still accurate. If
    the Appraiser's office pulled a sales feed on some date, any deed
    corrected after that date is wrong in their copy and right here.

    A correction usually means the deed type or the arm's length
    qualification changed. That matters: a sale reclassified as a QUIT CLAIM
    or as not-arm's-length must be REMOVED from any ratio study that used it,
    because its price is not evidence of market value.

    Returns the corrected record as it stands today. Compare it against the
    copy you hold.

    Args:
        since_date: A date in YYYY-MM-DD form. Deeds corrected on or after
                    this date are returned, e.g. "2026-01-15".
    """
    date = (since_date or "").strip()
    if len(date) != 10 or date[4] != "-" or date[7] != "-":
        return {"ok": False,
                "error": f"{since_date!r} is not a date.",
                "hint": "Write the date as YYYY-MM-DD, e.g. 2026-01-15."}

    rows = query("SELECT instrument_number, parcel_id, recorded_date, "
                 "sale_price, grantor, grantee, deed_type, arms_length, "
                 "corrected_on FROM sales WHERE corrected_on IS NOT NULL "
                 "AND corrected_on >= ? ORDER BY corrected_on DESC LIMIT ?",
                 (date, MAX_ROWS))
    if not rows:
        return {"ok": True, "office": "Clerk of Courts", "corrections": 0,
                "data": [],
                "note": f"No deeds have been corrected since {date}. A copy "
                        f"of the sales file taken on that date is still "
                        f"accurate."}
    return {"ok": True, "office": "Clerk of Courts",
            "corrections": len(rows), "data": rows,
            "note": f"{len(rows)} deed(s) corrected since {date}. Any ratio "
                    f"study run on a sales file older than the correction "
                    f"date used the superseded values."}


if __name__ == "__main__":
    mcp.run()
