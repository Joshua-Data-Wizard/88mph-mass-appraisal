"""
tax_server.py — the Tax Collector, exposed over MCP.

    uv run python mcp_server/tax_server.py

Section 10. The third office, and the one that decides the answer.

The Appraiser says how big the lot is. The Clerk says who owns it. Neither of
them knows whether the owner has paid, and neither of them can find out
without picking up a telephone. Policy 4.2 turns on exactly that fact, which
is why a question as ordinary as *"may I split my lot?"* cannot be answered
by any single office in the county.

That is the whole argument of this section, and it is not a technology
argument. It is an argument about who holds which piece of the answer.

**The docstring under @mcp.tool() is the text the model reads.** Say what the
numbers mean and what units they are in — a model that mistakes a levy for a
balance gives a taxpayer the wrong answer very confidently.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcp.server.fastmcp import FastMCP           # noqa: E402

from delorean.data import TAX_DB                 # noqa: E402
from delorean.tools.parcel import normalize_parcel_id   # noqa: E402

mcp = FastMCP("hill-valley-tax")


def query(sql: str, params: tuple = ()) -> list[dict]:
    """Run one read-only query against the Tax Collector's ledger."""
    con = sqlite3.connect(f"file:{TAX_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


@mcp.tool()
def get_tax_status(parcel_id: str) -> dict:
    """Look up the tax account for one parcel at the Tax Collector.

    Returns, in dollars: the levy for the year, how much has been paid, the
    balance still due, whether any prior year is also delinquent, and the
    total outstanding across all years.

    Use this whenever money owed on a parcel could matter — a split, a
    combination, a transfer, or a taxpayer asking what they owe. The
    Appraiser's roll and the Clerk's deeds do NOT carry this information.

    `total_outstanding` is the number that matters for eligibility questions:
    it includes the prior year balance, and a parcel with any outstanding
    amount at all is barred from a split under Policy 4.2.

    Args:
        parcel_id: The parcel number, written NN-NNN-NNNN, e.g. 04-127-3891.
    """
    canonical = normalize_parcel_id(parcel_id)
    if canonical is None:
        return {"ok": False,
                "error": f"{parcel_id!r} is not a Hill Valley parcel number.",
                "hint": "Parcel numbers have nine digits, NN-NNN-NNNN, for "
                        "example 04-127-3891."}

    rows = query("SELECT parcel_id, tax_year, owner_name, taxable_value, "
                 "levy_amount, amount_paid, balance_due, status, "
                 "last_payment_date, prior_year_delinquent, "
                 "prior_year_balance FROM tax_accounts WHERE parcel_id = ? "
                 "ORDER BY tax_year DESC", (canonical,))
    if not rows:
        return {"ok": False,
                "error": f"No tax account for {canonical}.",
                "hint": "Check the parcel number. Every parcel on the roll "
                        "should have an account."}

    latest = rows[0]
    total = round(latest["balance_due"] + (latest["prior_year_balance"] or 0), 2)

    return {
        "ok": True,
        "office": "Tax Collector",
        "data": {
            **latest,
            "total_outstanding": total,
            "clear_to_transfer": total == 0,
        },
        "note": ("Nothing is owed on this parcel." if total == 0 else
                 f"${total:,.2f} is outstanding across "
                 f"{'two tax years' if latest['prior_year_delinquent'] else 'the current tax year'}."),
    }


if __name__ == "__main__":
    mcp.run()
