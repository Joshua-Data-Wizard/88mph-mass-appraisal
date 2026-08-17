"""
data.py — getting at Hill Valley County's three office systems.

Every demo from Section 4 onward reads its data through this file, so nobody
has to hardcode a path to a .db and none of the notebooks drift apart.

The important thing about this module is what it does NOT provide: there is no
function that queries all three systems at once, and no shared connection. You
open the Appraiser's database, or the Clerk's, or the Tax Collector's. Three
offices, three connections, no join.

    from delorean import data

    data.check()                      # is the data built? build it if not.

    with data.cama() as con:          # the Appraiser's office
        row = con.execute(
            "SELECT * FROM parcels WHERE parcel_id = ?", ("04-127-3891",)
        ).fetchone()
        print(row["owner_name"], row["assessed_value"])

    df = data.parcels_df()            # or just hand me a DataFrame

That separation is not an inconvenience we are working around — it IS the
subject of Section 10. A single database would quietly answer questions that,
in a real county, require three phone calls.

ALL OF THIS DATA IS SYNTHETIC. See data/README.md.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# ─────────────────────────────────────────────────────────────────────────
#  Where things live
# ─────────────────────────────────────────────────────────────────────────

# src/delorean/data.py -> src/delorean -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
POLICY_DIR = DATA_DIR / "policy"
CSV_DIR = DATA_DIR / "csv"

CAMA_DB = DATA_DIR / "cama.db"
CLERK_DB = DATA_DIR / "clerk.db"
TAX_DB = DATA_DIR / "tax.db"

GENERATOR = REPO_ROOT / "scripts" / "generate_hill_valley_data.py"

#: Which office owns which database, and what it can tell you. Printed by
#: `data.systems()` and used in the Section 10 notebook to introduce the
#: three MCP servers.
SYSTEMS = {
    "cama": {
        "office": "Appraiser's Office",
        "path": CAMA_DB,
        "table": "parcels",
        "knows": "characteristics, ownership of record, assessed value, location",
        "does_not_know": "what anything sold for, or whether taxes were paid",
    },
    "clerk": {
        "office": "Clerk of Courts",
        "path": CLERK_DB,
        "table": "sales",
        "knows": "recorded deeds, sale prices, grantor and grantee, instrument numbers",
        "does_not_know": "what the property looks like, or what it is assessed at",
    },
    "tax": {
        "office": "Tax Collector",
        "path": TAX_DB,
        "table": "tax_accounts",
        "knows": "amounts levied, amounts paid, outstanding balances, delinquency",
        "does_not_know": "why the value is what it is, or who sold it to whom",
    },
}


# ─────────────────────────────────────────────────────────────────────────
#  Making sure the data exists
# ─────────────────────────────────────────────────────────────────────────

def is_built() -> bool:
    """Have the three databases been generated yet?"""
    return all(spec["path"].exists() for spec in SYSTEMS.values())


def build(quiet: bool = False) -> None:
    """Run the generator script to create the three databases.

    The data is deterministic, so running this again on any machine produces
    byte-identical databases. Safe to re-run at any time.
    """
    result = subprocess.run(
        [sys.executable, str(GENERATOR)],
        capture_output=quiet,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Data generation failed.\nRun it directly to see why:\n"
            f"    uv run python scripts/generate_hill_valley_data.py"
        )


def check(auto_build: bool = True) -> None:
    """Confirm the data is present, and build it if it is not.

    Call this at the top of any notebook that reads county data. It turns a
    confusing "unable to open database file" into either a working dataset or
    a message that says exactly what to run.
    """
    if is_built():
        return
    if not auto_build:
        raise FileNotFoundError(
            "Hill Valley County data has not been generated yet. Run:\n"
            "    uv run python scripts/generate_hill_valley_data.py"
        )
    # ASCII in console output: the default Windows terminal codepage mangles
    # em-dashes, and most attendees are on Windows.
    print("County data not found - generating it now (this takes a second)...")
    build(quiet=True)
    print("Done. Three office databases are ready in data/.")


# ─────────────────────────────────────────────────────────────────────────
#  Connections — one office at a time
# ─────────────────────────────────────────────────────────────────────────

def _connect(path: Path) -> sqlite3.Connection:
    """Open one database read-only-ish, with dict-like rows.

    `row_factory = sqlite3.Row` is what lets you write `row["owner_name"]`
    instead of `row[2]`. Worth it every time — the tools in Section 7 build
    their answers out of column names.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is missing. Run:\n"
            f"    uv run python scripts/generate_hill_valley_data.py"
        )
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def cama() -> sqlite3.Connection:
    """The Appraiser's office: parcels, characteristics, assessed values."""
    return _connect(CAMA_DB)


def clerk() -> sqlite3.Connection:
    """The Clerk of Courts: recorded deeds and sale prices."""
    return _connect(CLERK_DB)


def tax() -> sqlite3.Connection:
    """The Tax Collector: levies, payments, and outstanding balances."""
    return _connect(TAX_DB)


# ─────────────────────────────────────────────────────────────────────────
#  DataFrames — for the ratio study in Section 7
# ─────────────────────────────────────────────────────────────────────────

def _frame(con: sqlite3.Connection, sql: str) -> "pd.DataFrame":
    """Read a query into a DataFrame and close the connection behind us.

    Worth knowing, because it trips people up: `with sqlite3.connect(...)`
    manages a TRANSACTION, not the connection. It does not close anything on
    exit. Since these helpers open a connection the caller never sees, we close
    it explicitly in a finally block.
    """
    import pandas as pd

    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()


def parcels_df() -> "pd.DataFrame":
    """Every parcel on the roll, as a DataFrame."""
    return _frame(cama(), "SELECT * FROM parcels")


def sales_df() -> "pd.DataFrame":
    """Every recorded deed, as a DataFrame.

    Note this includes deeds that are NOT usable market sales — quit claims,
    sheriff's deeds, interfamily transfers. Filtering them is the assessor's
    job, not the Clerk's, and Section 7 does it explicitly.
    """
    return _frame(clerk(), "SELECT * FROM sales")


def tax_df() -> "pd.DataFrame":
    """Every tax account, as a DataFrame."""
    return _frame(tax(), "SELECT * FROM tax_accounts")


# ─────────────────────────────────────────────────────────────────────────
#  Office policy documents — the Section 8 RAG corpus
# ─────────────────────────────────────────────────────────────────────────

def policy_documents() -> list[Path]:
    """Every office policy document, for the Section 8 retrieval index.

    Returns both .md and .pdf so an attendee can drop their OWN office's PDFs
    into data/policy/ and re-index without touching any code.
    """
    if not POLICY_DIR.exists():
        return []
    found = sorted(POLICY_DIR.glob("*.md")) + sorted(POLICY_DIR.glob("*.pdf"))
    return found


# ─────────────────────────────────────────────────────────────────────────
#  A readable summary, for the notebooks
# ─────────────────────────────────────────────────────────────────────────

def systems() -> None:
    """Print the three offices, what each knows, and what it cannot tell you.

    Used to open Section 10: the gap between "knows" and "does not know" is
    the reason MCP exists.
    """
    print("Hill Valley County - three offices, three systems, no join\n")
    for key, spec in SYSTEMS.items():
        built = "ready" if spec["path"].exists() else "NOT GENERATED"
        print(f"  {spec['office']}   ({key}.db - {built})")
        print(f"      knows        : {spec['knows']}")
        print(f"      cannot tell you: {spec['does_not_know']}")
        print()
