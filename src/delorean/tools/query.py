"""
query.py — letting the model write its own SQL, safely.

Section 9's first tool, and the one that makes people in the audience sit
forward. The agent is not picking from a menu of canned reports. It writes a
query, we run it, and it reads the answer.

The obvious objection comes from the room within about four seconds — *"you
are letting an AI loose on my assessment roll?"* — and it deserves a real
answer rather than a shrug. There are four, and they are all visible in this
file:

1. **The connection is opened read-only.** Not "we check for the word DELETE"
   -- SQLite itself refuses to write. `mode=ro` on the URI.
2. **One statement, and it must be a SELECT.** Anything else is refused with
   an explanation before it ever reaches the database.
3. **Row counts are capped**, so a careless `SELECT *` cannot flood the
   model's context or the screen.
4. **It only ever sees the Appraiser's own database.** The Clerk's records and
   the Tax Collector's are not reachable from here. Three offices, three
   systems -- that gap is Section 10's whole subject.

The other half of this file is the schema blurb the model is shown. That
blurb is a prompt, and it is the highest-leverage text in Section 9: it is
what turns "write me some SQL" into a query that runs first time.

Measured on Qwen3.5-4B: 7 of 7 questions produced runnable SQL, about four
seconds each.
"""

from __future__ import annotations

import difflib
import re
import sqlite3
from typing import Any

from .. import data
from .base import fail, tool

# --------------------------------------------------------------------------
#  What the model is allowed to know about
# --------------------------------------------------------------------------
#
# Curated on purpose. `parcels` really has sixteen columns; the model is shown
# eleven. Bedrooms, bathrooms, the neighborhood code and the class description
# are all real, and all left out, because every column offered is one more
# thing to reach for and get wrong. Nothing in Section 9 needs them.

TABLES: dict[str, list[str]] = {
    "parcels": [
        "parcel_id", "situs_address", "owner_name", "neighborhood",
        "property_class", "year_built", "living_area_sqft", "lot_size_sqft",
        "assessed_value", "latitude", "longitude",
    ],
    "sales_feed": [
        "parcel_id", "sale_date", "sale_price", "assessed_value", "ratio",
        "arms_length", "deed_type", "neighborhood", "property_class",
        "year_built", "living_area_sqft", "lot_size_sqft",
    ],
}

NEIGHBORHOODS = [
    "Courthouse Square", "Eastwood Ravine", "Hilldale",
    "Lyon Estates", "Riverside Heights", "Twin Pines Ranch",
]

#: The schema exactly as the model sees it. Kept as a module constant so the
#: notebook and the Streamlit app can put the same text on screen -- an
#: attendee asking "how did it know that?" gets to read the actual answer.
SCHEMA_HELP = f"""\
Two tables, both in the Appraiser's office database. They do NOT need to be
joined -- sales_feed already carries the parcel characteristics you need.

TABLE parcels -- every parcel in the county (1,200 rows)
  parcel_id         TEXT     'NN-NNN-NNNN', e.g. '04-127-3891'
  situs_address     TEXT     street address of the property
  owner_name        TEXT     name on the roll, e.g. 'Biff Tannen'
  neighborhood      TEXT     one of: {', '.join(NEIGHBORHOODS)}
  property_class    TEXT     'R01' single family, 'R02' condominium,
                             'C01' commercial, 'V01' vacant land
  year_built        INTEGER  year of construction
  living_area_sqft  INTEGER  heated living area
  lot_size_sqft     INTEGER  size of the lot
  assessed_value    INTEGER  the value we assessed, in dollars
  latitude          REAL     needed to put a parcel on a map
  longitude         REAL

TABLE sales_feed -- sales sent over from the Clerk of Courts, with our
                    assessment already attached (244 rows)
  parcel_id         TEXT     matches parcels.parcel_id
  sale_date         TEXT     'YYYY-MM-DD'
  sale_price        INTEGER  price on the recorded deed, in dollars
  assessed_value    INTEGER  our assessed value, in dollars
  ratio             REAL     assessed_value / sale_price, already worked out
  arms_length       INTEGER  1 = a genuine market sale, 0 = exclude it
  deed_type         TEXT     e.g. 'Warranty Deed', 'QUIT CLAIM'
  neighborhood      TEXT     same six names as above
  property_class    TEXT     same four codes as above
  year_built        INTEGER
  living_area_sqft  INTEGER
  lot_size_sqft     INTEGER
"""

#: How many rows come back to the model. Enough to answer "the five most
#: expensive parcels", small enough that nothing floods the screen on stage.
MAX_ROWS = 40

#: What ratio_study and map_parcels are allowed internally. They summarise
#: rather than repeat them, so the model never sees this many.
MAX_ROWS_INTERNAL = 5_000

_FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create",
              "attach", "detach", "pragma", "vacuum", "replace", "reindex")


# --------------------------------------------------------------------------
#  Running a query without letting it hurt anybody
# --------------------------------------------------------------------------

def check_sql(sql: str) -> str | None:
    """Return an explanation if this SQL should not run, or None if it is fine.

    Deliberately checked BEFORE the database sees it, even though the
    connection is read-only anyway. A model that gets `attempt to write a
    readonly database` back learns nothing; a model told *"I can only run
    SELECT statements -- this database is read-only"* rewrites its query.
    """
    text = (sql or "").strip().rstrip(";").strip()

    if not text:
        return "No SQL was provided."

    # One statement only. A stray semicolon in the middle means two.
    if ";" in text:
        return ("That looks like more than one statement. Send a single "
                "SELECT.")

    lowered = text.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return ("Only SELECT statements can be run here. This database is "
                "opened read-only.")

    for word in _FORBIDDEN:
        if re.search(rf"\b{word}\b", lowered):
            return (f"'{word.upper()}' is not allowed. This is the live "
                    f"assessment roll, opened read-only - you can read it, "
                    f"but nothing here can change it.")

    return None


def _apply_limit(sql: str, limit: int) -> str:
    """Cap the rows a query can return, unless it already caps itself."""
    text = sql.strip().rstrip(";").strip()
    if re.search(r"\blimit\s+\d+\s*$", text, re.I):
        return text
    return f"{text} LIMIT {limit}"


def _hint_for(message: str) -> str:
    """Turn a SQLite error into something the model can act on.

    This is the file's whole philosophy in one function. `no such column:
    sale_date` is a fact; *"sales_feed has sale_date, sale_price... did you
    mean sale_date?"* is an instruction. The model reads the second one and
    fixes itself on the next turn, in front of the audience.
    """
    lowered = message.lower()

    match = re.search(r"no such column:?\s*([\w.]+)", lowered)
    if match:
        wanted = match.group(1).split(".")[-1]

        # The most common version of this mistake is not a typo at all: the
        # column is real, it just lives on the other table. Say which one,
        # rather than offering it back as a spelling suggestion.
        holders = [t for t, cols in TABLES.items()
                   if wanted in [c.lower() for c in cols]]
        if holders:
            return (f"'{wanted}' is a real column, but it is on "
                    f"{' and '.join(holders)}, not on the table you queried. "
                    f"Query {holders[0]} instead - it already carries the "
                    f"parcel characteristics, so you do not need a join.")

        every = sorted({c for cols in TABLES.values() for c in cols})
        close = difflib.get_close_matches(wanted, every, n=2, cutoff=0.5)
        hint = ""
        if close:
            hint = f"Did you mean {' or '.join(close)}? "
        return hint + "The columns you can use are:\n" + columns_help()

    match = re.search(r"no such table:?\s*(\w+)", lowered)
    if match:
        return (f"There is no table called {match.group(1)}. The only two "
                f"tables are 'parcels' and 'sales_feed'. Note the Clerk's and "
                f"the Tax Collector's records are NOT reachable from here - "
                f"they are separate offices with separate systems.")

    if "syntax error" in lowered:
        return ("Check the SQL for a typo - an unclosed quote or a missing "
                "comma is the usual cause. Remember string values need single "
                "quotes, e.g. neighborhood = 'Hilldale'.")

    if "ambiguous column" in lowered:
        return ("That column exists in both tables, so it needs a table name "
                "in front of it. You probably do not need a join at all - "
                "sales_feed already carries the parcel characteristics.")

    return ("Check the query against the schema. The columns you can use "
            "are:\n" + columns_help())


def columns_help() -> str:
    """A compact table-and-column listing, for error hints."""
    lines = []
    for table, cols in TABLES.items():
        lines.append(f"  {table}: {', '.join(cols)}")
    return "\n".join(lines)


def execute(sql: str, limit: int = MAX_ROWS) -> tuple[list[str], list[tuple]]:
    """Run a checked SELECT against the Appraiser's database, read-only.

    Raises `sqlite3.Error` on a bad query -- callers turn that into a hint.
    Shared by all three Section 9 tools so there is one place where a query
    actually touches the data.
    """
    # mode=ro is the real guarantee. SQLite refuses writes at the file level,
    # so a query that slipped past check_sql() still cannot do damage.
    uri = f"file:{data.CAMA_DB.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        cursor = con.execute(_apply_limit(sql, limit))
        columns = [d[0] for d in cursor.description or []]
        return columns, cursor.fetchall()
    finally:
        con.close()


def frame(sql: str, limit: int = MAX_ROWS_INTERNAL):
    """Same thing, as a DataFrame. Used by ratio_study and map_parcels."""
    import pandas as pd

    columns, rows = execute(sql, limit)
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------
#  The tool
# --------------------------------------------------------------------------

@tool(
    name="run_sql",
    description=(
        "Run a read-only SQL SELECT against the Hill Valley County assessment "
        "roll and get the rows back. Use this to look up parcels, owners, "
        "addresses and sales, and to count, sort or filter them.\n\n"
        "Do NOT use this to work out an assessment ratio study - use "
        "ratio_study for that, because a median, a COD and a PRD are not "
        "things to attempt in SQL.\n\n"
        + SCHEMA_HELP
    ),
    parameters={
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": (
                    "One SQLite SELECT statement. For example: "
                    "SELECT parcel_id, owner_name, assessed_value FROM parcels "
                    "WHERE neighborhood = 'Hilldale' ORDER BY assessed_value "
                    "DESC LIMIT 5"
                ),
            },
        },
        "required": ["sql"],
    },
    name_aliases=("sql", "query", "run_query", "sql_query", "execute_sql",
                  "query_database", "select"),
    param_aliases={"sql": ("query", "statement", "q", "sql_query", "command")},
)
def run_sql(sql: str) -> dict | str:
    """Run one SELECT and hand back the rows."""
    problem = check_sql(sql)
    if problem:
        return fail(problem,
                    "Send a single read-only SELECT. The columns available "
                    "are:\n" + columns_help())

    try:
        columns, rows = execute(sql)
    except sqlite3.Error as error:
        return fail(f"That query did not run: {error}", _hint_for(str(error)))

    result: dict[str, Any] = {
        "columns": columns,
        "row_count": len(rows),
        "rows": [dict(zip(columns, row)) for row in rows],
    }

    if len(rows) == MAX_ROWS:
        result["note"] = (
            f"Showing the first {MAX_ROWS} rows. There may be more - add a "
            f"COUNT(*) query if you need the total.")

    if not rows:
        result["note"] = (
            "That query ran but matched nothing. Check the spelling of any "
            "text value - neighborhood names are written exactly as: "
            + ", ".join(NEIGHBORHOODS) + ".")

    # A join between these two tables is never necessary and is a measured
    # failure mode: an inner join drops every parcel that has no sale, which
    # looks identical to "no such owner". Say so rather than let it pass.
    if re.search(r"\bjoin\b", sql, re.I):
        result["warning"] = (
            "This query joins the two tables, which is not needed - "
            "sales_feed already carries neighborhood, property_class, "
            "year_built and the parcel's size. A join here silently drops "
            "every parcel that has never sold.")

    return result
