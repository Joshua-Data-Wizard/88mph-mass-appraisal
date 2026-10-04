"""
review.py — the two tools that turn rows into appraisal work.

Section 9 gives the agent three tools and they compose deliberately:

    run_sql       chooses WHICH properties we are talking about
    ratio_study   judges them against the IAAO standards
    map_parcels   puts them on the ground

The second and third both take a **SQL query** rather than a list of parcels,
and that one decision is what makes the demo open-ended instead of a menu.
"Ratio study by neighborhood" and "ratio study for every home built after
1980" and "map the sales where we came in under 80% of price" are all the same
two tool calls with a different WHERE clause. Nobody had to anticipate the
question.

It also sidesteps a real problem. The alternative -- having the model pass a
list of two hundred parcel numbers -- would mean two hundred parcel numbers
travelling through a 4B model's tool-call format, and Section 2 already showed
the audience what that model does to a parcel number.

**Both tools are deliberately tolerant of what comes back.** We measured the
model returning whatever columns it felt like rather than a tidy parcel_id
list, so these tools take the frame they are given, find what they need, and
go and fetch the rest themselves. Same principle as `normalize_parcel_id`:
absorb the sloppiness, do not lecture about it.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .. import ratio as ratio_maths
from . import query
from .base import fail, tool

#: The last map that was drawn, as a DataFrame, for whoever is rendering.
#:
#: A tool result goes back into the model's context as text, and a map is not
#: text. So the model gets a summary -- how many parcels, where, how many
#: outside the standard -- and the notebook or the Streamlit app reads the
#: full frame from here and draws the picture. The audience sees a map; the
#: model sees a sentence. Same precedent as `agent.local_responder.last_raw`.
_LAST_MAP: Any = None


def last_map():
    """The frame behind the most recent `map_parcels` call, or None."""
    return _LAST_MAP


# --------------------------------------------------------------------------
#  Getting from "whatever the model's query returned" to "what we need"
# --------------------------------------------------------------------------

def _sales_for(frame):
    """Make sure we have sale prices and assessed values to work with.

    Three cases, in order of how tidy the model was:

    1. The query already returned sale_price and assessed_value. Use them.
    2. It returned parcel numbers and nothing else -- the case we actually
       asked for. Go and fetch the sales for those parcels.
    3. It returned neither. Nothing we can do, so say what was needed.
    """
    import pandas as pd

    if {"sale_price", "assessed_value"} <= set(frame.columns):
        return frame, None

    if "parcel_id" not in frame.columns:
        return None, (
            "That query returned no parcel_id column, so there is no way to "
            "tell which properties you meant. Select parcel_id - for example: "
            "SELECT parcel_id FROM sales_feed WHERE neighborhood = 'Hilldale'")

    ids = [str(v) for v in frame["parcel_id"].dropna().unique()]
    if not ids:
        return None, "That query matched no parcels at all."

    placeholders = ",".join("?" for _ in ids)
    uri = f"file:{query.data.CAMA_DB.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        cursor = con.execute(
            f"SELECT * FROM sales_feed WHERE parcel_id IN ({placeholders})", ids)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
    finally:
        con.close()

    fetched = pd.DataFrame(rows, columns=columns)
    if fetched.empty:
        return None, (
            f"Found {len(ids)} parcel(s), but none of them has ever sold, so "
            f"there is nothing to compare our assessment against. A ratio "
            f"study needs sales - try querying sales_feed instead of parcels.")

    return fetched, None


# --------------------------------------------------------------------------
#  ratio_study
# --------------------------------------------------------------------------

@tool(
    name="ratio_study",
    description=(
        "Run an IAAO assessment ratio study over a set of sales and report "
        "whether it meets the standards. Use this for ANY question about how "
        "well the county, a neighborhood or a group of properties is being "
        "assessed - whether values are high or low, consistent, or fair "
        "between cheap and expensive homes.\n\n"
        "You give it a SQL SELECT that picks the sales to study; it does the "
        "statistics. Never try to compute a median, a COD or a PRD in SQL "
        "yourself - IAAO requires the median rather than the average, and "
        "this tool also applies the arm's length rules for you.\n\n"
        "It reports three tests: Level (the median ratio, standard 0.90 to "
        "1.10), Uniformity (COD, standard 5 to 15) and Equity (PRD, standard "
        "0.98 to 1.03)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": (
                    "A SELECT against sales_feed choosing which sales to "
                    "study. Selecting just parcel_id is enough. For the whole "
                    "county use: SELECT parcel_id FROM sales_feed"
                ),
            },
            "group_by": {
                "type": "string",
                "description": (
                    "Optional. A column to run a SEPARATE study for each value "
                    "of - most usefully 'neighborhood'. Leave empty for one "
                    "study over everything."
                ),
            },
        },
        "required": ["sql"],
    },
    name_aliases=("ratio", "run_ratio_study", "ratio_analysis",
                  "assessment_ratio", "ratio_stats", "sales_ratio_study"),
    param_aliases={
        "sql": ("query", "statement", "select", "sales", "q"),
        "group_by": ("group", "groupby", "by", "breakdown", "stratify"),
    },
)
def ratio_study(sql: str, group_by: str = "") -> dict | str:
    """Judge a set of sales against the IAAO standards."""
    problem = query.check_sql(sql)
    if problem:
        return fail(problem,
                    "Send a single read-only SELECT choosing the sales to "
                    "study, for example: SELECT parcel_id FROM sales_feed "
                    "WHERE neighborhood = 'Hilldale'")

    try:
        selected = query.frame(sql)
    except sqlite3.Error as error:
        return fail(f"That query did not run: {error}",
                    query._hint_for(str(error)))

    if selected.empty:
        return fail(
            "That query matched no sales.",
            "Check any text values - neighborhood names are written exactly "
            "as: " + ", ".join(query.NEIGHBORHOODS) + ".")

    sales, problem = _sales_for(selected)
    if problem:
        return fail("Cannot run a ratio study on that selection.", problem)

    if group_by:
        group_by = group_by.strip().strip("'\"")
        if group_by not in sales.columns:
            return fail(
                f"Cannot group by {group_by!r}.",
                f"Available columns are: {', '.join(sales.columns)}. For a "
                f"breakdown by area use group_by='neighborhood'.")

        table = ratio_maths.ratio_by_group(sales, group_by)
        groups = table.to_dict(orient="records")
        failing = [g for g in groups if g.get("verdict") == "FAIL"]
        return {
            "grouped_by": group_by,
            "groups_studied": len(groups),
            "groups_failing": len(failing),
            "results": groups,
            "summary": _grouped_summary(group_by, groups, failing),
        }

    # `summary` is filled in by ratio_stats itself, so the sentence the agent
    # reports and the sentence a notebook prints are the same sentence.
    result = ratio_maths.ratio_stats(sales)

    # IAAO says stratify: a ratio study that mixes houses with shops is
    # comparing two different markets. Worth saying rather than silently
    # averaging them together.
    if "property_class" in sales.columns and sales["property_class"].nunique() > 1:
        classes = ", ".join(sorted(sales["property_class"].dropna().unique()))
        result["stratification_note"] = (
            f"This selection mixes property classes ({classes}). IAAO asks for "
            f"separate studies per class. To limit it to houses, add: "
            f"WHERE property_class IN ('R01','R02')")

    return result


def _grouped_summary(group_by: str, groups: list[dict],
                     failing: list[dict]) -> str:
    """The sentence Section 9 is built to produce."""
    if not failing:
        return f"All {len(groups)} {group_by}s meet the IAAO standards."

    named = []
    for group in failing:
        named.append(f"{group[group_by]} fails {group['failed']}")
    return (f"{len(failing)} of {len(groups)} {group_by}s fail at least one "
            f"IAAO standard: " + "; ".join(named) + ".")


# --------------------------------------------------------------------------
#  map_parcels
# --------------------------------------------------------------------------

@tool(
    name="map_parcels",
    description=(
        "Plot a set of parcels on a map of Hill Valley County. Use this only "
        "when someone asks to SEE properties on a map, or to see where a "
        "problem is concentrated.\n\n"
        "You give it a SQL SELECT choosing which parcels to plot. The map is "
        "drawn for the user to look at; you get back a short description of "
        "what is on it. Describe what the map shows - do not list the "
        "individual parcels."
    ),
    parameters={
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": (
                    "A SELECT choosing which parcels to plot. Selecting just "
                    "parcel_id is enough, for example: SELECT parcel_id FROM "
                    "sales_feed WHERE neighborhood = 'Riverside Heights'"
                ),
            },
            "color_by": {
                "type": "string",
                "description": (
                    "What the colours should mean: 'ratio' (default, shows "
                    "under- and over-assessment), 'assessed_value', or "
                    "'neighborhood'."
                ),
            },
        },
        "required": ["sql"],
    },
    name_aliases=("map", "show_map", "plot_parcels", "draw_map", "map_parcel",
                  "plot_map", "show_parcels_on_map"),
    param_aliases={
        "sql": ("query", "statement", "select", "parcels", "q"),
        "color_by": ("colour_by", "color", "colour", "shade_by", "by"),
    },
)
def map_parcels(sql: str, color_by: str = "ratio") -> dict | str:
    """Put a selection of parcels on the map."""
    global _LAST_MAP

    problem = query.check_sql(sql)
    if problem:
        return fail(problem,
                    "Send a single read-only SELECT choosing the parcels to "
                    "plot, for example: SELECT parcel_id FROM parcels WHERE "
                    "neighborhood = 'Hilldale'")

    try:
        selected = query.frame(sql)
    except sqlite3.Error as error:
        return fail(f"That query did not run: {error}",
                    query._hint_for(str(error)))

    if selected.empty or "parcel_id" not in selected.columns:
        return fail(
            "Nothing to map.",
            "The query must return a parcel_id column and at least one row. "
            "For example: SELECT parcel_id FROM sales_feed WHERE "
            "neighborhood = 'Riverside Heights'")

    frame = _coordinates_for(selected["parcel_id"])
    if frame is None or frame.empty:
        return fail("Those parcels have no coordinates on file.",
                    "Try a different selection.")

    color_by = (color_by or "ratio").strip().strip("'\"").lower()
    if color_by not in ("ratio", "assessed_value", "neighborhood"):
        color_by = "ratio"

    frame.attrs["color_by"] = color_by
    _LAST_MAP = frame

    result: dict[str, Any] = {
        "map_drawn": True,
        "parcels_plotted": int(len(frame)),
        "coloured_by": color_by,
        "neighborhoods": sorted(frame["neighborhood"].dropna().unique().tolist()),
        "centre": [round(float(frame["latitude"].mean()), 5),
                   round(float(frame["longitude"].mean()), 5)],
    }

    rated = frame["ratio"].dropna() if "ratio" in frame.columns else None
    if rated is not None and len(rated):
        low = int((rated < 0.90).sum())
        high = int((rated > 1.10).sum())
        result["below_iaao_range"] = low
        result["above_iaao_range"] = high
        result["summary"] = (
            f"A map of {len(frame)} parcels is now on screen, coloured by "
            f"assessment ratio. {low} are assessed below the IAAO range and "
            f"{high} above it.")
    else:
        result["summary"] = (
            f"A map of {len(frame)} parcels is now on screen.")

    return result


def _coordinates_for(parcel_ids):
    """Fetch what it takes to draw a parcel: where it is, and how it is doing.

    Coordinates live on `parcels` and the ratio lives on `sales_feed`, so this
    is the one place in Section 9 that genuinely needs both. It is a LEFT join
    on purpose -- a parcel with no sale still belongs on the map, it just has
    no ratio to colour it by.
    """
    import pandas as pd

    ids = [str(v) for v in pd.Series(parcel_ids).dropna().unique()]
    if not ids:
        return None

    placeholders = ",".join("?" for _ in ids)
    uri = f"file:{query.data.CAMA_DB.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        cursor = con.execute(
            f"""SELECT p.parcel_id, p.situs_address, p.owner_name,
                       p.neighborhood, p.property_class, p.assessed_value,
                       p.latitude, p.longitude,
                       s.sale_price, s.sale_date, s.ratio
                  FROM parcels p
                  LEFT JOIN sales_feed s ON s.parcel_id = p.parcel_id
                 WHERE p.parcel_id IN ({placeholders})
                   AND p.latitude IS NOT NULL""",
            ids)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
    finally:
        con.close()

    return pd.DataFrame(rows, columns=columns)
