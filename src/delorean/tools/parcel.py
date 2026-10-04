"""
parcel.py — looking a property up in the Appraiser's office roll.

This is the first real tool the agent gets, in Section 7.

The interesting part is not the database query. It is `normalize_parcel_id()`,
and the reason it exists goes all the way back to Section 2.

    Section 2 showed the audience that "parcel 04-127-3891" is not one thing to
    a language model. It is THIRTEEN separate tokens, every digit its own
    fragment. A model that sees fragments cannot be trusted to put them back
    together the same way twice.

We measured exactly that. Ten different phrasings of the same question, one
model, and the parcel number came back three different ways:

    "04-127-3891"     the canonical form
    "04127-3891"      a hyphen dropped
    "4-127-3891"      the leading zero gone

All three mean the same parcel. `WHERE parcel_id = ?` finds one of them.

So the tool normalizes before it queries, and when it still cannot find
anything it says what it looked for and offers the nearest parcels on file --
because that message is the model's next instruction, not an error log.
"""

from __future__ import annotations

import re

from .. import data
from .base import fail, tool

# Hill Valley County parcel numbers are NN-NNN-NNNN: two digits of book, three
# of page, four of item. Nine digits, always.
_DIGITS_IN_A_PARCEL_ID = 9

# Below this many digits we are looking at a fragment, not a parcel number, and
# padding it out would invent a parcel that was never asked for.
_MIN_USABLE_DIGITS = 8


def normalize_parcel_id(raw: str) -> str | None:
    """Turn whatever the model sent into a canonical NN-NNN-NNNN parcel number.

        "04-127-3891"        -> "04-127-3891"
        "04127-3891"         -> "04-127-3891"
        "4-127-3891"         -> "04-127-3891"     (the padding does the work)
        "parcel #04 127 3891" -> "04-127-3891"
        "3891"               -> None              (a fragment, not a parcel)

    Returns None when the input cannot be a parcel number, so the caller can
    say something useful instead of guessing.
    """
    if raw is None:
        return None

    digits = re.sub(r"\D", "", str(raw))

    if len(digits) < _MIN_USABLE_DIGITS or len(digits) > _DIGITS_IN_A_PARCEL_ID:
        return None

    # A dropped leading zero is the common slip, and zero-padding fixes it
    # without ambiguity: the format is fixed-width, so there is only one way
    # eight digits can become nine.
    digits = digits.zfill(_DIGITS_IN_A_PARCEL_ID)
    return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"


def _nearby(parcel_id: str, limit: int = 3) -> list[str]:
    """Parcels sharing a book and page with the one we could not find.

    Turns a dead end into a lead. If the model asked for 04-127-9999 we can at
    least tell it what does exist in book 04, page 127.
    """
    book_page = parcel_id[:6]           # "04-127"
    con = data.cama()
    try:
        rows = con.execute(
            "SELECT parcel_id FROM parcels WHERE parcel_id LIKE ? "
            "ORDER BY parcel_id LIMIT ?",
            (book_page + "%", limit),
        ).fetchall()
    finally:
        con.close()
    return [r["parcel_id"] for r in rows]


@tool(
    name="lookup_parcel",
    description=(
        "Look up one parcel in the Hill Valley County assessment roll by its "
        "parcel number. Returns the owner of record, the situs address, the "
        "property characteristics and the current assessed value. Use this "
        "whenever a question names a specific parcel number."
    ),
    parameters={
        "type": "object",
        "properties": {
            "parcel_id": {
                "type": "string",
                "description": ("The parcel number, formatted NN-NNN-NNNN, "
                                "for example 04-127-3891."),
            },
        },
        "required": ["parcel_id"],
    },
    # Names a small model plausibly reaches for instead of the real one.
    name_aliases=("parcel_lookup", "get_parcel", "lookup_parcel_info",
                  "parcel_info", "get_parcel_info", "find_parcel"),
    param_aliases={"parcel_id": ("parcel", "id", "parcel_number", "parcelid",
                                 "parcel_no", "pid", "number", "parcelnumber")},
)
def lookup_parcel(parcel_id: str) -> dict | str:
    """Fetch one parcel from the Appraiser's roll."""
    canonical = normalize_parcel_id(parcel_id)

    if canonical is None:
        return fail(
            f"{parcel_id!r} does not look like a Hill Valley County parcel number.",
            "Parcel numbers have nine digits, written NN-NNN-NNNN, for example "
            "04-127-3891. If you only have an address or an owner name, use "
            "search_parcels instead.",
        )

    con = data.cama()
    try:
        row = con.execute(
            "SELECT * FROM parcels WHERE parcel_id = ?", (canonical,)
        ).fetchone()
    finally:
        con.close()

    if row is None:
        suggestions = _nearby(canonical)
        hint = ("Check the number and try again. "
                "You can also search by owner or address with search_parcels.")
        if suggestions:
            hint = (f"Parcels that do exist in that book and page: "
                    f"{', '.join(suggestions)}. " + hint)
        return fail(f"No parcel {canonical} on the Hill Valley County roll.", hint)

    return {
        "parcel_id": row["parcel_id"],
        "situs_address": row["situs_address"],
        "owner_name": row["owner_name"],
        "neighborhood": row["neighborhood"],
        "property_class": row["property_class_desc"],
        "year_built": row["year_built"],
        "living_area_sqft": row["living_area_sqft"],
        "lot_size_sqft": row["lot_size_sqft"],
        "bedrooms": row["bedrooms"],
        "bathrooms": row["bathrooms"],
        "assessed_value": row["assessed_value"],
        "assessment_year": row["assessment_year"],
    }


@tool(
    name="search_parcels",
    description=(
        "Find parcels in the Hill Valley County roll by owner name, street "
        "address, or neighborhood, when the parcel number is not known. "
        "Returns a short list of matches with their parcel numbers."
    ),
    parameters={
        "type": "object",
        "properties": {
            "owner": {"type": "string",
                      "description": "All or part of the owner's name."},
            "address": {"type": "string",
                        "description": "All or part of the street address."},
            "neighborhood": {"type": "string",
                             "description": ("Neighborhood name, e.g. Hilldale, "
                                             "Lyon Estates, Riverside Heights.")},
            "limit": {"type": "integer",
                      "description": "How many matches to return. Default 5."},
        },
        "required": [],
    },
    name_aliases=("find_parcels", "parcel_search", "search_parcel",
                  "lookup_by_owner", "find_property"),
    param_aliases={
        "owner": ("owner_name", "name", "ownername"),
        "address": ("situs_address", "street", "situs", "property_address"),
        "neighborhood": ("nbhd", "neighbourhood", "subdivision", "area"),
        "limit": ("max", "count", "top", "n"),
    },
)
def search_parcels(owner: str = "", address: str = "",
                   neighborhood: str = "", limit: int = 5) -> dict | str:
    """Find parcels without knowing the parcel number."""
    clauses, params = [], []
    if owner:
        clauses.append("owner_name LIKE ?")
        params.append(f"%{owner}%")
    if address:
        clauses.append("situs_address LIKE ?")
        params.append(f"%{address}%")
    if neighborhood:
        clauses.append("neighborhood LIKE ?")
        params.append(f"%{neighborhood}%")

    if not clauses:
        return fail(
            "search_parcels needs at least one of owner, address or neighborhood.",
            "Say who or where you are looking for, for example "
            'owner="Tannen" or neighborhood="Hilldale".',
        )

    limit = max(1, min(int(limit or 5), 25))
    con = data.cama()
    try:
        rows = con.execute(
            "SELECT parcel_id, situs_address, owner_name, neighborhood, "
            "       assessed_value "
            "FROM parcels WHERE " + " AND ".join(clauses) +
            " ORDER BY parcel_id LIMIT ?",
            (*params, limit),
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return fail(
            "No parcels matched that search.",
            "Try a shorter piece of the name or address -- partial matches are "
            "allowed, so \"Tannen\" works better than \"Biff H Tannen Jr\".",
        )

    return {
        "match_count": len(rows),
        "matches": [dict(r) for r in rows],
    }
