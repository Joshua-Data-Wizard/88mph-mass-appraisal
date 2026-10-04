"""
test_tools.py — prove the tools survive the mistakes a small model actually makes.

    uv run python scripts/test_tools.py

No pytest, no new dependency. Same idea as check_setup.py: run it, read the
report, and know whether the machine is ready.

Every case in here is either measured from a real model run or a slip that is
one keystroke away from one. The point is not code coverage -- it is that a
demo in front of 200 people does not die because the model dropped a hyphen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from delorean import data, tools                                  # noqa: E402
from delorean.tools.parcel import normalize_parcel_id             # noqa: E402

HERO = "04-127-3891"
FAILURES: list[str] = []


def check(label: str, passed: bool, detail: str = "") -> None:
    mark = "PASS" if passed else "FAIL"
    print(f"  {mark}  {label}" + (f"   {detail}" if detail else ""))
    if not passed:
        FAILURES.append(label)


def call(name: str, args: dict) -> dict:
    """Dispatch and decode, so tests can assert on the envelope."""
    return json.loads(tools.dispatch(name, args))


def section(title: str) -> None:
    print("\n" + "-" * 74)
    print(f"  {title}")
    print("-" * 74)


# ─────────────────────────────────────────────────────────────────────────
print("=" * 74)
print("  Tool robustness — can the agent survive a sloppy caller?")
print("=" * 74)

data.check()

# ── 1. The normalizer ────────────────────────────────────────────────────
section("1. Parcel number normalization")
print("     The three starred forms were emitted by Qwen3.5-4B across ten")
print("     phrasings of the same question. All must reach the same row.\n")

NORMALIZE_CASES = [
    ("04-127-3891", HERO, "canonical                      *measured"),
    ("04127-3891", HERO, "hyphen dropped                 *measured"),
    ("4-127-3891", HERO, "leading zero dropped           *measured"),
    ("041273891", HERO, "no punctuation at all"),
    ("  04-127-3891  ", HERO, "surrounding whitespace"),
    ("parcel 04-127-3891", HERO, "the word 'parcel' included"),
    ("#04-127-3891", HERO, "a hash in front"),
    ("04 127 3891", HERO, "spaces instead of hyphens"),
    ("04-127-3891.", HERO, "trailing full stop"),
    ("04/127/3891", HERO, "slashes instead of hyphens"),
    ("3891", None, "a fragment -> refused, not guessed"),
    ("", None, "empty -> refused"),
    ("not a parcel", None, "prose -> refused"),
    ("0412738910000", None, "far too many digits -> refused"),
]
for raw, want, why in NORMALIZE_CASES:
    got = normalize_parcel_id(raw)
    check(f"{raw!r:<22} -> {str(got):<14} {why}", got == want,
          "" if got == want else f"expected {want}")

# ── 2. The measured variants reach the real record ───────────────────────
section("2. Every measured variant returns Biff Tannen's parcel")
for variant in ("04-127-3891", "04127-3891", "4-127-3891", "parcel 04 127 3891"):
    r = call("lookup_parcel", {"parcel_id": variant})
    good = r["ok"] and r["data"]["owner_name"] == "Biff Tannen"
    check(f"lookup_parcel({variant!r})", good,
          r["data"]["situs_address"] if good else r.get("error", ""))

# ── 3. Parameter names the model might guess ─────────────────────────────
section("3. The model calls the argument something else")
for key in ("parcel_id", "parcel", "id", "parcelId", "parcel_number",
            "PARCEL_ID", "pid", "parcel_no"):
    r = call("lookup_parcel", {key: HERO})
    check(f"argument named {key!r}", r["ok"],
          "" if r["ok"] else r.get("error", "")[:60])

# ── 4. Tool names the model might guess ──────────────────────────────────
section("4. The model calls the tool something else")
for name in ("lookup_parcel", "lookup-parcel", "lookupParcel", "Lookup_Parcel",
             "parcel_lookup", "get_parcel", "lookup_parcels", "find_parcel"):
    r = call(name, {"parcel_id": HERO})
    check(f"tool named {name!r}", r["ok"],
          "" if r["ok"] else r.get("error", "")[:60])

# ── 5. Genuinely wrong input gets a USEFUL failure ───────────────────────
section("5. Failures have to teach the model something")

r = call("lookup_parcel", {"parcel_id": "04-127-9999"})
check("nonexistent parcel does not raise", not r["ok"] and "error" in r)
check("  ...and the hint names real nearby parcels",
      "hint" in r and "04-127" in r["hint"], r.get("hint", "")[:78])

r = call("lookup_parcel", {"parcel_id": "3891"})
check("fragment is refused, not guessed at", not r["ok"])
check("  ...and the hint explains the format",
      "NN-NNN-NNNN" in r.get("hint", ""), r.get("hint", "")[:78])

r = call("lookup_parcel", {})
check("missing required argument", not r["ok"])
check("  ...and the hint names the missing parameter",
      "parcel_id" in r.get("hint", ""), r.get("hint", "")[:78])

r = call("teleport_parcel", {"parcel_id": HERO})
check("invented tool name", not r["ok"])
check("  ...and the hint lists the real tools",
      "lookup_parcel" in r.get("hint", ""), r.get("hint", "")[:78])

# ── 6. Junk that should not crash anything ───────────────────────────────
section("6. Nothing raises, whatever arrives")
JUNK = [
    ("lookup_parcel", {"parcel_id": None}),
    ("lookup_parcel", {"parcel_id": ""}),
    ("lookup_parcel", {"parcel_id": '"04-127-3891"'}),      # quoted value
    ("lookup_parcel", {"parcel_id": HERO, "colour": "blue"}),  # extra arg
    ("lookup_parcel", {"parcel_id": 41273891}),             # an int, not a string
    ("lookup_parcel", None),
    ("", {}),
    ("search_parcels", {}),
]
for name, args in JUNK:
    try:
        out = tools.dispatch(name, args)
        decoded = json.loads(out)
        check(f"dispatch({name!r}, {str(args)[:34]}) returned an envelope",
              isinstance(decoded, dict) and "ok" in decoded)
    except Exception as e:  # noqa: BLE001
        check(f"dispatch({name!r}, {str(args)[:34]})", False,
              f"RAISED {type(e).__name__}: {e}")

r = call("lookup_parcel", {"parcel_id": '"04-127-3891"'})
check("a quoted value still finds the parcel", r["ok"])
r = call("lookup_parcel", {"parcel_id": 41273891})
check("an integer argument still finds the parcel", r["ok"])

# ── 7. Type coercion ─────────────────────────────────────────────────────
section("7. XML tool calls carry no types - everything arrives as a string")
r = call("search_parcels", {"neighborhood": "Hilldale", "limit": "3"})
check("limit='3' (a string) is coerced to an integer",
      r["ok"] and r["data"]["match_count"] == 3,
      f"got {r['data']['match_count'] if r['ok'] else r.get('error')}")
r = call("search_parcels", {"neighborhood": "Hilldale", "limit": "three"})
check("limit='three' fails cleanly instead of crashing", not r["ok"],
      r.get("error", "")[:60])

# ── 8. search_parcels does its job ───────────────────────────────────────
section("8. search_parcels")
r = call("search_parcels", {"owner": "Tannen"})
check("finds parcels by partial owner name", r["ok"] and r["data"]["match_count"] > 0,
      f"{r['data']['match_count']} matches" if r["ok"] else r.get("error", ""))
r = call("search_parcels", {"name": "Tannen"})
check("...even when the argument is called 'name'", r["ok"])
r = call("search_parcels", {"neighborhood": "Nowhere At All"})
check("no matches fails helpfully", not r["ok"] and "hint" in r)

# ── 9. What the model is shown ───────────────────────────────────────────
section("9. The schemas handed to the model")
all_schemas = tools.schemas()
one = tools.schemas(["lookup_parcel"])
check(f"{len(all_schemas)} tools registered", len(all_schemas) >= 2,
      ", ".join(sorted(tools.REGISTRY)))
check("schemas() can expose a single tool for Section 7",
      len(one) == 1 and one[0]["function"]["name"] == "lookup_parcel")
check("every schema has the shape apply_chat_template wants",
      all(s.get("type") == "function" and "parameters" in s["function"]
          for s in all_schemas))
check("descriptions are written for a model to read",
      all(len(s["function"]["description"]) > 60 for s in all_schemas))

# ── 9b. search_policy — the RAG tool ─────────────────────────────────────
section("9b. search_policy (Section 8's RAG tool)")
print("     First call builds the index, which loads the embedding model.\n")

r = call("search_policy", {"question": "minimum lot size to split an R-1 parcel"})
check("finds the split policy", r["ok"] and r["data"]["passages"])
if r["ok"]:
    top = r["data"]["passages"][0]
    check("  ...and it is the right section", top["cite"].endswith("4.3"), top["cite"])
    check("  ...every passage carries a citation",
          all("§" in p["cite"] for p in r["data"]["passages"]))
    check("  ...no passage cites an empty section",
          all(not p["cite"].endswith("§") for p in r["data"]["passages"]),
          "front matter must stay out of the index")
    check("  ...the model is told to cite what it used",
          "cite" in r["data"].get("instruction", "").lower())

r = call("search_policy", {"question": "can a parcel be split if taxes are owed?"})
check("finds the tax bar on splits (Section 10 depends on this)",
      r["ok"] and r["data"]["passages"][0]["cite"].endswith("4.2"),
      r["data"]["passages"][0]["cite"] if r["ok"] else r.get("error", ""))

for key in ("question", "query", "q", "text", "search_query"):
    r = call("search_policy", {key: "appeal deadline"})
    check(f"argument named {key!r}", r["ok"])

r = call("search_policy", {"question": "PRD", "top_k": "2"})
check("top_k='2' as a string is coerced", r["ok"] and len(r["data"]["passages"]) == 2)
r = call("search_policy", {"question": "PRD", "k": 1})
check("k=1 aliases to top_k", r["ok"] and len(r["data"]["passages"]) == 1)

r = call("search_policy", {"question": "   "})
check("an empty question fails with a hint", not r["ok"] and "hint" in r)
r = call("policy_search", {"query": "appeal deadline"})
check("tool nickname 'policy_search' resolves", r["ok"])


# ── 10. Parsing what the model emitted ───────────────────────────────────
section("10. The parser handles all FOUR registry formats")
print("     Four models, four conventions. Two wrap in <tool_call> with")
print("     different contents, one omits the tags entirely, and Gemma 4")
print("     brings its own delimiters. A tag check alone fools you.\n")

SHAPE_A = (  # Qwen3.5 and Nemotron -- measured
    "<tool_call>\n<function=lookup_parcel>\n"
    "<parameter=parcel_id>\n04-127-3891\n</parameter>\n</function>\n</tool_call>"
)
SHAPE_B = (  # SmolLM3 -- what its chat template ASKS for
    '<tool_call>\n{"name": "lookup_parcel", '
    '"arguments": {"parcel_id": "04-127-3891"}}\n</tool_call>'
)
SHAPE_C = (  # SmolLM3 -- what it ACTUALLY writes. No tags at all.
    '{"name": "lookup_parcel", "arguments": {"parcel_id": "04-127-3891"}}'
)
SHAPE_D = (  # Gemma 4 -- its own delimiters, <|"|> instead of quotes
    '<|tool_call>call:lookup_parcel{parcel_id:<|"|>04-127-3891<|"|>}<tool_call|>'
)

for label, blob in (("Shape A (Qwen3.5 / Nemotron)", SHAPE_A),
                    ("Shape B (SmolLM3 template)", SHAPE_B),
                    ("Shape C (SmolLM3 actual)", SHAPE_C),
                    ("Shape D (Gemma 4)", SHAPE_D),
                    ("Shape D (Phi-4-mini, = Shape C)",
                     '{"name": "lookup_parcel", '
                     '"parameters": {"parcel_id": "04-127-3891"}}')):
    calls = tools.parse_tool_calls(blob)
    good = (len(calls) == 1 and calls[0].name == "lookup_parcel"
            and calls[0].args.get("parcel_id") == HERO)
    check(f"{label} parses", good, str(calls[0].args) if calls else "no call")

check("prose with no tool call yields nothing",
      tools.parse_tool_calls("Parcel 04-127-3891 is owned by Biff Tannen.") == [])

TRUNCATED = "<tool_call>\n<function=lookup_parcel>\n<parameter=parcel_id>\n04-127"
check("a truncated call yields no bogus call",
      tools.parse_tool_calls(TRUNCATED) == [])
check("  ...but is detectable, so the agent can raise max_new_tokens",
      tools.attempted_tool_call(TRUNCATED))

check("reasoning before a call is kept separate from it",
      tools.strip_tool_calls("I should look this up.\n" + SHAPE_A)
      == "I should look this up.")

MULTI = SHAPE_A + "\n" + SHAPE_A.replace("04-127-3891", "01-133-3269")
check("parallel tool calls are all returned, not just the first",
      len(tools.parse_tool_calls(MULTI)) == 2)

# Shape C is the loose one -- bare JSON with no delimiters at all -- so it
# carries two guards the others do not need. Both matter: without them, any
# answer containing a brace could fire a tool.
check("Shape C ignores prose that merely contains JSON",
      tools.parse_tool_calls('Store it as {"a": 1} in the config.') == [])
check("Shape C ignores well-formed JSON naming no real tool",
      tools.parse_tool_calls(
          '{"name": "launch_missiles", "arguments": {"x": 1}}') == [])

# Shape D has explicit delimiters, so an unknown NAME is a genuine call to a
# tool that does not exist -- parsed on purpose, so dispatch() can hand the
# model back the list of tools it really has. See section 6 above.
unknown = tools.parse_tool_calls(
    '<|tool_call>call:launch_missiles{x:<|"|>1<|"|>}<tool_call|>')
check("Shape D parses an unknown tool rather than silently dropping it",
      len(unknown) == 1 and unknown[0].name == "launch_missiles")
check("  ...and dispatch turns that into a usable hint",
      not json.loads(tools.dispatch("launch_missiles", {}))["ok"])

# Gemma emits <|tool_response> straight after the call and, unchecked, invents
# a result for a tool that has not run. That must never reach a human.
GEMMA_RUNON = (SHAPE_D + '<|tool_response>The owner is Marty McFly.')
check("Gemma's invented tool response is stripped from the prose",
      "Marty" not in tools.strip_tool_calls(GEMMA_RUNON))
check("  ...and the real call still parses out of it",
      tools.parse_tool_calls(GEMMA_RUNON)[0].name == "lookup_parcel")

# End to end: parse, then dispatch, exactly as the agent node will.
for label, blob in (("Shape A", SHAPE_A), ("Shape B", SHAPE_B),
                    ("Shape C", SHAPE_C), ("Shape D", SHAPE_D)):
    c = tools.parse_tool_calls(blob)[0]
    out = json.loads(tools.dispatch(c.name, c.args))
    check(f"{label}: parse -> dispatch -> Biff Tannen",
          out["ok"] and out["data"]["owner_name"] == "Biff Tannen")

# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 74)
if FAILURES:
    print(f"  {len(FAILURES)} FAILED:")
    for f in FAILURES:
        print("    -", f)
    raise SystemExit(1)
print("  ALL TOOL ROBUSTNESS CHECKS PASSED")
print("=" * 74)
