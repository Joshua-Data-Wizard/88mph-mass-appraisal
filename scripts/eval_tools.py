"""
eval_tools.py — measure how well a model actually uses our tools.

    uv run python scripts/eval_tools.py                # the default workhorse
    uv run python scripts/eval_tools.py qwen smol      # compare models

`scripts/test_tools.py` proves the tools survive bad input. This is the other
half: given a real question, does the model call the *right* tool, with usable
arguments, and — just as importantly — does it know when to call **no tool at
all**?

That last one matters more than it sounds. An agent that reaches for a database
every time somebody says "good morning" is worse than useless: it is slow, it
looks broken, and in front of an audience it is the thing that goes wrong.

⚠️  Only Qwen3.5-4B is cached on the dev box. Naming `smol` or `nemotron` will
    download several GB of weights the first time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json  # noqa: E402

import torch  # noqa: E402

from delorean import data, llm, models, tools  # noqa: E402

HERO = "04-127-3891"

# Which tools the agent is offered. Section 6 exposes one on the live path;
# the eval offers both so we can measure whether it picks correctly.
OFFERED = ["lookup_parcel", "search_parcels", "search_policy"]


def build_cases() -> list[dict]:
    """The question set.

    `expect` is the tool that should be called, or None for "answer directly".
    `outcome` is what dispatching it should produce:
        "hero"     -> the record for 04-127-3891
        "any"      -> any successful result
        "graceful" -> a failure WITH a hint (the tool must not raise)
    """
    # A second real parcel, read from the roll so this stays true if the data
    # is ever regenerated. Guards against a model that has simply memorised
    # the hero parcel from the rest of the demo.
    con = data.cama()
    try:
        other = con.execute(
            "SELECT parcel_id, owner_name, situs_address FROM parcels "
            "WHERE neighborhood_code = 'LYON' AND property_class = 'R01' "
            "ORDER BY parcel_id LIMIT 1"
        ).fetchone()
    finally:
        con.close()

    cases = [
        # ── A. clean parcel lookups ──────────────────────────────────────
        ("A. clean lookup", "Who owns parcel 04-127-3891?", "lookup_parcel", "hero"),
        ("A. clean lookup", "What is parcel 04-127-3891 assessed at?", "lookup_parcel", "hero"),
        ("A. clean lookup", "Can you pull the record for parcel #04-127-3891?", "lookup_parcel", "hero"),
        ("A. clean lookup", "Tell me about PARCEL 04-127-3891 please", "lookup_parcel", "hero"),
        ("A. clean lookup", "Give me the assessed value for parcel 04-127-3891 and the owner too.", "lookup_parcel", "hero"),

        # ── B. the model mangles the identifier ──────────────────────────
        ("B. mangled id", "I need details on 04127-3891.", "lookup_parcel", "hero"),
        ("B. mangled id", "Pull up 4-127-3891.", "lookup_parcel", "hero"),
        ("B. mangled id", "parcel 04 127 3891 - who owns it", "lookup_parcel", "hero"),
        ("B. mangled id", "Look up parcel 04/127/3891 for me", "lookup_parcel", "hero"),

        # ── C. a different real parcel ───────────────────────────────────
        ("C. other parcel", f"Who owns parcel {other['parcel_id']}?", "lookup_parcel", "any"),
        ("C. other parcel", f"What year was {other['parcel_id']} built?", "lookup_parcel", "any"),

        # ── D. no parcel number given -> must search instead ─────────────
        ("D. search", "Which parcels does Biff Tannen own?", "search_parcels", "any"),
        ("D. search", "Find properties in the Hilldale neighborhood.", "search_parcels", "any"),
        ("D. search", "Show me parcels owned by someone named Tannen.", "search_parcels", "any"),
        ("D. search", "How big is the house at 1955 Hilldale Ct?", "search_parcels", "any"),

        # ── E. answer directly, DO NOT call a tool ───────────────────────
        # Deliberately general knowledge or small talk. Nothing here lives in
        # a parcel record OR in the policy manual, so any tool call is wrong.
        ("E. no tool", "What is the difference between market value and assessed value?", None, None),
        ("E. no tool", "What are the three approaches to value in real property appraisal?", None, None),
        ("E. no tool", "Good morning!", None, None),
        ("E. no tool", "Thanks, that's all I needed.", None, None),
        ("E. no tool", "Why do property taxes exist?", None, None),

        # ── G. the answer is in the policy manual, not a database ────────
        # Adding search_policy makes the choice harder: these are questions a
        # model could plausibly answer from memory, and the whole point of
        # Section 8 is that it should go and read the manual instead.
        ("G. policy", "What is the minimum lot size to split an R-1 parcel in this county?", "search_policy", "any"),
        ("G. policy", "Can a parcel be split if the owner owes back taxes?", "search_policy", "any"),
        ("G. policy", "When is the deadline to file an assessment appeal?", "search_policy", "any"),
        ("G. policy", "What COD does this office consider acceptable for improved residential?", "search_policy", "any"),
        ("G. policy", "Which deed types does this office treat as qualified sales?", "search_policy", "any"),
        ("G. policy", "What is our policy on combining two parcels?", "search_policy", "any"),

        # ── F. the tool must fail gracefully ─────────────────────────────
        ("F. graceful fail", "Who owns parcel 99-999-9999?", "lookup_parcel", "graceful"),
        ("F. graceful fail", "Look up parcel 3891.", "lookup_parcel", "graceful"),
        ("F. graceful fail", "Find parcels owned by Zorblax the Destroyer.", "search_parcels", "graceful"),
    ]
    return [{"cat": c, "q": q, "expect": e, "outcome": o} for c, q, e, o in cases]


def run_model(key: str, cases: list[dict], max_new_tokens: int = 160) -> dict:
    spec = models.get(key)
    print("\n" + "=" * 92)
    print(f"  {spec.label}   ({spec.hf_id})")
    print("=" * 92)

    model, tok = llm.load_local(key, verbose=False)
    schemas = tools.schemas(OFFERED)

    rows, t_start = [], time.time()
    for case in cases:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": case["q"]}],
            tools=schemas, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                         skip_special_tokens=True)

        calls = tools.parse_tool_calls(raw)
        truncated = not calls and tools.attempted_tool_call(raw)

        verdict, detail = "FAIL", ""

        if case["expect"] is None:
            # Should have answered directly.
            if not calls:
                verdict, detail = "PASS", "answered directly"
            else:
                detail = f"called {calls[0].name} unnecessarily"
        elif truncated:
            detail = "tool call truncated - raise max_new_tokens"
        elif not calls:
            detail = "no tool call emitted"
        else:
            call = calls[0]
            resolved = tools.resolve_tool(call.name)
            if resolved is None or resolved.name != case["expect"]:
                detail = (f"called {call.name!r} "
                          f"(wanted {case['expect']})")
            else:
                result = json.loads(tools.dispatch(call.name, call.args))
                if case["outcome"] == "graceful":
                    if not result["ok"] and result.get("hint"):
                        verdict, detail = "PASS", "failed with a usable hint"
                    else:
                        detail = "expected a graceful failure"
                elif not result["ok"]:
                    detail = f"tool said: {result.get('error','')[:46]}"
                elif case["outcome"] == "hero":
                    got = result["data"].get("owner_name")
                    if got == "Biff Tannen":
                        verdict, detail = "PASS", "-> Biff Tannen"
                    else:
                        detail = f"resolved to {got!r}"
                else:
                    verdict, detail = "PASS", "ok"

        if len(calls) > 1:
            detail += f"  [+{len(calls)-1} extra call(s)]"

        rows.append({**case, "verdict": verdict, "detail": detail,
                     "n_calls": len(calls)})

    elapsed = time.time() - t_start

    # ── report ──────────────────────────────────────────────────────────
    print(f"\n  {'category':<18} {'question':<52} {'result'}")
    print("  " + "-" * 88)
    for r in rows:
        mark = "PASS" if r["verdict"] == "PASS" else "FAIL"
        print(f"  {r['cat']:<18} {r['q'][:50]:<52} {mark}  {r['detail'][:32]}")

    passed = sum(r["verdict"] == "PASS" for r in rows)
    print("  " + "-" * 88)

    print(f"\n  By category:")
    cats: dict[str, list] = {}
    for r in rows:
        cats.setdefault(r["cat"], []).append(r)
    for cat, group in sorted(cats.items()):
        good = sum(g["verdict"] == "PASS" for g in group)
        bar = "#" * good + "." * (len(group) - good)
        print(f"    {cat:<20} {good}/{len(group):<4} {bar}")

    over_eager = sum(1 for r in rows if r["expect"] is None and r["n_calls"] > 0)
    parallel = sum(1 for r in rows if r["n_calls"] > 1)
    print(f"\n  SCORE  {passed}/{len(rows)}  ({100*passed/len(rows):.0f}%)"
          f"   {elapsed:.0f}s   {elapsed/len(rows):.1f}s per question")
    print(f"  Unnecessary tool calls on 'answer directly' questions: {over_eager}")
    print(f"  Questions that produced more than one tool call:       {parallel}")

    return {"model": spec.label, "passed": passed, "total": len(rows),
            "seconds": elapsed, "over_eager": over_eager, "rows": rows}


def main() -> None:
    keys = sys.argv[1:] or ["qwen"]
    unknown = [k for k in keys if k not in models.REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown model key(s): {unknown}. "
                         f"Choose from {list(models.REGISTRY)}")

    data.check()
    cases = build_cases()

    print("=" * 92)
    print("  Tool-use evaluation")
    print("=" * 92)
    print(f"  {len(cases)} questions, offering the model: {', '.join(OFFERED)}")
    print("  Scored on: did it pick the right tool, and do the arguments resolve?")

    results = [run_model(k, cases) for k in keys]

    if len(results) > 1:
        print("\n" + "=" * 92)
        print("  SUMMARY")
        print("=" * 92)
        print(f"  {'model':<28} {'score':>10} {'over-eager':>12} {'time':>9}")
        for r in results:
            print(f"  {r['model']:<28} {r['passed']:>4}/{r['total']:<5} "
                  f"{r['over_eager']:>12} {r['seconds']:>7.0f}s")

    if any(r["passed"] < r["total"] for r in results):
        print("\n  Some cases failed. That is data, not necessarily a bug -- read the")
        print("  detail column and decide whether the tool or the prompt should change.")


if __name__ == "__main__":
    main()
