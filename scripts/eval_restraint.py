"""eval_restraint.py — does a bigger toolbelt cost restraint, and does the
system prompt buy it back?

    uv run python scripts/eval_restraint.py            # the default workhorse
    uv run python scripts/eval_restraint.py smol       # another model

Why this exists
---------------
The Section 7/8 app prints a four-row table on screen, and DEMOS.md quotes it
onto a slide:

    toolbelt            restraint   recall
    1 tool              6/6         1/1
    2 tools             6/6         2/2
    3 tools             5/6         4/4
    3 tools + guard     6/6         4/4

**Nothing committed reproduced it.** `eval_tools.py` offers a fixed belt of
three tools and never sets a system prompt, so it cannot produce those rows.
The table came from a script that was thrown away -- the same mistake that left
two apps looking tested when they were not.

This script is that measurement, committed. **It does not reproduce the old
table — it corrects it.**

Measured 2026-10-04, `Qwen3.5-4B`, greedy decoding:

    toolbelt                                 restraint   recall
    1 tool   [lookup]                              5/5      2/2
    2 tools  [lookup, policy]   <- tab 2           4/5      4/4
    3 tools  [lookup, policy, parcels]             5/5      6/6
    3 tools  [lookup, parcels, policy]             4/5      6/6
    1 tool   [lookup]                 + guard      5/5      2/2
    2 tools  [lookup, policy]         + guard      5/5      4/4   <- fixed
    3 tools  [lookup, policy, parcels] + guard     5/5      6/6
    3 tools  [lookup, parcels, policy] + guard     4/5      6/6   <- NOT fixed

Three things changed from what the app and DEMOS.md had been claiming, and the
third one was a surprise.

**The slip is at TWO tools, not three** -- and two tools is precisely the
Almanac tab, so the failure happens in the configuration the audience is
looking at. The old table said two tools was clean, which quietly contradicted
the stage move printed right underneath it.

**Belt LENGTH is not the variable. Belt ORDER is.** The last two rows hold the
same three tools and differ only in the sequence they are declared in, and they
score differently. `tools.schemas()` preserves the order it is given, so this
is the model reading a list and being swayed by position -- not a harness
artifact. Listing `search_parcels` before `search_policy` is enough to pull the
definition question toward the policy manual.

**And the guardrail does not always rescue it.** On the app's own belts it
does, cleanly: 4/5 -> 5/5 with no loss of recall, which is what the stage move
depends on. On `[lookup, parcels, policy]` the slip survives the prompt
entirely. **A system prompt is a strong lever, not a guarantee** -- so a
configuration that has not been measured has not been established, even when a
measured neighbour of it looks fine.

Every slip in every configuration is the same single question:
*"What is the difference between market value and assessed value?"*, answered
by searching the policy manual. `scripts/eval_tools.py` reproduces it
independently -- 28/29 overall, with that one case as the only failure.

⚠️ Greedy decoding, so a re-run on the same model is deterministic. Re-run it
when the question set, the prompts, the tool ORDER, or the default model
change.

What the two numbers mean
-------------------------
**Restraint** -- the question is general knowledge or small talk. Nothing in a
parcel record or the policy manual answers it, so *any* tool call is wrong. An
agent that hits the database because somebody said "good morning" looks broken
on stage.

**Recall** -- the question genuinely needs a tool that is on the belt. Restraint
bought by making the agent reluctant to use tools at all is not a win, so the
two have to be read together.

Both sets are reused verbatim from `eval_tools.py` so the figures stay
comparable with the 29-case eval.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from delorean import data, llm, models, prompts, tools  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── The questions ───────────────────────────────────────────────────────────
#
# Restraint: category E of `eval_tools.py`, verbatim. Deliberately general
# knowledge and small talk -- nothing here lives in a parcel record OR in the
# policy manual, so a tool call is always the wrong move.
RESTRAINT = [
    "What is the difference between market value and assessed value?",
    "What are the three approaches to value in real property appraisal?",
    "Good morning!",
    "Thanks, that's all I needed.",
    "Why do property taxes exist?",
]

# Recall: two questions per tool, so every belt is scored only on the tools it
# actually has. Taken from categories A/B (lookup_parcel), D (search_parcels)
# and G (search_policy) of `eval_tools.py`.
RECALL: dict[str, list[str]] = {
    "lookup_parcel": [
        "Who owns parcel 04-127-3891?",
        "Pull up 4-127-3891.",
    ],
    "search_policy": [
        "Can a parcel be split if the owner owes back taxes?",
        "When is the deadline to file an assessment appeal?",
    ],
    "search_parcels": [
        "Which parcels does Biff Tannen own?",
        "Find properties in the Hilldale neighborhood.",
    ],
}

# The belts. ⚠️ ORDER IS A VARIABLE, not a formatting choice -- the last two
# hold the same three tools and score differently, so they are both measured.
# `tools.schemas()` passes this sequence through untouched.
#
# `[lookup]` is Section 7's tab 1 and `[lookup, policy]` is Section 8's tab 2,
# so the first two rows are the belts this app actually puts on stage.
BELTS: list[tuple[str, list[str]]] = [
    ("1 tool   [lookup]",               ["lookup_parcel"]),
    ("2 tools  [lookup, policy]",       ["lookup_parcel", "search_policy"]),
    ("3 tools  [lookup, policy, parcels]",
     ["lookup_parcel", "search_policy", "search_parcels"]),
    ("3 tools  [lookup, parcels, policy]",
     ["lookup_parcel", "search_parcels", "search_policy"]),
]

# Each belt is measured bare and with the guardrail. Bare is what the app's
# "🤖 No system prompt at all" preset does, which is the stage move.
CONFIGS: list[tuple[str, list[str], str]] = (
    [(label, belt, "") for label, belt in BELTS]
    + [(f"{label}  + guard", belt, prompts.FRONT_COUNTER)
       for label, belt in BELTS]
)


def asked_for_a_tool(model, tok, question: str, belt: list[str], system: str,
                     max_new_tokens: int = 160) -> tuple[bool, str, str]:
    """One turn. Did the model reach for a tool, and which one?

    Mirrors `eval_tools.py`: a single generation with the tool schemas rendered
    into the chat template, decoded with `skip_special_tokens=False` because
    Gemma's calls are made of special tokens and vanish without it.
    """
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    prompt = tok.apply_chat_template(
        messages, tools=tools.schemas(belt), tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=False)
    calls = tools.parse_tool_calls(text)
    return bool(calls), (calls[0].name if calls else ""), text


def main() -> int:
    key = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    spec = models.get(key)
    data.check()

    print("=" * 78)
    print(f"  Restraint vs toolbelt size   ·   {spec.label}  ({spec.hf_id})")
    print("=" * 78)
    print("  restraint = answered directly when it should have")
    print("  recall    = called the right tool when it genuinely needed one\n")

    model, tok = llm.load_local(key, verbose=False)
    rows, t0 = [], time.time()

    for label, belt, system in CONFIGS:
        print(f"── {label}   ({', '.join(belt)})")

        kept, slips = 0, []
        for q in RESTRAINT:
            called, name, _ = asked_for_a_tool(model, tok, q, belt, system)
            if called:
                slips.append((q, name))
            else:
                kept += 1
            print(f"     {'OK ' if not called else 'XX '} restraint  "
                  f"{q[:52]:52s} {'-> ' + name if called else ''}")

        hits, misses = 0, []
        recall_qs = [(t, q) for t in belt for q in RECALL[t]]
        for want, q in recall_qs:
            called, name, _ = asked_for_a_tool(model, tok, q, belt, system)
            good = called and name == want
            hits += good
            if not good:
                misses.append((q, want, name or "no tool"))
            print(f"     {'OK ' if good else 'XX '} recall     "
                  f"{q[:52]:52s} -> {name or 'no tool'}")

        rows.append((label, kept, len(RESTRAINT), hits, len(recall_qs),
                     slips, misses))
        print()

    print("=" * 78)
    print(f"  {spec.label}   ·   {time.time() - t0:.0f}s total")
    print("=" * 78)
    print("=" * 78)
    print(f"  {'toolbelt':44s} {'restraint':>10s} {'recall':>8s}")
    for label, kept, nr, hits, nc, slips, misses in rows:
        flag = "" if kept == nr else "  <- slips"
        print(f"  {label:44s} {f'{kept}/{nr}':>10s} {f'{hits}/{nc}':>8s}{flag}")

    print("\n  Every restraint slip, verbatim:")
    any_slip = False
    for label, _, _, _, _, slips, misses in rows:
        for q, name in slips:
            any_slip = True
            print(f"    {label:44s} {q!r} -> {name}")
        for q, want, got in misses:
            print(f"    {label:44s} MISSED RECALL {q!r} "
                  f"wanted {want}, got {got}")
    if not any_slip:
        print("    (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
