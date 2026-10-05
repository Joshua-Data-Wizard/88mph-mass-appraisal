"""AppTest for the Section 9 app.

Real click SEQUENCES, not jumps to an end state -- CLAUDE.md records a bug
that `set_value()` sailed straight past because it skipped the intermediate
rerun where the bug lived.
"""
import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

# The default Windows console codepage cannot encode the app's emoji.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Resolved from this file, never hardcoded. This was an absolute path with a
# username in it until 2026-10-03, which meant the test ran on exactly one
# machine in the world -- in a repository whose entire purpose is being handed
# to somebody else.
REPO = Path(__file__).resolve().parent.parent
APP = str(REPO / "app" / "09_88mph_Valuation_Review_local_Streamlit.py")
TIMEOUT = 400


def fresh():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    return at


def button(at, key):
    for b in at.button:
        if b.key == key:
            return b
    raise AssertionError(f"no button {key!r}; have {[b.key for b in at.button]}")


def report(at, label):
    bad = [str(e.value) for e in at.exception]
    status = "OK " if not bad else "XX "
    print(f"  [{status}] {label}")
    for b in bad:
        print("        ", b.splitlines()[0] if b else b)
    return not bad


def main():
    print("\n=== 1. cold start ===")
    t0 = time.time()
    at = fresh()
    ok = report(at, f"app renders ({time.time()-t0:.0f}s incl. model load)")
    print(f"        buttons: {len(at.button)}  text inputs: {len(at.text_input)}")
    print(f"        info shown: {len(at.info)}  title: {at.title[0].value[:44]!r}")

    print("\n=== 2. the three story questions, in order ===")
    for key, label in [("rev_county", "1 county"),
                       ("rev_hoods", "2 by neighborhood"),
                       ("rev_map", "3 map")]:
        t0 = time.time()
        button(at, key).click().run()
        ok &= report(at, f"{label}  ({time.time()-t0:.0f}s)")
        blocks = list(at.get("html")) + list(at.get("iframe"))
        markup = " ".join((getattr(b.proto,"body","") or getattr(b.proto,"srcdoc","")) for b in blocks)
        drawn = [name for name, needle in
                 [("scorecard", "IAAO ratio study"),
                  ("neighborhood table", "fail at least one IAAO"),
                  ("map", "coloured by how their assessment"),
                  ("speedometer", "Median assessment ratio")] if needle in markup]
        print(f"        html blocks {len(blocks)}  drew: {', '.join(drawn)}")

    print("\n=== 3. back to 1  (does a repeat click still work?) ===")
    button(at, "rev_county").click().run()
    ok &= report(at, "1 again after 3")

    print("\n=== 4. free text, typed then Run ===")
    at.text_input[0].set_value("Who owns parcel 04-127-3891?").run()
    ok &= report(at, "typing does not fire the agent")
    at.button[0].click().run()   # the Run button
    ok &= report(at, "Run executes the typed question")

    print("\n=== 5. the no-ratio-tool probe, then straight back ===")
    button(at, "try_noratio").click().run()
    ok &= report(at, "no-ratio probe runs")
    warns = [w.value.lower() for w in at.warning]
    banner = any("ratio_study` tool for this run" in w for w in warns)
    state = at.session_state["state"]
    print(f"        banner persists after rerun: {banner}")
    print(f"        toolbelt was: {state.get('toolbelt')}")
    print(f"        garbage table rendered: {len(at.dataframe) == 1}")
    assert banner, "the no-ratio banner must survive the rerun"
    assert state.get("toolbelt") == ["run_sql", "map_parcels"]
    button(at, "rev_hoods").click().run()
    ok &= report(at, "toolbelt switches BACK to three tools")

    print("\n=== 6. restraint case draws no findings ===")
    button(at, "try_define").click().run()
    ok &= report(at, "answers directly, no tool")

    print("\n=== 7. the story order — trace, then answer, then the figures ===")
    # §9 used to open with the polished answer and fold the loop away below
    # it. That is the conclusion arriving before the argument, and worse than
    # untidy: hiding the steps turns a demonstration into an assertion, in
    # the one section whose job is convincing a room that a machine can be
    # trusted with an assessment roll. The trace is the EVIDENCE.
    #
    # AppTest gives one flat list per element type, so an st.html cannot be
    # compared against an st.markdown. Captions can: the trace ends with
    # "Toolbelt for this run" and the answer and findings each carry one, so
    # their order inside at.caption IS document order.
    button(at, "rev_county").click().run()
    ok &= report(at, "county question re-run for the ordering check")

    caps = [c.value for c in at.caption]
    def where(needle):
        return next((i for i, c in enumerate(caps) if needle in c), None)

    spots = [("the trace", where("Every step, in the order")),
             ("the answer", where("identical characters")),
             ("the figures", where("believe the numbers"))]
    missing = [name for name, i in spots if i is None]
    ok &= not missing
    print(f"  [{'OK ' if not missing else 'XX '}] all three sections present "
          f"{ {n: i for n, i in spots} }")
    if not missing:
        order = [i for _, i in spots]
        good = order == sorted(order)
        ok &= good
        print(f"  [{'OK ' if good else 'XX '}] trace → answer → figures, in "
              f"that order")

    for what, needle in (("the trace is NOT behind an expander",
                          "##### 🔁 How it got there"),
                         ("the answer is labelled plainly",
                          "##### 🏁 What it concluded"),
                         ("the figures invite checking, not believing",
                          "##### 📊 Now check it yourself")):
        good = any(needle in m.value for m in at.markdown)
        ok &= good
        print(f"  [{'OK ' if good else 'XX '}] {what}")

    # The SQL is quoted above the trace as highlighted code, deliberately as
    # well as inside the trace's Act step -- the presenter wants it readable.
    good = len(at.code) >= 1
    ok &= good
    print(f"  [{'OK ' if good else 'XX '}] the SQL it wrote is quoted as code "
          f"({len(at.code)} block(s))")

    answer = at.session_state["state"]["answer"]
    rendered = any(m.value == answer for m in at.markdown)
    ok &= bool(answer) and rendered
    print(f"  [{'OK ' if rendered else 'XX '}] the answer is its OWN "
          f"st.markdown element, so markdown RENDERS")

    print("\n=== 8. the needle: one median, or none at all ===")
    # Reported from the screen: question 2 left the countywide median glowing
    # above a table of six different ones. The old code kept the stale value
    # on purpose and explained it in a subtitle, which was defensible while
    # the dial lived in the HEADER as a standing instrument -- and is not,
    # now that it is drawn with the figures for the run on screen.
    #
    # ⚠️ Counted as IFRAMES. The dial is SVG, and SVG through `st.html` is
    # stripped by DOMPurify -- which is what `svg_card()` exists to avoid. A
    # probe that looked in `get("html")` reported "not drawn" when it was.
    #
    # ⚠️ And it is a SEQUENCE, 1 -> 2 -> 1, not a jump to the end state.
    # CLAUDE.md records a bug that slipped past exactly that shortcut.
    def needle():
        return at.session_state["dial"], len(at.get("iframe"))

    button(at, "rev_county").click().run()
    dial, frames = needle()
    good = dial is not None and frames >= 1
    ok &= good
    print(f"  [{'OK ' if good else 'XX '}] Q1 countywide → needle shows "
          f"{dial!r} ({frames} iframe)")

    button(at, "rev_hoods").click().run()
    dial, frames = needle()
    good = dial is None and frames == 0
    ok &= good
    print(f"  [{'OK ' if good else 'XX '}] Q2 by neighborhood → needle "
          f"CLEARED (dial={dial!r}, {frames} iframe)")
    grouped = any("Ratio study by" in (getattr(b.proto, "body", "") or "")
                  for b in at.get("html"))
    ok &= grouped
    print(f"  [{'OK ' if grouped else 'XX '}] …and the grouped table is still "
          f"drawn")

    button(at, "rev_county").click().run()
    dial, frames = needle()
    good = dial is not None and frames >= 1
    ok &= good
    print(f"  [{'OK ' if good else 'XX '}] back to Q1 → needle returns "
          f"({dial!r})")

    print("\n" + "=" * 60)
    print("RESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
