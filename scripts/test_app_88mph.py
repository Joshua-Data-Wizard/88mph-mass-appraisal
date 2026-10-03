"""AppTest for the Section 7 app.

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
APP = str(REPO / "app" / "05_88mph_Valuation_Review_local_Streamlit.py")
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

    print("\n" + "=" * 60)
    print("RESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
