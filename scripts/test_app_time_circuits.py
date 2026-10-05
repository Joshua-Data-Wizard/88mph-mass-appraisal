"""AppTest for the Sections 4 and 5 app — Time Circuits and Plutonium.

    uv run python scripts/test_app_time_circuits.py

Runs on the 4B, because §4 needs a model that can actually hold a character,
so this is slower than `test_app_flux.py`. Budget a couple of minutes.

Why this exists
---------------
This app and the §2 app were the two most-edited in the repository and the
only two with **no committed test** — which is worse than having none, because
it looks covered and is not. Every check below is a bug that shipped:

  * the wait estimate     -- hardcoded 14 tok/s in a five-model picker
  * the answer card       -- kept the preset name over a hand-edited prompt
  * the §5 PRD probe      -- quoted a figure that matched no stratification
  * the memory button     -- said nothing, so it looked broken
  * the time circuits     -- a quote in a font stack ate every row colour
  * the footer            -- announced §5's punchline during §4
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
APP = str(REPO / "app" / "04_05_Time_Circuits_and_Plutonium_local_Streamlit.py")
TIMEOUT = 900

# The three DeLorean rows, in order, from viz._CIRCUIT_ROWS.
CIRCUIT_COLOURS = {"#ff3b30": "DESTINATION TIME (red)",
                   "#34e05a": "PRESENT TIME (green)",
                   "#ffb000": "LAST TIME DEPARTED (amber)"}


def report(at: AppTest, label: str) -> bool:
    bad = [str(e.value) for e in at.exception]
    errs = [e.value for e in at.error]
    print(f"  [{'OK ' if not bad and not errs else 'XX '}] {label}")
    for b in bad + errs:
        print(f"         !! {b[:240]}")
    return not bad and not errs


def button(at: AppTest, contains: str):
    for b in at.button:
        if contains in (b.label or ""):
            return b
    raise AssertionError(
        f"no button containing {contains!r}; have {[b.label for b in at.button][:8]}")


def main() -> int:
    ok = True
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()

    print("=== 1. loads, on the 4B, with both sections named ===")
    ok &= report(at, "loads clean")
    caps = [c.value for c in at.caption]
    good = any("Sections 4 & 5" in c for c in caps)
    ok &= good
    print(f"  [{'OK ' if good else 'XX '}] sidebar says 'Sections 4 & 5' "
          f"(renumbering left this as 'Sections 5 & 5' once)")

    print("\n=== 2. the time circuits render all three row colours ===")
    # viz.MONO once held 'Courier New' inside single-quoted style attributes,
    # which ended them early and discarded everything after font-family --
    # including these colours. The readouts rendered in browser-default grey
    # for weeks and nothing failed.
    htmls = [e.proto.body for e in at.get("html")]
    circuit = next((h for h in htmls if "DESTINATION TIME" in h.upper()
                    or "Destination Time" in h), "")
    for colour, what in CIRCUIT_COLOURS.items():
        good = colour in circuit
        ok &= good
        print(f"  [{'OK ' if good else 'XX '}] {colour}  {what}")

    broken = sum(h.count("style='") - len(re.findall(r"style='[^']*'", h))
                 for h in htmls)
    ok &= broken == 0
    print(f"  [{'OK ' if not broken else 'XX '}] {len(htmls)} html blocks, "
          f"{broken} truncated style attributes")

    print("\n=== 3. the wait estimate is honest before it is measured ===")
    before = [c for c in (x.value for x in at.caption)
              if "seconds** of generation" in c]
    good = bool(before) and "estimated at 14 tokens a second" in before[0]
    ok &= good
    print(f"  [{'OK ' if good else 'XX '}] labelled an estimate until something runs")

    print("\n=== 4. a persona preset loads its prompt ===")
    button(at, "🔬 Doc Brown").click().run()
    ok &= report(at, "preset click")
    loaded = "Doc Emmett Brown" in at.session_state["system_text"]
    ok &= loaded
    print(f"  [{'OK ' if loaded else 'XX '}] system prompt is now "
          f"{len(at.session_state['system_text'])} chars")

    print("\n=== 5. ⚡ Ask, and the estimate becomes a measurement ===")
    button(at, "⚡ Ask").click().run()
    ok &= report(at, "generated an answer")
    after = [c for c in (x.value for x in at.caption)
             if "seconds** of generation" in c]
    good = bool(after) and "measured" in after[0]
    ok &= good
    print(f"  [{'OK ' if good else 'XX '}] "
          f"{after[0][-66:] if after else '(no estimate caption)'}")

    print("\n=== 6. an edited prompt is not labelled with the old preset ===")
    # destination_label() exists because "a display that confidently reports
    # the wrong destination is worse than no display at all" -- and the answer
    # card used to do exactly that.
    at.session_state["system_text"] = "You are a licensed land surveyor."
    at.run()
    button(at, "⚡ Ask").click().run()
    ok &= report(at, "asked with a hand-edited prompt")
    cards = [h for h in (e.proto.body for e in at.get("html")) if "system:" in h]
    edited = any("edited" in c for c in cards)
    ok &= edited
    print(f"  [{'OK ' if edited else 'XX '}] the card says '✏️ edited', not the "
          f"stale preset name")

    print("\n=== 7. §5's PRD probe quotes the figure §9 will show ===")
    # Hilldale has two true PRDs: 1.042 residential-only (IAAO's
    # stratification, what CLAUDE.md prints) and 1.043 across all classes
    # (what §9's agent computes and DEMOS.md prints). This probe promises §9
    # will find the number, so it must be 1.043. It read 1.041 until
    # 2026-10-04, which matched neither.
    at.radio(key="probe_key").set_value("📊 What does PRD stand for?").run()
    q = at.session_state["probe_question"]
    good = "1.043" in q and "1.041" not in q and "1.042" not in q
    ok &= good
    print(f"  [{'OK ' if good else 'XX '}] {q}")

    print("\n=== 8. the memory button says what it actually did ===")
    button(at, "Free unused memory").click().run()
    ok &= report(at, "first click")
    said = [c for c in (x.value for x in at.caption)
            if "Returned" in c or "Nothing to return" in c]
    ok &= bool(said)
    print(f"  [{'OK ' if said else 'XX '}] "
          f"{said[0][:76] if said else '(reported nothing)'}")

    print("\n=== 9. §5 earns its title, and hands off to §6 ===")
    blobs = [m.value for m in at.markdown]
    for what, needle in (
            ("the plutonium framing", "needs plutonium to make the jump"),
            ("the wrap-up", "That's Sections 4 and 5"),
            ("the §6 hand-off", "Section 6 starts to fix")):
        good = any(needle in b for b in blobs)
        ok &= good
        print(f"  [{'OK ' if good else 'XX '}] {what}")
    closer = any("don't need roads" in c for c in (x.value for x in at.caption))
    ok &= closer
    print(f"  [{'OK ' if closer else 'XX '}] the closing quote")

    print("\n" + "=" * 60)
    print("RESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
