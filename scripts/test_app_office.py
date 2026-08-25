"""AppTest for the Section 6 + Section 8 app.

    uv run python scripts/test_app_office.py

This app had no test at all until 2026-08-25, which is how a change to
`local_responder` reached it unnoticed. Section 7's app has had one since it
was built; this is the matching cover for the other agent app.

What it actually checks, beyond "does it render":

  * the ONE tool in tab 1 and the TWO in tab 2 -- the toolbelt growing is the
    §8 lesson, so a regression there is a broken demo, not a broken widget
  * the raw reveal, which is §6's whole payload. It has to contain the
    model's tool-call markup, not its prose. That distinction is exactly what
    broke when `last_raw` briefly became the special-token-stripped decode.
  * restraint -- the question that must NOT reach for a tool
  * the mangled parcel id, which is §2's lesson collecting its debt
  * the system-prompt presets, which are click SEQUENCES through an on_click
    callback, and CLAUDE.md records that jumping to an end state hides bugs

⚠️ AppTest runs the script but not a browser, so it cannot see anything
DOMPurify strips. See the §7 app's `svg_card` note. Nothing here draws SVG.
"""
import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

# The default Windows console codepage cannot encode the app's emoji.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
APP = str(REPO / "app" / "04_06_Office_Assistant_local_Streamlit.py")
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


def report(at, label, extra=""):
    bad = [str(e.value) for e in at.exception]
    print(f"  [{'OK ' if not bad else 'XX '}] {label}{('  ' + extra) if extra else ''}")
    for b in bad:
        print("        ", b.splitlines()[0] if b else b)
    return not bad


def markup(at):
    """Every st.html body on the page, concatenated."""
    return " ".join(getattr(b.proto, "body", "") or "" for b in at.get("html"))


def main():
    ok = True

    print("\n=== 1. cold start ===")
    t0 = time.time()
    at = fresh()
    ok &= report(at, f"app renders ({time.time() - t0:.0f}s incl. model load)")
    print(f"        buttons {len(at.button)} · text inputs {len(at.text_input)} "
          f"· tabs {len(at.tabs)}")

    print("\n=== 2. the model picker hides what will not run ===")
    opts = at.selectbox[0].options
    ok &= report(at, f"picker offers {len(opts)} models, nemotron excluded",
                 str(list(opts)))
    if "nemotron" in opts:
        print("        XX  nemotron is selectable and should not be")
        ok = False
    captions = " ".join(c.value for c in at.caption)
    ok &= report(at, "the absence is explained, not silent",
                 "Nemotron named" if "Nemotron" in captions else "!! NO EXPLANATION")

    print("\n=== 3. §6 tab — one tool, and the raw reveal ===")
    t0 = time.time()
    button(at, "hover_go").click().run()
    used = at.session_state["last_used"]
    ok &= report(at, f"default question runs ({time.time() - t0:.0f}s)", f"tools={used}")
    if used != ["lookup_parcel"]:
        print(f"        XX  expected ['lookup_parcel'], got {used}")
        ok = False

    raw = at.session_state["last_raw"]
    is_call = any(t in raw for t in ("<tool_call>", "<|tool_call>", '"name"'))
    ok &= report(at, "the reveal shows the model's TOOL CALL, not its prose",
                 repr(raw.strip()[:70]))
    if not is_call:
        print("        XX  last_raw holds no tool-call markup at all")
        ok = False

    print("\n=== 4. §6 — the mangled parcel id (Section 2 collecting its debt) ===")
    button(at, "hq1").click().run()          # "Pull up 4-127-3891."
    ok &= report(at, "preset button fills the box",
                 repr(at.session_state["hover_q"][:40]))
    t0 = time.time()
    button(at, "hover_go").click().run()
    used = at.session_state["last_used"]
    body = markup(at)
    ok &= report(at, f"mangled id still resolves ({time.time() - t0:.0f}s)",
                 f"tools={used}")
    if "Biff Tannen" not in body:
        print("        XX  the trace does not show Biff Tannen")
        ok = False

    print("\n=== 5. §6 — restraint: this one must NOT call a tool ===")
    button(at, "hq3").click().run()          # market vs assessed value
    t0 = time.time()
    button(at, "hover_go").click().run()
    used = at.session_state["last_used"]
    ok &= report(at, f"answers a definition directly ({time.time() - t0:.0f}s)",
                 f"tools={used or 'none'}")
    if used:
        print(f"        XX  reached for {used} on a general question")
        ok = False

    print("\n=== 6. §8 tab — the toolbelt has grown to two ===")
    button(at, "aq0").click().run()          # back taxes / parcel split
    t0 = time.time()
    button(at, "almanac_go").click().run()
    used = at.session_state["last_used"]
    offered = at.session_state["last_offered"]
    ok &= report(at, f"policy question runs ({time.time() - t0:.0f}s)",
                 f"offered={offered} used={used}")
    if offered != ["lookup_parcel", "search_policy"]:
        print(f"        XX  expected a two-tool belt, got {offered}")
        ok = False
    if used != ["search_policy"]:
        print(f"        XX  expected search_policy, got {used}")
        ok = False
    body = markup(at)
    cited = "Policy" in body or "§" in body
    ok &= report(at, "the answer carries a checkable citation",
                 "found" if cited else "!! no document/section reference")

    print("\n=== 7. the system-prompt presets, clicked in sequence ===")
    seq = ["🤖 No system prompt at all",
           "🏛️ The Section 4 expert",
           "🎛️ Front counter (the guardrail)"]
    for name in seq:
        button(at, f"sysbtn_{name}").click().run()
        text = at.session_state["system_text"]
        shown = at.session_state["preset_name"]
        good = shown == name
        print(f"  [{'OK ' if good else 'XX '}] {name:<34} "
              f"-> {len(text):>3} chars")
        ok &= good and report(at, f"   no exception on {name!r}")

    print("\n=== 8. empty system prompt does not break the run ===")
    button(at, "sysbtn_🤖 No system prompt at all").click().run()
    t0 = time.time()
    button(at, "hover_go").click().run()
    ok &= report(at, f"runs with no system prompt ({time.time() - t0:.0f}s)",
                 f"tools={at.session_state['last_used'] or 'none'}")

    print("\n" + "=" * 60)
    print("RESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
