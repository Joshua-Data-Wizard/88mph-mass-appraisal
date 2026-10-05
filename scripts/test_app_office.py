"""AppTest for the Section 7 + Section 8 app.

    uv run python scripts/test_app_office.py

This app had no test at all until 2026-08-25, which is how a change to
`local_responder` reached it unnoticed. Section 9's app has had one since it
was built; this is the matching cover for the other agent app.

What it actually checks, beyond "does it render":

  * the ONE tool in tab 1 and the TWO in tab 2 -- the toolbelt growing is the
    §8 lesson, so a regression there is a broken demo, not a broken widget
  * the raw reveal, which is §7's whole payload. It has to contain the
    model's tool-call markup, not its prose. That distinction is exactly what
    broke when `last_raw` briefly became the special-token-stripped decode.
  * restraint -- the question that must NOT reach for a tool
  * the mangled parcel id, which is §2's lesson collecting its debt
  * the system-prompt presets, which are click SEQUENCES through an on_click
    callback, and CLAUDE.md records that jumping to an end state hides bugs
  * no token count printed as a LITERAL -- the panel said "thirteen fragments"
    for a string that is 10 to Qwen and 6 to SmolLM3
  * the theme and the close: the hoverboard metaphor, the wrap-up naming both
    sections, and the hand-off to §9

⚠️ AppTest runs the script but not a browser, so it cannot see anything
DOMPurify strips. See the §9 app's `svg_card` note. Nothing here draws SVG.

⚠️ It also flattens tabs, so a footer assertion proves the footer EXISTS, not
which tab it sits in. That placement is the whole point of the footer rule, and
it is held by indentation in the app plus the comment beside it.
"""
import html as html_mod
import re
import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

# The default Windows console codepage cannot encode the app's emoji.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
APP = str(REPO / "app" / "07_08_Office_Assistant_local_Streamlit.py")
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

    print("\n=== 3. §7 tab — one tool, and the raw reveal ===")
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

    # ── The answer appears TWICE, on purpose ────────────────────────────────
    # The trace keeps the raw, escaped answer because that is the honest
    # picture -- the model emitted characters, `**` and all, which is §7's
    # whole argument. The box below it is the same text through st.markdown,
    # which is what an office would ship. Losing EITHER is a regression:
    #   * lose the box  -> `*   **Owner:** Biff Tannen` on the projector
    #   * lose the trace-> the mechanics go behind a curtain
    # ⚠️ EXACT match against `last_answer`, not "does this text appear
    # somewhere". The loose version passed with the box deleted, because the
    # §8 expander mentions the same parcel and the same owner in static copy.
    # Watching the guard fail is what exposed that.
    answer = at.session_state["last_answer"]
    rendered = any(m.value == answer for m in at.markdown)
    ok &= bool(answer) and rendered
    print(f"  [{'OK ' if rendered else 'XX '}] the answer is its OWN "
          f"st.markdown element, so markdown RENDERS  {answer[:44]!r}")

    heading = any("The same answer, rendered" in m.value for m in at.markdown)
    ok &= heading
    print(f"  [{'OK ' if heading else 'XX '}] the box is labelled as the same "
          f"text, not a second answer")

    # The trace must still carry its own Answer step. Checked by looking for
    # the answer's first words HTML-ESCAPED inside the markup, which is both
    # "the step is there" and "it is escaped, not interpolated raw".
    blob = markup(at)
    probe = html_mod.escape(answer.strip()[:40]) if answer else ""
    in_trace = bool(probe) and probe in blob
    ok &= in_trace
    print(f"  [{'OK ' if in_trace else 'XX '}] the trace still carries the raw "
          f"Answer step, escaped — nothing went behind a curtain")

    print("\n=== 4. §7 — the mangled parcel id (Section 2 collecting its debt) ===")
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

    print("\n=== 5. §7 — restraint: this one must NOT call a tool ===")
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
           "🏛️ The Section 5 expert",
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

    print("\n=== 9. no token count is printed as a literal ===")
    # §7's "two things to try" panel said the model sees the mangled parcel id
    # as "thirteen fragments". Three separate problems in one word: 13 is the
    # count for `parcel 04-127-3891` (a different string -- §2's preset, the
    # word and space included), the bare id is 11, and the id this panel
    # actually names is 10. Worse, it is tokenizer-dependent in a five-model
    # picker: SmolLM3 and Phi-4-mini see 6. Same bug class as §2's hardcoded
    # vocabulary size, and the repo's rule is to never print one as a literal.
    at = fresh()
    sys.path.insert(0, str(REPO / "src"))
    from delorean import llm  # noqa: E402

    blobs = [m.value for m in at.markdown]
    panel = next((b for b in blobs if "mangled parcel number" in b), "")
    ok &= bool(panel)
    print(f"  [{'OK ' if panel else 'XX '}] found the panel")

    # ⚠️ Word boundaries, not `in`. "ten" is a substring of "often",
    # "written" and "listen", so a plain containment check would fail this
    # test the first time somebody wrote a perfectly good sentence.
    spelled_out = re.findall(
        r"\b(?:thirteen|twelve|eleven|ten|nine|eight|six)\b", panel.lower())
    ok &= not spelled_out
    print(f"  [{'OK ' if not spelled_out else 'XX '}] no count spelled out in "
          f"words{'  !! ' + ', '.join(spelled_out) if spelled_out else ''}")

    for key, want in (("tiny", 10), ("smol", 6)):
        if at.selectbox[0].value != key:
            at.selectbox[0].set_value(key).run()
        live = len(llm.load_tokenizer(key).encode("4-127-3891"))
        blobs = [m.value for m in at.markdown]
        panel = next((b for b in blobs if "mangled parcel number" in b), "")
        good = live == want and f"{want} separate fragments" in panel
        ok &= good
        print(f"  [{'OK ' if good else 'XX '}] {key:5s} tokenizes "
              f"'4-127-3891' as {live} and the panel says so")

    print("\n=== 10. the theme lands, and the section closes ===")
    # §7 is called The Hoverboard and for a long time nothing but the tab label
    # said so. The metaphor is load-bearing, not decoration: the hoverboard is
    # a BORROWED capability, which is the technical claim of the whole section.
    #
    # ⚠️ AppTest flattens tabs, so this asserts the footer EXISTS, not which
    # tab it sits in. Placement is held by indentation in the app and by the
    # comment beside it -- and it matters: a footer under the tab strip would
    # announce "that's Sections 7 and 8" while the room is still on tab 1.
    at = fresh()
    blobs = [m.value for m in at.markdown]
    caps = [c.value for c in at.caption]

    for what, needle, where in (
            ("the hoverboard is BORROWED, not faster",
             "borrowed a board off a kid", blobs),
            ("...and ties it back to §2's unchanged model",
             "Nothing about the model has changed", blobs),
            ("the wrap-up names both sections", "That's Sections 7 and 8", blobs),
            ("§7's own §5 lie — the invented comps",
             "inventing comparable sales", blobs),
            ("§8's one-chip-to-two argument",
             "grow from one chip to two", blobs),
            ("the hand-off to §9", "That's Section 9", blobs),
            ("a closing quote, like the other two apps",
             "don't work on water", caps)):
        good = any(needle in b for b in where)
        ok &= good
        print(f"  [{'OK ' if good else 'XX '}] {what}")

    # Griff Tannen leads the 2015 hoverboard chase; Biff's gang is the 1955
    # SKATEBOARD scene the hoverboard sequence mirrors. Getting this backwards
    # in a Back to the Future themed talk is the kind of thing the room spots.
    hover = next((b for b in blobs if "borrowed a board" in b), "")
    good = "Griff" in hover and "Biff" not in hover
    ok &= good
    print(f"  [{'OK ' if good else 'XX '}] the hoverboard chase is Griff's "
          f"gang, not Biff's")

    print("\n" + "=" * 60)
    print("RESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
