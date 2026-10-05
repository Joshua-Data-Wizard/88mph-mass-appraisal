"""AppTest for the Section 10 app — three MCP servers, two demos.

    uv run python scripts/test_app_adapter.py

This one exercises more moving parts than any other test in the repository:
a language model, three subprocesses speaking MCP over stdio, a local RAG
tool, and a Streamlit script thread coordinating all of it.

⚠️ **This test is why `Offices` no longer uses `run_in_executor`.** It
appeared to hang for an hour; it had in fact finished in twenty seconds and
then could not exit, because a blocked thread-pool worker kept the interpreter
alive holding several GB of VRAM. See CLAUDE.md, section 13.

If it ever stalls again, check **CPU time against wall clock** before anything
else — a working process burns CPU and a stuck one does not — and run
`Offices(quiet=False)`, because a server that dies on startup says so on
stderr and nowhere else.

Run it with output to a file, never piped through `grep`: grep buffers, and a
finished-but-hung run then looks exactly like one that never started.
"""
import html
import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

# The default Windows console codepage cannot encode the app's emoji.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
APP = str(REPO / "app" / "10_Universal_Adapter_local_Streamlit.py")
TIMEOUT = 400


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
    """Every st.html body on the page, as readable text.

    ⚠️ Unescaped on the way out, and that is not tidiness. The first version
    of this helper returned the raw markup, and the check for the panel titled
    "with the Clerk's correction" failed against a perfectly good app: every
    builder in viz.py runs its title through `html.escape`, so the apostrophe
    arrives as `Clerk&#x27;s` and no substring search for it can ever match.

    An assertion that fails on correct output is worse than no assertion --
    it sends you looking for a bug that is not there.
    """
    raw = " ".join(getattr(b.proto, "body", "") or ""
                   for b in at.get("html"))
    return html.unescape(raw)


def answer_text(at):
    return " ".join(m.value for m in at.markdown)


def main():
    ok = True

    print("\n=== 1. cold start: model + three MCP servers ===")
    t0 = time.time()
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    ok &= report(at, f"app renders ({time.time() - t0:.0f}s)")

    body = markup(at)
    for office in ("Appraiser", "Clerk of Courts", "Tax Collector",
                   "This office"):
        good = office in body
        ok &= good
        print(f"  [{'OK ' if good else 'XX '}] panel present: {office}")

    # The catalogue is the claim. These names came off the servers.
    for tool in ("get_parcel", "find_parcels", "get_deeds",
                 "corrections_since", "get_tax_status"):
        good = tool in body
        ok &= good
        print(f"  [{'OK ' if good else 'XX '}] discovered from a server: {tool}")

    print("\n=== 2. DEMO 7 — may Biff split his lot? ===")
    t0 = time.time()
    button(at, "q_split").click().run()
    secs = time.time() - t0
    ok &= report(at, f"question runs ({secs:.0f}s)")

    said = answer_text(at)
    checks = {
        "says no / denied": any(w in said.lower() for w in
                                ("may not", "cannot", "not be", "no,", "denied")),
        "quotes the balance $8,432.13": "8,432.13" in said or "8432.13" in said,
        "names policy 4.2": "4.2" in said,
        "names the Appraiser's lot size": "15,800" in said or "15800" in said,
    }
    for label, good in checks.items():
        ok &= good
        print(f"  [{'OK ' if good else 'XX '}] {label}")

    body = markup(at)
    answered = body.count("answered")
    print(f"        office panels marked answered: {answered}")
    if answered < 3:
        print("        XX  fewer than three offices answered")
        ok = False

    # §10's novelty is NOT "several tool calls" -- §9 already does that over
    # one database. It is several calls to separate PROGRAMS, three of them
    # somebody else's. The app states both numbers in a caption, so both are
    # read off the trace rather than written into the source.
    acts = [s for s in at.session_state["state"]["trace"]
            if s.get("step") == "act"]
    names = [s["tool"] for s in acts]
    lead = next((c.value for c in at.caption
                 if "separate programs" in c.value), "")
    good = bool(lead) and f"{len(acts)} tool calls" in lead
    ok &= good
    print(f"  [{'OK ' if good else 'XX '}] the caption counts the real calls "
          f"({len(acts)}), it does not claim a number")
    good = len(acts) >= 4 and len(set(names)) >= 4
    ok &= good
    print(f"  [{'OK ' if good else 'XX '}] four distinct sources consulted "
          f"{names}")

    print("\n=== 3. DEMO 8 — is our sales file still true? ===")
    t0 = time.time()
    button(at, "q_sync").click().run()
    secs = time.time() - t0
    ok &= report(at, f"sync check runs ({secs:.0f}s)")

    said = answer_text(at)
    checks = {
        "finds the corrected parcel": "12-393-7896" in said,
        "says QUIT CLAIM": "quit claim" in said.lower(),
        "impact panel drew a before/after": (
            markup(at).lower().count("courthouse square") >= 2
            and "our sales file" in markup(at).lower()
            and "clerk's correction" in markup(at).lower()),
        # The one theme reference in this app, and it earns its place: a
        # stale snapshot of a record edited behind you IS the photograph.
        "the photograph lands": any(
            "fade out of it one at a time" in i.value for i in at.info),
        "and it names the pull date": any(
            "pulled from the Clerk on" in i.value for i in at.info),
        "the cost section has its own heading": any(
            "##### 📊 What that costs you" in m.value for m in at.markdown),
    }
    for label, good in checks.items():
        ok &= good
        print(f"  [{'OK ' if good else 'XX '}] {label}")

    errs = " ".join(e.value for e in at.error)
    hidden = "hiding" in errs or "flattering" in errs
    print(f"  [{'OK ' if hidden else 'XX '}] calls out the hidden second failure")
    ok &= hidden

    print("\n=== 4. back to demo 7 (does a repeat click still work?) ===")
    t0 = time.time()
    button(at, "q_split").click().run()
    ok &= report(at, f"repeat click ({time.time() - t0:.0f}s)")

    print("\n=== 5. a free-text question through the same offices ===")
    t0 = time.time()
    at.text_input[0].set_value(
        "Who owns parcel 04-127-3891 and what did they pay for it?").run()
    button(at, "q_owner").click().run()
    ok &= report(at, f"also-try button runs ({time.time() - t0:.0f}s)")

    print("\n=== 6. the story order, and the trace is OPEN ===")
    # Two earlier versions were wrong in opposite directions: the answer at
    # the very top, then the answer last but the trace still shut inside a
    # collapsed expander. §10 is the section where a machine tells somebody
    # they may not split their own land, so the evidence cannot sit behind a
    # click. Order must be: how it got there -> the decision -> what it costs.
    #
    # AppTest exposes one flat list per element type, so an st.html cannot be
    # compared against an st.markdown. Captions can: each section carries one,
    # so their order inside at.caption IS document order.
    caps = [c.value for c in at.caption]

    def where(needle):
        return next((i for i, c in enumerate(caps) if needle in c), None)

    trace_at = where("separate programs")
    answer_at = where("identical characters the trace above")
    found = trace_at is not None and answer_at is not None
    ok &= found
    print(f"  [{'OK ' if found else 'XX '}] both sections present "
          f"(trace #{trace_at}, decision #{answer_at})")
    if found:
        good = trace_at < answer_at
        ok &= good
        print(f"  [{'OK ' if good else 'XX '}] the trace comes BEFORE the "
              f"decision")

    for what, needle in (
            ("the trace leads, and is not an expander",
             "##### 🔁 How it got there"),
            ("the decision is labelled, after the offices",
             "##### 🏁 The decision, with every office consulted")):
        good = any(needle in m.value for m in at.markdown)
        ok &= good
        print(f"  [{'OK ' if good else 'XX '}] {what}")

    answer = at.session_state["state"]["answer"]
    rendered = any(m.value == answer for m in at.markdown)
    ok &= bool(answer) and rendered
    print(f"  [{'OK ' if rendered else 'XX '}] the answer is its OWN "
          f"st.markdown element, so markdown RENDERS")

    print("\n" + "=" * 60)
    print("RESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
