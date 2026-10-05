"""AppTest for the Section 2 app — the Flux Capacitor.

    uv run python scripts/test_app_flux.py

Runs on the 0.8B, so it is the quickest of the app tests.

Why this exists
---------------
§2 and §4/§5 were the two most-edited apps in the repository and the only two
with **no committed test**. Every bug found in them was caught by throwaway
scripts that were never checked in, which is a worse position than having no
tests at all: it looks covered and is not.

Every check below corresponds to a bug that actually shipped:

  * the candidate buttons       -- a duplicate-label crash waiting to happen
  * the vocabulary heading      -- hardcoded 248,320 in a five-model picker
  * the token warning           -- stated no count, and drifted from its twin
  * the memory button           -- said nothing, so it looked broken
  * the style attributes        -- a quote in a font stack ate every colour
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

# The default Windows console codepage cannot encode the app's emoji.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Resolved from this file, never hardcoded -- `test_app_88mph.py` carried an
# absolute path with a username in it for a while, in a repository whose whole
# purpose is being handed to somebody else.
REPO = Path(__file__).resolve().parent.parent
APP = str(REPO / "app" / "02_Flux_Capacitor_local_Streamlit.py")
TIMEOUT = 600


def fresh() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    return at


def report(at: AppTest, label: str) -> bool:
    bad = [str(e.value) for e in at.exception]
    print(f"  [{'OK ' if not bad else 'XX '}] {label}")
    for b in bad:
        print(f"         !! {b[:240]}")
    return not bad


def button(at: AppTest, contains: str):
    for b in at.button:
        if contains in (b.label or ""):
            return b
    raise AssertionError(
        f"no button containing {contains!r}; have {[b.label for b in at.button][:8]}")


def picker(at: AppTest):
    for box in at.selectbox:
        if box.label == "Model":
            return box
    raise AssertionError("no selectbox labelled 'Model'")


def main() -> int:
    ok = True
    at = fresh()
    print("=== 1. the app loads, on the model §2 is supposed to use ===")
    ok &= report(at, "loads clean")
    default = picker(at).value
    good = default == "tiny"
    ok &= good
    print(f"  [{'OK ' if good else 'XX '}] default model is {default!r} "
          f"(DEMOS.md pins §2 to the 0.8B; `index=0` used to decide this by "
          f"registry order)")

    print("\n=== 2. three candidate buttons, each carrying its own token ===")
    # Not one button. Being able to take second or third place is what makes
    # the temperature point, and each needs an explicit key: two token ids can
    # decode to nearly the same text (' location' and ' Location' are 1st and
    # 3rd on the default prompt) and identical labels without keys are one
    # widget to Streamlit -- a DuplicateWidgetID crash mid-demo.
    labels = [b.label for b in at.button]
    takes = [l for l in labels if "Take the winner" in l or "2nd place" in l
             or "3rd place" in l]
    good = len(takes) == 3
    ok &= good
    print(f"  [{'OK ' if good else 'XX '}] found {len(takes)} of 3")
    for l in takes:
        print(f"         {l}")

    print("\n=== 3. taking 2nd place goes somewhere greedy would not ===")
    start = at.session_state["predict_text"]
    button(at, "2nd place").click().run()
    second = at.session_state["predict_text"]
    ok &= report(at, "2nd place click ran")
    diverged = second != start and second.startswith(start)
    ok &= diverged
    print(f"  [{'OK ' if diverged else 'XX '}] {start!r} -> {second!r}")

    button(at, "Take the winner").click().run()
    onward = at.session_state["predict_text"]
    ok &= report(at, "then greedy from there")
    print(f"         continued to {onward!r}")

    button(at, "Reset the sentence").click().run()
    reset = at.session_state["predict_text"] == start
    ok &= reset
    print(f"  [{'OK ' if reset else 'XX '}] reset restores the default sentence")

    print("\n=== 4. the temperature expander explains why 2nd place exists ===")
    blobs = [m.value for m in at.markdown]
    has = any("do_sample=False" in b for b in blobs)
    ok &= has
    print(f"  [{'OK ' if has else 'XX '}] names the one argument the whole repo "
          f"pins (`do_sample=False`)")

    print("\n=== 5. the token warning states its own count ===")
    # Matches the notebook twin's wording, including the figure. DEMOS.md
    # quotes 13 tokens for the parcel and 8 for the value, so the app must
    # agree with the slide.
    for preset, expect in (("parcel 04-127-3891", 13), ("$412,500", 8),
                           ("appraisal", None)):
        button(at, preset).click().run()
        warns = [w.value for w in at.warning]
        hit = [w for w in warns if "Great Scott" in w]
        if expect is None:
            good = not hit
            print(f"  [{'OK ' if good else 'XX '}] {preset!r} raises no warning")
        else:
            good = bool(hit) and f"{expect} tokens" in hit[0]
            print(f"  [{'OK ' if good else 'XX '}] {preset!r} -> "
                  f"{hit[0].splitlines()[0][:60] if hit else '(no warning)'}")
        ok &= good

    print("\n=== 6. the vocabulary heading is read from the loaded model ===")
    # It was the literal 248,320 -- Qwen3.5's figure -- while the picker offers
    # vocabularies from 128,256 to 262,144, so the heading contradicted the
    # banner directly above it the moment anybody switched.
    for key, want in (("tiny", "248,320"), ("smol", "128,256")):
        if picker(at).value != key:
            picker(at).set_value(key).run()
        heads = [s.value for s in at.subheader]
        banners = [s.value for s in at.success]
        head = next((h for h in heads if "One guess" in h), "")
        banner = next((b for b in banners if "chooses between" in b), "")
        good = want in head and want in banner
        ok &= good
        print(f"  [{'OK ' if good else 'XX '}] {key:5s} heading and banner both "
              f"say {want}")

    print("\n=== 7. the memory button says what it actually did ===")
    at = fresh()
    button(at, "Free unused memory").click().run()
    ok &= report(at, "first click")
    caps = [c.value for c in at.caption]
    said = [c for c in caps if "Returned" in c or "Nothing to return" in c]
    ok &= bool(said)
    print(f"  [{'OK ' if said else 'XX '}] "
          f"{said[0][:76] if said else '(reported nothing)'}")
    button(at, "Free unused memory").click().run()
    ok &= report(at, "second click, when there is nothing left to hand back")

    print("\n=== 8. generation, and the wrap-up that follows it ===")
    button(at, "⚡ Generate").click().run()
    ok &= report(at, "generated")
    errs = [e.value for e in at.error]
    ok &= not errs
    for e in errs:
        print(f"         !! error box: {e[:150]}")
    caps = [c.value for c in at.caption]
    for what, needle in (("pass count", "passes** through the model"),
                         ("the §3 hook", "1.21 gigawatts")):
        good = any(needle in c for c in caps)
        ok &= good
        print(f"  [{'OK ' if good else 'XX '}] {what}")
    # The footer lives inside tab 3 so it cannot announce the conclusion while
    # the room is still on tab 1. ⚠️ AppTest flattens tabs, so this asserts the
    # footer EXISTS, not where it sits -- placement is enforced by indentation
    # in the app and by the comment there.
    blobs = [m.value for m in at.markdown]
    good = any("That's Section 2" in b for b in blobs)
    ok &= good
    print(f"  [{'OK ' if good else 'XX '}] the wrap-up renders")

    print("\n=== 9. no style attribute was eaten by a quote ===")
    # viz.MONO once held 'Courier New' inside single-quoted style attributes,
    # which ended them early and silently discarded every colour after
    # font-family. scripts/test_viz_markup.py guards the builders directly;
    # this checks what the APP actually put on the page.
    htmls = [e.proto.body for e in at.get("html")]
    broken = 0
    for h in htmls:
        opens = h.count("style='")
        closes = len(re.findall(r"style='[^']*'", h))
        broken += opens - closes
    ok &= broken == 0
    print(f"  [{'OK ' if not broken else 'XX '}] {len(htmls)} html blocks, "
          f"{broken} truncated style attributes")

    print("\n" + "=" * 60)
    print("RESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
