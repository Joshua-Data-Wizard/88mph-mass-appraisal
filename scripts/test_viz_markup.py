"""Every builder in viz.py must emit style attributes a browser can parse.

    uv run python scripts/test_viz_markup.py

Why this exists
---------------
`MONO` used to be `"…Consolas,'Courier New',monospace"` — a font stack with
**single quotes inside it** — and every style attribute in `viz.py` is
delimited with single quotes. So the browser ended the attribute early:

    style='…font-family:…Consolas,'        <- attribute ends HERE
    Courier New',monospace;color:#34e05a   <- discarded as junk attributes

**Everything written after `font-family:` was silently thrown away** —
`color`, `font-size`, `font-weight`, `text-shadow`, padding, borders — in 12
of this file's builders. Nothing errored. Nothing looked broken enough to
investigate. The DeLorean readouts just rendered in default grey, and the
presenter spotted it before any test did.

This runs no model and needs no GPU. It is a markup check, not a look check:
it cannot tell you a colour is ugly, only that the colour survives to the
browser at all.
"""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from delorean import viz  # noqa: E402


class FakeTokenizer:
    """Enough of a tokenizer for `tokens_html`, with no model to load."""

    def encode(self, text, add_special_tokens=False):
        return [100 + i for i, _ in enumerate(text.split() or [""])]

    def decode(self, ids):
        return "tok"

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]


# One representative call per public builder. Values chosen to exercise the
# interesting branches: a real string, an empty string, and a verdict.
CASES = {
    "tokens_html": lambda: viz.tokens_html(FakeTokenizer(), "parcel 04-127-3891"),
    "probabilities_html": lambda: viz.probabilities_html(
        [(" location", 0.81), (" and", 0.047), (" Location", 0.008)],
        prompt="Location, location,", vocab_size=248_320),
    "canvas_html": lambda: viz.canvas_html(["An ", "appraiser ", "estimates"],
                                           title="Qwen3.5-0.8B"),
    "time_circuits_html": lambda: viz.time_circuits_html(
        "🏛️ Senior appraiser", "My assessment went up 15%", ""),
    "persona_card_html": lambda: viz.persona_card_html(
        "😤 Angry taxpayer", "This is outrageous.",
        system_prompt="You are an angry taxpayer.", pending=True),
    "verification_html": lambda: viz.verification_html(
        "0.25 acres, per the County's Zoning Ordinance.",
        "6,000 sq ft per resulting parcel.",
        source="Policy 4.3", question="What is the minimum lot size?",
        consequence="It reverses the decision."),
    "note_html": lambda: viz.note_html("A title", "A body", kind="warn"),
}


class AttrSpy(HTMLParser):
    """Collect every attribute name a real HTML parser sees."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.names: set[str] = set()
        self.tags = 0

    def handle_starttag(self, tag, attrs):
        self.tags += 1
        for name, _value in attrs:
            self.names.add(name)


# Everything viz.py legitimately writes. Anything else means an attribute was
# truncated and the leftovers were parsed as new attributes.
ALLOWED = {"style", "class", "colspan", "rowspan", "title", "srcdoc", "src",
           "width", "height", "alt", "href", "target", "rel", "id"}


def check(name: str, markup: str) -> list[str]:
    """Return a list of problems with this markup, empty if it is sound."""
    problems = []

    # 1. Every style attribute must close. A quote inside one ends it early.
    opens = markup.count("style='")
    closes = len(re.findall(r"style='[^']*'", markup))
    if opens != closes:
        problems.append(f"{opens} style attrs opened, only {closes} parse cleanly")

    # 2. The decisive check: hand it to a real parser and see what attributes
    #    it finds. With the old MONO this produced attributes called `courier`,
    #    `new'`, `9px` and friends -- the shrapnel of a truncated style.
    spy = AttrSpy()
    spy.feed(markup)
    junk = sorted(spy.names - ALLOWED)
    if junk:
        problems.append(f"{len(junk)} bogus attribute(s) from truncated markup: "
                        + ", ".join(repr(j) for j in junk[:6]))

    # 3. Tag soup check: every < must find its >.
    if markup.count("<") != markup.count(">"):
        problems.append(f"{markup.count('<')} '<' vs {markup.count('>')} '>'")

    return problems


def main() -> int:
    print("=" * 66)
    print("  viz.py markup — every style attribute must parse")
    print("=" * 66)

    ok = True
    for name, build in CASES.items():
        try:
            markup = build()
        except Exception as exc:  # noqa: BLE001
            print(f"  [XX ] {name:22s} raised {type(exc).__name__}: {exc}")
            ok = False
            continue
        problems = check(name, markup)
        ok &= not problems
        attrs = len(re.findall(r"style='[^']*'", markup))
        print(f"  [{'OK ' if not problems else 'XX '}] {name:22s} "
              f"{attrs:>3} style attrs, {len(markup):>5} chars")
        for p in problems:
            print(f"         !! {p}")

    # The specific constant that caused it, asserted directly.
    quoted = "'" in viz.MONO
    ok &= not quoted
    print(f"\n  [{'OK ' if not quoted else 'XX '}] viz.MONO contains no single "
          f"quote  ({viz.MONO[:46]}…)")
    if quoted:
        print("         !! a single quote here ends every style attribute early")

    print("\n" + "=" * 66)
    print("RESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
