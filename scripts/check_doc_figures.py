"""Do the ratio tables in the docs match what the data actually says?

    uv run python scripts/check_doc_figures.py

Parses the neighborhood tables out of `data/README.md` (residential only) and
`DEMOS.md` (all classes), recomputes both from the committed databases, and
reports any cell that disagrees.

This exists because `data/README.md` carried pre-regeneration figures in six of
seven rows and nobody noticed. CLAUDE.md warned it would happen; a warning is
not a check.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from delorean import data, ratio  # noqa: E402

HOODS = ("Lyon Estates", "Twin Pines Ranch", "Eastwood Ravine",
         "Riverside Heights", "Courthouse Square", "Hilldale")


def computed(residential: bool) -> dict[str, tuple[int, float, float, float]]:
    feed = data.sales_feed_df()
    frame = feed[feed.arms_length == 1]
    if residential:
        frame = frame[frame.property_class.isin(["R01", "R02"])]
    out = {}
    g = ratio.ratio_by_group(frame, "neighborhood").set_index("neighborhood")
    for name in HOODS:
        r = g.loc[name]
        out[name] = (int(r["sales_used"]), round(r["Median ratio"], 3),
                     round(r["COD"], 2), round(r["PRD"], 3))
    st = ratio.ratio_stats(frame)
    v = {t["metric"]: t["value"] for t in st["tests"]}
    out["COUNTYWIDE"] = (len(frame), round(v["Median ratio"], 3),
                         round(v["COD"], 2), round(v["PRD"], 3))
    return out


def parse(path: Path) -> dict[str, tuple[int, float, float, float]]:
    """Pull `| Name | n | median | cod | prd |` rows out of a markdown table."""
    found = {}
    # Cells carry bold and ✕/✓ verdict marks: `**✕ 22.08**`.
    num = r"\*{0,2}\s*[✕✓]?\s*([\d.]+)\s*\*{0,2}"
    row = re.compile(
        rf"^\|\s*\*{{0,2}}([A-Za-z ]+?)\*{{0,2}}\s*\|\s*{num}\s*\|\s*{num}"
        rf"\s*\|\s*{num}\s*\|\s*{num}\s*\|", re.M)
    for m in row.finditer(path.read_text(encoding="utf-8")):
        name = m.group(1).strip()
        if name in HOODS or name == "COUNTYWIDE":
            found.setdefault(name, (int(float(m.group(2))), float(m.group(3)),
                                    float(m.group(4)), float(m.group(5))))
    return found


def compare(label: str, path: Path, residential: bool) -> bool:
    print(f"\n=== {label} ({path.name}) ===")
    want = computed(residential)
    got = parse(path)
    if not got:
        print("  [XX ] no table rows parsed — has the format changed?")
        return False
    ok = True
    for name in list(HOODS) + ["COUNTYWIDE"]:
        if name not in got:
            print(f"  [-- ] {name:19s} not in this table")
            continue
        bad = [f"{f}: {g}≠{w}" for f, g, w in
               zip(("n", "median", "COD", "PRD"), got[name], want[name]) if g != w]
        ok &= not bad
        print(f"  [{'OK ' if not bad else 'XX '}] {name:19s} "
              + (", ".join(bad) if bad else "matches"))
    return ok


def check_notebook_sections() -> bool:
    """Every notebook is numbered BY its talk section. Do the docs agree?

    Since the 2026-10-04 reorder the filename prefix *is* the section number,
    which makes this mechanically checkable — and it needs checking, because
    this exact bug class has shipped twice. A renumbering regex looking for
    `§N` cannot see a bare `| 7 |` in a table cell, so DEMOS.md kept a whole
    notebook table numbered *and sorted* by the old running order: five of
    eight rows wrong, every filename beside them right.
    """
    print("\n── notebook filename vs the section each doc claims ──")
    pattern = re.compile(r"\|\s*\**`(\d\d)_[^`]*`\**\s*\|\s*\**(\d+)\**\s*\|")
    ok, seen = True, 0
    for doc in ("DEMOS.md", "README.md"):
        text = (REPO / doc).read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            seen += 1
            good = int(m.group(1)) == int(m.group(2))
            ok &= good
            print(f"  [{'OK ' if good else 'XX '}] {doc:10s} "
                  f"{m.group(1)}_… claims section {m.group(2)}")
    if not seen:
        print("  [-- ] no notebook rows found — has the table format changed?")
    return ok


ok = compare("residential only", REPO / "data/README.md", True)
ok &= compare("all classes", REPO / "DEMOS.md", False)
ok &= check_notebook_sections()

print("\nRESULT:", "ALL FIGURES MATCH" if ok else "MISMATCHES ABOVE")
sys.exit(0 if ok else 1)
