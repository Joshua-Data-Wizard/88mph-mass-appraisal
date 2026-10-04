"""Switching models in an app must RELEASE the model you were using.

    uv run python scripts/test_model_switch.py

Why this test exists
--------------------
Every app's sidebar offers five models. Until 2026-10-03 picking a different
one loaded the new model *without* letting go of the old, so the second switch
put two models on the graphics card and the third could run it out of memory
in the middle of a demo. On the team's 8 GB floor that is a crash on stage.

Two assertions, and the first one is the sharp one
--------------------------------------------------
1. **Go away and come back.** Each app visits two other models and then
   returns to the one it started on. That final reading must match the first
   one. With the bug it did not: `02_03` started at 2.90 GB and came back to
   6.47 GB, because by then it was holding three models at once.
2. **The card holds no more than the model you asked for.** Each reading is
   checked against that model's measured VRAM from the registry. This catches
   a switch that leaves two models resident even if the round trip happens to
   come out even.

What a failure here means
-------------------------
Something is holding a reference to the previous model. There are four known
holders and the fourth is genuinely surprising -- a stale widget callback
keeps the previous script run's whole namespace alive. The full story, the
measurements, and the three wrong diagnoses are in
`Plans/vram_model_switching.md`.

This runs no generations, so the figures are weights only, not peak VRAM.
That is why the registry's peak figure works as a ceiling.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

# The default Windows console codepage cannot encode the app's emoji.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from delorean import llm, models  # noqa: E402

# Every switch is a real model load, and there are three per app.
TIMEOUT = 600

# Headroom on the registry's peak-VRAM ceiling. The registry figure is peak
# during real agent work and this test never generates, so weights-only always
# lands below it -- but only by about 30 MB for the 0.8B model, which is too
# close to call without a little slack.
CEILING_SLACK_GB = 0.25

# Visited in this order, skipping whichever one the app already has loaded.
# Small, local, already cached on any machine that has run the demos.
CANDIDATES = ("qwen", "tiny", "smol")

APPS = [
    "01_Flux_Capacitor_local_Streamlit.py",
    "02_03_Time_Circuits_and_Plutonium_local_Streamlit.py",
    "04_06_Office_Assistant_local_Streamlit.py",
    "05_88mph_Valuation_Review_local_Streamlit.py",
    # §10 last on purpose: it starts three MCP subprocesses, and `Offices`
    # closes them itself at interpreter exit.
    "08_09_Universal_Adapter_local_Streamlit.py",
]


def allocated_gb() -> float:
    """What PyTorch has handed out to live tensors, in GB.

    Not `mem_get_info()`: that reports what the *driver* has given the
    process, which includes blocks PyTorch is caching for reuse. Those blocks
    are not a leak, and counting them would fail this test for the wrong
    reason.
    """
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 2**30


def picker(at):
    """The sidebar's model dropdown, found by label rather than by position."""
    for box in at.selectbox:
        if box.label == "Model":
            return box
    raise AssertionError(
        f"no selectbox labelled 'Model'; have {[b.label for b in at.selectbox]}")


def ceiling(key: str) -> float:
    spec = models.REGISTRY[key]
    return (spec.measured_vram_gb or 99.0) + CEILING_SLACK_GB


def sweep() -> None:
    """Reclaim the card once the AppTest that was using it is unreachable.

    ⚠️ Call this from the CALLER, after `check_app()` has returned. Leaving it
    out produced a thoroughly confusing first result: all five apps "failed"
    with figures that were exactly the previous app's model plus this one's.

    An `AppTest` holds the session state it ran with, which holds the widget
    metadata, which holds the script's callbacks, which hold the script's
    namespace -- and the model in it. **That is the same chain this whole test
    is about**, so a test that keeps its AppTests alive measures its own
    residue and blames the app.

    🩹 And a `del at` inside a helper taking `at` as a parameter does NOT fix
    it -- that only unbinds the parameter while the caller's name still holds
    the object. Letting `check_app()`'s frame go is what releases it. The
    baseline subtraction below stays anyway, as a belt to this brace.
    """
    import gc

    import streamlit as st
    import torch

    # ⚠️ The GLOBAL clear, which CLAUDE.md forbids inside an app -- and the
    # reason it is right here is the reason it is wrong there.
    #
    # `@st.cache_resource` registers its cache in a PROCESS-GLOBAL registry, so
    # a cached model outlives the `AppTest` that created it. Nothing short of
    # this reaches the previous app's entry, because `load_model.clear()` can
    # only be called on *this* app's function object.
    #
    # In an app the same call would drop §10's three MCP servers without
    # calling close(). Here, nothing is running: the sweep happens before each
    # app starts, and §10 is deliberately last so nothing sweeps after it.
    st.cache_resource.clear()
    gc.collect()
    llm.unload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_app(name: str) -> bool:
    print(f"\n=== {name} ===")

    # Clear the card BEFORE this app starts, then record what is left. The
    # baseline should be zero; it is measured anyway so that a failure of the
    # sweep shows up as a number rather than as a mysterious app bug.
    sweep()
    base = allocated_gb()
    at = AppTest.from_file(str(REPO / "app" / name), default_timeout=TIMEOUT)
    at.run()

    if at.exception:
        print(f"  [XX ] app raised on load: {at.exception[0].value}")
        return False

    home = picker(at).value
    first = allocated_gb() - base

    # ⚠️ A skip must never read as a pass. `main()` has already established
    # that there is a GPU, so "the app loaded and nothing appeared on the card"
    # means the measurement is broken, not that the app is fine.
    #
    # 🩹 This branch used to `return True`, and it fired for two of the five
    # apps while the summary line still said ALL PASS. The cause was a stale
    # baseline: PyTorch collects garbage and retries when an allocation is
    # about to fail, which freed a previous app's leftovers mid-load and made
    # `allocated - base` come out NEGATIVE.
    if first < 0.05:
        print(f"  [XX ] nothing measurable loaded (first={first:+.2f} GB, "
              f"base={base:.2f} GB). A negative figure means the baseline went "
              f"stale — the sweep between apps is not working.")
        return False

    if base > 0.30:
        print(f"  [XX ] {base:.2f} GB was already on the card before this app "
              f"started. The sweep between apps is not working, and these "
              f"figures cannot be trusted.")
        return False

    ok = first <= ceiling(home)
    print(f"  [{'OK ' if ok else 'XX '}] loaded {home:6s} -> {first:.2f} GB "
          f"(ceiling {ceiling(home):.2f})")

    for want in [k for k in CANDIDATES if k != home][:2]:
        picker(at).set_value(want).run()
        if at.exception:
            print(f"  [XX ] app raised switching to {want}: "
                  f"{at.exception[0].value}")
            return False
        now = allocated_gb() - base
        within = now <= ceiling(want)
        ok &= within
        print(f"  [{'OK ' if within else 'XX '}] switch {want:6s} -> "
              f"{now:.2f} GB (ceiling {ceiling(want):.2f} — above it means "
              f"the previous model is still resident)")

    # ── The sharp one: go home again ────────────────────────────────────────
    # A guard that releases on the way out but not on the way back would still
    # fill the card during a demo.
    picker(at).set_value(home).run()
    back = allocated_gb() - base
    budget = first * 1.15
    same = back <= budget
    ok &= same
    print(f"  [{'OK ' if same else 'XX '}] back to {home:6s} -> {back:.2f} GB "
          f"(started at {first:.2f}, must be ≤ {budget:.2f})")

    return ok


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("No GPU on this machine — nothing to measure. Skipping.")
        return 0

    print("=" * 70)
    print("  Switching models must release the model you were using")
    print("=" * 70)

    ok = True
    for name in APPS:
        ok &= check_app(name)

    llm.unload()
    print("\n" + "=" * 70)
    print("RESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    if not ok:
        print("A model is being held somewhere. Four known holders, and the "
              "surprising one, are in Plans/vram_model_switching.md.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
