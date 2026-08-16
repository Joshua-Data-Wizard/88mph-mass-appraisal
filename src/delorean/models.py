"""
models.py — which language model to use, and why.

Every notebook in this repo picks its model from this one file. If you want to
try a different model, change it here (or pass a different key) and everything
downstream follows.

Why a registry instead of hardcoding a model name in each notebook: part of the
point of this repository is that you are NOT locked into one vendor's model.
Swapping the engine should be a one-line change, and you should be able to see
what you are trading away when you do it.

    from delorean import models

    models.compare()                  # print the comparison table
    spec = models.get("qwen")         # -> ModelSpec
    spec = models.default_local()     # the workhorse for agent demos
    spec = models.tiny()              # the little one used in Section 2
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Everything we know about one model.

    `measured_*` fields are real numbers from an RTX 3080 in this repo's
    scripts/check_setup.py — not marketing claims. Where we have not measured a
    model ourselves, the field is None and says so.
    """

    key: str                      # short name you type, e.g. "qwen"
    hf_id: str                    # the exact HuggingFace repository id
    label: str                    # human-friendly name
    params: str                   # parameter count, e.g. "4B"
    license: str                  # what you are allowed to do with it
    role: str                     # "tiny" | "workhorse" | "alternate"
    quantize: bool                # does it need 4-bit to fit comfortably in 8 GB?
    good_at: tuple[str, ...]
    watch_out_for: tuple[str, ...]
    measured_vram_gb: float | None = None
    measured_speed: str | None = None

    @property
    def precision(self) -> str:
        return "4-bit NF4" if self.quantize else "fp16"


# ── The registry ─────────────────────────────────────────────────────────────
#
# All four are ungated: you can download them without accepting a license
# agreement first. That matters for a repository people take back to work.

REGISTRY: dict[str, ModelSpec] = {
    "tiny": ModelSpec(
        key="tiny",
        hf_id="Qwen/Qwen3.5-0.8B",
        label="Qwen3.5 0.8B",
        params="0.8B",
        license="Apache 2.0",
        role="tiny",
        quantize=False,
        good_at=(
            "Starting up in about three seconds",
            "Running on a laptop with no GPU at all",
            "Showing how generation works, one token at a time",
        ),
        watch_out_for=(
            "Not for agent work — it is here to be understood, not to reason",
            "Will confidently get facts wrong",
        ),
        measured_vram_gb=1.44,
        measured_speed="~14 tokens/sec",
    ),
    "qwen": ModelSpec(
        key="qwen",
        hf_id="Qwen/Qwen3.5-4B",
        label="Qwen3.5 4B",
        params="4B",
        license="Apache 2.0",
        role="workhorse",
        quantize=True,
        good_at=(
            "Tool calling — ranked 1st of 13 local models in a 2026 evaluation",
            "Long documents (262,144 token context)",
            "Punching far above its size; it beat models 5-7x larger",
        ),
        watch_out_for=(
            "Eager to call several tools at once when one at a time would do",
            "Thinking mode is ON by default and adds a pause before answering",
        ),
        measured_vram_gb=2.98,
        measured_speed="~14 tokens/sec",
    ),
    "smol": ModelSpec(
        key="smol",
        hf_id="HuggingFaceTB/SmolLM3-3B",
        label="SmolLM3 3B",
        params="3B",
        license="Apache 2.0",
        role="alternate",
        quantize=True,
        good_at=(
            "The smallest model here that can still do real agent work",
            "The best 'look how little this needs' story for Section 9",
        ),
        watch_out_for=(
            "Did not place in the top tier for tool calling in 2026 testing",
            "Not measured in this repo yet — try it and see",
        ),
    ),
    "nemotron": ModelSpec(
        key="nemotron",
        hf_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        label="NVIDIA Nemotron 3 Nano 4B",
        params="4B",
        license="NVIDIA Open Model License (not Apache — read it)",
        role="alternate",
        quantize=True,
        good_at=(
            "Working through steps in order, one after another",
            "Scored 95% on the same tool-calling evaluation",
        ),
        watch_out_for=(
            "Struggles when asked to call the same tool several times at once",
            "Its license is NVIDIA's own, not Apache 2.0",
            "Not measured in this repo yet — try it and see",
        ),
    ),
}


# ── Lookup helpers ───────────────────────────────────────────────────────────

def get(key: str) -> ModelSpec:
    """Look up a model by short key ('qwen') or full HuggingFace id."""
    if key in REGISTRY:
        return REGISTRY[key]
    for spec in REGISTRY.values():
        if spec.hf_id.lower() == key.lower():
            return spec
    known = ", ".join(REGISTRY)
    raise KeyError(f"Unknown model {key!r}. Known keys: {known}")


def tiny() -> ModelSpec:
    """The little model Section 2 uses to show generation happening."""
    return REGISTRY["tiny"]


def default_local() -> ModelSpec:
    """The workhorse behind the agent demos, Sections 6 through 11."""
    return REGISTRY["qwen"]


# ── Comparison table ─────────────────────────────────────────────────────────

def compare() -> None:
    """Print the side-by-side comparison. Use this to pick a model for YOUR office."""
    for spec in REGISTRY.values():
        vram = (
            f"{spec.measured_vram_gb} GB measured"
            if spec.measured_vram_gb
            else "not measured here"
        )
        print(f"\n{spec.label}  ({spec.params}, {spec.precision})")
        print(f"  {spec.hf_id}")
        print(f"  License : {spec.license}")
        print(f"  VRAM    : {vram}" + (f"   Speed: {spec.measured_speed}" if spec.measured_speed else ""))
        print("  Good at :")
        for line in spec.good_at:
            print(f"      + {line}")
        print("  Watch   :")
        for line in spec.watch_out_for:
            print(f"      - {line}")
    print()
