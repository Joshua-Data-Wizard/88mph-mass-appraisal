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

    `measured_*` fields are real numbers from an RTX 3080, not marketing
    claims. Where we have not measured a model ourselves, the field is None.

    ⚠️ `measured_vram_gb` is **peak VRAM during real agent work** — the
    Section 9 toolbelt, via scripts/compare_models.py — and not the weight
    footprint. Weights are the smaller, friendlier number and the wrong one to
    plan with: Qwen3.5-4B holds 2.90 GB of weights and peaks at 4.00 GB once
    the KV cache fills. Quote the peak, or somebody runs out of memory on
    stage. (`scripts/check_setup.py` reports a lower figure because its probe
    is a few tokens long; that number is a smoke test, not a budget.)
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

    #: How much this model costs to download, in GB. Measured from a finished
    #: cache, not read off a model card.
    #:
    #: Normally this is also what it occupies on disk. HuggingFace stores each
    #: file once in `blobs/` and links to it from `snapshots/`.
    #:
    #: ⚠️ On a filesystem that cannot make symlinks — Windows without
    #: Developer Mode, some network shares — it copies instead and the model
    #: costs TWICE this. `scripts/fetch_models.py` probes for that before it
    #: downloads rather than after you have run out of disk.
    #:
    #: (An earlier note here claimed the doubling was happening on the dev box.
    #: It was not: the measurement was following its own symlinks. See
    #: `_bytes_under`.)
    download_gb: float | None = None

    #: Cap the context window at load time, or None to use the model's own.
    #:
    #: A sensible guard for a model advertising a 262,144-token context when
    #: the longest prompt in this repository is about 2,000 tokens.
    #:
    #: ⚠️ It is NOT what makes Nemotron usable, and the story of finding that
    #: out is worth keeping. Nemotron loads in 3 GB and then asks for 21 GB on
    #: the first token, and 262144 x 1024 x 42 layers x 2 happens to come to
    #: almost exactly 21 GB -- so a preallocated key-value cache looked like an
    #: open-and-shut diagnosis. Capping the context to 8,192 changed nothing:
    #: same failure, same byte count, to the byte.
    #:
    #: The real cause is architectural. Nemotron is a hybrid Mamba model, and
    #: without the mamba-ssm / causal-conv1d / triton CUDA kernels it falls
    #: back to a PyTorch scan that costs about **12 MB per prompt token**.
    #: Measured: 256 tokens -> 6.3 GB, 1,024 -> 15.6 GB, 1,280 -> out of
    #: memory. No context setting fixes that; only the kernels do, and they do
    #: not install cleanly on Windows.
    #:
    #: The lesson: a number that matches your theory is not evidence your
    #: theory is right. Change it and see whether the failure moves.
    max_context: int | None = None

    #: Will this model actually run the demos on the machine this repository
    #: targets — Windows, 8 GB of VRAM? False hides it from every app's model
    #: picker. See `selectable()`.
    #:
    #: A blocked model is NOT deleted from the registry, for two reasons.
    #:
    #: The first is that *why* a model will not run is the single most useful
    #: page in Section 3, and you cannot teach it from an empty space. The
    #: second is honesty: "does not run here" is not "does not run". Nemotron
    #: needs the mamba-ssm / causal-conv1d / triton CUDA kernels, and those
    #: build cleanly on Linux — it is a platform limit, not a bad model. An
    #: office running Linux should still be able to try it, and can:
    #: `llm.load_local("nemotron")` is untouched by this flag.
    #:
    #: What the flag prevents is narrower and specific — a dropdown is a
    #: promise, and offering a model that runs out of memory halfway through a
    #: live demo is a promise we cannot keep.
    runs_here: bool = True

    #: One plain sentence for the reader when `runs_here` is False.
    blocked_reason: str = ""

    @property
    def precision(self) -> str:
        return "4-bit NF4" if self.quantize else "fp16"


# ── The registry ─────────────────────────────────────────────────────────────
#
# Every one of these is ungated: you can download it without accepting a
# licence agreement first. That matters for a repository people take back
# to work, and it is why Gemma 3 is not here — it required a click-through.
# Gemma 4 does not, which is what made a familiar-brand option possible.

REGISTRY: dict[str, ModelSpec] = {
    "tiny": ModelSpec(
        key="tiny",
        hf_id="Qwen/Qwen3.5-0.8B",
        download_gb=1.65,
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
        download_gb=8.7,
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
        measured_vram_gb=4.00,
        measured_speed="8.7s per agent question (Section 9 toolbelt)",
    ),
    "smol": ModelSpec(
        key="smol",
        hf_id="HuggingFaceTB/SmolLM3-3B",
        download_gb=5.74,
        label="SmolLM3 3B",
        params="3B",
        license="Apache 2.0",
        role="alternate",
        quantize=True,
        good_at=(
            "The smallest model here that can still do real agent work",
            "Ran Section 9 perfectly: 6/6 tools, 4/4 restraint, 6/6 SQL",
            "The fastest of the three at 6.7s a question, on 1.85 GB",
            "The best 'look how little this needs' story for Section 3",
        ),
        watch_out_for=(
            "Writes its tool calls as BARE JSON with no <tool_call> tags, "
            "even though its own chat template asks for them — see "
            "tools/base.py, Shape C",
            "Did not place in the top tier for tool calling in 2026 testing, "
            "which our own measurements do not support",
        ),
        measured_vram_gb=2.77,
        measured_speed="6.7s per agent question (Section 9 toolbelt)",
    ),
    "phi": ModelSpec(
        key="phi",
        hf_id="microsoft/Phi-4-mini-instruct",
        download_gb=7.17,
        label="Microsoft Phi-4 mini",
        params="3.8B",
        license="MIT",
        role="alternate",
        quantize=True,
        good_at=(
            "A name every county IT department already trusts",
            "MIT licence — the most permissive here, and ungated",
            "The smallest download of the agent-capable models (7.2 GB)",
            "Section 9 toolbelt: recall 5/6, restraint 4/4, 7.0s a question, "
            "2.70 GB of weights and a 4.15 GB peak",
        ),
        watch_out_for=(
            "Its chat template IGNORES the standard tools= argument and looks "
            "for the tool list on the SYSTEM message instead. Passing tools "
            "the usual way renders a prompt with no tools in it and raises "
            "nothing — see agent.render_prompt(), which detects this.",
            "Wants its tool list UNWRAPPED. Handed the usual OpenAI nesting "
            "it echoes the tool definition back and calls nothing — see "
            "agent._unwrap_schemas().",
            "The over-eager parallel calling the registry once attributed to "
            "Qwen3.5 and never reproduced. This is where it lives: asked one "
            "question it called four tools, and 3 of its 10 SQL queries "
            "failed — the worst SQL record of the four.",
            "Released February 2025, the oldest model in this registry",
        ),
        measured_vram_gb=4.15,
        measured_speed="7.0s per agent question (Section 9 toolbelt)",
    ),
    "gemma": ModelSpec(
        key="gemma",
        hf_id="google/gemma-4-E2B-it",
        download_gb=9.57,
        label="Google Gemma 4 E2B",
        params="2.3B",
        license="Apache 2.0 (check the linked Gemma terms)",
        role="alternate",
        quantize=True,
        good_at=(
            "Another name nobody has to be talked into",
            "Gemma 4 is ungated — Gemma 3 required accepting a licence first",
            "Section 9 toolbelt: recall 5/6, restraint 4/4, and 5/5 SQL "
            "queries that ran — the cleanest SQL record after Qwen",
            "The fastest of the four at 5.7s a question",
            "The most downloaded model here by a wide margin",
        ),
        watch_out_for=(
            "⚠️ 6.29 GB on the card, NOT the ~2.8 GB '2.3B effective' implies. "
            "The Per-Layer Embedding tables stay resident and are not "
            "quantized. It fits an 8 GB machine with about 1.5 GB to spare, "
            "so close everything else first. This is the second model here "
            "whose advertised size did not predict what it needs — see "
            "Nemotron for the first.",
            "Writes <|tool_response> straight after its call and, unchecked, "
            "invents a plausible result for a tool that has not run. "
            "strip_tool_calls() removes it; do not skip that step.",
            "Its tool-call delimiters are SPECIAL TOKENS, so decoding with "
            "skip_special_tokens=True silently deletes them. Measured 0/6 "
            "that way while emitting correct calls the whole time.",
            "Multimodal (text, image, audio in), so its config nests under "
            "text_config — use llm.vocab_size() rather than cfg.vocab_size",
            "HuggingFace declares Apache 2.0 and the repo is ungated, but the "
            "card also links Google's own Gemma terms. Worth ten minutes of "
            "your county attorney's time before it goes into production.",
        ),
        measured_vram_gb=6.55,
        measured_speed="5.7s per agent question — the fastest of the four",
    ),
    "nemotron": ModelSpec(
        key="nemotron",
        hf_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        download_gb=7.42,
        label="NVIDIA Nemotron 3 Nano 4B",
        params="4B",
        license="NVIDIA Open Model License (not Apache — read it)",
        role="alternate",
        quantize=True,
        good_at=(
            "Working through steps in order, one after another",
            "Scored 95% on the same tool-calling evaluation",
            "Fine on short prompts — a 221-token question answered normally",
        ),
        watch_out_for=(
            "⚠️ UNUSABLE FOR AGENT WORK ON WINDOWS. It is a hybrid "
            "Mamba/attention model, and its fast path needs the mamba-ssm, "
            "causal-conv1d and triton CUDA kernels, none of which install "
            "cleanly here. The PyTorch fallback costs about 12 MB of VRAM PER "
            "PROMPT TOKEN — roughly 100x a normal transformer.",
            "Measured on a 10 GB card: 1,024-token prompt peaks at 15.6 GB "
            "and 1,280 tokens runs out of memory. Section 9's prompt is 1,600 "
            "tokens, so it never gets started.",
            "Its license is NVIDIA's own, not Apache 2.0",
        ),
        measured_vram_gb=15.57,
        measured_speed="OOM past ~1,100 prompt tokens without Mamba kernels",
        runs_here=False,
        blocked_reason=(
            "Needs the mamba-ssm / causal-conv1d / triton CUDA kernels, which "
            "do not build cleanly on Windows. Without them it costs about "
            "12 MB of VRAM per prompt token and runs out of memory partway "
            "through Section 9's prompt. Expected to be fine on Linux with "
            "those kernels installed — this is a platform limit, not a "
            "verdict on the model."
        ),
        # See the note on ModelSpec.max_context. Without this it loads in 3 GB
        # and then tries to allocate 21 GB for the first token.
        max_context=8192,
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
    """The workhorse behind the agent demos, Sections 7 through 11."""
    return REGISTRY["qwen"]


# ── Is it already on this machine? ───────────────────────────────────────────
#
# Worth knowing before you pick one, because the alternates are a 5-7 GB
# download and nothing warns you. Choosing an uncached model from a dropdown
# in front of an audience starts that download immediately, with no progress
# bar anywhere the room can see. This is what lets the app say so first.

def _folder_for(key: str):
    """Where Hugging Face keeps this model on disk."""
    import os
    from pathlib import Path

    spec = get(key)
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return root / "hub" / ("models--" + spec.hf_id.replace("/", "--"))


def _bytes_under(folder) -> int:
    """Bytes actually occupied under a folder.

    ⚠️ Symlinks are skipped, and that is the whole point of this function.
    `os.path.getsize()` FOLLOWS a symlink and returns the target's size, so
    walking a HuggingFace cache naively counts every file twice — once as the
    blob and once as the snapshot link pointing at it.

    Measured wrong that way, SmolLM3 came out at 11.47 GB and was written up
    as "Windows copied it instead of linking". It is 5.74 GB and it is
    properly linked. The double was our own arithmetic.
    """
    import os

    total = 0
    for base, _dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(base, name)
            try:
                if os.path.islink(path):
                    continue        # the blob it points at is counted once
                total += os.path.getsize(path)
            except OSError:
                pass                # mid-download, or a broken link
    return total


def cache_size_gb(key: str) -> float:
    """How big this model is on disk, in GB. 0.0 if it is not here.

    Reports the LARGER of `blobs/` and `snapshots/` rather than their sum,
    which is correct in both of the layouts a cache can be in:

      * **symlinked** (the normal case) — the bytes live in `blobs/` and
        `snapshots/` holds links. `_bytes_under` skips links, so snapshots
        measures ~0 and blobs carries the size.
      * **copied** — Windows without Developer Mode cannot make the link, so
        HuggingFace writes a real file in both places. Either directory gives
        the true size; adding them would double it.

    Some older entries have an empty `blobs/` entirely. Taking the maximum is
    right in every one of these cases; summing is right in none of them.
    """
    folder = _folder_for(key)
    if not folder.exists():
        return 0.0
    biggest = max(_bytes_under(folder / "blobs"),
                  _bytes_under(folder / "snapshots"))
    return round(biggest / 1024 ** 3, 2)


def is_cached(key: str) -> bool:
    """True if this model is fully downloaded and ready to load.

    Size alone is not enough, and getting this wrong defeats the point. A
    download in progress leaves a directory that grows steadily, so a
    size-only check reports "ready" while the model is still arriving --
    measured, it said ready at 3.16 GB of a download that was not finished.

    Hugging Face parks each blob it is still fetching under a `.incomplete`
    name, so the presence of one of those is a definite no.
    """
    folder = _folder_for(key)
    if not folder.exists():
        return False
    if any(folder.rglob("*.incomplete")):
        return False
    return cache_size_gb(key) >= 1.0


# ── What an app should offer ─────────────────────────────────────────────────

def selectable() -> list[str]:
    """The model keys an app may show in a picker.

    Deliberately not `list(REGISTRY)`. Everything here has been seen to run on
    this repository's baseline machine; anything marked `runs_here=False` is
    filtered out so a presenter cannot pick it by accident thirty seconds
    before it matters.

    Blocked models stay in the registry and stay loadable by key, so
    `compare()`, `scripts/compare_models.py` and Section 3's notebook all still
    tell the whole story.
    """
    return [key for key, spec in REGISTRY.items() if spec.runs_here]


def blocked() -> list[ModelSpec]:
    """Models that are documented here but will not run on this machine.

    Apps use this to say *why* a model is missing rather than quietly showing
    a shorter list — the absence is itself worth a sentence.
    """
    return [spec for spec in REGISTRY.values() if not spec.runs_here]


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
