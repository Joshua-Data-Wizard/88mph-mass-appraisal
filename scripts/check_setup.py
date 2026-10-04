"""
check_setup.py — run this first. It answers one question:

    "Will the demos in this repo actually run on my machine?"

Run it with:

    uv run python scripts/check_setup.py            # full check
    uv run python scripts/check_setup.py --quick    # skip the 4B model download

The check runs in four stages, each one proving more than the last:

    1. Environment    Is PyTorch installed and does it see your GPU?
    2. Quantization   Does bitsandbytes work on this GPU + driver combo?
    3. Tiny model     Can we load and run a real model end to end? (~1.6 GB)
    4. Workhorse      Does the 4B model we use in the demos fit in VRAM? (~8 GB)

If stage 4 passes, every local demo in this repo will run on your machine.
If you have no GPU, stages 1-2 will tell you so, and you should use the
"_api" notebook variants instead of the "_local" ones.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Load .env before anything else touches the HuggingFace libraries.
# Without this, HF_TOKEN is never seen and downloads run unauthenticated —
# which works for public models but gets rate-limited and is slower.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover - dotenv is a declared dependency
    pass

# The model the agent demos actually use. See src/delorean/models.py.
WORKHORSE_MODEL = "Qwen/Qwen3.5-4B"

# The little one used in Section 2 to show token-by-token generation.
TINY_MODEL = "Qwen/Qwen3.5-0.8B"

# What we need free on the GPU for the demos, in GB. The team floor is 8 GB.
TARGET_VRAM_GB = 8.0


# ─────────────────────────────────────────────────────────────────────────
#  Small helpers so the output is readable from the back of a room
# ─────────────────────────────────────────────────────────────────────────

def header(stage: int, title: str) -> None:
    print(f"\n{'=' * 68}")
    print(f"  STAGE {stage}: {title}")
    print("=" * 68)


def ok(msg: str) -> None:
    print(f"  [ OK ]  {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN]  {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL]  {msg}")


def gb(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


# ─────────────────────────────────────────────────────────────────────────
#  Stage 1 — is PyTorch here, and does it see the GPU?
# ─────────────────────────────────────────────────────────────────────────

def stage_1_environment() -> tuple[bool, float]:
    """Returns (has_working_gpu, total_vram_gb)."""
    header(1, "Environment")

    print(f"  Python           {sys.version.split()[0]}")

    token = os.environ.get("HF_TOKEN", "")
    if token and not token.startswith("ENTER"):
        print("  HF_TOKEN         set (authenticated downloads)")
    else:
        warn("HF_TOKEN not set — downloads work but are rate-limited.")
        print("           Copy .env.example to .env and add your token.")

    try:
        import torch
    except ImportError:
        fail("PyTorch is not installed. Run:  uv sync")
        return False, 0.0

    print(f"  PyTorch          {torch.__version__}")
    print(f"  Built for CUDA   {torch.version.cuda or 'CPU-only build'}")

    if not torch.cuda.is_available():
        fail("PyTorch cannot see a CUDA GPU.")
        print()
        print("  This is usually one of three things:")
        print("    1. No NVIDIA GPU in this machine.")
        print("    2. A CPU-only build of PyTorch got installed.")
        print("       Check that pyproject.toml points at the cu130 index.")
        print("    3. Your NVIDIA driver is too old for this PyTorch build.")
        print()
        print("  You can still run every demo using the *_api notebooks,")
        print("  which call a hosted model instead of a local one.")
        return False, 0.0

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_gb = gb(props.total_memory)
    capability = f"{props.major}.{props.minor}"

    ok(f"GPU detected: {name}")
    print(f"  VRAM             {total_gb:.1f} GB")
    print(f"  Compute capability {capability}")

    # bitsandbytes CUDA 13 builds require Turing (7.5) or newer.
    if props.major < 7 or (props.major == 7 and props.minor < 5):
        warn(
            f"Compute capability {capability} is older than Turing (7.5). "
            "4-bit quantization may not be available."
        )

    if total_gb < TARGET_VRAM_GB - 0.5:
        warn(
            f"{total_gb:.1f} GB VRAM is below the {TARGET_VRAM_GB:.0f} GB "
            "these demos target. The 4B model may not fit."
        )

    return True, total_gb


# ─────────────────────────────────────────────────────────────────────────
#  Stage 2 — does bitsandbytes work here?
#
#  This is the piece most likely to fail on Windows, so we test it on a
#  throwaway tensor before spending bandwidth on a model download.
# ─────────────────────────────────────────────────────────────────────────

def stage_2_quantization() -> bool:
    header(2, "4-bit quantization (bitsandbytes)")

    try:
        import bitsandbytes as bnb
    except ImportError:
        fail("bitsandbytes is not installed. Run:  uv sync")
        return False

    print(f"  bitsandbytes     {bnb.__version__}")

    try:
        import torch
        from bitsandbytes.nn import Linear4bit

        # Build a small 4-bit layer, move it to the GPU, push a tensor through.
        # If the CUDA kernels are missing or mismatched, this is where it breaks.
        layer = Linear4bit(64, 64, bias=False, compute_dtype=torch.float16)
        layer = layer.to("cuda")
        result = layer(torch.randn(1, 64, dtype=torch.float16, device="cuda"))
        torch.cuda.synchronize()

        assert result.shape == (1, 64), f"unexpected output shape {result.shape}"
        ok("4-bit CUDA kernels loaded and ran correctly.")
        return True

    except Exception as exc:
        fail(f"bitsandbytes could not run on this GPU: {type(exc).__name__}: {exc}")
        print()
        print("  Without 4-bit quantization the 4B model needs ~8 GB just for")
        print("  weights, leaving no room to actually run it. Options:")
        print("    - Use the smaller SmolLM3-3B model, or")
        print("    - Use the *_api notebooks instead.")
        return False


# ─────────────────────────────────────────────────────────────────────────
#  Stages 3 and 4 — load a real model and generate real text
# ─────────────────────────────────────────────────────────────────────────

def load_and_generate(model_id: str, quantize: bool) -> float | None:
    """Load a model, generate a few tokens, report peak VRAM. Returns GB used."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"  Loading {model_id} ...")
    print("  (first run downloads the weights; later runs read from cache)")
    started = time.time()

    kwargs: dict = {"dtype": torch.float16, "device_map": "cuda:0"}

    if quantize:
        from transformers import BitsAndBytesConfig

        # NF4 is the 4-bit format designed for neural network weights.
        # double_quant squeezes out a little more by quantizing the
        # quantization constants themselves.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    load_seconds = time.time() - started

    weights_gb = gb(torch.cuda.max_memory_allocated())
    ok(f"Loaded in {load_seconds:.0f}s — weights occupy {weights_gb:.2f} GB")

    # Now actually generate. Loading proves the weights fit; generating
    # proves there is room left over for the KV cache, which is what
    # actually bites you on an 8 GB card.
    messages = [{"role": "user", "content": "In one sentence, what is a property appraisal?"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3.5 thinks by default; that is dead air on stage
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    started = time.time()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    gen_seconds = time.time() - started

    new_tokens = output.shape[-1] - inputs["input_ids"].shape[-1]
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    peak_gb = gb(torch.cuda.max_memory_allocated())
    speed = new_tokens / gen_seconds if gen_seconds else 0

    ok(f"Generated {new_tokens} tokens in {gen_seconds:.1f}s ({speed:.1f} tok/s)")
    print(f"\n  Model said: {text.strip()[:200]}\n")
    ok(f"Peak VRAM including KV cache: {peak_gb:.2f} GB")

    del model
    torch.cuda.empty_cache()
    return peak_gb


def stage_3_tiny_model() -> bool:
    header(3, f"Tiny model — {TINY_MODEL}")
    print("  This is the model Section 2 uses to show next-word prediction.")
    print("  It is small enough to run almost anywhere.\n")
    try:
        load_and_generate(TINY_MODEL, quantize=False)
        return True
    except Exception as exc:
        fail(f"{type(exc).__name__}: {exc}")
        return False


def stage_4_workhorse(total_vram_gb: float) -> bool:
    header(4, f"Workhorse model — {WORKHORSE_MODEL} in 4-bit")
    print("  This is the model the agent demos use in Sections 7 through 11.")
    print("  First run downloads roughly 8 GB. Use --quick to skip.\n")
    try:
        peak_gb = load_and_generate(WORKHORSE_MODEL, quantize=True)
        if peak_gb is None:
            return False

        headroom = total_vram_gb - peak_gb
        print(f"  Headroom left:   {headroom:.2f} GB of {total_vram_gb:.1f} GB")

        if headroom < 1.0:
            warn(
                "Under 1 GB of headroom. Longer conversations may run out of "
                "memory. Cap the context length in the notebooks."
            )
        return True

    except Exception as exc:
        fail(f"{type(exc).__name__}: {exc}")
        print()
        print("  If this is an out-of-memory error, try SmolLM3-3B instead:")
        print("    it is smaller and still handles the demos.")
        return False


# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Check that this machine can run the demos.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip stage 4 (avoids the ~8 GB workhorse model download)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 68)
    print("  88mph-mass-appraisal — setup check")
    print("  \"Roads? Where we're going, we don't need roads.\"")
    print("=" * 68)

    has_gpu, total_vram = stage_1_environment()

    if not has_gpu:
        print("\n" + "-" * 68)
        print("  RESULT: No GPU available. Use the *_api notebooks.")
        print("-" * 68 + "\n")
        return 1

    if not stage_2_quantization():
        print("\n" + "-" * 68)
        print("  RESULT: Quantization unavailable. See the notes above.")
        print("-" * 68 + "\n")
        return 1

    if not stage_3_tiny_model():
        print("\n" + "-" * 68)
        print("  RESULT: Could not run the tiny model. See the error above.")
        print("-" * 68 + "\n")
        return 1

    if args.quick:
        print("\n" + "-" * 68)
        print("  RESULT: Stages 1-3 passed. Stage 4 skipped (--quick).")
        print("  Re-run without --quick to confirm the 4B model fits.")
        print("-" * 68)
        # --quick skipped the 4B download, so the agent demos would fetch it
        # mid-click. Better to know now.
        print("\n  ⚠️  The 4B model every agent demo uses is NOT downloaded yet.")
        print("  Get it in the terminal, where you can watch the progress:")
        print("      uv run python scripts/fetch_models.py")
        print()
        return 0

    if not stage_4_workhorse(total_vram):
        print("\n" + "-" * 68)
        print("  RESULT: The 4B model did not run. See the notes above.")
        print("-" * 68 + "\n")
        return 1

    print("\n" + "-" * 68)
    print("  RESULT: All checks passed. Every local demo will run here.")
    print("  Your machine is ready. Let's see some serious speed.")
    print("-" * 68)
    # Stages 3 and 4 have just cached the two models the live demos use, so
    # nothing else is required to start. Say that plainly -- the alternates
    # are a 22 GB opt-in and nobody should discover that from a progress bar
    # inside a Streamlit app.
    print("\n  The two models the demos use are now downloaded.")
    print("  For the alternates (SmolLM3, Phi-4-mini, Gemma 4):")
    print("      uv run python scripts/fetch_models.py --list")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
