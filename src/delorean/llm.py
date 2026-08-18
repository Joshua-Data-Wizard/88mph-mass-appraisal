"""
llm.py — loading a model, and talking to one.

This is the boring plumbing that every notebook needs and nobody needs to read
twice. The interesting parts (how generation actually works, how tool calling
works) stay in the notebooks where you can see them.

    from delorean import llm

    model, tok = llm.load_local("tiny")          # local, on your GPU
    prompt = llm.chat_prompt(tok, [{"role": "user", "content": "Hi"}])
    client = llm.get_api_client()                # hosted, via the Anthropic API
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import models

# Load .env from the repository root so HF_TOKEN and ANTHROPIC_API_KEY are
# available. Doing it here means notebooks never have to think about it.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


# Loading a model takes seconds and gigabytes. Re-running a notebook cell should
# not pay that twice, so we keep what we have already loaded.
_CACHE: dict[str, tuple[Any, Any]] = {}


# ── Local models ─────────────────────────────────────────────────────────────

def load_local(
    which: str = "tiny",
    *,
    quantize: bool | None = None,
    verbose: bool = True,
) -> tuple[Any, Any]:
    """Load a model from the registry onto your GPU.

    Args:
        which:    a registry key ("tiny", "qwen", "smol", "nemotron") or a
                  full HuggingFace id.
        quantize: squeeze the model to 4 bits so it fits in less memory.
                  Defaults to whatever the registry recommends for that model.
        verbose:  print what is happening and how much memory it took.

    Returns:
        (model, tokenizer)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = models.get(which)
    use_4bit = spec.quantize if quantize is None else quantize
    cache_key = f"{spec.hf_id}:{'4bit' if use_4bit else 'fp16'}"

    if cache_key in _CACHE:
        if verbose:
            print(f"Using the copy of {spec.label} already in memory.")
        return _CACHE[cache_key]

    has_cuda = torch.cuda.is_available()
    if verbose:
        where = torch.cuda.get_device_name(0) if has_cuda else "CPU (this will be slow)"
        print(f"Loading {spec.label} onto {where} ...")

    kwargs: dict[str, Any] = {
        "dtype": torch.float16 if has_cuda else torch.float32,
        "device_map": "cuda:0" if has_cuda else "cpu",
    }

    if use_4bit and has_cuda:
        from transformers import BitsAndBytesConfig

        # NF4 is a 4-bit number format designed for neural network weights.
        # Double quantization compresses the compression constants themselves.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    if has_cuda:
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
    model.eval()

    if verbose and has_cuda:
        used = torch.cuda.max_memory_allocated() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Ready. Using {used:.2f} GB of {total:.1f} GB ({spec.precision}).")

    _CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def load_tokenizer(which: str = "tiny") -> Any:
    """Load only the tokenizer, without the model weights.

    A tokenizer is a few megabytes; the weights are gigabytes. The *_api
    notebooks use this to show how text is split into tokens without needing a
    GPU or a large download.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(models.get(which).hf_id)


def chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool = False,
) -> str:
    """Turn a list of messages into the exact text the model expects.

    Every chat model has its own format for marking who said what. The
    tokenizer knows it, so we ask the tokenizer rather than guessing.

    `enable_thinking` is off by default. Qwen3.5 will otherwise reason at
    length before answering, which is useful for hard problems and is dead air
    in front of an audience.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def where_it_runs(model: Any | None = None) -> dict[str, Any]:
    """Report where the model's weights actually live, and what memory is free.

    Worth checking rather than assuming. "The GPU is being used" and "the CPU is
    busy" are both true at once during generation: the weights and the maths sit
    on the graphics card, while Python spends its time launching one small
    kernel per token. High CPU does not mean the model fell back to the CPU --
    but this is how you find out for certain.

    Pass a loaded model to count its parameter tensors per device. Call it with
    nothing for just the hardware summary.
    """
    import torch

    report: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": None,
        "on_gpu": 0,
        "on_cpu": 0,
        "allocated_gb": None,
        "free_gb": None,
        "total_gb": None,
    }

    if report["cuda_available"]:
        report["device_name"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        report["free_gb"] = free / 1024**3
        report["total_gb"] = total / 1024**3
        report["allocated_gb"] = torch.cuda.memory_allocated() / 1024**3

    if model is not None:
        for param in model.parameters():
            if param.device.type == "cuda":
                report["on_gpu"] += 1
            else:
                report["on_cpu"] += 1

    return report


def vocab_size(model: Any) -> int:
    """How many different tokens this model can choose from.

    Qwen3.5 is a multimodal model, so its configuration is nested one level
    deeper than you might expect: `config.text_config.vocab_size` rather than
    `config.vocab_size`. This helper hides that difference.
    """
    cfg = model.config
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "vocab_size"):
        return int(cfg.text_config.vocab_size)
    return int(cfg.vocab_size)


# ── Hosted models ────────────────────────────────────────────────────────────

def get_api_client() -> Any:
    """An Anthropic client, for the *_api notebooks.

    Requires ANTHROPIC_API_KEY in your .env file. If you are running everything
    locally you never need this.
    """
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key or key.startswith("ENTER"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key, or use the *_local "
            "notebooks instead — they need no API key at all."
        )
    return anthropic.Anthropic(api_key=key)


def api_model() -> str:
    """Which hosted model the *_api notebooks should use.

    Set ANTHROPIC_MODEL in .env to change it. Rough guide, per million tokens
    of input / output:
        claude-opus-5      $5 / $25    most capable
        claude-sonnet-5    $3 / $15    balanced
        claude-haiku-4-5   $1 / $5     cheapest

    We default to the cheapest. These demos are about showing how the machinery
    works, not about squeezing out the last few points of answer quality — and
    nobody should get a surprise bill from running a conference demo.
    """
    return os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5").strip() or "claude-haiku-4-5"
