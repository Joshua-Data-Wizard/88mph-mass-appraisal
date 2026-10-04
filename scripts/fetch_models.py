"""
fetch_models.py — download models ahead of time, and prove they run.

    uv run python scripts/fetch_models.py                # tiny + qwen
    uv run python scripts/fetch_models.py smol gemma     # named models
    uv run python scripts/fetch_models.py --all          # every usable model
    uv run python scripts/fetch_models.py --list         # what exists, what you have
    uv run python scripts/fetch_models.py --download-only # skip the GPU proof

WHY THIS EXISTS
---------------
Nothing in this repository downloads a model until you run a notebook or start
an app. The first time you pick a model that is not on your machine, several
gigabytes begin arriving **with no progress shown anywhere in the Streamlit
interface** — so a perfectly healthy app looks frozen for ten minutes, and the
natural conclusion is that it is broken.

Run this first, in a terminal, where HuggingFace's own progress bar is visible.

It does a second thing the apps cannot: after the download it **loads the model
onto your GPU and generates a few tokens**, so you learn that it actually runs
on *your* hardware rather than merely that the files arrived. A model that
downloads fine and then runs out of memory is a much worse surprise on stage
than one that never downloaded.

⚠️ `scripts/check_setup.py` already fetches `tiny` and `qwen` as a side effect
of its stages 3 and 4 — those two are what the live demos need. This script is
for the alternates (`smol`, `phi`, `gemma`), for anyone who ran check_setup
with `--quick`, and for proving a specific model before you present.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import textwrap
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# The default Windows console codepage cannot encode this file's output.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load .env before anything touches the HuggingFace libraries: without HF_TOKEN
# downloads run unauthenticated, which works for these public models but is
# rate-limited and slower.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:                                  # pragma: no cover
    pass

from delorean import models                          # noqa: E402

#: What the live demos actually need. §2 uses the little one; everything from
#: §7 onward uses the 4B.
DEFAULT_KEYS = ["tiny", "qwen"]

BAR = "=" * 78


# ─────────────────────────────────────────────────────────────────────────
#  Output
# ─────────────────────────────────────────────────────────────────────────

def head(title: str) -> None:
    print(f"\n{BAR}\n  {title}\n{BAR}")


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠️  {msg}")


def fail(msg: str) -> None:
    print(f"  ❌ {msg}")


def mins(seconds: float) -> str:
    return (f"{seconds:.0f}s" if seconds < 90
            else f"{seconds / 60:.0f}m {seconds % 60:.0f}s")


# ─────────────────────────────────────────────────────────────────────────
#  Disk
# ─────────────────────────────────────────────────────────────────────────

def symlinks_work(where: Path) -> bool:
    """Can this filesystem make a symlink? Probed, not assumed.

    It decides whether a download costs its stated size or twice it.
    HuggingFace stores each file once under `blobs/` and points at it from
    `snapshots/`. Where symlinks are unavailable — Windows without Developer
    Mode, and some network shares — it copies instead, and the model occupies
    both directories at full size.

    On this dev box symlinks DO work and the cache is not doubled — the six
    models occupy the 40.25 GB they report. The warning below fires only where
    the probe actually fails, which is what makes it worth probing rather than
    assuming either way.
    """
    where.mkdir(parents=True, exist_ok=True)
    target = where / ".delorean_symlink_probe"
    link = where / ".delorean_symlink_probe_link"
    try:
        target.touch()
        link.symlink_to(target)
        return True
    except (OSError, NotImplementedError, AttributeError):
        return False
    finally:
        for path in (link, target):
            try:
                path.unlink()
            except OSError:
                pass


def hf_home() -> Path:
    import os

    return Path(os.environ.get("HF_HOME",
                               Path.home() / ".cache" / "huggingface")) / "hub"


# ─────────────────────────────────────────────────────────────────────────
#  The two things this script does
# ─────────────────────────────────────────────────────────────────────────

def download(spec) -> bool:
    """Fetch one model, with HuggingFace's progress bar left visible."""
    from huggingface_hub import snapshot_download

    print(f"\n  ⬇️  {spec.label} — {spec.download_gb:.2f} GB "
          f"from {spec.hf_id}")
    started = time.time()
    try:
        snapshot_download(spec.hf_id)
    except OSError as exc:
        fail(f"{spec.label} did not download: {type(exc).__name__}: {exc}")
        # Windows caps a path at 260 characters unless long paths are enabled,
        # and a HuggingFace cache path is already long before your username and
        # a 40-character revision hash are added. It surfaces as a baffling
        # "cannot find the path specified" for a file it is in the middle of
        # writing. Measured: a 270-character path failed, the same download to
        # a short path succeeded in 18 seconds.
        if getattr(exc, "winerror", None) in (3, 206) or "path" in str(exc).lower():
            print()
            warn("On Windows this is usually the 260-character path limit.")
            print(f"     Your cache is at:\n       {hf_home()}")
            print("     Move it somewhere short by setting HF_HOME in .env:"
                  "\n       HF_HOME=C:/hf")
        return False
    except Exception as exc:                          # noqa: BLE001
        fail(f"{spec.label} did not download: {type(exc).__name__}: {exc}")
        return False
    ok(f"{spec.label} downloaded in {mins(time.time() - started)}")
    return True


def prove(key: str) -> bool:
    """Load onto the GPU and generate, so we know it runs HERE.

    The download only proves the files arrived. This proves bitsandbytes, the
    driver, the architecture and the card all agree — which is the part that
    fails, and the part you want failing at your desk.
    """
    import torch

    from delorean import llm

    spec = models.REGISTRY[key]
    llm.unload()                    # never measure on top of another model
    torch.cuda.reset_peak_memory_stats()

    try:
        started = time.time()
        model, tokenizer = llm.load_local(key, verbose=False)
        load_s = time.time() - started

        inputs = tokenizer("An appraiser estimates the fair market",
                           return_tensors="pt").to(model.device)
        started = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        gen_s = time.time() - started
        made = out.shape[1] - inputs["input_ids"].shape[1]

        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        ok(f"{spec.label:<22} runs here — {peak:.2f} GB peak · "
           f"{made / gen_s:.0f} tok/s · loaded in {load_s:.0f}s")
        return True

    except torch.cuda.OutOfMemoryError:
        fail(f"{spec.label} ran OUT OF MEMORY on this card.")
        print(f"     It downloaded fine — it just does not fit here. "
              f"Peak needed is about {spec.measured_vram_gb} GB.")
        return False
    except Exception as exc:                          # noqa: BLE001
        fail(f"{spec.label} would not run: {type(exc).__name__}: {exc}")
        return False
    finally:
        llm.unload()


# ─────────────────────────────────────────────────────────────────────────
#  Listing
# ─────────────────────────────────────────────────────────────────────────

def show_list() -> int:
    head("MODELS IN THE REGISTRY")
    print(f"  {'key':<10}{'model':<26}{'download':>10}{'on disk':>10}  status")
    print("  " + "-" * 74)
    for key, spec in models.REGISTRY.items():
        here = models.is_cached(key)
        size = models.cache_size_gb(key)
        if not spec.runs_here:
            status = "⚠️  will not run on this platform"
        elif here:
            status = "✅ ready"
        else:
            status = "⬇️  not downloaded"
        print(f"  {key:<10}{spec.label:<26}{spec.download_gb:>9.2f}G"
              f"{size:>9.2f}G  {status}")

    missing = [k for k in models.selectable() if not models.is_cached(k)]
    print()
    if missing:
        print(f"  To get the rest:  uv run python scripts/fetch_models.py "
              f"{' '.join(missing)}")
    else:
        ok("Every usable model is on this machine.")
    return 0


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download models ahead of time, and prove they run.")
    parser.add_argument("keys", nargs="*",
                        help=f"registry keys. Default: {' '.join(DEFAULT_KEYS)}")
    parser.add_argument("--all", action="store_true",
                        help="every model that runs on this platform")
    parser.add_argument("--list", action="store_true",
                        help="show what exists and what you already have")
    parser.add_argument("--download-only", action="store_true",
                        help="skip the load-onto-the-GPU proof")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="do not ask before downloading")
    args = parser.parse_args()

    if args.list:
        return show_list()

    if args.all:
        keys = models.selectable()
    elif args.keys:
        keys = args.keys
    else:
        keys = list(DEFAULT_KEYS)

    unknown = [k for k in keys if k not in models.REGISTRY]
    if unknown:
        fail(f"Unknown model(s): {', '.join(unknown)}")
        print(f"     Known keys: {', '.join(models.REGISTRY)}")
        return 1

    head(f"FETCH MODELS — {len(keys)} requested")

    # ── what is already here ────────────────────────────────────────────
    have = [k for k in keys if models.is_cached(k)]
    need = [k for k in keys if k not in have]

    for key in keys:
        spec = models.REGISTRY[key]
        mark = "✅ already here" if key in have else "⬇️  needs downloading"
        print(f"  {spec.label:<26}{key:<10}{spec.download_gb:>7.2f} GB   {mark}")

    # ── budget, before anything starts ──────────────────────────────────
    if need:
        wanted = sum(models.REGISTRY[k].download_gb for k in need)
        copies = not symlinks_work(hf_home())
        on_disk = wanted * 2 if copies else wanted
        free = shutil.disk_usage(hf_home()).free / 2 ** 30

        print(f"\n  to download : {wanted:.2f} GB")
        if copies:
            print(f"  on disk     : {on_disk:.2f} GB   "
                  f"⚠️  twice the download")
            warn("This filesystem cannot make symlinks, so HuggingFace copies "
                 "instead of\n     linking and each model is stored twice. "
                 "Nothing is broken and the\n     models work normally — it "
                 "just costs double the disk.")
            print("     (Optional, Windows: turning on Developer Mode lets it "
                  "symlink and\n     halves this. It is a system setting your "
                  "IT policy may not allow,\n     and everything works fine "
                  "without it.)")
        else:
            print(f"  on disk     : {on_disk:.2f} GB")
        print(f"  free space  : {free:.0f} GB")

        if free < on_disk * 1.1:
            fail(f"Not enough room. Need about {on_disk:.0f} GB, "
                 f"have {free:.0f} GB.")
            return 1

        if not args.yes:
            print()
            try:
                answer = input(f"  Download {len(need)} model(s), "
                               f"{wanted:.2f} GB? [Y/n] ").strip().lower()
            except EOFError:
                answer = "y"
            if answer and not answer.startswith("y"):
                print("  Nothing downloaded.")
                return 0

        for key in need:
            if not download(models.REGISTRY[key]):
                return 1
    else:
        print("\n  Nothing to download.")

    # ── prove they run ──────────────────────────────────────────────────
    if args.download_only:
        print("\n  Skipping the GPU check (--download-only).")
        return 0

    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    if not has_gpu:
        print()
        warn("No GPU found, so nothing was loaded to prove it runs.")
        print("     The models are downloaded and the `_api` notebooks will "
              "work.\n     For the `_local` ones you need a CUDA GPU — "
              "see scripts/check_setup.py.")
        return 0

    print("\n  Loading each one to prove it runs on this machine...\n")
    proved = 0
    for key in keys:
        spec = models.REGISTRY[key]
        if not spec.runs_here:
            warn(f"{spec.label} downloaded, but not loaded onto the GPU.")
            for line in textwrap.wrap(spec.blocked_reason, width=68):
                print(f"     {line}")
            continue
        proved += prove(key)

    runnable = [k for k in keys if models.REGISTRY[k].runs_here]

    if not runnable:
        # Everything asked for is blocked on this platform. The files are
        # here and that is all this script can honestly claim.
        head("Downloaded, but nothing to run")
        print("  Every model you asked for is marked as not running on this\n"
              "  platform, so none was loaded. The files are on disk.")
        return 0

    head(f"{proved} of {len(runnable)} ready to use")
    if proved == len(runnable):
        print("  Start an app:\n"
              "      uv run streamlit run "
              "app/02_Flux_Capacitor_local_Streamlit.py")
    else:
        print("  Some models downloaded but would not run. The message above\n"
              "  says which, and why. The rest are fine to use.")
    return 0 if proved == len(runnable) else 1


if __name__ == "__main__":
    raise SystemExit(main())
