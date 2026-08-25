"""Run Section 7's toolbelt against every local model in the registry.

    uv run python scripts/compare_models.py qwen smol
    uv run python scripts/compare_models.py            # all of them

One model at a time, unloaded in between -- three 4-bit models resident is
about 8.5 GB and the team floor is 8 GB.

Answers the questions we cannot answer from a third-party leaderboard:
  * does our dual-shape tool-call parser actually work on this model?
  * can it write SQL that runs?
  * does it reach the right tool, and leave the tools alone when it should?
  * how fast, and how much VRAM?
"""
import gc
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

from delorean import agent, llm, models, prompts, tools  # noqa: E402

TOOLBELT = ["run_sql", "ratio_study", "map_parcels"]

CASES = [
    ("recall", "ratio_study", "How did the county do on the ratio study this year?"),
    ("recall", "ratio_study", "Run the assessment ratio study broken down by neighborhood."),
    ("recall", "map_parcels", "Show me the sales in Riverside Heights on a map."),
    ("recall", "run_sql", "What are the five highest value parcels in the county?"),
    ("recall", "run_sql", "Who owns parcel 04-127-3891?"),
    ("recall", "ratio_study", "Run a ratio study for homes built after 1980."),
    ("restraint", None, "What is the difference between market value and assessed value?"),
    ("restraint", None, "Good morning!"),
    ("restraint", None, "What does COD stand for in a ratio study?"),
    ("restraint", None, "Why would an office use a median instead of an average?"),
]


def vram():
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 2 ** 30


def run(key):
    import torch

    spec = models.REGISTRY[key]
    print("\n" + "=" * 78)
    print(f"  {spec.label}   ({spec.params}, {spec.license})")
    print(f"  {spec.hf_id}")
    print("=" * 78)

    llm.unload()                 # nothing from a previous model on the card
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, tok = llm.load_local(key, verbose=False)
    load_s = time.time() - t0
    weights_gb = vram()

    respond = agent.local_responder(model, tok, max_new_tokens=420)
    graph = agent.build_agent(respond, TOOLBELT, max_steps=5)

    recall_ok = recall_n = restraint_ok = restraint_n = 0
    sql_written = sql_ran = 0
    first_raw = ""
    total_s = 0.0

    for kind, want, question in CASES:
        t = time.time()
        try:
            state = agent.ask(graph, question, system=prompts.REVIEW_ASSISTANT)
        except Exception as exc:                      # noqa: BLE001
            print(f"  [ERR] {question[:48]:<50} {type(exc).__name__}: {exc}")
            continue
        elapsed = time.time() - t
        total_s += elapsed

        called = [s["tool"] for s in state["trace"] if s["step"] == "act"]
        for step in state["trace"]:
            if step["step"] == "act" and step["args"].get("sql"):
                sql_written += 1
                if step["result"].get("ok"):
                    sql_ran += 1
        if not first_raw:
            raw = getattr(respond, "last_raw", "") or ""
            if tools.attempted_tool_call(raw):
                first_raw = raw

        if kind == "recall":
            recall_n += 1
            good = want in called
            recall_ok += good
        else:
            restraint_n += 1
            good = not called
            restraint_ok += good

        print(f"  [{'OK ' if good else 'XX '}] {kind:<9} {elapsed:5.1f}s  "
              f"{question[:44]:<46} -> {called or 'no tool'}")

    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0

    print(f"\n  load {load_s:.0f}s · weights {weights_gb:.2f} GB · peak {peak:.2f} GB")
    print(f"  RECALL    {recall_ok}/{recall_n}")
    print(f"  RESTRAINT {restraint_ok}/{restraint_n}")
    print(f"  SQL       {sql_ran}/{sql_written} queries ran without error")
    print(f"  average   {total_s / max(len(CASES), 1):.1f}s per question")
    if first_raw:
        print("\n  the first tool call it wrote, verbatim:")
        for line in first_raw.strip().splitlines()[:8]:
            print("     " + line[:100])

    result = {
        "key": key, "label": spec.label, "params": spec.params,
        "recall": f"{recall_ok}/{recall_n}",
        "restraint": f"{restraint_ok}/{restraint_n}",
        "sql": f"{sql_ran}/{sql_written}",
        "weights_gb": round(weights_gb, 2), "peak_gb": round(peak, 2),
        "avg_s": round(total_s / max(len(CASES), 1), 1),
        "load_s": round(load_s),
    }

    # Give the card back before the next one loads.
    #
    # `del` on these names is not enough on its own -- llm.load_local() keeps
    # every model it has loaded in a module-level cache, so the weights stay
    # resident and the NEXT model's "weights" reading includes this one. That
    # bug reported Gemma at 13.75 GB when it uses 6.30.
    del graph, respond, model, tok
    llm.unload()
    gc.collect()
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    keys = sys.argv[1:] or ["qwen", "smol", "nemotron"]
    rows = []
    for k in keys:
        try:
            rows.append(run(k))
        except Exception as exc:                      # noqa: BLE001
            print(f"\n  !! {k} FAILED TO RUN: {type(exc).__name__}: {exc}")
            rows.append({"key": k, "label": k, "params": "?", "recall": "—",
                         "restraint": "—", "sql": "—", "weights_gb": 0,
                         "peak_gb": 0, "avg_s": 0, "load_s": 0})

    print("\n\n" + "=" * 78)
    print("  SECTION 7 TOOLBELT — every local model, same ten questions")
    print("=" * 78)
    print(f"  {'Model':<26}{'Recall':>8}{'Restr':>7}{'SQL ok':>8}"
          f"{'Weights':>9}{'Peak':>7}{'s/q':>6}")
    print("  " + "-" * 74)
    for r in rows:
        print(f"  {r['label']:<26}{r['recall']:>8}{r['restraint']:>7}{r['sql']:>8}"
              f"{r['weights_gb']:>8.2f}G{r['peak_gb']:>6.2f}G{r['avg_s']:>6.1f}")
