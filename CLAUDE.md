# CLAUDE.md — 88mph-mass-appraisal

Working agreement for this repository. Read this first every session.

---

## 1. What this is

A companion repository for the IAAO Calgary conference session **"Search, Extract & Value: The Lightning Fast Assistant"** (90 minutes, Back to the Future theme). The session outline is the source of truth for scope and lives in [Presentation_Material/](Presentation_Material/) — 12 sections, 9 live demos, exactly 90:00 with zero slack.

Two constraints come from the presenters, not from the outline, and they drive every decision below:

1. **The audience is senior property-appraisal officials, not engineers.** A few technical staff are mixed in. Explanations of LLM / agent / tool / ReAct / RAG / MCP must land for a non-technical director while the code underneath stays real.
2. **This repo is the take-home.** The presenters' pet peeve is talks that demo cool things and never show how. An attendee must be able to hand this repo to their county IT staff and actually get started. Readability beats cleverness, everywhere.

---

## 2. Ground rules

- **Code is teaching material.** Comment density higher than a normal production repo. Name things after what they *are* to an appraiser, not what they are to a framework.
- **Keep the Back to the Future theme in names and docs.** It is a deliberate engagement device, not noise. `delorean`, `flux_capacitor`, `88mph` are intentional. Do not "clean this up."
- **Every demo maps to a numbered outline section.** An attendee who liked §7 must be able to find §7's code in under ten seconds. Never renumber.
- **Assume a county IT laptop.** Windows, no admin rights guaranteed, corporate proxy possible, 8 GB VRAM. If a step needs more than `uv sync`, it needs a documented reason.
- **Every live demo needs an offline fallback.** Conference wifi is the single largest risk to this talk. Notebooks are committed *with outputs* so a dead cell can be narrated from the saved result. See §8.

---

## 3. Hardware & environment baseline

Measured on the primary dev box (2026-08-16):

| | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3080, **10240 MiB** (10 GB) |
| Driver | 595.95, reports **CUDA 13.2** |
| Architecture | Ampere (sm_86) — Turing or newer, so CUDA 13 builds apply |
| uv | 0.12.3 |
| System Python | **none** — uv manages the interpreter |
| git | 2.52.0.windows.1 |

**Design to 8 GB VRAM, not 10.** The team floor is 8 GB. This box's extra 2 GB is headroom for the presenter, never budget for the demo.

`nvidia-smi` reporting "CUDA Version: 13.2" is the *driver's* maximum supported runtime, not an installed toolkit. `nvcc` is not on PATH and does not need to be — PyTorch ships its own CUDA runtime in the wheel.

---

## 4. Stack

### Environment
- **uv + `pyproject.toml`.** Never `pip install` into the venv directly; if a package is needed it goes in `pyproject.toml` and we re-sync.
- **PyTorch from the CUDA 13.0 index**, not PyPI: `https://download.pytorch.org/whl/cu130`. Configure as an explicit uv index (`explicit = true`) so only torch/torchvision/torchaudio resolve there and everything else stays on PyPI. Known upstream issue: `--index-url` can fail on a missing `cuda-bindings` pin; `--extra-index-url` is the workaround.

- **`link-mode = "copy"` is set in `[tool.uv]` and must stay.** uv hardlinks by default; hardlinks fail on OneDrive, Dropbox, Google Drive, and many network shares with `os error 396`. This repo lives in OneDrive, and so will most attendees'. Costs a little disk, buys "works anywhere."

### Pinned facts (verified 2026-08-16 — re-check at venv build)
| Thing | Constraint | Why it matters |
|---|---|---|
| `transformers` | **≥ 5.2** — released 5.15.0 satisfies it | Qwen3.5 architectures are unsupported on 4.x. **Resolved: no `git+` dependency needed.** |
| `bitsandbytes` | **≥ 0.50** | CUDA 13.0 wheels for Windows x86-64 landed in 0.50.0; hardware support is Turing+ (RTX 3080 qualifies) |
| `langgraph` | **≥ 1.0** | Orchestration only — see the architecture rule below |
| Vector store | **ChromaDB** (not FAISS) | Chroma stores documents + metadata + vectors together; FAISS stores vectors only and makes you maintain a parallel docstore |
| GUI | **Streamlit** | Team is already comfortable with it |
| Web search | **Tavily** (`tavily-python`) | §6 demo requires a real live-internet search |

### Architecture rule: LangGraph only, hand-built nodes

**We use `langgraph`. We do not use `langchain` or its provider wrappers.**
Removed on purpose: `langchain`, `langchain-huggingface`, `langchain-anthropic`,
`langchain-tavily`, `langchain-mcp-adapters`. Do not reintroduce them.

Why:
1. **The graph is the lesson.** `create_react_agent()` builds the loop for you,
   which means the audience never sees Reason → Act → Observe happen. Hand-built
   nodes make §5's concept slide and §7's code the same thing.
2. **Every graph prints its own diagram.** `graph.get_graph().draw_mermaid()`
   needs no extra dependency and makes no network call. Verified working.
3. **Robustness across the model registry.** Local models format tool calls
   differently from one another. Owning the parsing means all three registry
   models work, not just whichever one a wrapper happens to support.
4. **Full control for attendees.** Someone taking this home can change one node
   without reverse-engineering an abstraction.

Providers are called directly: `transformers` for local models, the `anthropic`
SDK for the hosted variants, `tavily` for search, `mcp` for §10. Each call site
is roughly ten readable lines.

**Message format is plain dicts** — `{"role": ..., "content": ...}` — matching
what the model APIs actually take. No `langchain_core.messages` wrappers.

**Every notebook that builds a graph must print its mermaid diagram**, and the
`.md`-rendered version belongs in the notebook so GitHub shows the picture.

### Hosted API (for the `_api` notebook variants)
Current model IDs and list pricing per MTok (input / output):

| Model | ID | Price | Use for |
|---|---|---|---|
| Claude Opus 5 | `claude-opus-5` | $5 / $25 | Default in the `_api` notebooks |
| Claude Sonnet 5 | `claude-sonnet-5` | $3 / $15 ($2 / $10 intro thru 2026-08-31) | Cost-conscious attendees |
| Claude Haiku 4.5 | `claude-haiku-4-5` | $1 / $5 | Cheapest path for an office trialing this |

Document all three in the `_api` notebooks so an attendee can pick by budget. Call the API through the **`anthropic` SDK directly** — no LangChain wrapper (see the architecture rule above). `.env.example` defaults to `claude-haiku-4-5` so nobody gets a surprise bill from a conference demo; `llm.api_model()` uses the same fallback.

**`effort` / `output_config` errors on `claude-haiku-4-5`.** It is only supported on newer models. Since Haiku is our default, do not put `output_config={"effort": ...}` in any `_api` notebook — it will 400 for most attendees.

---

## 5. Repo layout

```
88mph-mass-appraisal/
├── notebooks/                   # one pair per demo, numbered to the outline
│   ├── 01_flux_capacitor_local.ipynb      DONE
│   └── 01_flux_capacitor_api.ipynb        DONE
├── app/                         # Streamlit — always .py, never a notebook
│   └── 01_Flux_Capacitor_local_Streamlit.py   DONE
├── src/delorean/                # importable package — the shared engine
│   ├── models.py                # ModelSpec registry: get/tiny/default_local/compare
│   ├── llm.py                   # load_local, load_tokenizer, chat_prompt,
│   │                            #   vocab_size, get_api_client, api_model
│   ├── viz.py                   # *_html() builders + show_*() notebook wrappers
│   ├── tools/                   # (§6+) parcel query, ratio study, mapping
│   └── rag/                     # (§8) index + retrieve
├── scripts/check_setup.py       # 4-stage machine check — run this first
├── Guides/UV_Readme.md          # uv for newcomers
├── mcp_server/                  # §10 — must be .py
├── data/                        # synthetic parcels, sales, office policy PDFs
└── Presentation_Material/       # outline (source of truth for scope)
```

**`llm.vocab_size(model)` exists because Qwen3.5 is multimodal** — its config
nests under `cfg.text_config`, so plain `cfg.vocab_size` raises `AttributeError`.
Use the helper rather than reaching into the config.

**Why hybrid and not pure notebooks:** the parcel / ratio-study / mapping tools are reused across demos 5–9. Duplicated across six notebooks they would drift within a week. §10's MCP server and §11's Streamlit app cannot be notebooks at all.

**Where tools live is a per-demo decision:**
- **§6 Hoverboard — define tools inline in the notebook.** The tool *is* the lesson. An attendee must see a plain Python function become something the model can call.
- **§7 onward — import from `src/delorean/tools/`.** By then the point is the agent, not the plumbing.

---

## 6. Demo map

| § | Title | Demo | Artifact |
|---|---|---|---|
| 1 | Welcome to Hill Valley | — | slides |
| 2 | The Flux Capacitor | 1 | ✅ **DONE** — `01_flux_capacitor_{local,api}.ipynb` + `app/01_Flux_Capacitor_local_Streamlit.py` |
| 3 | Setting the Time Circuits | 2 | `02_time_circuits` — swap the system prompt |
| 4 | Running out of Plutonium | 3 | `03_plutonium` — watch it fail / hallucinate |
| 5 | Introducing Mr. Fusion | — | slides: Reason → Act → Observe |
| 6 | The Hoverboard | 4 | `04_hoverboard` — dummy weather, dummy sale lookup, **live Tavily** |
| 7 | Reaching 88 MPH | 5 | `05_88mph` — SQL + ratio study (COD, PRD) + mapping |
| 8 | Grays Sports Almanac | 6 | `06_almanac` — RAG over office policy docs |
| 9 | The DeLorean Engine | 7 | `07_delorean_engine` — **local only**, an API variant defeats the point |
| 10 | The Universal Adapter | 8 | `mcp_server/` + `08_universal_adapter` |
| 11 | The Dashboard | 9 | `app/` Streamlit |
| 12 | Wrap Up & Q&A | — | slides |

Demos 1–3 share one conceptual thread and demo 4's weather tool is the payoff for demo 3's failure. **Build them in outline order** so each artifact can lean on the previous one.

---

## 7. Model registry

Primary workhorse for §6–§11, and the default everywhere:

**`Qwen/Qwen3.5-4B`** — Apache 2.0, released 2026-03-02. Ranked **#1 of 13 local models** on a 2026 tool-calling eval (97.5%, one failure in 40 cases), beating models 5–7× its size. 262K native context. Hybrid thinking mode. ~2.5–3 GB at 4-bit NF4.

**Build out these three first**, all swappable behind the same interface, with a strengths/weaknesses writeup attendees can use to pick for their own office. All three are ≥3B so they can genuinely reason through a valuation problem — parameter count is not the thing to economize on in the agent demos.

| Model | Params | License | Strength | Weakness |
|---|---|---|---|---|
| `Qwen/Qwen3.5-4B` | 4B | Apache 2.0 | Best-in-class tool calling (97.5%); 262K context | Over-eager parallel tool calls |
| `HuggingFaceTB/SmolLM3-3B` | 3B | Apache 2.0 | Smallest of the three; best "look how tiny" story for §9 | Did not place in the 2026 tool-calling top tier |
| `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | 4B | NVIDIA Open Model | Sequential reasoning (95%) | Struggles with parallel same-tool calls |

⚠️ **That Nemotron id is exact and non-obvious.** It is *Nemotron-**3**-Nano*, not `Nemotron-Nano-4B` — the plausible-looking names all 404. All four registry models are **ungated** (no license click-through), which matters for a take-home repo. The canonical list lives in `src/delorean/models.py`; this table is a summary of it.

**`Qwen/Qwen3.5-0.8B` is for demo 1 only.** A tiny model is pedagogically *better* for showing next-word prediction — fast, dramatic, and it runs on a laptop with no GPU at all. It is **not** in the agent registry; §6 onward needs a model that can actually think.

### Measured on the dev box (2026-08-16, RTX 3080)
`scripts/check_setup.py` results — all four stages passed:

| Model | Precision | Weights | Peak inc. KV cache | Speed |
|---|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | fp16 | 1.40 GB | 1.44 GB | 12.0 tok/s |
| `Qwen/Qwen3.5-4B` | 4-bit NF4 | 2.95 GB | 2.98 GB | 13.6 tok/s |

**The 4B model uses under 3 GB.** That leaves ~5 GB free even on the team's
8 GB floor — far more headroom than planned for. Consequences:
- Context can be generous, not the 8–16K originally assumed. Determine the real
  ceiling empirically when building §7 rather than guessing.
- Running two models side by side for a live comparison is feasible.
- SmolLM3-3B and Nemotron-Nano-4B will both fit comfortably alongside.

### Traps to avoid
- **`Qwen/Qwen3.5-4B-Base` is a base model.** No chat template, no instruction tuning, no tool calling. It exists and looks right; it will not work for any agent demo. Always the non-`-Base` checkpoint.
- **bf16 does not fit.** A 4B model in bf16 is ~8 GB of weights alone — no room for KV cache on an 8 GB card. **4-bit NF4 via bitsandbytes is required**, not optional. Confirmed working on Ampere (compute 8.6) under CUDA 13.
- **Thinking mode is ON by default in Qwen3.5.** Long thinking traces mean dead air in front of an audience. Disable it (`enable_thinking=False`) for speed everywhere *except* §5/§7, where showing the model reason is the whole point.
- **Windows symlinks.** Without Developer Mode enabled, the HuggingFace cache falls back to copying instead of symlinking, so duplicate model files eat extra disk. Not fatal, but worth enabling Developer Mode on the presenter machine.

---

## 8. Conventions

### The twin rule (`_local` / `_api`)
Every demo ships two notebooks so an attendee grabs the one matching their situation — a data-security-constrained office takes `_local`, an office with no GPU budget takes `_api`. No flags, no branching, no config to understand.

**The two variants must differ *only* in the model-construction cell.** Everything downstream — tools, prompts, graph, output cells — is identical, because both import the same objects from `src/delorean/`. If a change touches anything but that one cell in one twin, it belongs in `src/` where both get it.

§9 is local-only by design. §10 and §11 are `.py` apps with a backend switch, not twins.

### Streamlit apps
Some demos ship a clickable app alongside the notebook (`app/NN_..._Streamlit.py`).
The notebook is the *explanation*; the app is the *instrument* — for when
someone in the audience asks "what about MY parcel number?"

Two rules learned the hard way in §2:

- **`viz.py` returns HTML strings, `show_*()` displays them.** `IPython.display`
  does nothing in Streamlit. Keep the `*_html()` builders as the single source
  of the markup so the notebook and the app can never drift apart visually.
- **Any button that edits a text box must use `on_click=`.** Streamlit forbids
  writing to `st.session_state[key]` after the widget with that key exists in
  the same run, and it only fails *when clicked* — never on load. Use the
  `set_field()` callback pattern.

Always test an app with `streamlit.testing.v1.AppTest`, which actually executes
the script and clicks the widgets. An HTTP 200 from the server proves nothing:
Streamlit does not run your script until a client connects.

### Notebooks
- **Commit with outputs.** This is deliberate: it makes the repo readable on GitHub without running anything, and it is the stage fallback when a cell dies live. Accept the diff noise.
- Every notebook opens with a markdown cell naming its outline section, what it teaches, and what it needs (GPU? API key? internet?).
- Cells small enough to narrate one at a time.

### Code
- Type hints on anything in `src/`. Docstrings written for a reader who does not know the library.
- No abstraction that exists only to be elegant. If a for-loop is clearer than a comprehension for this audience, write the for-loop.

---

## 9. Security

- **`.gitignore` is the first file in this repo**, before any code. It must cover `.env`, `.venv/`, `__pycache__/`, model caches (`*.safetensors`, `*.gguf`), and `.ipynb_checkpoints/`.
- **Never commit `.env`.** It currently holds a live `HF_TOKEN`. That token has been exposed in a session transcript and **should be rotated** at https://huggingface.co/settings/tokens.
- Ship a committed **`.env.example`** listing required variable *names* with empty values — attendees need to know what to set.
- No real parcel data, no real owner names, no real addresses. All demo data is synthetic. This is a public repo about property records; treat it that way.

---

## 10. Build order

1. ~~`.gitignore` + `.env.example`~~ — **done**
2. ~~`pyproject.toml` + `uv sync` + `scripts/check_setup.py`~~ — **done, all 4 stages pass**
3. ~~`src/delorean/` skeleton: `models.py`, `llm.py`, `viz.py`~~ — **done**
4. ~~**§2 The Flux Capacitor**~~ — **done**: local notebook, API notebook, Streamlit app
5. Synthetic `data/` — parcels, sales, office policy docs — **next**
6. Remaining demos in outline order: §3 → §4 → §6 → §7 → §8 → §9 → §10 → §11
7. Model comparison writeup (registry → attendee-facing doc)
8. Full dry run against the 90-minute clock

### What §2 established (reuse it)
- **`viz.py` is the shared visual language.** Token chips, probability bars, and
  the generated-text canvas are used by notebook and app alike. Later demos
  should extend it rather than inventing new styling.
- **The four-act shape works**: set up → show the mechanism → show it looping →
  ask something real, then optional "explore on your own" sections below the
  live path. Keep the live path to ~4 cells.
- **Hunt for the honest example.** The planned math demo *succeeded* (`1,247 +
  3,891` is correct), which would have contradicted the narration. Testing found
  `17% of $412,500 → 68,125` (truth 70,125) — off by $2,000 and entirely
  plausible-looking. That reframed the lesson from "it can't do math" to "it
  hands you a wrong number that passes the eyeball test," which is far more
  useful. **Always run the demo before writing the story about it.**

### Commands
```powershell
uv sync                                                        # build the env
uv run python scripts/check_setup.py                           # 4-stage machine check
uv run jupyter lab                                             # notebooks
uv run streamlit run app/01_Flux_Capacitor_local_Streamlit.py  # the §2 app

# after authoring a notebook, embed its outputs:
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 <path>
```

Notebooks are authored by writing a `nbformat` builder script to the scratchpad,
running it, then executing the notebook to embed outputs. Do **not** hand-write
`.ipynb` JSON. Watch for escaped quotes: use `r'''...'''` for any code cell
containing a `"""docstring"""`.

### Presenter workflow (goes on a card in Calgary)
Notebooks are committed **with outputs** — that is the stage fallback. Before
presenting: **Kernel → Restart Kernel and Clear All Outputs**, run cells live,
then *don't save* (or `git checkout` the file) to restore the fallback.
**F11** fullscreen, **Ctrl +** to zoom, **View → Simple Interface**.
Close other browser windows and apps first — desktop software holds 2–3 GB of
GPU memory. `nvidia-smi` overstates this on Windows; trust
`torch.cuda.mem_get_info()` instead.

**Open items:**
- ~~Verify a released `transformers` covers Qwen3.5~~ — **resolved**, 5.15.0 on PyPI, no `git+` pin.
- ~~Confirm bitsandbytes 4-bit loads under CUDA 13 on the RTX 3080~~ — **resolved**, works; 4B fits in 2.98 GB.
- ~~Does `ChatHuggingFace` surface tool calling for local Qwen3.5?~~ — **moot.** The LangGraph-only decision means we parse tool calls ourselves, so the wrapper is gone along with the risk.
- **Parse each registry model's tool-call format.** Qwen3.5, SmolLM3, and Nemotron each emit tool calls differently (Qwen uses `<tool_call>` tags; the others vary). The tool node needs a small per-model parser. Confirm the exact format for all three by inspecting their chat templates before building §6 — this is now the riskiest remaining unknown.

### Known environment gotchas

**OneDrive is actively hostile to Python installs — recommend moving the repo.**
This repo currently lives in `OneDrive\Documents\GitHub\`. That has already
caused two distinct failures in one session:

1. `uv sync` died with `os error 396` — OneDrive's cloud filesystem rejects the
   hardlinks uv uses by default. Fixed with `link-mode = "copy"` in `[tool.uv]`.
2. `scikit-learn` installed **silently corrupt** — 40 of 1055 files, no
   `__init__.py`, but valid `.dist-info` metadata, so uv and `pip list` both
   reported success. Symptom was a nonsense error from transformers. Fixed with
   `uv sync --reinstall-package scikit-learn`.

Failure 2 is the dangerous one: a partially-written package looks installed.
Moving the repo to a non-synced path (`C:\dev\88mph-mass-appraisal`) removes
the whole class of problem. Model weights are unaffected either way — they
cache to `~/.cache/huggingface`, outside OneDrive.

**If a package behaves impossibly, suspect a partial install first.**
```powershell
uv sync --reinstall-package <name>
```

**Chroma downloads a 79 MB ONNX model on first use** if you let it pick its own
embedding function. That is a surprise network call in the middle of a live
demo. Always pass an explicit embedding function backed by
`sentence-transformers` — it uses the GPU we already have, avoids the download,
and makes the "text becomes numbers" step visible in the code, which §8 needs
the audience to see anyway.

**transformers 5.15.0 has an undeclared scikit-learn dependency.**
`generation/candidate_generator.py` imports `sklearn.metrics` at module level,
so any use of `AutoModelForCausalLM` fails with the misleading
`Could not import module 'GenerationMixin'`. We declare `scikit-learn`
explicitly in `pyproject.toml`. Re-check on transformers upgrades.
