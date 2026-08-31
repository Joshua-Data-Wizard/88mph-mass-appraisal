# CLAUDE.md — 88mph-mass-appraisal

Working agreement for this repository. Read this first every session.

> ### 📂 This is one of two files. The other is [FINDINGS.md](FINDINGS.md).
> **CLAUDE.md is the agreement — what to do and what not to break.**
> **FINDINGS.md is the evidence — what we measured and what it cost.**
> Read this one every session. Go to the other when something surprises you,
> when a test fails impossibly, or before you "fix" something that looks
> wrong on purpose.

> **Two numbering systems, and they collide.** `§2`–`§12` in prose are
> **sections of the 90-minute talk** (§7 is the ratio study, §10 is MCP).
> `## 1.`–`## 15.` are **sections of this document**. Where this file refers
> to itself it says so by name — *"see §11 Tool calling"* — because `§11`
> alone could mean either the dashboard or the parser.

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
- **Give each demo the smallest toolbelt its lesson needs.** §6 hands the agent
  *one* tool because watching it choose between five is a different lesson than
  watching it use one. Where a later demo introduces something new, prefer
  **swapping a tool out** over piling another on. This is a teaching rule
  first — a short tool list keeps the screen readable and the point singular.
  It also happens to help the model: restraint measurably degrades at three
  tools and a guardrail system prompt buys it back (§13). **Small toolbelts
  because it suits the demo; the prompt is what handles scale.**

---

### ⛔ Non-negotiables that came from bugs

Every line here cost somebody an afternoon. **The rule is here; the story is in
[FINDINGS.md](FINDINGS.md).** If one of these looks unnecessary, read the story
before removing it — most of them look unnecessary right up until they don't.

| Rule | Break it and |
|---|---|
| **Apps build model pickers from `models.selectable()`**, never `list(models.REGISTRY)` | a presenter picks a model that OOMs mid-demo |
| **Draw SVG with `st.iframe`, never `st.html`** | DOMPurify silently deletes the drawing and leaves the card |
| **Model output never goes into an HTML string** | markdown tables render as raw pipes, and untrusted text lands in markup |
| **Call `llm.unload()` between models** in any multi-model script | VRAM accumulates and every measurement after the first is wrong |
| **Never block a thread you did not create** (`run_in_executor` on a blocking `get`) | the interpreter cannot exit and holds the GPU |
| **Parse tool calls with the full decode** (`skip_special_tokens=False`) | Gemma's calls vanish and it scores 0/6 |
| **Write system-prompt checklist items as nouns, not negations** | the model repeats *"no tax is outstanding"* back as a finding |
| **`.streamlit/config.toml` keeps `fileWatcherType = "none"`** | the watcher walks 5,500 `transformers` modules on every rerun |
| **`link-mode = "copy"` stays in `[tool.uv]`** | `uv sync` dies on OneDrive with `os error 396` |
| **Pass Chroma an explicit embedding function** | it downloads 79 MB mid-demo |
| **Never widen a tool's guessing to make a demo pass** | you ship the §4 failure the talk spends three minutes warning about |
| **Never renumber a demo** | an attendee who liked §7 cannot find §7 |

⚠️ **When a model scores zero, suspect the harness before the model.** It has
happened three times in this repository and the model was fine every time.
Print the raw generation with `repr()` before you believe any scoreboard.

⚠️ **When a process seems to hang, compare CPU time against wall clock.** A
working process burns CPU; one stuck at exit does not. That ratio was sitting
in plain sight for an hour once.

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
| Web search | **Tavily** (`tavily-python`) | §6 **explore section only** — never on the live path. Kept so attendees have a working web-search tool; off stage because it needs wifi *and* an API key |

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
├── notebooks/                   # 15 notebooks — a _local/_api pair per demo
│   ├── 01_flux_capacitor_{local,api}.ipynb    §2  what an LLM is
│   ├── 02_time_circuits_{local,api}.ipynb     §3  system prompts
│   ├── 03_plutonium_{local,api}.ipynb         §4  where it goes blind
│   ├── 04_hoverboard_{local,api}.ipynb        §6  tools, from nothing
│   ├── 05_88mph_{local,api}.ipynb             §7  ratio studies, SQL, maps
│   ├── 06_almanac_{local,api}.ipynb           §8  RAG as one more tool
│   ├── 07_delorean_smol.ipynb                 §9  local only, by design
│   └── 08_universal_adapter_{local,api}.ipynb §10 three offices over MCP
├── app/                         # Streamlit — always .py, never a notebook
│   ├── 01_Flux_Capacitor_local_Streamlit.py                  §2
│   ├── 02_03_Time_Circuits_and_Plutonium_local_Streamlit.py  §3 + §4
│   ├── 04_06_Office_Assistant_local_Streamlit.py             §6 + §8
│   ├── 05_88mph_Valuation_Review_local_Streamlit.py          §7
│   └── 08_09_Universal_Adapter_local_Streamlit.py            §10
├── src/delorean/                # importable package — the shared engine
│   ├── models.py                # ModelSpec registry; selectable() / blocked()
│   ├── llm.py                   # load_local, unload, chat_prompt, vocab_size,
│   │                            #   where_it_runs, get_api_client, api_model
│   ├── agent.py                 # the graph; responders; render_prompt; toolbox
│   ├── prompts.py               # the measured system prompts, with their why
│   ├── viz.py                   # *_html() builders + show_*() wrappers (~1760 ln)
│   ├── data.py                  # check/build, cama/clerk/tax, *_df frames
│   ├── ratio.py                 # IAAO maths — no model anywhere near it
│   ├── rag.py                   # §8 index + retrieve (bge-small, Chroma)
│   ├── offices.py               # §10 MCP client — discovery, routing, sync API
│   └── tools/                   # base, parcel, policy, query, review
├── mcp_server/                  # §10 — three servers, three offices
│   ├── appraiser_server.py      #   get_parcel, find_parcels
│   ├── clerk_server.py          #   get_deeds, corrections_since
│   └── tax_server.py            #   get_tax_status
├── scripts/
│   ├── check_setup.py           # 4-stage machine check — run this first
│   ├── fetch_models.py          # pre-download models; prove each one runs
│   ├── generate_hill_valley_data.py   # seeded, deterministic, re-runnable
│   ├── test_tools.py            # 20 parser checks, no model needed
│   ├── eval_tools.py            # tool-use score against a live model
│   ├── compare_models.py        # every model, same questions, same tools
│   ├── test_app_88mph.py        # §7 app, clicked
│   ├── test_app_office.py       # §6/§8 app, clicked
│   └── test_app_adapter.py      # §10 app, clicked
├── Guides/UV_Readme.md          # uv for newcomers
├── data/                        # DONE — see data/README.md
│   ├── cama.db                  #   Appraiser's office   1,200 parcels
│   ├── clerk.db                 #   Clerk of Courts        244 deeds
│   ├── tax.db                   #   Tax Collector        1,200 accounts
│   ├── csv/                     #   CSV mirrors, so GitHub can render the data
│   └── policy/                  #   4 office policy docs — the §8 RAG corpus
└── Presentation_Material/       # outline (source of truth for scope)
```

**`.vscode/settings.json` and `.vscode/extensions.json` are committed on
purpose** so the repo opens correctly in VS Code without registering a Jupyter
kernel. `.gitignore` uses `.vscode/*` plus negations — **not** `.vscode/`,
which would make the negations dead, because git never descends into an
excluded directory to re-include a file.

**`llm.vocab_size(model)` exists because Qwen3.5 is multimodal** — its config
nests under `cfg.text_config`, so plain `cfg.vocab_size` raises `AttributeError`.
Use the helper rather than reaching into the config.

**Why hybrid and not pure notebooks:** the parcel / ratio-study / mapping tools are reused across demos 5–9. Duplicated across six notebooks they would drift within a week. §10's MCP server and §11's Streamlit app cannot be notebooks at all.

**Where tools live is a per-demo decision:**
- **§6 Hoverboard — define tools inline in the notebook.** The tool *is* the lesson. An attendee must see a plain Python function become something the model can call.
- **§7 onward — import from `src/delorean/tools/`.** By then the point is the agent, not the plumbing.

### The dataset — three offices, deliberately not one database

Full documentation is in **[data/README.md](data/README.md)**; this is the part
that constrains future work.

**Three separate SQLite files with no foreign keys between them**, because that
is the real situation in a county: three elected offices, three vendors, no
integration. `data.py` deliberately offers **no** function that queries all
three at once. That gap is the subject of §10 — do not "helpfully" close it.

| File | Office | Holds |
|---|---|---|
| `cama.db` | Appraiser | characteristics, ownership, assessed value, lat/lon |
| `clerk.db` | Clerk of Courts | recorded deeds, sale prices, grantor/grantee |
| `tax.db` | Tax Collector | levies, payments, balances, delinquency |

**Systems agree on identifiers.** Same `parcel_id` string, same owner spelling,
same address in all three. Format normalization was considered and rejected —
it is ETL noise, and §10's lesson is connectivity, not string cleaning.

**The flaws are planted and measured, not incidental.** Each neighborhood fails
a *different* IAAO test so §7 has three distinct findings:

| Neighborhood | Median | COD | PRD | Demonstrates |
|---|---:|---:|---:|---|
| Riverside Heights | **0.807** | 6.27 | 1.002 | level failure |
| Courthouse Square | 0.938 | **22.20** | 1.026 | uniformity failure |
| Hilldale | 0.930 | 13.59 | **1.042** | equity / regressivity |
| **Countywide** | 0.927 | 12.37 | 1.027 | **all three pass** |

Residential only (`property_class IN ('R01','R02')`), which is how IAAO
stratifies and what `verify()` in the generator reports. Run it across every
class and the verdicts are identical, the figures move in the third decimal.

That last row is the payload: the county looks healthy in aggregate while half
its neighborhoods have real problems. **If you regenerate or retune the data,
re-verify that row still passes** — the §7 narration depends on it.

⚠️ **Anything that changes how often `rng` is called moves every number after
it.** Bounding the map scatter (below) swapped two `rng.gauss` calls for two
others and shifted Hilldale's median from 0.902 to 0.930. The three planted
failures still landed and the county still passed, because the generator
prints the whole table on every run — but nothing warns you, so read it.

**`04-127-3891` is the hero parcel and threads the whole talk.** §2 already
tokenized that exact string; §10 asks whether it can be split. Biff Tannen,
15,800 sq ft lot in Hilldale (splits fine under the 6,000 sq ft R-1 minimum),
owns it outright per the Clerk — **but owes $8,432.13 across two tax years**,
and Policy 4.2 bars a split over any balance. Lot ✅, ownership ✅, taxes ❌.
The agent must consult all three systems *and* the policy corpus, then name the
specific blocking reason. Its assessed value ($412,500) is also the number §2's
model does bad arithmetic on.

**Neighborhoods are contiguous, and that took a second attempt.** Parcels were
originally scattered with `rng.gauss(0, 0.0075)` around each neighborhood
centre. A gaussian has no edge, and that spread was wider than the gaps
between the six centres — Hilldale's outline ended up containing the centres
of three other neighborhoods. Every statistic was still exactly right; the
county map was an unreadable blob. Parcels are now sampled inside a bounded
disc (`NEIGHBORHOOD_RADIUS = 0.0038`, about 420 m) and `verify_geography()`
asserts on every run that no two neighborhoods touch. Tightest gap: 0.10 km.

**The generator is seeded (`SEED = 1985`) and byte-identical on re-run** —
verified. That is what lets committed notebook outputs match an attendee's
local data. Re-run it freely; never hand-edit a `.db`.

**Policy docs are `.md`, not PDF.** They render on GitHub, which matters for a
take-home repo, and it avoids a `reportlab` dependency. §8's loader must accept
both so an attendee can drop in their own office's real PDFs.

---

## 6. Demo map

| § | Title | Demo | Artifact |
|---|---|---|---|
| 1 | Welcome to Hill Valley | — | slides |
| 2 | The Flux Capacitor | 1 | ✅ **DONE** — `01_flux_capacitor_{local,api}.ipynb` + `app/01_Flux_Capacitor_local_Streamlit.py` |
| 3 | Setting the Time Circuits | 2 | ✅ **DONE** — `02_time_circuits_{local,api}.ipynb` + shared app |
| 4 | Running out of Plutonium | 3 | ✅ **DONE** — `03_plutonium_{local,api}.ipynb` + shared app |
| 5 | Introducing Mr. Fusion | — | slides: Reason → Act → Observe |
| 6 | The Hoverboard | 4 | ✅ **DONE** — `04_hoverboard_{local,api}.ipynb` + shared app tab 1 |
| 7 | Reaching 88 MPH | 5 | ✅ **DONE** — `05_88mph_{local,api}.ipynb` + `app/05_88mph_Valuation_Review_local_Streamlit.py` |
| 8 | Grays Sports Almanac | 6 | ✅ **DONE** — `06_almanac_{local,api}.ipynb` + shared app tab 2 |
| 9 | The DeLorean Engine | 7 | ✅ **DONE** — `07_delorean_smol.ipynb` — **local only**, an API variant defeats the point |
| 10 | The Universal Adapter | 8 **and 9** | ✅ **DONE** — `mcp_server/` × 3 + `08_universal_adapter_{local,api}.ipynb` + `app/08_09_Universal_Adapter_local_Streamlit.py` |
| 11 | Putting a Bow On It | — | ✅ **DECIDED — no new artifact.** The take-home close; see §15 |
| 12 | Wrap Up & Q&A | — | slides |

Demos 1–3 share one conceptual thread. **Build them in outline order** so each artifact can lean on the previous one.

### §6 and §8 are one app, and the toolbelt grows (decided 2026-08-16)

The outline's §6 called for three tools — dummy weather, sale lookup, live
Tavily. **We build one live tool: `lookup_parcel`.** Weather and Tavily move to
the notebook's explore section.

Why, beyond saving stage time:

1. **Weather is the one §4 failure where the model behaved *well*.** It refused
   honestly. The two failures where it actually *lied* are both appraisal
   questions, and they map cleanly onto the two demos:

   | §4 lie | Fixed by |
   |---|---|
   | invented comps — *"1952 Hilldale Ct, $485,000"* | **§6** `lookup_parcel` |
   | invented ordinance — *"0.25 acres per the County's Zoning Ordinance"* | **§8** RAG over the policy manual |

2. **Wifi is the #1 risk to this talk** (see §2). Tavily needs live internet
   *and* an API key, on stage. Off the live path, that failure mode is gone.
3. **"RAG is just another tool" should be shown, not asserted** — and the
   outline itself frames it that way. One app, one agent, and the visible tool
   list grows from `[lookup_parcel]` in tab 1 to
   `[lookup_parcel, search_policy]` in tab 2.

**§7 sits between them** (15 min, the core payload), so the presenter reopens
the app after it. That is deliberate, not an oversight.

**Watch the §11 boundary.** §11 is *the* dashboard. Keep the split: **§6/§8 is
the teaching instrument** (the Reason → Act → Observe loop, the raw tool call
and the retrieved chunks are all visible) and **§11 is the product** (clean
chat, machinery hidden). Both run the same engine from `src/delorean/`.

**§10 asks a decision, not a lookup.** The outline's original framing — verify a
deed against CAMA — proves the plumbing works but invites "couldn't you just
write a SQL join?" Instead: *"May Biff Tannen split parcel 04-127-3891?"* That
cannot be answered from one system, because the lot size lives with the
Appraiser, ownership with the Clerk, the balance with the Tax Collector, and
**the rule itself in a policy document**. §8's RAG supplies the rule and §10's
MCP supplies the facts, so the two demos interlock instead of sitting side by
side. Preserve that link — it is what makes §8 setup for the finale rather than
a detour.

---

## 7. Model registry

> 📎 Why Phi-4-mini and Gemma 4 were added, and what each model measured, are in **[FINDINGS.md](FINDINGS.md)**.

Primary workhorse for §6–§11, and the default everywhere:

**`Qwen/Qwen3.5-4B`** — Apache 2.0, released 2026-03-02. Ranked **#1 of 13 local models** on a 2026 tool-calling eval (97.5%, one failure in 40 cases), beating models 5–7× its size. 262K native context. Hybrid thinking mode. ~2.5–3 GB at 4-bit NF4.

All swappable behind the same interface, with a strengths/weaknesses writeup attendees can use to pick for their own office. All are ≥2.3B so they can genuinely reason through a valuation problem — parameter count is not the thing to economize on in the agent demos.

**Every figure here is measured** by `scripts/compare_models.py` on the §7 toolbelt: ten questions, three tools. **Peak VRAM, not weights.**

| Model | Params | License | Recall | Restr | SQL | Peak | s/q |
|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3.5-4B` ⭐ | 4B | Apache 2.0 | **6/6** | 4/4 | **6/6** | 4.00 GB | 8.7 |
| `HuggingFaceTB/SmolLM3-3B` | 3B | Apache 2.0 | **6/6** | 4/4 | **6/6** | **2.77 GB** | 6.8 |
| `microsoft/Phi-4-mini-instruct` | 3.8B | **MIT** | 5/6 | 4/4 | 7/10 | 4.15 GB | 7.0 |
| `google/gemma-4-E2B-it` | 2.3B | Apache 2.0 | 5/6 | 4/4 | **5/5** | 6.55 GB | **5.7** |
| `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | 4B | NVIDIA Open | — | — | — | ⚠️ OOM | — |

⚠️ **That Nemotron id is exact and non-obvious.** It is *Nemotron-**3**-Nano*, not `Nemotron-Nano-4B` — the plausible-looking names all 404. **Every registry model is ungated** (no license click-through), which matters for a take-home repo. The canonical list lives in `src/delorean/models.py`; this table is a summary of it.

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

Three rules learned the hard way:

- **`viz.py` returns HTML strings, `show_*()` displays them.** `IPython.display`
  does nothing in Streamlit. Keep the `*_html()` builders as the single source
  of the markup so the notebook and the app can never drift apart visually.
- **Any button that edits a text box must use `on_click=`.** Streamlit forbids
  writing to `st.session_state[key]` after the widget with that key exists in
  the same run, and it only fails *when clicked* — never on load. Use the
  `set_field()` callback pattern.
- **Seed widget state with `key=` + `session_state.setdefault()`, never
  `default=`** (§3, tab 2). A `multiselect` given `default=` re-applies that
  default on every rerun, so removing an item silently undid itself; combined
  with `max_selections` it then insisted the limit was reached while showing
  fewer items than the limit. **Avoid `max_selections` on a stage app entirely**
  — it greys out the remaining options and demands "remove an option first",
  which is the worst possible behaviour when you are swapping one option for
  another in front of a room. Show the cost (generation seconds) instead of
  forbidding the choice.

Always test an app with `streamlit.testing.v1.AppTest`, which actually executes
the script and clicks the widgets. An HTTP 200 from the server proves nothing:
Streamlit does not run your script until a client connects.

⚠️ **Every app needs its own test, and for a while one did not have one.**
`test_app_88mph.py` existed from the day §7 was built; the §6/§8 app had
**no test at all**, and a change to `local_responder` reached it unnoticed
because nothing exercised it. `scripts/test_app_office.py` now covers it:
the one-tool and two-tool belts, the raw reveal, the restraint case, the
mangled parcel id, and the system-prompt presets clicked in sequence.

**The lesson is not "write tests."** It is that the two agent apps look
similar enough that testing one feels like testing both, and it is not.
They exercise different toolbelts and different parts of `agent.py`.

**But AppTest does not catch everything.** `max_selections` is enforced in the
browser, so `set_value()` jumped straight to the final state and sailed past a
bug a user hit within minutes. When a report is about *interaction*, reproduce
the actual click sequence — set, unset, re-set — not one jump to the end state.

To assert on `st.html()` output in a test, read `element.proto.body` —
`element.value` raises `AttributeError` for html elements.

**§2's app was re-audited with sequences after that bug and is clean**: four
consecutive "Take the winner" clicks, consecutive presets, type-then-preset,
slider-then-click, generate/toggle/generate, and model switching all hold. It
has no `multiselect` and its text widgets already use `key=` + a seed.

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
5. ~~Synthetic `data/` — three office systems + policy corpus~~ — **done**, flaws verified
6. ~~§3 + §4 — shared app + four notebooks~~ — **done**, all executed with outputs
7. ~~Remaining demos in outline order: §6 → §7 → §8 → §9 → §10~~ — **done**
8. ~~§11 The Dashboard~~ — **resolved without an artifact**, see *§15 Section 11* below

### ✅ Every demo is built. What is left is not code.

| | |
|---|---|
| **The 90-minute dry run** | never done, and it is now the biggest open risk. Nine demos, five apps, one clock. |
| **Model comparison writeup** | the registry has six models and real numbers; an attendee-facing page does not exist yet |
| **A presenter card** | the workflow below plus the pre-flight checks, on one printable page |

⚠️ **Do not add a tenth demo.** The outline is 90 minutes with zero slack and
every section is now spoken for. New ideas belong in the explore sections of
existing notebooks, where they cost no stage time.

### DEMOS.md is the presentation-facing companion
**[DEMOS.md](DEMOS.md)** holds, per demo: what is on screen, the click order,
**the measured numbers and verbatim model quotes** a slide can cite, the
one-sentence takeaway, timings, and fallbacks. It exists so the talk script and
the code cannot drift apart.

**Keep it current when a demo changes** — it is written to be quoted directly,
so a stale figure in it becomes a wrong figure on a slide. Mark unbuilt demos
clearly; the script must never be drafted against something that does not exist.

### Commands
```powershell
uv sync                                                        # build the env
uv run python scripts/check_setup.py                           # 4-stage machine check
uv run python scripts/fetch_models.py                          # pre-download tiny + qwen
uv run python scripts/fetch_models.py --list                   # what is cached
uv run python scripts/generate_hill_valley_data.py             # rebuild data/ (deterministic)
uv run jupyter lab                                             # notebooks
uv run streamlit run app/01_Flux_Capacitor_local_Streamlit.py  # the §2 app
uv run streamlit run app/02_03_Time_Circuits_and_Plutonium_local_Streamlit.py  # §3+§4
uv run streamlit run app/04_06_Office_Assistant_local_Streamlit.py            # §6+§8
uv run python scripts/test_tools.py                            # tool robustness
uv run python scripts/eval_tools.py                            # tool-use score
uv run python scripts/compare_models.py                        # every model, S7 toolbelt
uv run python scripts/test_app_88mph.py                        # the S7 app, clicked
uv run python scripts/test_app_office.py                       # the S6/S8 app, clicked

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
- ~~Parse each registry model's tool-call format.~~ — **RESOLVED, see *§11 Tool calling* below.**
  The old note here guessed Qwen emits JSON inside `<tool_call>`. **It does not.**

## 11. Tool calling — measured formats

> 📎 The full investigations — how a template silently drops your tools, why a model scoring zero is usually your fault, and what thinking mode costs — are in **[FINDINGS.md](FINDINGS.md)**.

**This was the riskiest open item in the project. It is now resolved, and the
previous assumption was wrong.** Verified by rendering each model's chat
template with `tools=` and by generating a real call on the 4B.

### Four shapes, and both ends of the pipe can betray you

> 🩹 **Corrected twice.** This section once said "two shapes, not three" and
> claimed all models wrap calls in `<tool_call>`. That cost SmolLM3 a score of
> **0/6**. It then said "three shapes"; adding Phi-4-mini and Gemma 4 on
> 2026-08-25 found a fourth *output* shape **and a second failure mode
> entirely — on the input side.** Assume there is a fifth.

| Model | Emits | Wrapped in `<tool_call>`? |
|---|---|---|
| `Qwen/Qwen3.5-4B` | nested XML | yes |
| `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | nested XML | yes |
| `HuggingFaceTB/SmolLM3-3B` | **bare JSON** | ❌ **no** |
| `microsoft/Phi-4-mini-instruct` | bare JSON (Shape C) | ❌ no |
| `google/gemma-4-E2B-it` | **`<\|tool_call>call:name{…}`** | ❌ **its own tags** |

```text
Shape D — Gemma 4. Note the pipe is INSIDE the brackets, and the closing
tag is mirrored, so a `"<tool_call>" in text` check does not see it at all:

    <|tool_call>call:map_parcels{sql:<|"|>SELECT parcel_id FROM …<|"|>}<tool_call|>

Strings are fenced with `<|"|>` rather than quoted, which is what makes it
safe to put a SQL statement — commas, quotes and braces included — inside a
brace-delimited argument list.
```

⚠️ **Gemma also writes `<|tool_response>` immediately after its call and, left
alone, invents a plausible result for a tool that has not run yet.**
`strip_tool_calls()` cuts everything from that marker, and there is a test for
it. That text must never reach a user — it is §4 wearing a tool call's
clothes.

### Passing results back
Append `{"role": "tool", "content": <string>}`. All three templates render that
as a `<tool_response>` block inside a **user** turn. Content must be a string —
`json.dumps()` the payload.

## 12. The tools layer — built for sloppy callers

> 📎 The measured tool-use scores, the over-eager-tool signal, and why the normalizer exists at all are in **[FINDINGS.md](FINDINGS.md)**.

`src/delorean/tools/` · verify with `uv run python scripts/test_tools.py`

### The one idea: errors are prompts

A tool's failure message does not go to a log. It goes back into the model's
context as the observation it reasons from next. A traceback teaches it nothing;
`"parcel numbers look like NN-NNN-NNNN, did you mean 04-127-3891?"` lets it fix
itself on the next turn. **Every failure returns a `hint` written for the model
to read.** One envelope, always:

```python
{"ok": True,  "data": {...}}
{"ok": False, "error": "...", "hint": "..."}
```

`dispatch()` **never raises.** An agent that sees an exception cannot recover.

### What `base.py` absorbs

Tool name punctuation/case/nicknames (`lookupParcel`, `parcel_lookup`,
`lookup_parcels`) · parameter aliases (`parcel`, `id`, `pid`, `parcelId` →
`parcel_id`) · quoted values · **type coercion**, because Shape A carries no
types at all (`"3"` → `3`, `"$412,500"` → `412500`) · missing required args ·
unknown extra args (dropped, never fatal) · any exception the tool throws.

**Schemas are hand-written dicts next to each function**, not derived. An
appraiser must be able to read the function and the description the model sees
side by side. The decorator only does what is genuinely about robustness.

`schemas(["lookup_parcel"])` exposes a subset — §6's live path gives the agent
**one** tool, because watching it choose between five is a different lesson.

## 13. The agent loop (`agent.py`) and the tool-count question

> 📎 The toolbelt-versus-restraint measurements, the Streamlit bugs, and what each demo's notebooks established are in **[FINDINGS.md](FINDINGS.md)**.

`build_agent(responder, tool_names, max_steps=4)` — two nodes, one conditional
edge, hand-written. `reason` asks the model what it wants; `act` runs it;
the edge loops back. `agent.get_graph().draw_mermaid()` prints the picture,
which is the same picture §5 puts on a slide. **Verified end to end: 7.2 s,
2 reasoning steps, and it resolved the mangled id `4-127-3891` correctly.**

**One graph, both twins.** A **responder** is the only model-aware part:
`(messages, schemas) -> (text, tool_calls)`. `local_responder()` parses text;
`api_responder()` reads Anthropic's native `tool_use` blocks and translates
message shapes. Everything above that line is identical, which is what lets
`_local` and `_api` share a graph.

`act` contains **no error handling on purpose** — `dispatch()` never raises, so
a tool failure arrives as a readable observation the model can act on.

### "Is it actually using the GPU?" — how to answer it

**`llm.where_it_runs(model)`** counts parameter tensors per device and reports
VRAM. Measured on the dev box with `qwen`: **426 of 426 tensors on `cuda:0`,
0 on CPU**, 2.90 GB allocated, 2.98 GB peak, 11.3 tok/s.

**High CPU during generation is normal and does not mean a CPU fallback.** The
weights and the maths are on the card; Python is launching one small kernel per
token. Two things *are* on the CPU on purpose: `bge-small` for retrieval (so an
office with no GPU can still do RAG, and so the VRAM stays free) and
tokenization.

The §6/§8 app puts this in the sidebar — part diagnostic, part demo. An audience
being told "this runs on the machine under your desk" should be able to see it.

## 14. Section 10 — MCP, and the three offices

> 📎 The measured timings, the negation-in-a-prompt failure and the capability ceiling between §7 and §10 are in **[FINDINGS.md](FINDINGS.md)**.

`mcp_server/` × 3 · `src/delorean/offices.py` · verify with
`uv run python scripts/test_app_adapter.py`

### The decision: three servers, not one more tool

The outline framed MCP as "tools, but standardised". **We build it as three
separate programs representing three elected offices**, because that is the
argument the room actually needs. A question a taxpayer asks at a counter —
*may I split my lot?* — cannot be answered by any single office in the county,
and today the only thing joining them is somebody making three phone calls.

Showing MCP as one more entry in a toolbelt makes it look like plumbing.
Showing three buildings makes it look like the problem it solves.

### `build_agent(toolbox=...)` is the whole section in one argument

```python
agent.build_agent(respond, toolbox=offices)     # three MCP servers
agent.build_agent(respond, ["run_sql"])         # local functions, §7
```

Anything with `.schemas(names)` and `.dispatch(name, args)` works. Reason →
Act → Observe does not change, the responder does not change, the notebook
does not change. **Only where the tools live changes**, which is the entire
claim MCP makes — and it is a three-line change to `agent.py` rather than a
new abstraction.

### ⚠️ The docstring under `@mcp.tool()` is the prompt, not documentation

It is the literal text a model reads when deciding whether to call the
function. Every server here names its units, says what a number means, and
says explicitly what it does **not** know (*"the Appraiser's roll and the
Clerk's deeds do NOT carry this information"*). A vague docstring on an MCP
server is indistinguishable from a broken tool.

### Traps specific to MCP

- **FastMCP generates the schema from the signature**, so parameter
  descriptions in an `Args:` block do **not** become per-property
  descriptions — they stay in the tool description. Both models coped; do not
  assume a future one will.
- **Tool names must be unique across servers.** `Offices` raises rather than
  letting two offices publish the same name, because the agent has no way to
  tell them apart.
- **Servers log every request at INFO to stderr.** `Offices(quiet=True)` is
  the default for that reason — and `quiet=False` is the first debugging move,
  because a server that dies on startup says so on stderr and nowhere else.
- **The `search_policy` tool stays LOCAL on purpose.** The policy manual
  belongs to this office; MCP is for reaching somebody else's system. The app
  draws it in a dashed panel to make that difference visible.

---

## 15. Section 11 — the take-home close (no artifact)

**Decided 2026-08-25.** The outline asks for *"a small Streamlit app running
the Mass Valuation Agent… a clean, ChatGPT-like interface."* That is §7's
engine with a different face — a reprise, not a new product.

Building a third app would end the talk on a **smaller** demo than the two
before it. §7 finds the problem; §10 reaches across the county to solve it;
anything after that is an anticlimax.

So §11 spends its five minutes on what the presenters care about most: **how
somebody actually starts on Monday.** Reopen either finished app with the
machinery collapsed, then show the repository itself.

⚠️ **The nine-demo count is preserved** — §10's app carries two distinct demo
beats (the cross-office question, and the sync check), which is also the
honest way to describe it.

**If a §11 artifact is ever wanted**, the cheapest honest version is a ~100
line chat wrapper over the same engine — `chat_input`, `agent.ask`,
`st.markdown`, no sidebar and no trace. The contrast *is* the message. It is
not built, and the talk does not need it.
