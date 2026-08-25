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
- **Give each demo the smallest toolbelt its lesson needs.** §6 hands the agent
  *one* tool because watching it choose between five is a different lesson than
  watching it use one. Where a later demo introduces something new, prefer
  **swapping a tool out** over piling another on. This is a teaching rule
  first — a short tool list keeps the screen readable and the point singular.
  It also happens to help the model: restraint measurably degrades at three
  tools and a guardrail system prompt buys it back (§13). **Small toolbelts
  because it suits the demo; the prompt is what handles scale.**

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
├── notebooks/                   # one pair per demo, numbered to the outline
│   ├── 01_flux_capacitor_local.ipynb      DONE
│   └── 01_flux_capacitor_api.ipynb        DONE
├── app/                         # Streamlit — always .py, never a notebook
│   ├── 01_Flux_Capacitor_local_Streamlit.py   DONE
│   ├── 02_03_Time_Circuits_and_Plutonium_local_Streamlit.py  DONE (§3+§4)
│   └── 04_06_Office_Assistant_local_Streamlit.py  DONE (§6+§8)
├── src/delorean/                # importable package — the shared engine
│   ├── models.py                # ModelSpec registry: get/tiny/default_local/compare
│   ├── llm.py                   # load_local, load_tokenizer, chat_prompt,
│   │                            #   vocab_size, get_api_client, api_model
│   ├── viz.py                   # *_html() builders + show_*() notebook wrappers
│   ├── data.py                  # DONE — check/build, cama/clerk/tax connections,
│   │                            #   *_df frames, policy_documents, systems
│   ├── tools/                   # base.py + parcel.py DONE — see §12
│   │                            #   (§7+) ratio study, mapping still to build
│   └── rag/                     # (§8) index + retrieve
├── scripts/
│   ├── check_setup.py           # 4-stage machine check — run this first
│   └── generate_hill_valley_data.py   # DONE — seeded, deterministic
├── Guides/UV_Readme.md          # uv for newcomers
├── mcp_server/                  # §10 — must be .py
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
| 9 | The DeLorean Engine | 7 | `07_delorean_engine` — **local only**, an API variant defeats the point |
| 10 | The Universal Adapter | 8 | `mcp_server/` + `08_universal_adapter` — **the parcel-split scenario**, see below |
| 11 | The Dashboard | 9 | `app/` Streamlit |
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
7. Remaining demos in outline order: ~~§6~~ → ~~§7~~ → ~~§8~~ → **§9 → next** → §10 → §11
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

### What §3/§4 established (measured, not assumed)

**§3 and §4 are cause and effect, and that is the spine of both demos.**
A bare model asked something it cannot know **politely refuses** — every probe,
both models. The outline's "watch it hallucinate" does not reproduce. What does
reproduce: give it the *confident expert persona from §3* and it invents
specific, cited, entirely fictional answers. **The prompt that makes it useful
is the prompt that makes it lie.** Verified on `qwen` 4B:

| Probe | Bare model | With the §3 expert persona | Truth |
|---|---|---|---|
| R-1 min lot size | "no such county… not defined" | "**0.25 acres.** Per the County's Zoning Ordinance…" | 6,000 sq ft |
| Appeal deadline | invents Jan 15, blames a *Community* episode | "**January 15**, late filings not accepted" | July 25 |
| Comps near 1955 Hilldale Ct | correctly refuses | "1952 Hilldale Ct — $485,000, closed 14 days ago" | no such address |
| PRD | — | invents "Property **Revenue Distribution**", then "Property **Reference Date**" | Price Related Differential |

**Use the lot-size probe as the hero.** 0.25 acres = 10,890 sq ft; Biff's
15,800 sq ft lot halves to 7,900, which *passes* the real 6,000 rule and *fails*
the invented one. The hallucination reverses §10's actual decision.

**§3/§4 use `qwen` (4B), not `tiny`.** The 0.8B model **breaks character** on
the angry-taxpayer persona — it stops and lectures you about being an AI. The
4B is superb at it. The app exposes the small model anyway so that failure can
be shown deliberately.

**"You are a helpful assistant" is a no-op.** On 4B it produced output nearly
identical to no system prompt at all, because that is already what the model
was trained to be. Value comes from specificity, not from having *a* prompt.

**The failure is unpredictable, and say so.** The same persona invented lavish
comps and then correctly refused to describe a sale that never happened. Do not
promise the audience a hallucination on any single question — promise them that
you cannot tell which you will get.

**Measured VRAM is higher in real use than `check_setup.py` reports.** The 4B
sat at **4.36 GB** during generation with these prompts, versus 2.98 GB for the
setup check's short probe. Still fine on an 8 GB floor; do not quote 2.98 as a
working figure.

**`viz.py` gained four §3/§4 helpers** — `time_circuits_html()` (the dark
three-row LED panel: destination = persona, present = question, last departed =
previous persona), `persona_card_html()`, `verification_html()` (the
model-said / county-record-says panel §4 rests on), and `show_html()`, a
generic notebook display for any builder's string.

**⚠️ The §4 twins reach genuinely different conclusions, and that is correct.**
This is the one place the twin rule bends: both notebooks run identical code,
but `claude-haiku-4-5` **refuses** the probe that `Qwen3.5-4B` **invents** an
answer to — often naming the persona while refusing (*"while I'm playing the
role of an experienced appraiser, I shouldn't guess"*). The `_api` notebook
therefore quotes the local model's fabrication verbatim and frames the gap as a
real local-vs-hosted trade-off an attendee should weigh, with the caveat that
**resistance is not immunity**. Haiku also identified Hill Valley as *Back to
the Future* correctly, where the 4B guessed *Toy Story*, *Community*, and
*Young Sheldon* across three runs. Do not "fix" this divergence by making the
twins agree — re-measure if models change, and rewrite the narration to match.

**Notebook authoring trap that cost a re-run:** the builder script had
`build("local")` and `build("api")` at the bottom, so re-running it to fix the
API variant silently wiped the already-executed local notebook's outputs.
Build scripts that emit twins should take the variant as an argument.

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
uv run python scripts/generate_hill_valley_data.py             # rebuild data/ (deterministic)
uv run jupyter lab                                             # notebooks
uv run streamlit run app/01_Flux_Capacitor_local_Streamlit.py  # the §2 app
uv run streamlit run app/02_03_Time_Circuits_and_Plutonium_local_Streamlit.py  # §3+§4
uv run streamlit run app/04_06_Office_Assistant_local_Streamlit.py            # §6+§8
uv run python scripts/test_tools.py                            # tool robustness
uv run python scripts/eval_tools.py                            # tool-use score

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
- ~~Parse each registry model's tool-call format.~~ — **RESOLVED, see §11 below.**
  The old note here guessed Qwen emits JSON inside `<tool_call>`. **It does not.**

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

---

## 11. Tool calling — measured formats (2026-08-16)

**This was the riskiest open item in the project. It is now resolved, and the
previous assumption was wrong.** Verified by rendering each model's chat
template with `tools=` and by generating a real call on the 4B.

### Two shapes, not three

| Model | Emits |
|---|---|
| `Qwen/Qwen3.5-4B` | **nested XML** |
| `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | **nested XML** — same as Qwen3.5 |
| `HuggingFaceTB/SmolLM3-3B` | **JSON** |

```text
Shape A — Qwen3.5 and Nemotron          Shape B — SmolLM3
<tool_call>                             <tool_call>
<function=lookup_parcel>                {"name": "lookup_parcel",
<parameter=parcel_id>                    "arguments": {"parcel_id": "04-127-3891"}}
04-127-3891                             </tool_call>
</parameter>
</function>
</tool_call>
```

⚠️ **All three wrap in `<tool_call>`, so a tag check alone will fool you.** The
contents differ. A parser that only does `json.loads()` on the tag body — the
obvious implementation, and what the old note implied — **fails on our own
default model.** This parser handles both and is verified end to end:

```python
def parse_tool_call(text: str):
    # Shape A -- nested XML. Qwen3.5 and Nemotron.
    m = re.search(r"<tool_call>\s*<function=([^>\s]+)\s*>(.*?)</function>", text, re.S)
    if m:
        args = dict(re.findall(
            r"<parameter=([^>\s]+)\s*>\s*(.*?)\s*</parameter>", m.group(2), re.S))
        return m.group(1).strip(), args
    # Shape B -- JSON. SmolLM3.
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if m:
        payload = json.loads(m.group(1))
        return payload.get("name"), payload.get("arguments", {})
    return None
```

**Arguments from Shape A are always strings.** There is no typing in the XML —
`<parameter=year>1971</parameter>` gives you `"1971"`. Coerce against the tool's
own JSON schema before calling the function.

### Passing results back
Append `{"role": "tool", "content": <string>}`. All three templates render that
as a `<tool_response>` block inside a **user** turn. Content must be a string —
`json.dumps()` the payload.

### Thinking mode is ON by default when you pass `tools=`
The rendered prompt ends with `<think>\n` whether or not you asked for it.
Measured on the 4B, one full Reason → Act → Observe → Answer round trip:

| | Wall clock | Notes |
|---|---|---|
| `enable_thinking=False` | **6.8 s** | clean tool call, correct answer |
| `enable_thinking=True` | **23.8 s** | emits a real reasoning trace first |

The trace is genuinely good — *"I have a tool called lookup_parcel… this seems
perfect for this request"* — and it **is** the Reason step §5 describes on a
slide. Worth showing once in §6, then switching off.

**Optimisation worth taking:** enable thinking on the *decide* call only, not on
the *report* call. Thinking on the final answer costs ~12 s and adds nothing —
the model has already decided. Roughly halves the visible cost.

**Budget ≥320 `max_new_tokens` for a tool call with thinking on.** At 120 the
call truncated mid-parameter and the parser got nothing.

---

## 12. The tools layer — built for sloppy callers

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

### Why it exists — measured, not assumed

Across ten phrasings of one question, Qwen3.5-4B picked the right tool **10/10**
but emitted the parcel number **three different ways**: `04-127-3891`,
`04127-3891`, `4-127-3891`. A plain `WHERE parcel_id = ?` resolved **8/10** —
and failed *silently*, looking exactly like a parcel that does not exist.

**With the hardening layer: 10/10 end to end against the live model.**

Small models are reliable at *deciding* and sloppy at *formatting*. Absorb the
formatting and they behave far better than their size suggests.

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

### The §2 callback worth narrating

`normalize_parcel_id()` exists *because* of §2. The model sees `04-127-3891` as
thirteen fragments, so it cannot reassemble it the same way twice. Extract the
digits, left-pad to nine, reformat — the fixed-width format makes it unambiguous.
Fragments (`"3891"`) are **refused, not guessed at**, so the tool never invents
a parcel nobody asked for.

**Never widen a tool's guessing to make a demo pass.** Returning the wrong
parcel confidently is the exact failure §4 spends three minutes warning about.

### Measured tool-use score (`scripts/eval_tools.py`)

24 questions, both tools offered, `Qwen3.5-4B`, 2026-08-16:

| Category | Score |
|---|---|
| Clean parcel lookup | 5/5 |
| **Mangled identifier** (`04127-3891`, `4-127-3891`, `04/127/3891`) | **4/4** |
| A different real parcel (not the memorised hero) | 2/2 |
| No parcel number given → must `search_parcels` | 4/4 |
| **Answer directly, call no tool** | **6/6** |
| Tool must fail gracefully | 3/3 |
| **TOTAL** | **24/24 (100%)** · 5.0 s per question |

**Zero unnecessary tool calls** on the six "just answer" questions, and **zero
parallel calls**. Category E is the one to watch when anything changes: an agent
that hits the database because somebody said "good morning" looks broken on
stage, and it is the failure a bigger tool list makes more likely.

⚠️ **`models.py` lists "over-eager parallel tool calls" as Qwen3.5's weakness.**
It did not reproduce here — 0 of 24, with two tools offered. Treat that registry
note as unverified at this scale rather than as fact, and re-measure when §7
raises the count to five or six tools.

⚠️ **100% on a suite we wrote ourselves is a good signal, not proof.** The cases
cover the failure modes we know about. Add a case whenever a new one turns up.

Only `qwen` is cached on the dev box; naming `smol` or `nemotron` downloads
5.7 GB / 7.4 GB. The dual-shape parser is unit-tested against both formats in
`scripts/test_tools.py` without needing either download.

### `search_policy` — §8's RAG tool (`rag.py` + `tools/policy.py`)

**RAG is built as an ordinary tool, alongside the others.** That is the §8
lesson made structural rather than asserted: same agent, same loop, one more
entry in the tool list.

| | |
|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` — MIT, ungated, 33M params, 384-dim, 128 MB |
| Device | **CPU by default.** The `_api` twin assumes no GPU, and RAG still needs embeddings computed locally. Also leaves VRAM for the 4B. |
| Store | Chroma, **in memory** — the corpus rebuilds in ~1 s, and an index that cannot go stale is one less thing to debug live |
| Corpus | **27 chunks** from 4 policy documents, split on `###` headings |
| Retrieval | **6/6 hero questions at rank 1**; queries take 10–20 ms |

**Every passage carries `document §section`.** Non-negotiable — §10 needs the
agent to say *"Policy 4.2 bars the split"* and a human to be able to check it.
It is also the direct answer to §4, where the model cited an ordinance that
does not exist.

**Chunks that cannot name their own section are excluded.** Document front
matter retrieved as `§` with no number — an uncheckable citation. Dropping it
took the corpus 31 → 27 chunks and *improved* the appeal-deadline score from
0.799 to 0.825.

**Traps hit while building this, all fixed:**
- `chromadb.EphemeralClient()` returns a **shared** in-memory database per
  process, not a fresh one. Building twice raises "collection already exists".
  `build_index()` now drops the collection first, so re-running a notebook cell
  is safe.
- Chroma 1.5.9 calls **`embed_query()`**, not `__call__`, for queries — it
  exists for *asymmetric* models that want questions phrased differently from
  passages. A custom embedding function without it fails only at query time.
- We deliberately skip BGE's `"Represent this sentence for searching relevant
  passages: "` query prefix. Measured: no difference, 6/6 either way. Revisit on
  a larger corpus.

### ⚠️ The over-eager-tool signal (re-measure at every tool added)

`scripts/eval_tools.py` with **three** tools offered: **28/29 (97%)**,
4.3 s/question. Perfect on lookup, mangled ids, search, policy, and graceful
failure. The single failure is the one that matters:

> **"What is the difference between market value and assessed value?"** →
> called `search_policy` instead of answering directly.

**Two tools scored 6/6 on that category; three tools scored 4/5.** The first
unnecessary call appeared exactly when the toolbelt grew. The case is genuinely
borderline — checking the manual before answering a policy-adjacent question is
defensible — and it has been **left in the suite unreclassified**, because
massaging a test case to reach 100% would throw away the leading indicator.

**§7 offers five or six tools. Re-run this eval then, and expect category E to
degrade further.** The mitigation is a system prompt that says when to search
and when to answer — which is §3's lesson doing real work.

---

## 13. The agent loop (`agent.py`) and the tool-count question

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

### ⚠️ Measured: a bigger toolbelt costs restraint, and a system prompt buys it back

Same questions at each toolbelt size. **Restraint** = answered directly when it
should have. **Recall** = called the right tool when it genuinely needed one.

| Config | Restraint | Recall |
|---|---|---|
| 1 tool | 6/6 | 1/1 |
| 2 tools | 6/6 | 2/2 |
| **3 tools** | **5/6** ❌ | 4/4 |
| **3 tools + guardrail prompt** | **6/6** ✅ | **4/4** ✅ |

The failure at three tools is reproducible and specific: *"What is the
difference between market value and assessed value?"* → searched the policy
manual instead of answering.

**The guardrail prompt fixes it completely, at no cost to recall.** That is the
important result — restraint was recovered *without* removing a tool:

```
Use lookup_parcel or search_parcels ONLY when the question is about a specific
property, owner or address.
Use search_policy ONLY for questions about THIS office's own rules, deadlines,
standards or procedures.
Answer general appraisal questions, definitions and small talk directly from
your own knowledge. Do not use a tool for those.
```

**So: keep toolbelts small because it suits each demo's lesson, not because the
model cannot cope.** §7 needs five or six tools and can have them — with a
guardrail prompt, and re-measured at that count before the dry run.

This is §3 doing real engineering work rather than being a party trick, and it
is worth saying so on stage: the same lever that made the model *sound* like an
appraiser is what stops it reaching for a database to answer "good morning".

### Streamlit + transformers: the torchvision traceback wall

Running any app printed **hundreds** of `ModuleNotFoundError: No module named
'torchvision'` tracebacks and burned noticeable CPU.

**Cause.** Streamlit's hot-reload watcher walks every loaded module on **every
rerun** to decide which files to watch. `transformers` exposes ~**5,500** lazily
imported submodules, so the watcher imports image processors for models we never
touch — many of which import `torchvision`, which this project deliberately does
not install.

**Fix: `.streamlit/config.toml`, committed.**

```toml
[server]
fileWatcherType = "none"   # the whole fix
[theme]
base = "light"             # viz.py components are light cards
[browser]
gatherUsageStats = false
[client]
toolbarMode = "minimal"
```

⚠️ **Do not "fix" this by installing torchvision.** ~2 GB, not a line of it
used, and the watcher would still walk 5,500 modules. The cost of the fix is
losing auto-reload on file save (press **R** in the browser) — which is an
improvement on stage, where nothing should rerun unless you ask.

`theme.base = "light"` matters too: everything `viz.py` produces is a light
card, so a presenter whose laptop is in dark mode would otherwise get white
cards on dark chrome.

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


### §6/§8 notebooks — what they establish

**§6 puts the naive-then-hardened arc on the MAIN path**, not in explore. Acts
1–4 build a working tool call with **nothing imported from `src/`** — a plain
function, a hand-written schema, a two-line regex over the model's raw output.
Act 5 breaks it with `4-127-3891`; Act 6 brings in the hardened tool. That
sequence is the repo's best answer to "but how did you actually do that".

**⚠️ §6's twins diverge at Act 3, structurally.** The Anthropic API does tool
parsing for you and returns `tool_use` objects, so the `_api` notebook **cannot**
show the raw `<tool_call>` text — the demystifying moment. It quotes the local
model's output instead and says why. Same precedent as §2's logits and §4's
refusal. Do not "fix" this.

**§8's twins are nearly identical, and that is a selling point.** Chunking,
embedding and search all run locally in both. Only generation differs. Say it
out loud: *your document index never leaves the building; only the retrieved
passages are sent.* That is a data-governance argument this audience cares
about and it falls out of the architecture for free.

**Measured, and quotable on stage** — the §8 Act 3 similarity demo against
Policy 4.3:

| Probe | Score |
|---|---|
| "What is the minimum lot size to split an R-1 parcel?" | 0.825 |
| **"Can I divide my property into two?"** — no shared keywords | **0.698** |
| "How small is a lot allowed to be before it can be subdivided?" | 0.748 |
| "What are the office's holiday hours?" | 0.432 |
| "Best pizza in Calgary" | 0.406 |

Act 1, same question, no tools: the local 4B invents *"0.25 acres, per the
County's Zoning Ordinance"*; Haiku refuses with *"while I'm roleplaying as an
experienced appraiser, I shouldn't…"*. Both notebooks report their own result.

### 🐛 `api_responder` and the system prompt (fixed)

The local models take a system prompt as an ordinary message in the list. **The
Anthropic API rejects that with a 400** — it must be a top-level `system`
parameter. `api_responder()` now lifts any `role: "system"` messages out of the
list and merges them into that parameter.

This never showed up until an `_api` notebook called `respond()` with a system
message, because nothing else exercised that path. **Any new responder must be
tested with a system prompt, not just a bare question.**

### 🐛 `st.html()` silently deletes every SVG (fixed)

Reported from the running §7 app: the map card drew its title, its parcel
count and its legend, and had **a hole where the map should be**. No error, no
warning, nothing in the console.

**Cause.** `st.html()` sanitises with DOMPurify configured
`USE_PROFILES: {html: true}`. That profile allows HTML elements and **strips
the SVG namespace entirely**. Confirmed in the shipped bundle:

```
streamlit/static/static/js/Html.*.js
    var s={USE_PROFILES:{html:!0},FORCE_BODY:!0}
```

The server sends the markup intact — verified, 75 KB with 1,200 `<circle>`
elements — and the browser throws the drawing away on arrival. The surrounding
`<div>` chrome survives, which is what makes it look like your own layout bug
rather than a sanitiser.

**It was hitting two things, not one.** The bug was reported against the map;
the speedometer was equally blank and had simply not been mentioned.

**Fix: `st.iframe(markup, height=N)`.** An iframe is not sanitised.
`st.components.v1.html()` works too but is deprecated in favour of `st.iframe`.

**Pass a real number for `height`, not `'content'`.** Auto-measure is tidier
right until it measures zero, and that failure looks identical to the bug
being fixed. Both §7 pictures cap their width (`viz.SPEEDOMETER_MAX_WIDTH`,
`viz.MAP_MAX_WIDTH`) so the height is arithmetic — see `viz.map_frame_height()`.

**Only `speedometer_html` and `parcel_map_html` emit SVG.** Every other
`viz.py` builder is divs and tables, so §2, §3/§4 and §6/§8 were never
affected. Check this before adding SVG to any shared builder.

**Notebooks are unaffected** — `IPython.display.HTML` sanitises nothing.
So a visual can work perfectly in Jupyter and vanish in Streamlit, which is
the reverse of the usual direction and worth remembering.

⚠️ **`AppTest` cannot catch this.** The element is present and correct in the
proto either way; the stripping happens in the browser. The test now asserts
the SVG rides in an `iframe` element rather than an `html` one, which is a
proxy — the only real check is looking at the running app.

### Model output does not go into an HTML string

The §7 answer panel originally interpolated the model's reply into a styled
`<div>` via `unsafe_allow_html=True`. Two problems, and the second is the one
that showed up immediately: the model **writes markdown** — given no ratio
tool it returns a full markdown table — which inside a `<div>` renders as raw
pipes and dashes. And model output is untrusted text that should not be
concatenated into markup in a repository other offices are told to copy.

Now `st.container(border=True)` + `st.markdown(answer)`. Renders tables
properly and nothing is interpolated.

### The §7 map: three rules, each learned by getting it wrong

`viz.parcel_map_html()` is hand-drawn SVG. The reasoning lives in its
docstring; these are the rules not to undo.

**1. No mapping library.** folium renders **ten** external resources and
`leaflet.js` is one of them — on dead conference wifi you get no map at all,
not a map missing its streets. It would also stop committed notebook outputs
drawing on GitHub, which is most of why we commit them.

**2. The frame fits the SELECTION, not the county, and nothing outlines a
neighborhood.** Parcels are sampled inside a disc, so convex hulls came out as
six near-perfect circles — which is not what a county looks like to anyone who
values property for a living. **The outline was the problem, not the dots:** a
scatter reads as a cluster, but a boundary drawn round it reads as a *claim*
about a shape, and that claim was false. Asked to map Riverside Heights, the
map now shows Riverside Heights filling the frame. Nothing is asserted about
where a neighborhood ends because nothing needs to be — the name is in the
title. Parcels in shot but not selected are drawn as small grey dots, so a
selection sits among its neighbours instead of floating.

**3. A grid, a scale bar and a north arrow — never a fake basemap.** Inventing
streets under invented parcels doubles the fiction and invites "what road is
that?". A plain grid says *coordinate space*.

⚠️ **The scale bar is not decoration.** Once the frame fits whatever was
selected, 36 sales across 800 m and 244 across 5 km fill it identically.
Measured: the bar adapts 100 m → 200 m → 500 m → 2 km across the demo's
selections. Remove it and two maps at a sevenfold difference in scale look the
same, to a room full of people who read scaled maps professionally.

A minimum span (~250 m) stops a single-parcel selection zooming to infinity.

**`NEIGHBORHOOD_RADIUS` still matters even though hulls are gone** — it is what
keeps neighborhoods contiguous rather than interleaved, which is what makes
"the sales in Riverside Heights" a coherent place. `verify_geography()` checks
it on every generator run.

### §7 notebooks — what they establish

**§7 is the first demo whose twins CONVERGE, and that is the finding.**
Sections 4 and 6 both ended with the hosted model and the local one behaving
differently, and we wrote that up as an honest trade-off. Here they agree,
because `ratio_study` computes the statistics either way — the model never
touches them. Verified mechanically: the two notebooks differ in **exactly one
code cell** (the model-construction cell) plus three markdown cells that point
at each other.

That convergence is the strongest argument in this repository for putting
domain knowledge in tools, and it is the natural set-up for §9: **a small
model is not worse at arithmetic it never does.**

### ⚠️ Act 4 — measured, and it changed the whole act

The plan was to have the model attempt a ratio study in SQL and fail loudly.
Measuring it produced something better and much more uncomfortable.

| Asked | Qwen3.5-4B | Haiku 4.5 |
|---|---|---|
| *"Run the ratio study by neighborhood"* (vague) | `AVG(assessed_value / sale_price)` → SQLite integer division → Riverside Heights reads **0.000** | **declines** — "the available tools don't include a ratio_study function… what would you like me to calculate?" |
| *"Using SQL, work out the average assessment ratio…"* (directive) | `AVG(ratio)` | `AVG(ratio)` — **byte-identical output** |

So the notebook uses the **directive** phrasing, and both twins print the same
six numbers. The lesson lands harder that way:

- The numbers are **plausible** — 0.79 to 0.98, exactly where a ratio belongs.
  Riverside Heights even comes out lowest, which is correct.
- They are still wrong: an **average, not a median** (Twin Pines Ranch is off
  by 0.036), and **no COD and no PRD at all**. Two of the three IAAO tests
  quietly cease to exist.
- **It is not a small-model failure.** Both models wrote the same query. It
  does not go away by spending more money, because it is a *domain-knowledge*
  failure — and domain knowledge belongs somewhere the model cannot get it
  wrong.

The integer-division result is preserved as a **quoted aside** rather than an
executed cell, because reproducing it needs the vague phrasing, which makes
Haiku decline and breaks the twin. *"That one you would catch. The table above
is the one you would not."*

⚠️ **Do not "fix" the twins by making Act 4 diverge.** The convergence is the
point, and it was measured rather than hoped for.

### `ratio.summarize()` lives in ratio.py, not in the tool

The one-sentence verdict (*"Meets all three IAAO standards on 210 sales…"*)
was originally `_summary()` inside `tools/review.py`. It moved to `ratio.py`
and `ratio_stats()` now fills in `result["summary"]` itself, so an attendee
running `ratio_stats()` on their own sales gets the same wording the agent
reports. Found because the notebook printed `result["summary"]` and raised
`KeyError` — the maths module had no such key.

### The explore section runs on data that is not Hill Valley

The "point it at your own county" cell uses twenty invented sales from a
made-up two-neighborhood county, and reproduces the whole section in
miniature: **together they meet every IAAO standard; split in two, Southside
sits at 0.82 of market.** Tuned deliberately — the first attempt had six sales,
which failed COD as "suspiciously tight" and reported NOT ENOUGH SALES for
both groups, teaching the wrong thing entirely.
