<div align="center">

# ⚡ 88mph-mass-appraisal 🚗💨

### *Agentic AI, LLMs, and RAG for the Property Appraisal Office*

**"Where we're going, we don't need spreadsheets."**

<br>

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-managed-DE5FE9?style=for-the-badge&logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13_+_cu130-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agents-1C3C3C?style=for-the-badge&logo=langgraph&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Chroma](https://img.shields.io/badge/Chroma-RAG-FF6F61?style=for-the-badge)](https://www.trychroma.com/)
[![MCP](https://img.shields.io/badge/MCP-Model_Context_Protocol-6E56CF?style=for-the-badge)](https://modelcontextprotocol.io/)

[![Claude](https://img.shields.io/badge/Claude-Hosted_Option-D97757?style=for-the-badge&logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)](LICENSE)

<br>

[![Powered by](https://img.shields.io/badge/powered_by-1.21_gigabytes-FFC300?style=flat-square)](#)
[![Flux Capacitor](https://img.shields.io/badge/flux_capacitor-fluxing-00C2FF?style=flat-square)](#)
[![Roads](https://img.shields.io/badge/roads-not_required-lightgrey?style=flat-square)](#)

</div>

---

## 🎬 What is this?

> *"Roads? Where we're going, we don't need roads."* — Dr. Emmett Brown

This repository is for helping those of us in **Mass Appraisal** have a better
understanding of what Agentic LLMs are and how they can help us. 🏘️

It's the companion code for the IAAO Calgary session
**"Search, Extract & Value: The Lightning Fast Assistant"** — and it exists because
too many conference talks show you something amazing 🤩 and never show you how it
was done. Everything demonstrated on stage is in here, commented, and meant to be
taken back to your office. 📦

**You don't need to be a programmer to try it.** 🙌
**You don't need to send a single parcel record to the cloud.** 🔒

---

## 🚗 Fire up the DeLorean

Three commands. That's the whole setup. ⏱️

```powershell
uv sync                                    # ⚙️  build the environment
uv run python scripts/check_setup.py       # 🔍  will this run on my machine?
uv run jupyter lab                         # 📓  open the notebooks
```

> 🆕 **Never used `uv` before?** Start here → **[Guides/UV_Readme.md](Guides/UV_Readme.md)**
> It covers installing it, exactly what lands on your computer, and how to remove
> every trace of it. No Python knowledge assumed. 👍

`check_setup.py` tells you plainly whether your machine can run the local models —
and exactly what to do instead if it can't. No guessing. ✅

---

## ▶️ Start here — all nine demos are live

Every one of them runs **on your own machine**. Pick a road and go:

```powershell
uv run streamlit run app/01_Flux_Capacitor_local_Streamlit.py                  # §2
uv run streamlit run app/02_03_Time_Circuits_and_Plutonium_local_Streamlit.py  # §3 + §4
uv run streamlit run app/04_06_Office_Assistant_local_Streamlit.py             # §6 + §8
uv run streamlit run app/05_88mph_Valuation_Review_local_Streamlit.py          # §7
uv run streamlit run app/08_09_Universal_Adapter_local_Streamlit.py            # §10
uv run jupyter lab                                                             # 📓 15 notebooks
```

| 🖱️ The apps | What you click |
|---|---|
| **[⚡ The Flux Capacitor](app/01_Flux_Capacitor_local_Streamlit.py)** | Type in *your* parcel number and watch it shatter into tokens. Then watch the model predict one word at a time. |
| **[🎛️ Time Circuits & Plutonium](app/02_03_Time_Circuits_and_Plutonium_local_Streamlit.py)** | Punch a personality into the dashboard, then ask it something it cannot know and watch it invent an ordinance. |
| **[🛹 The Office Assistant](app/04_06_Office_Assistant_local_Streamlit.py)** | Hand it a tool and watch it stop guessing — then hand it your policy manual. |
| **[🏎️ 88 MPH — Valuation Review](app/05_88mph_Valuation_Review_local_Streamlit.py)** | Ask for a ratio study in plain English. It writes the SQL, runs the IAAO statistics and draws the map. |
| **[🔌 The Universal Adapter](app/08_09_Universal_Adapter_local_Streamlit.py)** | One question, three county offices, no integration project. |

---

## 🏆 The two people remember

If you only have ten minutes, run these two. Every figure below came off the
machine — nothing here is illustrative.

### 🏎️ *"How did the county do this year?"*

Hill Valley County **passes every IAAO standard.** Median 0.927, COD 12.22,
PRD 1.027. That is the number that goes in the annual report, and every word
of it is true.

Then you ask for it **by neighborhood**, and the assistant writes its own SQL:

| Neighborhood | n | Median | COD | PRD | |
|---|---:|---:|---:|---:|:---:|
| Courthouse Square | 30 | 0.926 | **✕ 22.08** | 1.028 | ✕ |
| Hilldale | 50 | 0.921 | 13.63 | **✕ 1.043** | ✕ |
| Riverside Heights | 32 | **✕ 0.809** | 6.21 | 1.001 | ✕ |
| Eastwood Ravine | 27 | 0.984 | 6.06 | 1.001 | ✓ |
| Lyon Estates | 42 | 0.948 | 6.04 | 1.001 | ✓ |
| Twin Pines Ranch | 29 | 0.929 | 9.33 | 1.002 | ✓ |
| **COUNTYWIDE** | **210** | **0.927** | **12.22** | **1.027** | **✓** |

**The red runs diagonally.** Three neighborhoods fail, and *no two fail the
same test* — one is assessed too low, one is wildly inconsistent, one is
quietly unfair to cheaper homes. Three different problems, three different
afternoons of work, and **the countywide figure shows none of them.**

Eleven seconds. On a laptop. Nothing left the building.

> 🔑 The agent decides **what to ask**. The tool decides **what is correct** —
> the IAAO maths is ordinary Python the model never touches.

### 🔌 *"I want to split my lot and sell half. Can I?"*

Nobody in the building can answer that. **The lot size** is with the Appraiser,
**ownership** with the Clerk of Courts, **the tax balance** with the Tax
Collector, and **the rule** in a policy manual. Three elected officials, three
vendors, three databases that have never spoken to each other.

Three MCP servers start up. The assistant asks each one what it can do — **we
wrote no tool code for any of them** — and then:

```
🏛️ Appraiser        15,800 sq ft → two 7,900 lots, over the 6,000 minimum   ✓
📜 Clerk of Courts   Warranty Deed 2025-000417, Biff Tannen is the owner     ✓
💰 Tax Collector     $8,432.13 outstanding across two tax years              ✕
📖 Policy 4.2        "no split while any ad valorem tax remains outstanding"
```

> **No, Biff Tannen may not split parcel 04-127-3891** because the Tax
> Collector reports a total outstanding balance of **$8,432.13**, which bars
> the split under **policy §4.2**…

Twenty-six seconds, for a question that today takes three phone calls and a
week. **And it checks all three even after one fails** — because somebody told
only the first reason their application was refused has to come back again to
hear the rest.

Then ask it the harder one: ***"is our sales file still true?"*** The Clerk
corrected a deed after your feed was pulled — and when you drop that one bad
sale, the neighborhood turns out to fail **a second test the stale file was
hiding.** 😬

---

**New here? Start with the Flux Capacitor.** Three minutes, and you will never
think about these things the same way again.

Everyone has watched a chatbot type an answer. Almost nobody has been shown **what
is actually happening** while it does. So we open it up:

**🔢 Your words stop being words.**

```
parcel 04-127-3891   →   13 tokens

  parcel │ · │ 0 │ 4 │ - │ 1 │ 2 │ 7 │ - │ 3 │ 8 │ 9 │ 1
```

Every digit is a separate piece. The model never sees a parcel number — it sees
thirteen unrelated fragments. *That one picture explains a whole category of
failure.*

**🎲 Then we reach inside mid-sentence** and pull out the real odds it assigns to
every one of its **248,320** possible next tokens:

```
An appraiser estimates the fair market ▮

  ·value  ████████████████████████████  99.4%
  ·price  ▏                              0.2%
  ·sale   ▏                              0.0%
```

Then a question where it is genuinely torn — `tax` **52.7%** / `value` **24.9%** /
`taxes` **10.4%** — and you can *see* the doubt.

**😬 And then the part that should worry you.** We ask it to do arithmetic:

```
CORRECT | 1,247 + 3,891   | model: 5,138       | truth: 5,138
WRONG   | 1,247 x 3,891   | model: 4,860,057   | truth: 4,852,077
WRONG   | 17% of $412,500 | model: 68,125      | truth: 70,125
```

Off by **$2,000** — and it looks completely reasonable. It didn't say *"I don't
know."* It handed back a confident, correctly-shaped number that passes any
eyeball test. **Now imagine that on a notice of value.** 📬

> 🛠️ The fix isn't a smarter model. It's a calculator — and that's what a *tool*
> is, which is exactly where §6 picks up.

**🖱️ And one thing only the app can do.** In the prediction tab there's a
**▶ Take the winner** button. Press it and the winning token is glued onto the
sentence and the model predicts *again* — so you walk the generation loop one
step at a time and watch the odds shift under you. Press it four times starting
from `Location, location,` and you have built a sentence by hand, the same way
the model does it a few hundred times a second.

---

## 🗺️ The Tour

Every demo maps to a numbered section of the talk, so if something caught your eye
on stage, you can find it in about ten seconds. 👀

| § | The Bit | ⚙️ What You Actually Learn |
|:---:|---|---|
| 2 | ⚡ **The Flux Capacitor** ✅<br/>[🖥️ local](notebooks/01_flux_capacitor_local.ipynb) · [☁️ hosted](notebooks/01_flux_capacitor_api.ipynb) · [🖱️ app](app/01_Flux_Capacitor_local_Streamlit.py) | What an LLM *is* — watch next-word prediction happen live |
| 3 | 🎛️ **Setting the Time Circuits** ✅<br/>[🖥️ local](notebooks/02_time_circuits_local.ipynb) · [☁️ hosted](notebooks/02_time_circuits_api.ipynb) · [🖱️ app ⑂](app/02_03_Time_Circuits_and_Plutonium_local_Streamlit.py) | The system prompt — one sentence that changes everything |
| 4 | 🔋 **Running out of Plutonium** ✅<br/>[🖥️ local](notebooks/03_plutonium_local.ipynb) · [☁️ hosted](notebooks/03_plutonium_api.ipynb) · [🖱️ app ⑂](app/02_03_Time_Circuits_and_Plutonium_local_Streamlit.py) | Where it goes blind — and invents an ordinance to cover it |
| 5 | 🗑️ **Introducing Mr. Fusion** | Agents & the ReAct loop: Reason → Act → Observe *(slides)* |
| 6 | 🛹 **The Hoverboard** ✅<br/>[🖥️ local](notebooks/04_hoverboard_local.ipynb) · [☁️ hosted](notebooks/04_hoverboard_api.ipynb) · [🖱️ app ⑂](app/04_06_Office_Assistant_local_Streamlit.py) | A tool is a Python function *you* wrote — build one from nothing |
| 7 | 🏎️ **Reaching 88 MPH** ✅<br/>[🖥️ local](notebooks/05_88mph_local.ipynb) · [☁️ hosted](notebooks/05_88mph_api.ipynb) · [🖱️ app](app/05_88mph_Valuation_Review_local_Streamlit.py) | The valuation review agent — SQL it writes itself, IAAO ratio studies, maps |
| 8 | 📖 **Grays Sports Almanac** ✅<br/>[🖥️ local](notebooks/06_almanac_local.ipynb) · [☁️ hosted](notebooks/06_almanac_api.ipynb) · [🖱️ app ⑂](app/04_06_Office_Assistant_local_Streamlit.py) | RAG in plain English — and it's just another tool |
| 9 | ⚙️ **The DeLorean Engine** ✅<br/>[🖥️ local](notebooks/07_delorean_smol.ipynb) | Small models — a 3B model runs §7 on **1.86 GB** of weights |
| 10 | 🔌 **The Universal Adapter** ✅<br/>[🖥️ local](notebooks/08_universal_adapter_local.ipynb) · [☁️ hosted](notebooks/08_universal_adapter_api.ipynb) · [🖱️ app](app/08_09_Universal_Adapter_local_Streamlit.py) · [🔌 servers](mcp_server/) | MCP — three county offices, one question, no integration project |
| 11 | 📊 **Putting a Bow On It** ✅ | Not another app — the take-home. How you actually start on Monday |

> ### ⑂ Three apps each cover two sections — on purpose.
>
> **§3 + §4** share one app because they are cause and effect: the system prompt
> that makes the model sound like an expert is the same one that makes it invent
> an ordinance. Splitting them would throw away the twist.
>
> **§6 + §8** share one app because the whole point of §8 is that RAG is **not**
> special — it is one more tool on the same agent. You watch the toolbelt grow
> from one chip to two. §7 sits between them in the talk, so you reopen it.
>
> **§10** is one app carrying **two** live demos — the cross-office question
> and the sync check — because they run on the same three servers and the
> second one only lands once you have seen the first.

## 🛣️ Two Roads

Every demo ships in **two versions**, so you can take whichever road fits your office:

| 🚦 Version | Runs on | 🎯 Use when |
|---|---|---|
| 🖥️ `*_local.ipynb` | **Your own GPU** | Data cannot leave the building |
| ☁️ `*_api.ipynb` | **A hosted model** | No GPU budget, or you want to try it in five minutes |

Same demo. Same tools. Same results. **Only the engine changes.** 🔧

> ℹ️ One honest exception: the hosted twin of Demo 1 **cannot** show you the
> probability bars. No provider returns those numbers. That limitation is the
> whole argument for running models yourself, so we say so plainly in the
> notebook instead of quietly skipping it.

---

## 🤖 Which Model?

Six are wired up and swappable with a one-word change. Every figure below is
**measured on an RTX 3080**, not a marketing claim, and VRAM is **peak during
real agent work** — not the weight footprint, which is the smaller, friendlier
number and the wrong one to plan with.

| Model | Size | License | Peak VRAM | Best at |
|---|:---:|---|:---:|---|
| `Qwen/Qwen3.5-0.8B` | 0.8B | Apache 2.0 | **1.44 GB** | Demo 1 — small enough to run with no GPU at all |
| `Qwen/Qwen3.5-4B` ⭐ | 4B | Apache 2.0 | **4.00 GB** | The workhorse — **1st of 13** local models for tool calling |
| `HuggingFaceTB/SmolLM3-3B` | 3B | Apache 2.0 | **2.77 GB** | The "look how little this needs" story — **runs all of §7** |
| 🪟 `microsoft/Phi-4-mini-instruct` | 3.8B | **MIT** | **4.15 GB** | A name your IT department already trusts |
| 🔷 `google/gemma-4-E2B-it` | 2.3B | Apache 2.0 | 6.55 GB | The fastest here — 5.7s a question |
| `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | 4B | NVIDIA Open | ⚠️ **15.6 GB** | ⚠️ **Not offered in the apps on Windows** — see below |

All six are **ungated** — no license form to sign before downloading. 🎉

### 📊 Measured on the same ten questions and the same three tools

`uv run python scripts/compare_models.py` — **recall** = called the right tool
when it needed one; **restraint** = answered directly when it should have.

| Model | Recall | Restraint | SQL that ran | s/question |
|---|:---:|:---:|:---:|:---:|
| Qwen3.5 4B ⭐ | **6/6** | 4/4 | **6/6** | 8.7 |
| SmolLM3 3B | **6/6** | 4/4 | **6/6** | 6.8 |
| Microsoft Phi-4 mini | 5/6 | 4/4 | 7/10 | 7.0 |
| Google Gemma 4 E2B | 5/6 | 4/4 | **5/5** | **5.7** |

**Every one of them answers "good morning" without hitting the database.** That
is the number to watch as you add tools — see [CLAUDE.md](CLAUDE.md) §13.

### 🔒 "I've never heard of these companies"

A fair question, and the honest answer is better than a brand name.

**Two of these are from Microsoft and Google**, if that helps — and it
legitimately might, when you are the one explaining this to a county
commission.

**But the stronger answer is that it does not matter.** A model is a file of
numbers sitting on your disk. It makes **no network calls** — pull the ethernet
cable out and every demo in this repository keeps working. It cannot phone
home, because there is no code in it to phone home *with*; it is weights, and
weights do not execute. Whoever trained the file has no more reach into your
county than the author of a PDF you downloaded.

What you should actually audit is the **code around** the model — this
repository — and that is why every line of it is here for you to read.

If it still helps to start with a name you know: `phi` and `gemma` are one word
each.

```python
from delorean import llm
model, tok = llm.load_local("phi")     # or "gemma", "qwen", "smol"
```

### ⚠️ Why Nemotron is documented here but not in the app dropdowns

It is a **hybrid Mamba model**, and its fast path needs the `mamba-ssm`,
`causal-conv1d` and `triton` CUDA kernels — none of which build cleanly on
Windows. Without them it falls back to a PyTorch scan costing roughly
**12 MB of VRAM per prompt token**, about 100× a normal transformer:

| prompt tokens | 256 | 512 | 768 | 1024 | 1280 |
|---|---|---|---|---|---|
| peak VRAM | 6.25 GB | 9.35 GB | 12.47 GB | 15.57 GB | **OOM** |

Demo 5's prompt is ~1,300 tokens and its peak sits past where Nemotron survives, so it never finishes. **On Linux with those
kernels installed it should be fine** — this is a platform limit, not a
verdict on the model, so it stays in the registry and stays loadable by key
(`llm.load_local("nemotron")`). The apps just do not offer it, because a
dropdown is a promise.

[`07_delorean_smol.ipynb`](notebooks/07_delorean_smol.ipynb) tells the whole
story. **"Check the architecture has working kernels on your OS"** is the
first question to ask about any model, and no leaderboard answers it.

```python
from delorean import models
models.compare()          # full strengths and weaknesses for each
```

---

## 🔌 What You Need

| | 🖥️ Local | ☁️ Hosted |
|---|---|---|
| **GPU** | 8 GB VRAM (NVIDIA, Turing or newer) | none 🎉 |
| **Disk** | ~15 GB *(this is heavy)* 📦 | ~3 GB |
| **Accounts** | [Hugging Face](https://huggingface.co/settings/tokens) 🤗 (free) | Hugging Face + an API key 🔑 |
| **Internet** | Only to download models once | Yes |
| **Cost** | $0 forever | ~**$1/day** at 500 questions on the cheapest model |

> 💡 **Demo 1 runs without a GPU** even in the local version — the 0.8B model is
> small enough for an ordinary laptop. You only need the graphics card once we
> get to the agent demos in §6.

Copy `.env.example` → `.env` and fill in your keys.

> ### ⚠️ Great Scott!
> `.env` holds real API keys and is **git-ignored on purpose**. Never commit it.
> Never paste it in a chat. Never email it. 🔐

---

## 📂 Where Everything Is

```
✅ Guides/
     UV_Readme.md            📘 never used uv? start here
✅ notebooks/                📓 one pair per demo — local and hosted
     01_flux_capacitor_{local,api}.ipynb   ⚡ §2  what an LLM is
     02_time_circuits_{local,api}.ipynb    🎛️ §3  system prompts
     03_plutonium_{local,api}.ipynb        🔋 §4  where it goes blind
     04_hoverboard_{local,api}.ipynb       🛹 §6  tools, built from nothing
     05_88mph_{local,api}.ipynb            🏎️ §7  ratio studies, SQL, maps
     06_almanac_{local,api}.ipynb          📖 §8  RAG as one more tool
     07_delorean_smol.ipynb                ⚙️ §9  the same demo on a 3B model
     08_universal_adapter_{local,api}.ipynb 🔌 §10 three offices over MCP
✅ app/                      🖱️ clickable versions, for showing someone
     01_Flux_Capacitor_local_Streamlit.py                  ⚡ §2
     02_03_Time_Circuits_and_Plutonium_local_Streamlit.py  🎛️ §3 + §4
     04_06_Office_Assistant_local_Streamlit.py             🛹 §6 + §8
     05_88mph_Valuation_Review_local_Streamlit.py          🏎️ §7
     08_09_Universal_Adapter_local_Streamlit.py            🔌 §10
✅ scripts/
     check_setup.py               🔍 run this first — can my machine do this?
     generate_hill_valley_data.py 🏘️ rebuilds data/ from scratch, seeded
     test_tools.py                🛡️ can the tools survive a sloppy model?
     eval_tools.py                📊 does the model actually use them well?
     compare_models.py            🤖 every model, same questions, same tools
     test_app_88mph.py            🖱️ the §7 app, clicked
     test_app_office.py           🖱️ the §6/§8 app, clicked
     test_app_adapter.py          🖱️ the §10 app, clicked
✅ src/delorean/             🚗 the shared engine every notebook imports
     models.py                  which model, and why
     llm.py                     loading models, local and hosted
     viz.py                     the visuals — tokens, time circuits, the agent loop
     data.py                    reaching the three county systems
     agent.py                   Reason → Act → Observe, hand-built
     prompts.py                 the system prompts, and why each word is there
     ratio.py                   the IAAO maths — no model anywhere near it
     rag.py                     chunk, embed, search — §8's machinery
     offices.py                 §10 — talking to three MCP servers
     tools/                     what the agent is allowed to do
✅ data/                     🏘️ Hill Valley County — 100% synthetic
     cama.db                    🏛️ Appraiser's office · 1,200 parcels
     clerk.db                   📜 Clerk of Courts · 244 recorded deeds
     tax.db                     💵 Tax Collector · 1,200 tax accounts
     csv/                       the same data, readable right on GitHub
     policy/                    📄 4 office policy docs — §8's RAG corpus
✅ mcp_server/               🔌 §10 — three offices, three programs
     appraiser_server.py        🏛️ the roll — lot size, class, value
     clerk_server.py            📜 recorded deeds, and what has been corrected
     tax_server.py              💵 the ledger — what is still owed
✅ .streamlit/config.toml    ⚙️ shared app settings (see below)
✅ DEMOS.md                  🎬 what each demo shows + the takeaway line
✅ CLAUDE.md                 🤖 the working agreement, if you build on this
✅ FINDINGS.md               🔬 every measurement, and every bug that taught us one
✅ Presentation_Material/    🎤 the session outline (source of truth for scope)
```

> ### 🏘️ About that data — [read data/README.md](data/README.md)
>
> Three **separate** databases with no foreign keys between them, because that
> is the real situation in a county: three elected offices, three vendors, and
> no integration anyone ever funded. You can't join them. That's the point, and
> it's what §10 is about.
>
> **The flaws in it are deliberate.** Three of the six neighborhoods fail a
> different IAAO test — one on level, one on uniformity, one on equity — while
> the countywide numbers pass every standard. An office reporting only the
> county figure would call it a good year. That gap is what §7's agent finds.
>
> Everything is invented. No real parcel, owner, address, or sale is in this
> repository.

### ▶️ Running the apps

```powershell
uv run streamlit run app/01_Flux_Capacitor_local_Streamlit.py                  # §2
uv run streamlit run app/02_03_Time_Circuits_and_Plutonium_local_Streamlit.py  # §3 + §4
uv run streamlit run app/04_06_Office_Assistant_local_Streamlit.py             # §6 + §8
```

Each opens at `localhost:8501` with a model picker in the sidebar, so you can
swap engines mid-demo. The Office Assistant also reports **which device the
weights are actually on**, so nobody has to take "it runs locally" on faith.

> ⚙️ **Settings are committed** in `.streamlit/config.toml`. The important one
> turns Streamlit's file watcher off: it otherwise walks ~5,500 `transformers`
> submodules on every click and prints a wall of harmless `torchvision` errors.
> The trade is that editing a `.py` no longer auto-reloads — press **R** in the
> browser.

> 🛡️ **All demo data is synthetic.** No real parcels, owners, or addresses live in
> this repository — and none should ever be committed to it.

### ▶️ Running a notebook

```powershell
uv run jupyter lab        # opens in your browser at localhost:8888
```

Then open `notebooks/` and pick a file. You can also open the `.ipynb` directly
in **VS Code** — just select the `.venv` interpreter when it asks.

> 🎤 **Presenting live?** The notebooks ship **with their outputs saved**, so they
> read properly on GitHub and give you a fallback if a cell dies on stage. Before
> you present, use **Kernel → Restart Kernel and Clear All Outputs** so the room
> watches it happen for real — then *don't save*, or `git checkout` the file
> afterwards to get your fallback back. **F11** for fullscreen, **Ctrl +** to zoom.

---

## 📅 Destination Time

Built in the open, in the order the session presents it. 🔨

| | Milestone |
|:---:|---|
| ✅ | Environment, dependency lockfile, and setup check |
| ✅ | Guides 📘 |
| ✅ | Model registry — **6 models** incl. Microsoft & Google, swappable in one word 🤖 |
| ✅ | **§2 — ⚡ The Flux Capacitor: what an LLM actually is** *(both roads)* |
| ✅ | §2 as an interactive Streamlit app 🖱️ |
| ✅ | Synthetic county data — 3 office systems + policy corpus 🏘️ |
| ✅ | **§3–4 — prompting, and where it breaks** 🕰️ *(4 notebooks + shared app)* |
| ✅ | **§6 + §8 — tools, and RAG as one more tool** 🛹📖 *(4 notebooks + shared app)* |
| ✅ | **§7 — 🏎️ the valuation review agent: SQL, ratio studies, maps** *(2 notebooks + app)* |
| ✅ | **§9 — ⚙️ small models: a 3B runs §7 on 1.86 GB** *(notebook, local only)* |
| ✅ | **§10 — 🔌 the Universal Adapter: three MCP servers, two demos** *(2 notebooks + app + 3 servers)* |
| ✅ | §11 — resolved as the take-home close, no new artifact needed 📊 |
| 🏁 | **All nine live demos are built.** What is left is rehearsal. |

---

## 🧯 Something Not Working?

The **[uv guide](Guides/UV_Readme.md)** has a troubleshooting section covering every
problem we actually hit while building this — cloud-sync folders breaking installs,
half-written packages, VS Code doing surprising things, and the ever-popular
`"Python was not found"` (which is harmless 😄).

When in doubt, the nuclear option is always safe:

```powershell
Remove-Item -Recurse -Force .venv
uv sync
```

---

## 📜 License

**MIT** — see [LICENSE](LICENSE). Take it, fork it, break it, ship it at your office. 🚀

---

<div align="center">

### 🕰️ *"Your future hasn't been written yet. No one's has.* 🕰️
### *Your future is whatever you make it. So make it a good one."*

**— Dr. Emmett Brown**

<br>

🚗💨 ⚡ 🏘️

</div>
