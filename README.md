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

## ▶️ Start here — Demo 1 is live

Three minutes, and you will never think about these things the same way again.
Two ways to take it in:

```powershell
uv run streamlit run app/01_Flux_Capacitor_local_Streamlit.py   # 🖱️ click around
uv run jupyter lab                                              # 📓 read the notebook
```

| | |
|---|---|
| 🖱️ **[The app](app/01_Flux_Capacitor_local_Streamlit.py)** | Point-and-click. Type in *your* parcel number and watch what happens. Best for showing someone. |
| 📓 **[The notebook](notebooks/01_flux_capacitor_local.ipynb)** | The full explanation, top to bottom, with the code visible. Best for understanding it yourself. |
| ☁️ **[Hosted notebook](notebooks/01_flux_capacitor_api.ipynb)** | Same demo with no GPU required. |

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
| 2 | ⚡ **The Flux Capacitor** ✅<br/>[🖥️ local](notebooks/01_flux_capacitor_local.ipynb) · [☁️ hosted](notebooks/01_flux_capacitor_api.ipynb) | What an LLM *is* — watch next-word prediction happen live |
| 3 | 🕰️ **Setting the Time Circuits** | System prompts: telling the model who it is |
| 4 | ☢️ **Running Out of Plutonium** | Where LLMs fail, and why they confidently make things up |
| 5 | 🗑️ **Introducing Mr. Fusion** | Agents & the ReAct loop: Reason → Act → Observe |
| 6 | 🛹 **The Hoverboard** | Tools — *any* Python function you let the model call |
| 7 | 🏎️ **Reaching 88 MPH** | The mass valuation agent: parcels, ratio studies, maps |
| 8 | 📖 **Grays Sports Almanac** | RAG — answers from **your** office documents |
| 9 | 🔋 **The DeLorean Engine** | Small models: tiny, free, and surprisingly mighty |
| 10 | 🔌 **The Universal Adapter** | MCP — the "USB-C of AI" ⚡ |
| 11 | 📊 **The Dashboard** | Wrapping it all in something *anyone* in the office can use |

---

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

Four are wired up and swappable with a one-word change. VRAM figures are
**measured on an RTX 3080**, not marketing claims.

| Model | Size | License | VRAM | Best at |
|---|:---:|---|:---:|---|
| `Qwen/Qwen3.5-0.8B` | 0.8B | Apache 2.0 | **1.44 GB** | Demo 1 — small enough to run with no GPU at all |
| `Qwen/Qwen3.5-4B` ⭐ | 4B | Apache 2.0 | **2.98 GB** | The workhorse — **1st of 13** local models for tool calling |
| `HuggingFaceTB/SmolLM3-3B` | 3B | Apache 2.0 | *untested* | The "look how little this needs" story |
| `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | 4B | NVIDIA Open | *untested* | Step-by-step sequential reasoning |

All four are **ungated** — no license form to sign before downloading. 🎉

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
✅ notebooks/
     01_flux_capacitor_local.ipynb   ⚡ §2, on your own GPU
     01_flux_capacitor_api.ipynb     ☁️ §2, hosted
✅ app/
     01_Flux_Capacitor_local_Streamlit.py  🖱️ §2 as a clickable app
✅ scripts/
     check_setup.py          🔍 run this first — can my machine do this?
✅ src/delorean/             🚗 the shared engine every notebook imports
     models.py                  which model, and why
     llm.py                     loading models, local and hosted
     viz.py                     the visuals — token chips, probability bars
✅ Presentation_Material/    🎤 the session outline (source of truth for scope)

⬜ data/                     🏘️ synthetic parcels, sales, office documents
⬜ mcp_server/               🔌 §10 — a local MCP server
⬜ app/dashboard.py          📊 §11 — the full Streamlit dashboard
```

### ▶️ Running the app

```powershell
uv run streamlit run app/01_Flux_Capacitor_local_Streamlit.py
```

Opens at `localhost:8501`. Three tabs — tokens, prediction, generation — with a
model picker in the sidebar so you can swap engines mid-demo.

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
| ✅ | Model registry — 4 models, swappable in one word 🤖 |
| ✅ | **§2 — ⚡ The Flux Capacitor: what an LLM actually is** *(both roads)* |
| ✅ | §2 as an interactive Streamlit app 🖱️ |
| ⬜ | Synthetic parcel and sales data 🏘️ |
| ⬜ | §3–4 — prompting, and where it breaks 🕰️ |
| ⬜ | §6–7 — tools, and the mass valuation agent 🏎️ |
| ⬜ | §8 — RAG over office documents 📖 |
| ⬜ | §9 — small models 🔋 |
| ⬜ | §10 — Model Context Protocol 🔌 |
| ⬜ | §11 — the dashboard 📊 |

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
