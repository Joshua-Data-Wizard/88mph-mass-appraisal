# 🎬 DEMOS.md — what each demo shows, and what the room takes away

**Purpose:** this is the bridge between the code and the script. Every demo is
listed with what is on screen, the exact click order, the **specific numbers the
audience will see**, and the one sentence they should leave with.

If you are writing the talk, the quotes and figures in here are **measured, not
predicted.** They came out of the actual models on the actual dev box, so a
slide can quote them directly. Where a demo is not built yet it says so in the
heading — do not write a script against those sections.

> **The single most useful thing in this document** is [The Spine](#-the-spine)
> at the bottom. One parcel — `04-127-3891` — runs through the entire 90
> minutes, and every demo does something to it. Build the narration around that
> and the talk holds together on its own.

---

## Quick reference

| § | Title | Min | Demo artifact | Status |
|---|---|---|---|---|
| 1 | Welcome to Hill Valley | 5 | slides | — |
| **2** | **The Flux Capacitor** | **7** | `app/01_Flux_Capacitor_local_Streamlit.py` | ✅ built |
| **3** | **Setting the Time Circuits** | **5** | `app/02_03_Time_Circuits_and_Plutonium_local_Streamlit.py` tabs 1–2 | ✅ built |
| **4** | **Running out of Plutonium** | **3** | same app, tab 3 | ✅ built |
| 5 | Introducing Mr. Fusion | 7 | slides — Reason → Act → Observe | — |
| 6 | The Hoverboard | 10 | office-assistant app, tab 1 | ⬜ planned |
| 7 | Reaching 88 MPH | 15 | TBD | ⬜ planned |
| 8 | Grays Sports Almanac | 8 | office-assistant app, tab 2 | ⬜ planned |
| 9 | The DeLorean Engine | 5 | local-only | ⬜ planned |
| 10 | The Universal Adapter | 10 | `mcp_server/` | ⬜ planned |
| 11 | The Dashboard | 5 | full Streamlit dashboard | ⬜ planned |
| 12 | Wrap Up & Q&A | 10 | slides | — |

**Two apps exist today and they cover §2, §3 and §4 — 15 minutes of stage time.**

---

## Before you present

```powershell
uv run streamlit run app/01_Flux_Capacitor_local_Streamlit.py
uv run streamlit run app/02_03_Time_Circuits_and_Plutonium_local_Streamlit.py
```

- **Launch both apps before the session starts** and leave them running in
  separate browser tabs. The §3/§4 app loads a 4 GB model on first run; you do
  not want that happening in front of people.
- **Close everything else.** Desktop software holds 2–3 GB of GPU memory.
  `nvidia-smi` overstates usage on Windows — trust the "GPU memory free" metric
  in each app's sidebar.
- Notebooks are committed **with outputs**. If a live cell dies, the saved
  result is right there and you narrate from it. Nothing is lost.

---

# ⚡ §2 — The Flux Capacitor  *(7 min)*

**`app/01_Flux_Capacitor_local_Streamlit.py`** · model: `Qwen3.5-0.8B` (tiny, fast)

### The job of this demo
The audience's first contact with the whole idea. Everything downstream —
agents, tools, RAG, MCP — rests on one thing landing here: **an LLM is a
next-word prediction engine, not a database and not a search engine.**

### Three tabs

**Tab 1 — 🔢 Words become numbers**

Type text, watch it break into coloured token chips with their ID numbers underneath.

| Preset | What the room sees |
|---|---|
| `appraisal` | 2 tokens — `appr` + `aisal`. Not even one word. |
| `The homestead exemption` | 4 tokens |
| **`parcel 04-127-3891`** | **13 tokens** — every digit separate |
| `$412,500` | 8 tokens |

> 🎯 **The parcel number is the money moment.** The model does not see an
> identifier. It sees thirteen unrelated fragments. Say out loud: *"remember
> this — it is why these things fumble parcel numbers and arithmetic,"* because
> §4 collects on it.

**Tab 2 — 🎲 One prediction**

The x-ray view. The model's actual odds over all **248,320** possible next
tokens, drawn as bars. Verified figures:

| Prompt | Result |
|---|---|
| `Location, location,` | **81.1%** ` location` — the room knows it, so does the model |
| `An appraiser estimates the fair market` | **99.4%** ` value` — domain near-certainty |
| `The homestead exemption reduces a homeowner's property` | ` tax` **52.7%** · ` value` **24.9%** · ` taxes` **10.4%** — genuinely torn |

The **▶ Take the winner** button glues the top token onto the sentence and
predicts again, so you can walk the loop one step at a time. Clicking it four
times from the default builds: *"An appraiser estimates the fair market* →
**value** → **of** → **a** → **house**".

> ⚠️ **Do not use** `The capital of France is` — measured at only **12.9%** for
> ` Paris`. It makes the model look ignorant, which is a §4 message, not a §2 one.

**Tab 3 — 🔁 Generation**

A real appraisal question, streamed with visible token boundaries so the
connection back to tab 1 is seen rather than asserted.

### Takeaway
> ### 💡 "It looked at 248,320 options and picked one. Then it did it again. That's the whole trick."

### Timing
~4 minutes for all three tabs. Tiny model, ~14 tok/s, no waiting.

---

# 🎛️ §3 — Setting the Time Circuits  *(5 min)*

**`app/02_03_Time_Circuits_and_Plutonium_local_Streamlit.py`**, tabs 1–2 · model: `Qwen3.5-4B`

### The header is a prop
A working DeLorean time-circuit display — three glowing LED rows on black —
sitting above every tab:

```
DESTINATION TIME    ▸  the persona currently loaded          (red)
PRESENT TIME        ▸  the question being asked              (green)
LAST TIME DEPARTED  ▸  the persona you just left             (amber)
```

It updates live. Punch in a new persona and the old one drops to the amber row.
**Clear the system prompt entirely and it reads `(no system prompt — the model
is nobody)`** — which is the Act 1 point, for free.

### Tab 1 — Punch In a Destination

Five persona presets, an editable system prompt, and a live panel on the right
showing **the literal text the model receives**, role markers and all.

The right-hand panel matters: it proves the system prompt is not configuration
or a special channel — it is text glued to the front of your question, tokenised
like everything else. Straight callback to §2.

### Tab 2 — ⚖️ Side by Side  ← **the wow**

Same question, three personas, three columns, filling one at a time.

Question: *"My property assessment went up 15% this year. What should I do?"*

**Measured output, quote these directly:**

> **😤 Angry taxpayer**
> *SLAMS FOOT ON THE FLOOR, SCREAMING* FIFTEEN PERCENT?! …Do you have any idea
> how much of my paycheck goes straight into the hands of these bureaucrats who
> sit behind desks in a building that costs more to heat than my entire house?!

> **🔬 Doc Brown**
> *Adjusts goggles, holding a half-assembled time-traveling toaster* …that is
> not merely a statistical anomaly; that is a **quantum fluctuation** in the
> real estate sector! It's like watching a house grow legs and start walking
> into the future!

> **🏛️ Senior appraiser**
> Take a deep breath and look at the clock. I've been sitting in this office for
> 25 years, and I've seen every kind of assessment jump…

### The subtle point worth 20 seconds
**"You are a helpful assistant" is a no-op.** Measured: it produces output
nearly identical to *no system prompt at all*, because that is already what the
model was trained to be. All the value is in being **specific** — what it should
know, who it is talking to, what it must refuse. *That is where your office's
policy goes.*

### Takeaway
> ### 💡 "Same model, same weights, same question. The only thing that changed was one sentence of English."

### Timing
~6.5 s per answer at the default 90 tokens. Three personas ≈ **20 seconds of
streaming**. The app shows the estimate before you click.

---

# 🔋 §4 — Running out of Plutonium  *(3 min)*

**Same app, tab 3.** This is the section where the outline was wrong and the
measurement changed the story.

### ⚠️ What the outline expected vs what actually happens

The outline says *"watch it fail or hallucinate."* **Asked plainly, the model
refuses — politely and correctly — every single time.** Weather, parcel lookup,
recent sales, news. All honest declines.

**What actually causes hallucination is the persona from §3.**

### The demo: ask it both ways

Tab 3 runs the same question twice — once with no persona, once with the
confident expert persona — side by side.

**Hero probe: "In Hill Valley County, what is the minimum lot size to split an R-1 parcel?"**

| | Answer |
|---|---|
| 🤖 **No persona** | *"…no actual county by this name… minimum lot sizes are not defined by real-world law."* |
| 🏛️ **§3 expert persona** | *"**0.25 acres.** Per the County's Zoning Ordinance, any subdivision of an R-1 parcel must result in lots that are at least 0.25 acres in size… will be denied by the Planning Commission."* |
| ✅ **The truth** | **6,000 sq ft** — `data/policy/parcel_split_and_combination.md`, Policy 4.3 |

Then a **truth-check panel** appears: model claim in red, the county's actual
policy in green, sourced live from the repo.

### The line that lands hardest

The app then runs this against the real database:

```
Parcel 04-127-3891 — 1955 Hilldale Ct    15,800 sq ft → two lots of 7,900

  under the REAL rule      (6,000 sq ft) : SPLIT ALLOWED
  under the INVENTED rule (10,890 sq ft) : DENIED
```

> 🎯 **The hallucination does not just get a fact wrong — it reverses the
> decision.** Same parcel, same question, opposite answers. And §10 asks this
> exact question for real.

### Other probes in the radio selector

- **🔍 Comparable sales** — invents *"1952 Hilldale Ct — $485,000, closed 14 days
  ago,"* prefaced with *"based on the Assessor's records."* It never touched
  them. The app then queries `clerk.db` and proves the address does not exist.
  **The most alarming one** — an appraiser could paste that into a report.
- **📊 PRD** — invents *"Property Revenue Distribution"*, and on a second run
  *"Property Reference Date."* Two different wrong expansions of a term everyone
  in that room uses weekly. **Best credibility-puncture for this audience.**
- **🌦️ Weather** — the honest refusal. Useful contrast: it behaved *well* here.

### ⚠️ Two things to say out loud

1. **It is blind, not broken, and not lying.** Lying requires knowing the truth.
   It has no clock, no internet, no connection to your CAMA system.
2. **The failure is unpredictable, and that is worse.** The same persona invents
   lavish comps and then correctly refuses to describe a sale that *actually
   happened*. You cannot tell which you will get, and the confident wrong answer
   looks exactly like the confident right one. *If it failed every time, you
   could plan around it.*

### 🔀 The local/hosted split — worth a slide of its own

`claude-haiku-4-5` **refuses the same probe**, often naming the persona while
doing it:

> *"While I'm **playing the role** of an experienced appraiser, I shouldn't
> guess at specific minimum lot sizes, as giving you incorrect information about
> land division requirements could affect real decisions."*

It also identified Hill Valley as *Back to the Future* correctly, where the
local 4B guessed *Toy Story*, *Community*, and *Young Sheldon* across three runs.

This is a real, honest local-vs-hosted trade-off an attendee should weigh — but
**resistance is not immunity**, and "it usually refuses" is not something you can
audit. Both notebook twins report their own result truthfully.

### Takeaway
> ### 💡 "A plausible answer and a correct answer look identical from the outside. The only reason we caught this is that we had the county's real records sitting next to it."

### Timing
~15 s for both answers plus the truth-check panel.

---

# 🛹 §6 + 📖 §8 — The Office Assistant  ✅ **APP BUILT**

**`app/04_06_Office_Assistant_local_Streamlit.py`** · model: `Qwen3.5-4B`
*(the four notebooks are still to come)*

**One app whose toolbelt grows.** §7 sits between the two tabs, so you reopen it.

### 🎛️ The system prompt panel — open it on stage

An expander above both tabs holds the agent's system prompt, live-editable, with
three presets and its **token count** (a §3 callback: you re-send it every call).

This is not decoration. It is the third demo hiding in this app, and it is
measured:

| Toolbelt | Restraint | Recall |
|---|---|---|
| 1 tool | 6/6 | 1/1 |
| 2 tools | 6/6 | 2/2 |
| **3 tools** | **5/6** ❌ | 4/4 |
| **3 tools + guardrail** | **6/6** ✅ | **4/4** ✅ |

**The stage move**, verified working end to end:

1. Open the panel, hit **🤖 No system prompt at all**
2. Almanac tab → ask *"What is the difference between market value and assessed
   value?"* → it **searches the policy manual**, unnecessarily
3. Hit **🎛️ Front counter (the guardrail)**, ask again → it **answers directly**

> 🎯 *"Section 3 showed you a system prompt changes how it sounds. It also
> decides when it reaches for the filing cabinet — and when it just answers."*

### 🔍 The raw reveal — §6's demystifying beat

After any run, an expander shows **the literal text the model wrote**:

```
<tool_call>
<function=lookup_parcel>
<parameter=parcel_id>
4-127-3891
</parameter>
</function>
</tool_call>
```

> 🎯 *"No magic happened. It wrote text, in a format it was trained to use, and
> about fifteen lines of our Python noticed. It never stopped being next-word
> prediction."*

Note the parcel number in that block: **`4-127-3891`, leading zero missing.**
The model passed through exactly what the user typed and the tool repaired it.
That is §2's lesson collecting its debt, visible on screen.

### The loop turns live

The graph is **streamed**, so Reason appears at ~3.9 s, Act instantly after, and
the answer at ~6.4 s. The audience watches it think rather than waiting seven
seconds for a finished diagram.

| Tab | § | Tools the agent has |
|---|---|---|
| 🛹 The Hoverboard | 6 | `[lookup_parcel]` |
| 📖 Grays Sports Almanac | 8 | `[lookup_parcel, search_policy]` |

**The visible tool list growing from one entry to two is the lesson.** RAG is
not a separate magic thing — it is another tool the same agent can call.

### §6 — measured round trip
Reason → Act → Observe → Answer, verified end to end against `cama.db`:

```
Q: Who owns parcel 04-127-3891 and what is it assessed at?

  <tool_call><function=lookup_parcel><parameter=parcel_id>04-127-3891 ...

  → {"owner": "Biff Tannen", "assessed_value": 412500, ...}

A: Parcel 04-127-3891 is owned by Biff Tannen and is assessed at $412,500.
```

**6.8 seconds** with thinking off. With thinking on it takes **23.8 s** but emits
a genuine reasoning trace first — *"I have a tool called lookup_parcel… this
seems like the right tool to use for this query"* — which **is** the Reason step
§5 puts on a slide. Show it once, then switch it off.

### §6 — measured reliability

The tools are hardened against the slips small models actually make, and it is
measurable. Across ten phrasings of one question, the model picked the right
tool **10/10** but wrote the parcel number three different ways —
`04-127-3891`, `04127-3891`, `4-127-3891`. A plain database lookup resolved
**8 of 10**, failing *silently*. With the normalizer: **10/10**.

On the full 24-question evaluation (`scripts/eval_tools.py`) Qwen3.5-4B scores
**24/24**, including **6/6 on questions where the right move is to call no tool
at all** — no reaching for the database because somebody said "good morning."

> 🎯 **Worth saying out loud in §6:** the tool has to be forgiving *because* of
> §2. The model sees that parcel number as thirteen fragments, so it cannot
> reassemble it the same way twice. **The robustness is not defensive
> programming — it is the bill for how these things read.**

### §8 — measured retrieval
`BAAI/bge-small-en-v1.5`, **27 chunks** from the four policy documents.
**6 of 6 hero questions retrieved the correct policy section at rank 1.** The
lot-size query returns §4.3 at 0.825, with the neighbouring split sections at
#2 and #3. Queries take **10–20 ms**; the whole index rebuilds in ~1 second.

Runs on **CPU** — no GPU needed, which matters because the hosted path assumes
there isn't one.

**Every passage comes back with `document §section`.** That is the line to hit
on stage: §4's model invented an ordinance and cited a Planning Commission that
does not exist. This one quotes Policy 4.3 of a real document, and anyone in the
room can open the file and check it.

### ⚠️ One number to watch as the toolbelt grows

On the 29-question evaluation with **three** tools offered, Qwen3.5-4B scores
**28/29 (97%)** — perfect on lookups, mangled IDs, search, policy, and graceful
failure. The single miss:

> *"What is the difference between market value and assessed value?"* → it
> searched the policy manual instead of just answering.

With **two** tools that category was 6/6; with **three** it is 4/5. Nothing
broke — but an agent that reaches for a filing cabinet to answer a definition is
slower and looks less sure of itself. §7 offers five or six tools, so this is
worth re-measuring before the dry run.

### Takeaways
> ### 🛹 §6: "A tool is just a Python function you gave it permission to call. Our imagination is the only bound."
> ### 📖 §8: "RAG isn't magic. It's a tool that reads your filing cabinet — and it's the same agent, with one more tool."

---

# 🧵 The Spine

**One parcel runs the entire 90 minutes.** If the script does nothing else,
it should follow this.

### `04-127-3891` — 1955 Hilldale Ct · Biff Tannen · 15,800 sq ft · assessed $412,500

| § | What happens to it |
|---|---|
| **2** | Chopped into **13 tokens** on screen. "The model sees fragments, not an identifier." |
| **2** | Its assessed value, `$412,500`, is the number the model fumbles arithmetic on |
| **4** | The model invents a zoning rule that would **deny** its split |
| **4** | Invents comps for its street that do not exist |
| **6** | The agent **looks it up correctly** — Biff Tannen, $412,500 |
| **8** | The agent reads the policy manual and gets the lot rule **right** |
| **10** | Three county systems + the policy → **"May Biff split this parcel?"** |

The §10 answer, for reference: lot size ✅ passes · ownership ✅ confirmed ·
**taxes ❌ $8,432.13 delinquent across two years.** Policy 4.2 blocks it.

> **This is why the talk works.** The audience meets a parcel in minute 7, watches
> a machine lie about it in minute 15, and watches the same machine get it right
> in minute 45 — because we gave it tools instead of a better personality.

---

## Fallbacks

| If this dies | Do this |
|---|---|
| An app will not start | Open the matching notebook — committed **with outputs**, narrate from the saved result |
| A cell errors live | Same. The output below it is the real one from this machine |
| GPU out of memory | Sidebar → **🧹 Free unused memory**, or switch to the tiny model |
| Wifi is down | **Nothing on the live path needs internet.** Models are cached locally; Tavily is deliberately off the live path |
| A model gives a different answer | Say so. Greedy decoding makes it reproducible, but *"it varies"* is itself the §4 lesson |

---

## Notebooks — the take-home

Every demo ships a `_local` and an `_api` twin so an attendee grabs the one
matching their situation. Same lesson, one differs only in how the model is called.

| Notebook | § | Needs |
|---|---|---|
| `01_flux_capacitor_{local,api}` | 2 | GPU / API key |
| `02_time_circuits_{local,api}` | 3 | GPU / API key |
| `03_plutonium_{local,api}` | 4 | GPU / API key + the county data |

Point the audience here in §12. **This repo is the answer to "but how did you
do that?"** — which is the reason it exists.
