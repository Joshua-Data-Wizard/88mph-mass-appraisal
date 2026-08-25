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
| **6** | **The Hoverboard** | **10** | `app/04_06_Office_Assistant_local_Streamlit.py` tab 1 | ✅ built |
| **7** | **Reaching 88 MPH** | **15** | `app/05_88mph_Valuation_Review_local_Streamlit.py` | ✅ built |
| **8** | **Grays Sports Almanac** | **8** | same app, tab 2 | ✅ built |
| **9** | **The DeLorean Engine** | **5** | slide reveal + `notebooks/07_delorean_smol.ipynb` | ✅ built |
| **10** | **The Universal Adapter** | **10** | `app/08_09_Universal_Adapter_local_Streamlit.py` + `mcp_server/` | ✅ built |
| **11** | **Putting a Bow On It** | **5** | no new artifact — the take-home close | ✅ decided |
| 12 | Wrap Up & Q&A | 10 | slides | — |

**Five apps cover §2, §3, §4, §6, §7, §8 and §10 — 58 minutes of stage time, and every demo in them is measured.**
**All nine live demos are built.** §11 needs no artifact; see its entry below.

---

## Before you present

```powershell
uv run streamlit run app/01_Flux_Capacitor_local_Streamlit.py
uv run streamlit run app/02_03_Time_Circuits_and_Plutonium_local_Streamlit.py
uv run streamlit run app/04_06_Office_Assistant_local_Streamlit.py
uv run streamlit run app/05_88mph_Valuation_Review_local_Streamlit.py
uv run streamlit run app/08_09_Universal_Adapter_local_Streamlit.py
```

⚠️ **The last two cannot run at the same time.** Each loads its own copy of the
4B model, and two will not fit on an 8 GB card. Stop one before starting the
other — a cold start is about 15 seconds, so do it during the intro slides for
whichever section is next. The §2 and §3/§4 apps are small enough to leave up.

### 🤖 The model picker — and a live moment worth having ready

Every app's sidebar now offers **five** models. The default is Qwen3.5-4B
everywhere and you should not need to touch it — but the list is there for
one specific question, which somebody in the room will ask:

> *"Do I have to run a model from a company I've never heard of?"*

**Change the dropdown to `Microsoft Phi-4 mini` or `Google Gemma 4 E2B`,
and ask the question again.** It works. That is a far better answer than a
slide, and it costs about fifteen seconds of model load.

| Model | Peak VRAM | Note |
|---|---|---|
| Qwen3.5 4B ⭐ | 4.00 GB | the default, 6/6 on tools |
| SmolLM3 3B | **2.77 GB** | smallest — this is §9's reveal |
| Microsoft Phi-4 mini | 4.15 GB | MIT licence |
| Google Gemma 4 E2B | **6.55 GB** | fastest, but the heaviest — see below |
| Qwen3.5 0.8B | 1.44 GB | §2 only; cannot drive tools |

⚠️ **Gemma needs 6.55 GB.** Fine on the presenter machine; on an 8 GB
laptop it leaves about 1.5 GB, so close everything else first. And
**NVIDIA Nemotron is deliberately not in the list** — it runs out of memory
on Windows. The app says so rather than leaving an unexplained gap.

🎤 *"The weights are a file on this disk. It makes no network calls — I
could pull the ethernet cable and this keeps working. Where the file was
trained doesn't change what it can reach, which is nothing."*

- **Launch the apps you need before the session starts** and leave them running
  in separate browser tabs. They load a model on first run; you do not want
  that happening in front of people.
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

# 🛹 §6 + 📖 §8 — The Office Assistant  ✅ **APP + NOTEBOOKS BUILT**

**`app/04_06_Office_Assistant_local_Streamlit.py`** · model: `Qwen3.5-4B`
· notebooks: [`04_hoverboard`](notebooks/04_hoverboard_local.ipynb) ·
[`06_almanac`](notebooks/06_almanac_local.ipynb) *(both with `_api` twins)*

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

# 🏎️ §7 — Reaching 88 MPH  ✅ **APP + BOTH NOTEBOOKS BUILT**

**`app/05_88mph_Valuation_Review_local_Streamlit.py`** · model: `Qwen3.5-4B`
· notebooks: [`05_88mph_local`](notebooks/05_88mph_local.ipynb) ·
[`05_88mph_api`](notebooks/05_88mph_api.ipynb)

**The longest section in the talk and the core IAAO payload.** Everything
before this earns the right to show it; everything after it is a variation.

**Start this app during the §7 intro slides.** Only one app can run at a time —
two model copies will not fit on an 8 GB card — and a cold start takes about
**15 seconds**. Stop the office-assistant app first, start this one while you
are still talking, and it is warm by the time you turn to it.

---

## 🎯 The one thing the room should take away

> ### The agent decides **what to ask**. The tool decides **what is correct**.

Everything on screen is arranged to make that land. The agent writes its own
SQL — visibly, checkably. The statistics are never its job.

---

## 🖥️ What is on screen

| Where | What |
|---|---|
| **Header** | A **speedometer**. The needle is the median assessment ratio; the green arc is the IAAO acceptance range 0.90–1.10; **1.00 is marked "88"** — a perfect ratio is 88 miles per hour. |
| **Middle** | The question box, then **three numbered buttons** for the story, then a separate *Also try* row. |
| **Below** | The answer in plain English, then whatever the agent produced — scorecard, table, or map. |
| **Bottom** | *"How it got there"* — collapsed. Opens to the full Reason → Act → Observe trace **and the SQL the model wrote**. |
| **Sidebar** | The three tools, the system prompt, the whole schema the model is shown, and the GPU it is running on. |

**This app deliberately leads with the finding, not the machinery** — that is
what separates it from §6/§8, where the loop *is* the show. The trace runs live
while it works, then folds itself away when the answer lands.

---

## 🖱️ The click order — three buttons, about 35 seconds of machine time

### 1️⃣ *"How did the county do?"* — **12 s**

Speedometer swings into the green. Three cards, three green ✓ PASS badges.

> **The county passed all three IAAO standards on 210 sales: the median ratio
> of 0.9268 indicates we are assessing slightly low overall, the COD of 12.2184
> shows consistent assessment across properties, and the PRD of 1.0262 confirms
> that cheap and expensive homes are treated equitably. Thirty-four sales were
> excluded as not arm's length, per IAAO guidelines.**
> — `Qwen3.5-4B`, verbatim

🎤 *"That is a clean year. That is the number that goes in the annual report,
and every word of it is true."*

**Note the last sentence of its answer.** Nobody asked it about arm's-length
sales — the tool excluded 34 of them and said so, and the model repeated it.
That is professional judgment living in the tool, and it is worth pointing at.

### 2️⃣ *"Break it down by neighborhood."* — **10 s**  ← **the moment**

> **Three of the six neighborhoods fail IAAO standards. Courthouse Square fails
> Uniformity (COD), Hilldale fails Equity (PRD), and Riverside Heights fails
> Level (Median ratio). Eastwood Ravine, Lyon Estates, and Twin Pines Ranch
> pass all three tests.**
> — `Qwen3.5-4B`, verbatim

🎤 *"Same sales. Same afternoon. Half the county is fine and half of it is not,
and the countywide number would have sent you home."*

### 3️⃣ *"Show me the sales in Riverside Heights on a map."* — **11 s**

> **A map of 36 Riverside Heights parcels is displayed, colored by
> assessment ratio. The majority (30) fall below the IAAO acceptable range,
> while only 3 exceed it. This indicates a general trend
> of under-assessment in the neighborhood.**
> — `Qwen3.5-4B`, verbatim

🎤 *"'Riverside Heights fails on level' is a sentence. Thirty dots in the wrong
colour is a work order."*

---

## 🔴 The money visual — three reds, in three different columns

This is the single most important image in the section. The table marks the
**failing cell**, not the failing row:

| Neighborhood | n | Median | COD | PRD | Verdict |
|---|---:|---:|---:|---:|:---:|
| Courthouse Square | 30 | 0.926 | **✕ 22.08** | 1.028 | ✕ FAIL |
| Hilldale | 50 | 0.921 | 13.63 | **✕ 1.043** | ✕ FAIL |
| Riverside Heights | 32 | **✕ 0.809** | 6.21 | 1.001 | ✕ FAIL |
| Eastwood Ravine | 27 | 0.984 | 6.06 | 1.001 | ✓ PASS |
| Lyon Estates | 42 | 0.948 | 6.04 | 1.001 | ✓ PASS |
| Twin Pines Ranch | 29 | 0.929 | 9.33 | 1.002 | ✓ PASS |
| **COUNTYWIDE** | **210** | **0.927** | **12.22** | **1.026** | **✓ PASS** |

**The red runs diagonally.** Nobody has to say *"and notice they fail for
different reasons"* — the table already said it.

🎤 *"Three neighborhoods, three different problems, three different afternoons
of work. One needs a revaluation. One needs the model rebuilt. One needs
somebody to look at how the low end is being valued."*

- **Riverside Heights — Level.** The market ran away and the roll did not
  follow. Everyone else in the county is quietly covering their share.
- **Courthouse Square — Uniformity.** The median is fine. The *spread* is
  enormous. **This is the neighborhood generating your appeals.**
- **Hilldale — Equity.** Cheaper homes carry a higher ratio than expensive
  ones. Regressive, and invisible without running the numbers.

---

## ⚙️ It writes its own SQL, and you can read it

Open *"How it got there"* and the query is right there:

```sql
SELECT parcel_id FROM sales_feed WHERE neighborhood = 'Riverside Heights'
```

**The objection arrives within four seconds of the room seeing that**, so have
the answer ready — it is four real things, all in
[`tools/query.py`](src/delorean/tools/query.py):

1. **The connection is opened read-only.** Not "we check for the word DELETE" —
   SQLite itself refuses to write.
2. **One statement, and it must be a `SELECT`.** Anything else is refused with
   an explanation before the database ever sees it.
3. **Row counts are capped**, so nothing floods the screen or the context.
4. **It can only see the Appraiser's own database.** The Clerk's records and
   the Tax Collector's are not reachable from here — which is §10's whole
   subject, arriving on cue.

🎤 *"It is not guessing at your data. It is writing a query, and you can read
the query."*

---

## 🎛️ Measured — the numbers behind the demo

| | |
|---|---|
| Cold start (incl. model load) | **~15 s** |
| Each of the three questions | **10–13 s** |
| Model | `Qwen3.5-4B`, 4-bit NF4 — **2.90 GB** of weights, **4.00 GB** peak in use |
| Tools offered | 3 — `run_sql`, `ratio_study`, `map_parcels` |
| **Recall** (called the right tool when it needed one) | **6/6** |
| **Restraint** (answered directly when it should) | **4/4** |

**Restraint held at three tools** — the count that broke it in §6 — because of
the guardrail prompt. Zero unnecessary tool calls on *"good morning"*, *"what
does COD stand for"*, *"market value versus assessed value"* and *"why a median
instead of an average"*.

### ⏱️ The system prompt line worth stealing

Adding one clause — *"keep every answer to three sentences; the figures are
already on screen"* — took question ② from **39 seconds to 12**. Same tools,
same answer, same accuracy.

🎤 *"Section 3 was not a party trick. That prompt is doing three jobs here: it
stops it reaching for a database to answer a definition, it stops it inventing
statistics, and it made this demo twice as fast."*

---

## 🔧 The bonus row — *"With no ratio tool"*

Sitting **off the story on purpose**, in the *Also try* row. Use it if the room
asks *"why not just let it do the maths?"* — and frame it as **what the tool is
doing for you**, never as *the model lies*.

Clicking it re-runs question ② with `ratio_study` taken away. The model writes
`AVG(assessed_value / sale_price)`, SQLite integer division rounds every ratio
to a whole number first, and it returns this — sorted, totalled, confident:

| Neighborhood | "avg ratio" | truth |
|---|---:|---:|
| Twin Pines Ranch | 0.379 | 0.929 |
| Hilldale | 0.240 | 0.921 |
| **Riverside Heights** | **0.000** | **0.809** |

🎤 *"It assessed thirty-two properties at nothing at all, and handed me a tidy
table about it. That is why the statistics are not in the prompt."*

⚠️ **This is a real failure and it is not on the live path deliberately.** §4
already spent three minutes on "you cannot trust it bare." §7 is the payoff
section and should feel like power. Keep this in your pocket for Q&A.

---

## 📓 What the notebooks add that the app cannot

The notebooks run the same engine, and **Act 4 contains the strongest single
finding in this section.** Asked precisely — *"using SQL, work out the average
assessment ratio for each neighborhood"* — the models write clean, sensible SQL
and return this:

| Neighborhood | model (average) | IAAO (median) | off by |
|---|---:|---:|---:|
| Twin Pines Ranch | 0.965 | 0.929 | **0.036** |
| Hilldale | 0.937 | 0.921 | 0.016 |
| Riverside Heights | 0.799 | 0.809 | 0.010 |

Nothing looks wrong. The numbers sit exactly where a ratio belongs, and
Riverside Heights even comes out lowest, which is correct. **But they are
averages, not medians — and there is no COD and no PRD at all.** Two of the
three IAAO tests have quietly ceased to exist.

> 🎯 **The local 4B and hosted Haiku wrote *different* SQL and returned the
> *same six numbers*, to the last decimal place.** Haiku aliases the column
> `average_ratio` and adds an `ORDER BY`; Qwen does neither. Both reached
> for an average, and neither produced a COD or a PRD.
>
> This is not a small-model failure and it does not go away by spending
> more money. It is a *domain-knowledge* failure, and domain knowledge has
> to live somewhere the model cannot get it wrong.

**§7 is the first demo whose twins agree**, and that is the point: §4 and §6
both ended with the two models behaving differently, but here `ratio_study`
computed the numbers either way. Verified mechanically — the two notebooks
differ in exactly **one code cell**.

🎤 *"That is the argument for §9 before we get there: a small model is not worse
at arithmetic it never does."*

---

## 🗺️ Two things worth saying out loud

**The map makes no network request of any kind.** It is hand-drawn SVG — no
tiles, no `leaflet.js`, nothing fetched. A `folium` map pulls ten external
files, and leaflet itself is one of them, so on dead conference wifi you do not
get a map missing its streets — you get no map. **This one cannot fail that
way.** It also carries a scale bar and a north arrow, and it draws no
neighborhood boundaries, because the parcels are invented and drawing invented
parcels over real streets would claim they are real.

**Nothing leaves the building.** The roll, the sales file, the model and the
map are all on the machine on stage. The sidebar shows **426 of 426 weight
tensors on the GPU** if anyone wants to check.

---

## 🚨 Fallbacks

| If | Then |
|---|---|
| A question hangs or the model misbehaves | Press the button again — it is deterministic, and each question is independent. |
| The app will not start | The notebooks are committed **with outputs**. Open `05_88mph_local.ipynb` and narrate from the saved run — every number in this section is in there. |
| The GPU is out of memory | The office-assistant app is probably still running. Stop it; only one app fits at a time. |
| Somebody asks for a slice you have not tried | The free-text box takes anything the tools can express. *Homes built after 1980* and *the five highest value parcels* are both verified working. |

---

### Takeaway
> ### 🏎️ §7: "The county passed every standard while half its neighborhoods failed. One number was never going to tell you that — and the assistant found it in eleven seconds, on a laptop, without sending your roll anywhere."

---

# 🔌 §10 — The Universal Adapter  ✅ **APP + BOTH NOTEBOOKS BUILT**

**`app/08_09_Universal_Adapter_local_Streamlit.py`** · model: `Qwen3.5-4B`
· notebooks: [`08_universal_adapter_local`](notebooks/08_universal_adapter_local.ipynb) ·
[`08_universal_adapter_api`](notebooks/08_universal_adapter_api.ipynb)

**The finale, and the only section where the architecture diagram IS the demo.**
Two live demos live in this one app.

---

## 🎯 The one thing the room should take away

> ### The tool did not get easier to write. It stopped being ours to write.

Section 6 cost us a function **and** a hand-written schema, per tool, forever.
Here three programs describe themselves and the agent assembles its own
toolbelt. **Nothing in this repository describes any of their tools.**

---

## 🖥️ What is on screen

Four panels across the top — three county offices and, dashed and set apart,
our own policy manual. They light up one at a time as each office answers.

```
🏛️ Appraiser          📜 Clerk of Courts     💰 Tax Collector      ┆📖 This office
the roll — lot size    recorded deeds —       the ledger —         ┆our own policy
and assessed value     who owns it, and how   what is still owed   ┆manual, not MCP
✓ answered             ✓ answered             ✓ answered           ┆✓ answered
▸ get_parcel           ▸ get_deeds            ▸ get_tax_status      ┆▸ search_policy
· find_parcels         · corrections_since                          ┆
```

Under it: the question box, two numbered buttons, the answer, and *"How it got
there"* collapsed — which lists **every office it walked into**.

> 🎤 *"Three separate programs. You can see them in Task Manager. I did not
> write a line of tool code for any of them — I asked each office what it had."*

---

## 🖱️ The click order — two buttons, about 40 seconds of machine time

### 8️⃣ *"May Biff split his lot?"* — **26 s**

The panels light 🏛️ → 📜 → 💰 → 📖.

> **No, Biff Tannen may not split parcel 04-127-3891 because the Tax Collector
> reports a total outstanding balance of $8,432.13, which bars the split under
> policy §4.2. The Appraiser shows a lot size of 15,800 sq ft, and the Clerk of
> Courts confirms Biff Tannen as the owner of record, but the delinquent tax
> balance disqualifies the application.**
> — `Qwen3.5-4B`, verbatim

🎤 *"Lot size from the Appraiser. Ownership from the Clerk. The balance from
the Tax Collector. The rule from our own manual. Today that is three phone
calls and a week — and nobody would have written it down anywhere."*

**Point at what it did NOT do.** It did not decide the lot was big enough, or
that Biff owns it, or estimate the balance. **The model chose the questions;
the offices supplied the facts.**

### 9️⃣ *"Is our sales file still true?"* — **15 s**  ← **the uncomfortable one**

> **Yes, the Clerk corrected one deed since 2026-01-15. The corrected deed is
> instrument number 2025-000069 for parcel 12-393-7896, which was reclassified
> as a QUIT CLAIM on 2026-03-04. This sale must be removed from any ratio study
> using the sales file pulled on 2026-01-15.**
> — `Qwen3.5-4B`, verbatim

It drew that last sentence itself. Nothing in the prompt mentions ratio
studies.

---

## 🔴 The money visual — Section 7, revisited

The app then measures what that one deed cost, side by side:

| Courthouse Square | Level | Uniformity (COD) | Equity (PRD) | Verdict |
|---|---:|---:|---:|:---:|
| **our sales file** `2026-01-15` | 0.9262 ✓ | **22.08 ✕** | 1.0278 ✓ | fails **one** |
| **with the Clerk's correction** | 0.9146 ✓ | **20.31 ✕** | **1.0419 ✕** | fails **two** |

**Removing a bad sale did not make it look better.** That $647,000 quit-claim
was a high-ratio outlier on an expensive property, and its presence was
propping the PRD up inside the standard.

> 🎤 *"The stale feed was not merely out of date. It was flattering me. An
> office would have signed off on an equity result that was not true, using a
> file it had every reason to trust — and nothing in its own systems could
> have told it otherwise."*

⚠️ **This is the argument for MCP that costs money if you ignore it.** Not
convenience, not fewer clicks — the difference between a number you can defend
at a hearing and one you cannot.

---

## 🎛️ Measured

| | |
|---|---|
| Three servers online | **5.5 s** |
| App cold start (incl. model) | **~21 s** |
| Demo 8 · local / hosted | **26 s / 36 s** |
| Demo 9 · local / hosted | **15 s / 3 s** |
| Sources consulted | **4 of 4**, both models |
| Reproducibility | byte-identical across runs (greedy decoding) |
| Robustness | 5 of 5 phrasings reach the right verdict — by address, as a taxpayer, with a mangled parcel number |
| App test | `scripts/test_app_adapter.py` — **ALL PASS**, 103 s |

### ⚠️ The capability ceiling — and it sits between §7 and §10

**SmolLM3-3B runs the whole of Section 7 and cannot do this.** Measured on the
same question: **1 office of 4**, and with the guided prompt it answered from
memory and **invented a policy structure that does not exist**.

> 🎤 *"The small model does the ratio study perfectly, because the arithmetic
> is in a tool. It cannot hold a four-step plan across three systems. That is
> where the money goes — and it is worth knowing which half of the job needs
> it."*

Use this if anyone asks whether §9's "run it on a 3B" claim covers everything.
It does not, and saying so is worth more than the claim.

---

## 🔀 The twins converge, for a NEW reason

Section 7's twins agreed because the **statistics** were in a tool. Section
10's agree because the **facts** are in three servers. Same principle, one
layer up — and the two notebooks differ in **exactly one code cell and one
markdown cell**, the tightest pair in the repository.

Haiku reads slightly further into the policy and adds *"payment plans do not
satisfy this requirement"*. Same verdict, same figures.

**The line to say when a hosted model touches county records:**

> 🎤 *"The three county servers are still on this machine. Only the model is
> remote. The databases never leave the building — the model is told the
> answers, it is not given the keys."*

---

## 🚨 Fallbacks

| If | Then |
|---|---|
| An office panel says **✕ did not answer** | That server died. The agent keeps going and says which office it could not reach — narrate that, it is the honest failure mode and it is designed in. |
| The app will not start | Notebooks are committed **with outputs**. Open `08_universal_adapter_local.ipynb` and narrate from the saved run. |
| The GPU is full | Another app is probably still up — only one fits. **And check for orphaned servers**: `nvidia-smi --query-gpu=memory.used --format=csv` should read ~1.1 GB idle. |
| You Ctrl-C'd the app and restarted | Three MCP servers may still be running. Check before relaunching — see CLAUDE.md §13. |
| Somebody asks a question you have not tried | The free-text box takes anything the three offices can answer. *"Who owns 1955 Hilldale Ct and do they owe anything?"* is verified working. |

---

### Takeaways
> ### 🔌 §10 Demo 8: "A taxpayer asked a question no single office could answer, and got a correct, sourced, citable answer in twenty-five seconds — from three systems that have never been integrated, on a machine under a desk."
> ### 🔌 §10 Demo 9: "Your sales file is a photograph of a filing cabinet that has since been rearranged. This is how you find out before the hearing does."

---

# 📊 §11 — Putting a Bow On It  *(5 min)*

**No new artifact, and that is deliberate.** Decided 2026-08-25.

The outline asks for *"a small Streamlit app running the Mass Valuation Agent…
a clean, ChatGPT-like interface."* That is Section 7's engine with a different
face — it was always a reprise, not a new product. Building a third app to
follow §7 and §10 would end the talk on a **smaller** demo than the two before
it, which is the wrong shape.

So §11 spends its five minutes on the thing the presenters care most about:
**how anybody in the room actually starts on Monday.**

### What is on screen

Reopen **either** finished app with the machinery collapsed — §7 for a single
office, §10 for the cross-office one — and let it sit there answering a
question while you talk over it. Then the repository itself.

### The beats

1. **"You have already seen the product."** Every app today is under 550 lines
   and imports the same engine from `src/delorean/`.
2. **"None of this needed a data scientist."** `uv sync`, one command, and
   `scripts/check_setup.py` tells you whether your machine can do it.
3. **"Take it home."** 15 notebooks, 5 apps, three MCP servers, and a county
   of synthetic data that is safe to show anybody.
4. **The four questions for choosing a model** — `07_delorean_smol.ipynb`.

### Takeaway
> ### 📊 §11: "Everything you watched today runs on a machine somebody in your office already owns, on data that never left the building. The hard part was never the technology. It was deciding to start."

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
| **7** | Its neighborhood, **Hilldale, fails the equity test** — the roll is regressive around it. *("Who owns 04-127-3891?" is a one-click probe in the app)* |
| **8** | The agent reads the policy manual and gets the lot rule **right** |
| **10** | Three county systems + the policy → **"May Biff split this parcel?"** — answered in **26 seconds** |

The §10 answer, measured: lot size ✅ **15,800 sq ft, halves to 7,900 against a
6,000 minimum** · ownership ✅ **Biff is grantee on the 2025 warranty deed** ·
taxes ❌ **$8,432.13 delinquent across two years.** Policy 4.2 blocks it — and
the same policy says how to cure it: pay in full, reinstate within 90 days, no
new filing fee. **The assistant can tell the taxpayer that at the counter.**

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
| `04_hoverboard_{local,api}` | 6 | GPU / API key + the county data |
| **`05_88mph_{local,api}`** | **7** | GPU / API key + the county data |
| `06_almanac_{local,api}` | 8 | GPU / API key + the policy corpus |
| **`07_delorean_smol`** | **9** | GPU only — **1.86 GB** of weights, no API key, local only |
| **`08_universal_adapter_{local,api}`** | **10** | GPU / API key + all three county databases |

Point the audience here in §12. **This repo is the answer to "but how did you
do that?"** — which is the reason it exists.
