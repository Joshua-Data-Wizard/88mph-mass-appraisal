# 🔬 FINDINGS.md — what we measured, and what broke

The evidence behind [CLAUDE.md](CLAUDE.md). **CLAUDE.md holds the rules; this
file holds the reasons.** If a rule there looks arbitrary, the forty lines
explaining what it cost to learn are in here.

> ### One rule governs this file: nothing in it is a guess.
> Every number was produced by running something on the dev box. Where a
> measurement contradicted what we expected, the expectation is written down
> too — those entries are the useful ones.

**If you are a language model working on this repository**, read CLAUDE.md
first. Come here when something surprises you, when a test fails in a way that
looks impossible, or before you "fix" something that looks wrong on purpose.

---

## The short version — every trap on one screen

| Trap | Where it bites |
|---|---|
| A chat template can **silently drop your tools** | any new model |
| A model scoring **0/6 is usually the harness**, not the model | any new model |
| `skip_special_tokens=True` **eats Gemma's tool calls** | `agent.local_responder` |
| A **negation in a system prompt** comes back as a fact | `prompts.py` |
| `st.html()` **strips every SVG**, silently | any app drawing a chart |
| `llm.load_local()` **caches forever** — call `llm.unload()` | any multi-model script |
| A **blocked thread-pool worker** stops the interpreter exiting | `offices.py` |
| Streamlit's watcher walks **5,500 transformers modules** | every app |
| OneDrive installs packages **silently corrupt** | `uv sync` |
| Chroma downloads **79 MB on first use** unless you stop it | `rag.py` |
| Model output **must not be interpolated into HTML** | every app |
| **`nvidia-smi` overstates** on Windows — trust `torch.cuda.mem_get_info()` | before presenting |
| `os.path.getsize()` **follows symlinks** — it double-counts a HF cache | measuring disk |
| Windows **MAX_PATH 260** breaks a model download | a deep `HF_HOME` |

---


## Cross-cutting — the traps that are not about one demo

### 🚨 The input side: a template can drop your tools silently

**This is the more dangerous discovery of the two, because there is no
wrong-looking output to notice.** Measured on `microsoft/Phi-4-mini-instruct`:

| How the tools were passed | Prompt | Tools in it? |
|---|---|---|
| `apply_chat_template(..., tools=schemas)` — the standard way | **99 chars** | ❌ **none** |
| tools placed on the **system** message | 643 chars | ✅ all three |

Phi's template ignores the `tools=` argument and looks for `message["tools"]`
on the system turn instead. **Passing tools the usual way raises nothing.** It
renders a valid prompt with no tools in it, the model calls nothing, and the
model looks too small to use tools.

And a second layer underneath that — **the declaration dialect matters too.**
Same model, same question, only the shape of the tool list changed:

| Declaration | What Phi-4-mini wrote |
|---|---|
| `{"type": "function", "function": {…}}` (OpenAI nesting, what we generate) | **echoes the tool DEFINITION back**, verbatim, calls nothing |
| the function object **unwrapped** | `{"name": "ratio_study", "arguments": {…}}` ✅ |

Handed a shape it did not recognise, it did the most plausible next-token
thing available and repeated it. **§2, still true, two hundred lines deeper
into the stack.**

**`agent.render_prompt()` handles both.** It does not keep a list of which
model needs what — it renders the prompt with the tools and without them, and
if the two come out identical the template ignored them, so the tools go on
the system turn in the unwrapped dialect. One extra string render, no
tokenizing, and it works for models released after it was written.

```text
Shape A — Qwen3.5, Nemotron       Shape B — what SmolLM3's TEMPLATE asks for
<tool_call>                       <tool_call>
<function=lookup_parcel>          {"name": "lookup_parcel",
<parameter=parcel_id>              "arguments": {"parcel_id": "04-127-3891"}}
04-127-3891                       </tool_call>
</parameter>
</function>                       Shape C — what SmolLM3 ACTUALLY WRITES
</tool_call>                      {"name": "lookup_parcel",
                                   "arguments": {"parcel_id": "04-127-3891"}}
```

⚠️ **A chat template tells you what a model was TRAINED to emit. It does not
tell you what the model emits.** SmolLM3's rendered prompt ends with an
explicit instruction to wrap calls in `<tool_call></tool_call>`. It writes the
payload perfectly and omits the tags every single time. We had "verified" the
formats by rendering each template and reading what it asked for, which felt
rigorous and was not — the only test that counts is generating from the model
and reading the raw output.

**The failure mode is the nasty kind: it looks like the model is incapable.**
SmolLM3 scored 0/6 on tool recall, never appearing to call anything, and the
obvious conclusion was that a 3B model cannot drive three tools. It was calling
them correctly the whole time. With Shape C added: **6/6, and 6/6 on SQL that
ran.**

Shape C is parsed with `json.JSONDecoder().raw_decode` rather than a regex —
the payload nests, so `\{.*?\}` stops at the first brace — and it only fires
when the name resolves to a tool that actually exists, so ordinary prose
containing a `{` cannot trigger it. It is tried **last**, being the loosest:
anything with real delimiters gets first refusal.

Shapes A, B and D deliberately do **not** carry that
does-this-tool-exist guard. Their delimiters are unambiguous, so a call naming
a tool that does not exist is a real call to a missing tool — parsing it lets
`dispatch()` hand the model back the list of tools it actually has, which is
the whole "errors are prompts" idea. Dropping it silently would leave the
agent with nothing to recover from.

⚠️ **All four shapes and every negative case are now covered in
`scripts/test_tools.py` — 20 parser checks.** This file previously claimed
Shape C was covered there. **It was not.** Adding Gemma is what surfaced the
gap; if a doc says something is tested, go and look.

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

### 🚨 When a model scores zero, suspect the harness first

Three times now a model in this registry has looked incapable, and **all three
times the model was fine and our code was wrong.** The pattern is consistent
enough to be a rule.

| Model | Looked like | Actually was | Cost |
|---|---|---|---|
| SmolLM3 | "a 3B can't drive three tools" — **0/6** | emitting perfect calls with **no `<tool_call>` tags**; our parser required them | an afternoon |
| Phi-4-mini | "echoes tool definitions, can't call anything" | handed **OpenAI-nested schemas** it did not recognise, and a template that **silently dropped `tools=`** | two probes |
| Gemma 4 | "never calls a tool" — **0/6** | its delimiters are **special tokens**, and `skip_special_tokens=True` deleted them before the parser ever saw them | one scored run |

The Gemma one is the sharpest, because the fix is a single keyword:

```python
tokenizer.decode(generated, skip_special_tokens=True)
#   <|tool_call>call:map_parcels{sql:<|"|>SELECT …<|"|>}<tool_call|>
#   becomes
#   call:map_parcels{sql:SELECT …}
```

Same generation, same weights. One argument decided whether the model scored
6/6 or 0/6. `local_responder()` now decodes **twice** — special tokens kept for
parsing, stripped for display — and `strip_tool_calls()` also removes the
delimiter-less remnant so nothing leaks into what a human reads.

**The diagnostic that works:** print the raw generation with `repr()` and
`skip_special_tokens=False` before believing any score. `scripts/` has no
shortage of scoreboards; none of them can tell you the difference between a
model that would not and a model we could not hear.

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

### ⚠️ `llm.load_local()` caches forever — call `llm.unload()` between models

The cache is right for a notebook and wrong for any process that loads several
models. `del model` in the caller frees nothing while `_CACHE` holds a
reference, so VRAM accumulates silently.

`scripts/compare_models.py` reported this before it was caught:

| Model | Reported | Actual |
|---|---|---|
| Qwen3.5 4B | 2.90 GB | 2.90 GB ✅ (ran first) |
| SmolLM3 3B | 4.76 GB | **1.86 GB** |
| Phi-4 mini | 7.45 GB | **2.69 GB** |
| Gemma 4 E2B | 13.75 GB | **6.30 GB** |

**The tell was that the figures grew monotonically in load order** — each one
was carrying every model before it. If a benchmark's memory numbers only ever
go up, it is measuring the process, not the model.

### 🐛 A blocked thread-pool worker stopped the interpreter exiting (fixed)

The most expensive bug in this project so far, and the diagnosis was wrong
twice before it was right. Worth keeping in full.

**Symptom.** Test runs that printed their final result and then never exited.
Two of them sat there for twenty minutes each holding **4.9 GB and 1.9 GB of
VRAM**, until the card was 9.8 GB of 10.2 GB full and nothing else could load.
An earlier run looked like it had hung for 61 minutes without producing
output at all.

**Cause.** `Offices._serve` waited for work with:

```python
job = await loop.run_in_executor(None, self._jobs.get)   # WRONG
```

That parks a worker of the **default ThreadPoolExecutor** on a blocking
`queue.get()` that never returns. Those workers are non-daemon, and
`concurrent.futures.thread` registers an atexit hook that **joins every one of
them**. The work finished; the interpreter could never shut down.

**Fix.** An `asyncio.Queue` fed through `call_soon_threadsafe` — which is what
that API exists for — so no thread we did not create is ever blocked. Plus
`atexit.register(self.close)`, because on Windows a child process is not
reaped when its parent exits and three orphaned servers is its own problem.

⚠️ **The tell, and the thing to check first next time: wall-clock climbing
while CPU time stays flat.** A process that is working burns CPU. A process
stuck in `atexit` does not. 61 minutes of wall clock against 152 seconds of
CPU was sitting in plain sight for an hour.

```powershell
Get-Process python | Where-Object { $_.StartTime -gt (Get-Date).AddHours(-1) } |
    Select-Object Id, StartTime, CPU, WorkingSet64
```

**Two wrong diagnoses on the way, both worth remembering.** First "transient",
which explained nothing. Then "TaskStop leaves orphaned children", which was a
real hazard but not the cause — the children were orphaned *because* the
parent could not exit, not the other way round. **Verify the mechanism before
writing the explanation down**; a plausible story that fits the symptom is not
the same as the cause, which is the same lesson `ModelSpec.max_context`
carries from Nemotron.

⚠️ **And do not pipe a long-running test through `grep`.** It buffers, so the
output file sits at zero bytes and a finished-but-hung run looks identical to
one that never started. That is what hid this for an hour. Redirect to a file
with `python -u` and tail it instead.

### 🐛 `api_responder` and the system prompt (fixed)

The local models take a system prompt as an ordinary message in the list. **The
Anthropic API rejects that with a 400** — it must be a top-level `system`
parameter. `api_responder()` now lifts any `role: "system"` messages out of the
list and merges them into that parameter.

This never showed up until an `_api` notebook called `respond()` with a system
message, because nothing else exercised that path. **Any new responder must be
tested with a system prompt, not just a bare question.**

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


### 🐛 A measurement that followed its own symlinks (and invented a cause)

**Claimed:** the six cached models occupy **70.12 GB** while reporting 40.25 GB,
because Windows cannot symlink and HuggingFace copies each file twice.

**Actual:** they occupy **40.25 GB**. They are symlinked — 7 to 21 links each.
Nothing was ever duplicated.

The bug is one function call. `os.path.getsize()` **follows a symlink and
returns the target's size**, so walking a HuggingFace cache counts every file
twice: once as the blob, once as the `snapshots/` link pointing at it.

```python
os.path.getsize(link)      # the TARGET's size — counted again
os.path.islink(link)       # what the walk actually needed to ask
```

**The part worth keeping is not the fix.** It is that a wrong number arrived
with a ready-made explanation attached. "Windows cannot symlink without
Developer Mode, so HuggingFace copies" is *true*, it is *well known*, and it
predicted exactly the doubling that had just been measured. It was written into
`models.py`, into `ModelSpec.download_gb`, and reported to the presenter as a
finding — and it was the wrong cause for a number that was itself wrong.

The check that would have caught it in ten seconds:

```powershell
Get-ChildItem -Recurse <cache> | Where-Object { $_.LinkType } | Measure-Object
```

⚠️ **A plausible mechanism is not evidence.** This is the third time in this
project — `ModelSpec.max_context` was added to fix a Nemotron OOM that matched
a KV-cache theory to the byte and was not a KV cache; the §10 hang was blamed
on orphaned child processes when the parent simply could not exit. In all three
the story fit the symptom perfectly and named the wrong cause.

**`_bytes_under()` now skips symlinks**, so `cache_size_gb()` measures real
occupied disk rather than being right by accident. The doubling warning in
`scripts/fetch_models.py` stays, because it is real *where it happens* — the
script probes the filesystem instead of assuming either way.

### 🪟 Windows MAX_PATH kills a model download

Found while testing `fetch_models.py` against a temporary cache directory:

```
FileNotFoundError: [WinError 3] The system cannot find the path specified:
  '..\blobs\f691cef...' -> '...\snapshots\2fc0636...\model.safetensors.index.json'
```

That path is **270 characters**. Windows caps a path at **260** unless long
paths are enabled. A HuggingFace cache path is already long before a username
and a 40-character revision hash are added, so a deep `HF_HOME` will fail —
and it fails while *writing a file it is midway through*, which reads like
corruption rather than a limit.

The same download to `C:/Users/<name>/hfprobe` finished in **18 seconds**.

`fetch_models.py` catches `WinError 3`/`206` and says so, with the fix:
`HF_HOME=C:/hf` in `.env`.

## Streamlit — four things that fail silently

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

### 🐛 `last_raw` must be the FULL decode, or §6's reveal lies

`local_responder` decodes twice — special tokens kept for parsing,
stripped for display. `respond.last_raw` feeds §6's *"what the model
literally wrote"* panel, and it has to be the **kept** version.

Gemma 4's tool-call delimiters are special tokens. With the stripped
decode, `attempted_tool_call()` finds no markup, the app's
`raw_with_call` stays empty, and it falls back to the **last** thing the
model said — which is the final prose answer, displayed under a heading
promising a tool call. Not an error. Not a blank box. Just a panel that
is quietly showing the wrong thing, in the one demo whose entire point is
that nothing is hidden.

`viz.raw_call_html()` highlights all three delimiter styles now, Gemma's
`<|tool_call>` and `<|"|>` included — matching only Qwen's left Gemma's
call as flat grey text.

**No notebook reads `last_raw`**, so this was safe to change without
re-executing anything. Check that before touching it again.

### Model output does not go into an HTML string

The §7 answer panel originally interpolated the model's reply into a styled
`<div>` via `unsafe_allow_html=True`. Two problems, and the second is the one
that showed up immediately: the model **writes markdown** — given no ratio
tool it returns a full markdown table — which inside a `<div>` renders as raw
pipes and dashes. And model output is untrusted text that should not be
concatenated into markup in a repository other offices are told to copy.

Now `st.container(border=True)` + `st.markdown(answer)`. Renders tables
properly and nothing is interpolated.


## Tools and restraint — measured

### Why it exists — measured, not assumed

Across ten phrasings of one question, Qwen3.5-4B picked the right tool **10/10**
but emitted the parcel number **three different ways**: `04-127-3891`,
`04127-3891`, `4-127-3891`. A plain `WHERE parcel_id = ?` resolved **8/10** —
and failed *silently*, looking exactly like a parcel that does not exist.

**With the hardening layer: 10/10 end to end against the live model.**

Small models are reliable at *deciding* and sloppy at *formatting*. Absorb the
formatting and they behave far better than their size suggests.

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

**All six registry models are now downloaded on the dev box** — `qwen`
8.7 GB, `smol` 5.7 GB, `phi` 7.2 GB, `gemma` 9.6 GB, `nemotron` 7.4 GB,
`tiny` 1.7 GB. Use
`models.is_cached(key)` before offering a model in a UI: choosing an uncached
one starts a multi-gigabyte download with no visible progress, which the §7
app now warns about rather than doing silently.

⚠️ `is_cached()` checks for Hugging Face's `.incomplete` blobs, not just
size — a download in progress reported "ready" at 3.16 GB of 5.7 GB otherwise.
And `cache_size_gb()` takes the LARGER of `blobs/` and `snapshots/` rather than
the sum: on Windows without Developer Mode the cache copies instead of
symlinking, so a freshly downloaded model sits in both at full size and naive
addition reported SmolLM3 as 11.47 GB.

All three tool-call shapes are unit-tested in `scripts/test_tools.py` without
needing any download. **`scripts/compare_models.py` runs the §7 toolbelt
against every model and is what produced the numbers in *§13 The agent loop*.**

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


## §2 · §3 · §4 — what the early demos established

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


## §6 · §8 — tools, and RAG as one more tool

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


## §7 — the valuation review agent

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
| *"Using SQL, work out the average assessment ratio…"* (directive) | `AVG(ratio) as avg_ratio` | `AVG(ratio) as average_ratio … ORDER BY` — different SQL, **identical numbers to the last decimal** |

So the notebook uses the **directive** phrasing, and both twins print the same
six numbers. The lesson lands harder that way:

- The numbers are **plausible** — 0.79 to 0.98, exactly where a ratio belongs.
  Riverside Heights even comes out lowest, which is correct.
- They are still wrong: an **average, not a median** (Twin Pines Ranch is off
  by 0.036), and **no COD and no PRD at all**. Two of the three IAAO tests
  quietly cease to exist.
- **It is not a small-model failure.** The two models wrote *different*
  queries — Haiku aliases the column differently and adds an `ORDER BY` —
  and returned **the same six numbers to the last decimal place**. It does
  not go away by spending more money, because it is a *domain-knowledge*
  failure, and domain knowledge belongs somewhere the model cannot get it
  wrong.

  ⚠️ **Re-measured 2026-08-25: this used to say the output was
  byte-identical, and it no longer is.** Do not re-tighten that claim; it
  was always the weaker version of the point. Two different models writing
  two different queries and arriving at the same wrong statistic is a
  better argument than two models emitting the same string.

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


## §9 — small models, and choosing one

### §9 notebook — what it established

**§9 is a slide reveal, not a live demo, and that was a deliberate change.**
The outline asked for a side-by-side of several small models. Three 4-bit
models resident is about 8.5 GB against an 8 GB floor, and swapping instead
costs 16 seconds of dead air twice inside a five-minute slot. The reveal —
*everything you have watched for the last hour ran on a small local model* —
costs nothing, and it is more persuasive because the audience already lived
it. The §7 app's sidebar (`426 of 426 tensors on cuda:0 · 2.95 GB`) supplies
the live moment for free.

`notebooks/07_delorean_smol.ipynb` is the take-home. **Local only, no `_api`
twin** — the section is about models you run yourself.

#### Measured: `scripts/compare_models.py`, same ten questions, same three tools

Re-measured 2026-08-25 in one run, after `llm.unload()` fixed the cumulative
VRAM bug and the tool-call plumbing was corrected for all four:

| Model | Recall | Restraint | SQL ran | Weights | Peak | s/question |
|---|---|---|---|---|---|---|
| Qwen3.5 4B | 6/6 | 4/4 | 6/6 | 2.90 GB | 4.00 GB | 8.7 |
| **SmolLM3 3B** | **6/6** | **4/4** | **6/6** | **1.86 GB** | **2.77 GB** | 6.8 |
| Microsoft Phi-4 mini | 5/6 | 4/4 | 7/10 | 2.70 GB | 4.15 GB | 7.0 |
| Google Gemma 4 E2B | 5/6 | 4/4 | **5/5** | 6.29 GB | 6.55 GB | **5.7** |
| Nemotron 3 Nano 4B | — | — | — | 3.05 GB | **OOM** | — |

⚠️ **Gemma is the second model here whose advertised size did not predict what
it needs.** "E2B" means *effective 2B* — 2.3B effective, 5.1B with embeddings —
and the Per-Layer Embedding tables stay resident and unquantized, so it peaks
at **6.55 GB**, not the ~2.8 GB the name suggests. It fits an 8 GB card with
about 1.5 GB to spare. Nemotron taught this lesson with hybrid Mamba kernels;
Gemma teaches it with parameter accounting. **Neither is discoverable from a
leaderboard, and both are the first thing an office should check.**

**SmolLM3 runs the entire §7 demo, smaller and faster than the model we use.**
That is the §9 argument, and it holds because `ratio_study` computes the
statistics either way — the model only decides what to ask.

#### ⚠️ Recall measures tool CHOICE, not argument quality

The notebook's third question exposes the gap and it is worth keeping. Asked
*"how did the county do?"*, SmolLM3 called `ratio_study` correctly — 6/6 recall
counts it as a win — and handed it:

```sql
... WHERE year_built BETWEEN 2010 AND 2020 AND ...
```

Nobody asked for a decade. Most neighborhoods then had too few qualifying
sales, so it reported on the two that survived and called it the county. Qwen
wrote `SELECT parcel_id FROM sales_feed` for the same question.

**The tool did not fail.** *"All 2 neighborhoods meet the IAAO standards"* is
true of the rows it was given, and nothing was fabricated. The mistake is
legible in the query — **which is the real argument for putting the SQL on
screen in §7.**

> The tool guarantees the arithmetic. It cannot guarantee the model asked for
> the right rows. Only a human reading the query can.

Do not "fix" this by swapping in a question SmolLM3 gets right. It is the most
useful page in the notebook.

#### Nemotron: the failure no benchmark reports

Loads in 3.05 GB, then asks for **21 GB** on the first token. It is a hybrid
Mamba model whose fast path needs `mamba-ssm`, `causal-conv1d` and `triton` —
none of which install cleanly on Windows. The PyTorch fallback costs about
**12 MB of VRAM per prompt token**, roughly 100× a normal transformer:

| prompt tokens | 256 | 512 | 768 | 1024 | 1280 |
|---|---|---|---|---|---|
| peak VRAM | 6.25 GB | 9.35 GB | 12.47 GB | 15.57 GB | **OOM** |

§7's prompt is ~1,600 tokens, so it never starts. Short prompts work fine.

⚠️ **A wrong diagnosis worth remembering.** The first explanation was a
preallocated KV cache, because `262144 × 1024 × 42 × 2` comes to almost
exactly 21 GB. `ModelSpec.max_context` was added to fix it. **It did not** —
same failure, same byte count, to the byte. *A number that matches your theory
is not evidence your theory is right; change the thing you blame and see
whether the failure moves.* `max_context` is kept as a sensible guard, with
that story in its docstring, and Nemotron is **not** built into any demo.

#### ⚠️ Nemotron is in the registry but out of every app picker

Decided 2026-08-25. `ModelSpec.runs_here` is False for `nemotron`, and all four
apps build their dropdown from **`models.selectable()`** rather than
`list(models.REGISTRY)`. Before this, every app offered it — and the §7 app,
whose prompt is ~1,600 tokens, would have hard-OOMed if a presenter picked it.

**It was not deleted, and that is the point.** Three reasons:

1. **The failure is the lesson.** "Check the architecture has working kernels on
   your OS" is the most useful page in §9 and you cannot teach it from a gap.
2. **"Does not run here" is not "does not run."** The mamba-ssm / causal-conv1d /
   triton kernels build cleanly on Linux. A county on Linux should be able to try
   it, and can — `llm.load_local("nemotron")` is untouched by the flag.
3. **Nothing else breaks.** `models.compare()`, `scripts/compare_models.py`,
   `07_delorean_smol.ipynb` and its committed outputs all still iterate
   `REGISTRY`. No notebook needed re-executing.

`models.blocked()` returns the filtered-out specs with a `blocked_reason`, and
the two agent apps print it under the picker — **a shorter dropdown with no
explanation reads as "we never tried it," which is the opposite of true.**

⚠️ **New apps must use `models.selectable()`, never `list(models.REGISTRY)`.**

#### The four questions for choosing a model (§9's take-home)

1. **Does it run on the hardware you actually have?** Check the architecture
   has working kernels on your OS. No leaderboard mentions this and it is what
   Nemotron cost us.
2. **Can it call tools reliably?** Measure it on *your* tools.
3. **What does the licence allow?** Two are Apache 2.0; Nemotron's is NVIDIA's own.
4. **Is it small enough to be boring?** 2 GB runs on a workstation nobody had
   to buy and can be replaced next year.

---

### Why Phi-4-mini and Gemma 4 are here (added 2026-08-25)

**Not for the benchmark — for the name on the box.** Raised by the presenters:
this audience is senior appraisal officials and county IT staff, and *Qwen* and
*SmolLM3* both read as unfamiliar and foreign to a room that has to justify
this to a commission. Nemotron used to cover that need and no longer runs here.

Both new entries are **ungated**, which is what made this possible at all:

- **Gemma 3 was gated** (`gated=manual`, a click-through licence) and would
  have been disqualified. **Gemma 4 is Apache 2.0 and ungated.** Re-check this
  on any future Google model rather than assuming either way.
- **Phi-4-mini is MIT** — the most permissive licence in the registry.

⚠️ **`microsoft/Fara1.5-4B` is a trap.** Microsoft, MIT, 4B, brand new — and
its architecture is `Qwen3_5ForConditionalGeneration`. It is a Qwen fine-tune
wearing a Microsoft badge. Using it to settle a provenance concern would blow
up the moment a curious IT person opened `config.json`, which is worse than
never raising the subject.

**And say the real answer out loud, because it is stronger than the brand
swap:** the weights are a file, the file makes no network calls, and pulling
the ethernet cable does not stop any demo in this repo. Whoever trained it has
no more reach into a county than the author of a downloaded PDF. What deserves
an audit is the *code around* the model — which is why this repository exists.
The familiar name gets you in the door; the architecture is what answers the
objection.

**Neither displaces Qwen3.5-4B as the default.** Qwen and SmolLM3 are the only
two at 6/6 recall with 6/6 runnable SQL. Phi and Gemma are there so an attendee
can start from a name they trust and still get a working agent.

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


## §10 — MCP and the three offices

### ⚠️ A negation in a system prompt came back as a fact

The `COUNTY_ASSISTANT` checklist originally read:

```
  3. no tax is outstanding    -> the Tax Collector
```

The model echoed the label straight back as a finding — *"the application
fails because **no tax is outstanding**"* — the exact opposite of the truth,
in the opening sentence, with every figure beneath it correct. Renaming the
item **"the tax balance"** fixed it outright.

**A phrase in a system prompt is a phrase the model may repeat.** Write
checklist items as nouns, not as negated conditions. This is a new failure
mode for this repo and it is cheap to avoid once you have seen it.

The same prompt's *"do not restate the checklist and do not explain your
process"* took the question from **59 s to 22 s** — the second independent
confirmation of §7's brevity finding.

### ⚠️ The capability ceiling sits between §7 and §10

**SmolLM3-3B runs the whole of §7 and cannot do this.** Same question, same
tools: **1 office of 4**, and with the guided prompt it answered from memory
and invented a policy structure that does not exist.

That is not a mark against §9 — it is the honest limit of its claim, and worth
saying on stage: *the small model does the ratio study perfectly because the
arithmetic is in a tool; it cannot hold a four-step plan across three systems.*
Four chained tool calls is the hardest thing in this repository.

### Measured (2026-08-25, Qwen3.5-4B unless stated)

| | |
|---|---|
| Three servers online | **5.5 s** |
| Tools discovered | **5**, none described by this repository |
| Demo 8 *(may Biff split?)* · local / hosted | **26 s / 36 s**, 4 of 4 sources |
| Demo 9 *(is our file still true?)* · local / hosted | **15 s / 3 s** |
| Reproducibility | byte-identical across runs |
| Phrasing robustness | **5 of 5** — by address, as a taxpayer, mangled parcel id |
| App test | ALL PASS, 103 s, exits cleanly |

**The twins converge, for a new reason.** §7's agreed because the *statistics*
were in a tool; §10's agree because the *facts* are in three servers. The two
notebooks differ in **exactly one code cell and one markdown cell** — the
tightest pair in the repo.


## Housekeeping

### 🔁 All four agent notebooks re-executed 2026-08-25

`04_hoverboard_{local,api}` and `05_88mph_{local,api}` were re-run after
the tool-call plumbing changed (see *§11 Tool calling*). **Every notebook in the repo is
at 0 errors and every committed output is from the current code.**

What moved, and what did not:

| | |
|---|---|
| **Unchanged** | every statistic. 210 sales, 0.9268 / 12.2184 / 1.0262, three of six neighborhoods failing three *different* tests, Act 4's 0.965 / 0.937 / 0.799 |
| **Unchanged** | §6's round trip — *"owned by Biff Tannen and is assessed at $412,500"* — and the naive-vs-hardened id arc |
| **Unchanged** | the twin rule: §7's twins still differ in **exactly one code cell**, verified mechanically |
| **Moved** | small wording in the model's prose. DEMOS.md's verbatim quotes were re-synced to match |
| **Moved** | Act 4 is **no longer byte-identical** across the twins — see *§11 Tool calling* |

⚠️ **Re-sync DEMOS.md's quotes whenever these notebooks are re-run.** They
are written to be pasted onto a slide, so a stale quote there becomes a
wrong quote in front of the room. The check is cheap: run the three §7
questions and diff the answers against DEMOS.md.
