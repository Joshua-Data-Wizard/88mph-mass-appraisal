"""
04_05_Time_Circuits_and_Plutonium_local_Streamlit.py
====================================================

Sections 4 and 5, in one app.

    uv run streamlit run app/04_05_Time_Circuits_and_Plutonium_local_Streamlit.py

    Section 4  "Setting the Time Circuits"    -> tabs 1 and 2
    Section 5  "Running out of Plutonium"     -> tab 3

Everything runs on your own machine. No API key, nothing leaves the building.

Why these two sections share one app
------------------------------------
They are not merely adjacent in the outline. They are cause and effect.

Section 4 shows that a system prompt turns a vague, generic model into
something that sounds like a 25-year appraiser. That is the good news, and it
is genuinely useful.

Section 5 shows the bill for it. Asked a question it cannot possibly know, a
model with NO persona will politely tell you it does not know. Give it the
confident expert persona from Section 4 and it will invent a specific,
plausible, completely fictional answer instead -- and cite an ordinance that
does not exist while doing it.

That is not two lessons. It is one lesson with a twist, and splitting it across
two apps would throw away the twist. So: one app, and tab 3 uses the persona
you were playing with in tab 1.

Verified behaviour, not assumed
-------------------------------
Every claim this app makes about the model was measured first, on the model
this app loads. Notably: the 0.8B model CANNOT hold the angry-taxpayer persona
(it breaks character and lectures you about being an AI), which is why this app
defaults to the 4B. Switch to the small one in the sidebar and watch it happen
-- that failure is worth seeing.
"""

from __future__ import annotations

import queue
import sqlite3
import sys
import time
from pathlib import Path
from threading import Thread

import streamlit as st

# Allow running this file directly from anywhere in the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from delorean import data, llm, models, viz  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
#  Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Setting the Time Circuits",
    page_icon="🎛️",
    layout="wide",
)

# The default model. Section 4 needs a model that can actually hold a character;
# the 0.8B one cannot, which is a finding in itself but a poor default.
DEFAULT_MODEL = "qwen"

# How many tokens each answer gets. Deliberately short: on stage, three
# personas at ~14 tok/s means this number is a wall-clock budget, not a
# quality setting. 90 tokens is roughly 6 seconds per answer.
ANSWER_TOKENS = 90

# How long to wait for the next token before deciding something has gone wrong.
# Generous: this is the gap between *consecutive* tokens, and the 4B produces
# about fourteen a second. See the comment at the streamer for why it exists.
STREAM_TIMEOUT = 60.0

# Tab 2 tells you how long comparing N personas will take, which means it needs
# a generation speed. This is only the starting guess -- Qwen3.5-4B at 4-bit on
# an RTX 3080 -- because the sidebar offers five models and they are not close
# to each other (the 0.8B is several times quicker). The first real answer
# replaces it with a measurement of THIS model on THIS machine.
TOKENS_PER_SEC_FALLBACK = 14.0


# ─────────────────────────────────────────────────────────────────────────────
#  The personas — Section 4's destinations
#
#  Punch one of these into the time circuits and the same question comes back
#  a completely different answer. That IS the demo.
# ─────────────────────────────────────────────────────────────────────────────

PERSONAS: dict[str, str] = {
    "🤖 Plain assistant": "You are a helpful assistant.",

    "🏛️ Senior appraiser": (
        "You are a senior property appraiser with 25 years of experience in a "
        "county assessor's office. You explain assessment matters plainly and "
        "accurately to members of the public, and you are careful to separate "
        "what you know from what you are guessing."
    ),

    "😤 Angry taxpayer": (
        "You are an angry taxpayer who is convinced the county has over-assessed "
        "your home and is wasting your money."
    ),

    "🔬 Doc Brown": (
        "You are Doc Emmett Brown, the eccentric inventor from Hill Valley, 1985. "
        "You speak with wild enthusiasm and constant scientific tangents."
    ),

    "📋 Hill Valley policy desk": (
        "You are the front counter of the Hill Valley County Assessor's Office. "
        "You answer in at most three sentences, in plain language, and you never "
        "guess. If the answer depends on a specific parcel or on county policy "
        "you have not been given, you say so and tell the caller where to look."
    ),
}

DEFAULT_PERSONA = "🤖 Plain assistant"

QUESTIONS = [
    "My property assessment went up 15% this year. What should I do?",
    "Why do I pay property taxes at all?",
    "What is the difference between market value and assessed value?",
    "The county says my house is worth more than I could sell it for. Is that legal?",
]

DEFAULT_QUESTION = QUESTIONS[0]


# ─────────────────────────────────────────────────────────────────────────────
#  The probes — Section 5's questions it cannot possibly know
#
#  Each of these was tested against this model before it went in here. The
#  `truth` is not an opinion: it comes out of data/policy/ or out of one of the
#  three county databases, and the app shows you which.
# ─────────────────────────────────────────────────────────────────────────────

PROBES: dict[str, dict] = {
    "🏠 Minimum lot size for a split": {
        "question": (
            "In Hill Valley County, what is the minimum lot size to split an "
            "R-1 parcel?"
        ),
        "truth": "6,000 sq ft per resulting parcel (R-1 Single Family).",
        "source": "data/policy/parcel_split_and_combination.md — Policy 4.3",
        "consequence": (
            "This is not a trivia question. Biff Tannen's parcel at 1955 Hilldale Ct "
            "is 15,800 sq ft, so a split produces two 7,900 sq ft lots. Under the "
            "real 6,000 sq ft rule that <b>passes</b>. Under the 0.25 acre "
            "(10,890 sq ft) rule the model invented, it <b>fails</b>. The made-up "
            "number does not just get a fact wrong — it reverses the decision. "
            "Section 10 asks this exact question for real."
        ),
    },

    "📅 Appeal deadline": {
        "question": (
            "What is the deadline to file a property assessment appeal in "
            "Hill Valley County?"
        ),
        "truth": (
            "July 25 — 40 days after the June 15 notice mailing. The deadline is "
            "jurisdictional; the office cannot accept a late appeal."
        ),
        "source": "data/policy/assessment_calendar.md — Policy 2.3",
        "consequence": (
            "A taxpayer who acts on the model's answer misses the real deadline by "
            "months and loses the right to appeal entirely. Nobody finds out until "
            "it is far too late to fix."
        ),
    },

    "📊 What does PRD stand for?": {
        # ⚠️ 1.043 on purpose, and it must stay matched to §7.
        #
        # Hilldale has two true PRDs depending on how you stratify:
        #   all arms-length sales (50)  -> 1.043   <- what §9's agent computes
        #   residential R01/R02 (47)    -> 1.042   <- IAAO's stratification
        # Both are correct. §9 does not filter by property class, so 1.043 is
        # the figure the room sees later, and this probe says §9 will find it.
        # Do not "correct" this to 1.042 without changing §9 too. It read
        # 1.041 until 2026-10-04, which matched neither -- it was a figure that
        # moved when the data was last regenerated and never got re-read.
        "question": (
            "In a ratio study, what does PRD stand for and what does a PRD of "
            "1.043 tell you?"
        ),
        "truth": (
            "Price Related Differential. Above 1.03 is regressive: lower-value "
            "properties are assessed at a higher percentage of value than "
            "higher-value ones. 1.043 fails the IAAO standard — and it is "
            "Hilldale's actual figure, which Section 9 finds for itself."
        ),
        "source": "data/policy/ratio_study_standards.md — Policy 9.4",
        "consequence": (
            "Watch it invent an expansion of the acronym. Run it twice and it will "
            "often invent a <i>different</i> one. This is a term every person in "
            "this room uses weekly — which is exactly why it is a good test. You "
            "can only catch this in a domain you already know."
        ),
    },

    "🔍 Comparable sales": {
        "question": (
            "Give me three comparable sales near 1955 Hilldale Ct in Hill Valley "
            "County, with addresses and sale prices."
        ),
        "truth": None,   # filled in live from clerk.db
        "source": "data/clerk.db — the Clerk of Courts' recorded deeds",
        "consequence": (
            "This is the dangerous one. The answer arrives formatted like real "
            "comps, with addresses, prices and closing dates, often prefaced with "
            "&ldquo;based on the Assessor's records.&rdquo; It never touched our "
            "records. An appraiser could paste this straight into a report."
        ),
    },

    "🌦️ The weather right now": {
        "question": "What is the weather in Calgary right now?",
        "truth": (
            "Unknowable. The model has no clock, no internet, and no sensors. "
            "It was trained once and frozen."
        ),
        "source": "no source exists — that is the point",
        "consequence": (
            "Here the model behaves <b>well</b>: it tells you it cannot know. That "
            "is the honest failure, and it is still a failure — a polite refusal "
            "gets an appraiser exactly as far as a wrong answer does. Section 7 "
            "hands it its first real tool, and questions it refused here start "
            "coming back answered."
        ),
    },
}

DEFAULT_PROBE = "🏠 Minimum lot size for a split"

PLAIN_PERSONA = "You are a helpful assistant."
EXPERT_PERSONA = (
    "You are a senior property appraiser with 25 years of experience in the "
    "Hill Valley County assessor's office. You know the county's parcels, "
    "policies, and sales history thoroughly. Answer confidently and concisely."
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_field(key: str, value: str) -> None:
    """Change the contents of a text box from a button press.

    Streamlit refuses to let you write to `st.session_state[key]` once the
    widget using that key has been created in this run. An `on_click` callback
    runs *before* the next run starts, when the write is still allowed — so
    every button that edits a text box goes through here.
    """
    st.session_state[key] = value


def set_persona(name: str) -> None:
    """Punch a new destination into the time circuits.

    Remembers the persona we are leaving, because that is what the third row
    of the display is for.
    """
    current = st.session_state.get("persona_name")
    if current and current != name:
        st.session_state["last_persona"] = current
    st.session_state["persona_name"] = name
    st.session_state["system_text"] = PERSONAS[name]


@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    """Load a model once and keep it between runs."""
    model, tokenizer = llm.load_local(model_key, verbose=False)
    return model, tokenizer


def gpu_status() -> tuple[float, float] | None:
    """(free_gb, total_gb) or None when there is no GPU."""
    import torch

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return free / 1024**3, total / 1024**3


def stream_answer(model, tokenizer, system: str, question: str,
                  render, *, max_tokens: int = ANSWER_TOKENS) -> str:
    """Generate an answer, calling `render(text_so_far, done)` as it arrives.

    Generation happens on a background thread so this one stays free to update
    the screen. That is the only reason the text appears to type itself rather
    than landing in a single block after a long pause.
    """
    import torch
    from transformers import TextIteratorStreamer

    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    text = llm.chat_prompt(tokenizer, messages)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # ⚠️ `timeout` is not optional. Without it the queue read below blocks
    # FOREVER. generate() runs on the thread further down, and if it raises --
    # an out-of-memory error from a long prompt, a CUDA fault -- nothing ever
    # calls `streamer.end()`. The loop then waits for a token that will never
    # arrive, with no error on screen, and the only way out is restarting the
    # server. In front of a room that is the worst failure available, so we
    # give up after a while and say so instead.
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True,
        timeout=STREAM_TIMEOUT,
    )

    kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        # Greedy, not sampled. On a stage you want the same answer you
        # rehearsed, and the persona differences are dramatic enough without
        # adding randomness on top.
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    def generate() -> None:
        # No gradients: we are not training, and skipping them saves memory.
        with torch.no_grad():
            model.generate(**kwargs)

    # `daemon=True` matters now that the join below is bounded: a generation
    # that has wedged must never be able to stop the interpreter exiting while
    # still holding the graphics card.
    thread = Thread(target=generate, daemon=True)
    thread.start()

    parts: list[str] = []
    stalled = False
    started = time.time()
    try:
        for chunk in streamer:
            parts.append(chunk)
            render("".join(parts), False)
    except queue.Empty:
        stalled = True
    elapsed = time.time() - started

    # Remember how fast this model actually is, so tab 2's "roughly N seconds"
    # stops being a guess after the first answer. The streamer yields one piece
    # per token in practice, which is close enough to call tokens per second.
    if not stalled and len(parts) > 5 and elapsed > 0.2:
        st.session_state[f"tok_per_sec_{st.session_state.get('loaded_model')}"] = (
            len(parts) / elapsed)

    # By here the thread has either finished or died. Never wait on it without
    # a limit -- that would reintroduce the hang the timeout exists to prevent.
    thread.join(timeout=5)

    answer = "".join(parts).strip()
    render(answer, True)

    if stalled:
        st.error(
            f"The model stopped producing text for {STREAM_TIMEOUT:.0f} seconds, "
            f"so this gave up rather than freezing. The usual cause is a very long "
            f"prompt filling the graphics card — shorten it, or switch to a "
            f"smaller model in the sidebar.",
            icon="⏱️",
        )
    return answer


def _fetch(connection, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
    """Run one query and close the connection behind us.

    Worth knowing: `with sqlite3.connect(...)` manages a TRANSACTION, not the
    connection — it does not close anything. Streamlit re-runs this script on
    every click, so a connection left open here would leak one per click.
    """
    try:
        return connection.execute(sql, params).fetchall()
    finally:
        connection.close()


@st.cache_data(show_spinner=False)
def comps_truth() -> str:
    """Build the real answer to 'give me comps near 1955 Hilldale Ct'.

    We do not simply assert that the model's comps are invented — we go to the
    Clerk of Courts and look. Cached because the answer cannot change.
    """
    # Comparables are selected by NEIGHBOURHOOD, not by street name. Streets in
    # this county cross neighbourhood lines the way real ones do, so filtering
    # on "Hilldale Ct" alone would offer a Twin Pines house at $1.3M as a comp
    # for a Hilldale bungalow. 1955 Hilldale Ct is in the Hilldale neighbourhood.
    neighbours = _fetch(
        data.cama(),
        "SELECT parcel_id, situs_address FROM parcels "
        "WHERE neighborhood_code = 'HILL' AND property_class IN ('R01','R02')",
    )
    addr = {r["parcel_id"]: r["situs_address"] for r in neighbours}
    ids = list(addr)

    placeholders = ",".join("?" for _ in ids)
    sales = _fetch(
        data.clerk(),
        f"SELECT parcel_id, recorded_date, sale_price FROM sales "
        f"WHERE parcel_id IN ({placeholders}) AND arms_length = 1 "
        f"ORDER BY recorded_date DESC",
        ids,
    )

    # Did the addresses it invented ever exist? Ask, rather than assert.
    on_street = _fetch(
        data.cama(),
        "SELECT situs_address FROM parcels WHERE situs_address LIKE '%Hilldale Ct'",
    )
    street_addrs = {r["situs_address"] for r in on_street}

    lines = [
        f"The three most recent qualified residential sales in the Hilldale "
        f"neighbourhood ({len(sales)} available):",
        "",
    ]
    for r in sales[:3]:
        lines.append(
            f"  {addr[r['parcel_id']]} — ${r['sale_price']:,} on {r['recorded_date']}"
        )

    # `street_addrs` is a SET, so this is distinct addresses, not parcels --
    # there are 87 parcels on Hilldale Ct but only 84 different addresses, and
    # calling 84 a parcel count would be wrong by three on screen. Distinct
    # addresses is also the number that matters here: the question is whether
    # the address the model invented exists at all.
    lines += ["", f"Distinct addresses on Hilldale Ct anywhere in the county: "
                  f"{len(street_addrs)}."]
    invented = [a for a in ("1952 Hilldale Ct", "1948 Hilldale Ct")
                if a not in street_addrs]
    if invented:
        lines.append("Addresses the model commonly invents here — "
                     + ", ".join(invented) + " — do not exist.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────

st.session_state.setdefault("persona_name", DEFAULT_PERSONA)
st.session_state.setdefault("system_text", PERSONAS[DEFAULT_PERSONA])
st.session_state.setdefault("last_persona", "")
st.session_state.setdefault("question_text", DEFAULT_QUESTION)
st.session_state.setdefault("compare_question", DEFAULT_QUESTION)
# Seeded here, before the widget is built, which is why the multiselect in tab 2
# takes `key=` and not `default=`. See the comment there.
st.session_state.setdefault(
    "compare_personas",
    ["🏛️ Senior appraiser", "😤 Angry taxpayer", "🔬 Doc Brown"],
)
st.session_state.setdefault("probe_question", PROBES[DEFAULT_PROBE]["question"])
st.session_state.setdefault("probe_key", DEFAULT_PROBE)
st.session_state.setdefault("probe_results", None)


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎛️ Time Circuits")
    st.caption("Sections 4 & 5 — prompting, and where it breaks")
    st.divider()

    keys = models.selectable()
    choice = st.selectbox(
        "Model",
        options=keys,
        index=keys.index(DEFAULT_MODEL) if DEFAULT_MODEL in keys else 0,
        format_func=lambda k: f"{models.REGISTRY[k].label}  ({models.REGISTRY[k].params})",
        help="All of these run entirely on your own machine.",
    )
    spec = models.REGISTRY[choice]
    st.caption(f"`{spec.hf_id}`")
    st.caption(f"**License:** {spec.license} · **Precision:** {spec.precision}")

    if spec.role == "tiny":
        st.warning(
            "The 0.8B model **cannot hold a character**. Ask it to be the angry "
            "taxpayer and it breaks role to tell you it is an AI assistant. "
            "That is worth seeing once — then switch back.",
            icon="⚠️",
        )

    st.divider()
    st.slider(
        "Answer length (tokens)", 40, 200, ANSWER_TOKENS, 10,
        key="answer_tokens",
        help="Shorter answers mean less waiting in front of an audience.",
    )

    st.divider()
    status = gpu_status()
    if status:
        free_gb, total_gb = status
        st.metric("GPU memory free", f"{free_gb:.1f} GB", f"of {total_gb:.0f} GB")

        # This button used to call empty_cache() and say nothing, which made it
        # look broken: in the situation a presenter reaches for it, it has
        # nothing to hand back and appears to do nothing at all. Now it reports
        # what actually happened, which is the honest and more interesting
        # answer -- PyTorch holds blocks it is not using, and those are the only
        # thing this button can return.
        if st.button("🧹 Free unused memory", use_container_width=True):
            import torch

            before = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()
            st.session_state["freed_gb"] = (
                before - torch.cuda.memory_reserved()) / 1024**3
            st.rerun()

        freed = st.session_state.pop("freed_gb", None)
        if freed is not None:
            st.caption(
                f"Returned **{freed:.2f} GB** of blocks PyTorch was holding "
                "but not using."
                if freed > 0.01 else
                "Nothing to return — PyTorch is using every block it holds. "
                "This button cannot unload a model; **switching models** is "
                "what does that."
            )
    else:
        st.warning("No GPU found — this will run on the CPU and be slower.", icon="🐢")

    st.divider()
    st.caption(
        "Notebook versions of both demos are in `notebooks/` "
        "(`04_time_circuits`, `05_plutonium`) and go deeper."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("🎛️ Setting the Time Circuits")
st.markdown(
    "A language model wakes up with no memory and no idea who it is. Every "
    "single time you call it.\n\n"
    "What it reads first is the **system prompt** — its purpose in life for "
    "that one call. Punch in a different destination and the same question "
    "comes back a fundamentally different answer."
)

# ── Switching models: let go of the old one before loading the new one ──────
#
# Picking a different model in the sidebar used to load the new one WITHOUT
# releasing the old, so two or three switches filled the card and the next
# answer died with an out-of-memory error.
#
# THREE separate things hold a loaded model, and all three have to let go:
#
#   1. Streamlit's @st.cache_resource     ->  load_model.clear()
#   2. delorean's own cache, llm._CACHE   ->  llm.unload()
#   3. this script's own `model` variable ->  dropped at the BOTTOM of this
#      file, and the comment there explains why it cannot be done here
#
# Measured on an RTX 3080: with 1 and 2 only, a switch leaves 2.90 GB still
# allocated and both models sit on the card at once. With all three, memory
# holds steady at one model however many times you switch.
previous = st.session_state.get("loaded_model")
if previous is not None and previous != choice:
    load_model.clear()      # 1
    llm.unload()            # 2 — also collects garbage and empties the CUDA cache
st.session_state["loaded_model"] = choice

with st.spinner(f"Loading {spec.label} …"):
    model, tokenizer = load_model(choice)

def destination_label() -> str:
    """What to show on the DESTINATION TIME row.

    The preset buttons set a persona *name*, but the system prompt is an
    editable text box. Once you hand-edit it, the name is no longer true — and
    a display that confidently reports the wrong destination is worse than no
    display at all, particularly in a section about how the system prompt is
    the thing that matters.
    """
    name = st.session_state["persona_name"]
    typed = st.session_state["system_text"].strip()

    if not typed:
        return "(no system prompt — the model is nobody)"

    preset = PERSONAS.get(name)
    if preset is not None and typed != preset.strip():
        return f"✏️ edited — was {name}"
    return name


st.html(
    viz.time_circuits_html(
        destination_label(),
        st.session_state["question_text"],
        st.session_state["last_persona"],
    )
)

tab1, tab2, tab3 = st.tabs([
    "🎛️  §4 · Punch In a Destination",
    "⚖️  §4 · Side by Side",
    "🔋  §5 · Running Out of Plutonium",
])


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 1 — one persona at a time
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.subheader("One question. One destination. Change it and watch.")

    st.markdown("**Presets** — click one to load it into the system prompt:")
    cols = st.columns(len(PERSONAS))
    for col, name in zip(cols, PERSONAS):
        col.button(
            name,
            use_container_width=True,
            on_click=set_persona,
            args=(name,),
            key=f"persona_btn_{name}",
        )

    left, right = st.columns([3, 2])

    with left:
        st.text_area(
            "System prompt — *who the model is this time*",
            key="system_text",
            height=140,
            help="This is the DESTINATION TIME row. It is just text, and it is "
                 "prepended to every call.",
        )
        st.text_input("Your question", key="question_text")

        qcols = st.columns(2)
        for n, q in enumerate(QUESTIONS[:2]):
            qcols[n].button(
                f"❓ {q[:38]}…",
                use_container_width=True,
                on_click=set_field,
                args=("question_text", q),
                key=f"q_btn_{n}",
            )

        go = st.button("⚡ Ask", type="primary", use_container_width=True)

    with right:
        st.markdown("**What the model actually receives**")
        st.caption(
            "Not a structure or an object — literally this text. The role "
            "markers are just more tokens."
        )
        preview_messages = []
        if st.session_state["system_text"].strip():
            preview_messages.append(
                {"role": "system", "content": st.session_state["system_text"]}
            )
        preview_messages.append(
            {"role": "user", "content": st.session_state["question_text"]}
        )
        st.code(llm.chat_prompt(tokenizer, preview_messages), language="text")

    if go:
        st.divider()
        placeholder = st.empty()
        name = st.session_state["persona_name"]
        accent = viz.PERSONA_COLORS[list(PERSONAS).index(name) % len(viz.PERSONA_COLORS)] \
            if name in PERSONAS else viz.PERSONA_COLORS[0]

        # The card's heading uses the same label as the DESTINATION TIME row,
        # not the raw preset name. Once you hand-edit the prompt, the preset
        # name is no longer true, and `destination_label()` says so -- a card
        # headed "🏛️ Senior appraiser" above an edited prompt is the exact
        # confidently-wrong display that function exists to prevent.
        card_title = destination_label()

        def render(text: str, done: bool) -> None:
            placeholder.html(
                viz.persona_card_html(
                    card_title, text,
                    system_prompt=st.session_state["system_text"],
                    accent=accent, pending=not done,
                )
            )

        stream_answer(
            model, tokenizer,
            st.session_state["system_text"],
            st.session_state["question_text"],
            render,
            max_tokens=st.session_state["answer_tokens"],
        )

    with st.expander("💡 Try this: delete the system prompt entirely"):
        st.markdown(
            "Clear the box and ask again. You will get something very close to "
            "what **“You are a helpful assistant”** produces — because that is "
            "already what the model was trained to be.\n\n"
            "A bland system prompt is a **no-op**. All of the value is in being "
            "specific: what the model should know, who it is talking to, what it "
            "should refuse to do. That is where your office's policy goes."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 2 — the same question, several destinations at once
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Same question. Same model. Same weights. Different destination.")
    st.caption(
        "Nothing changes between these columns except a sentence of English "
        "text at the top."
    )

    st.text_input("The question they all get", key="compare_question")

    # Two things worth knowing about this widget, both learned the hard way.
    #
    # 1. It uses `key=` and NOT `default=`. Passing `default` to a multiselect
    #    that also has to survive a rerun re-applies that default on every run,
    #    so removing a persona appeared to undo itself. Seeding session_state
    #    once, before the widget exists, is the pattern that actually works.
    #
    # 2. There is no `max_selections`. A hard cap makes Streamlit grey out the
    #    remaining options and demand you "remove an option first", which is
    #    exactly the wrong behaviour when you are standing in front of a room
    #    trying to swap one persona for another. The real cost of picking more
    #    is time, so we show the time instead of forbidding the choice.
    picked = st.multiselect(
        "Personas to compare",
        options=list(PERSONAS),
        key="compare_personas",
        help="Each one is a separate call to the model, so the wait grows with "
             "the number you pick.",
    )

    if picked:
        rate = st.session_state.get(f"tok_per_sec_{choice}")
        seconds = (len(picked) * st.session_state["answer_tokens"]
                   / (rate or TOKENS_PER_SEC_FALLBACK))
        how = (
            f"measured — this model does about **{rate:.0f} tokens a second** on "
            f"this machine"
            if rate else
            f"estimated at {TOKENS_PER_SEC_FALLBACK:.0f} tokens a second; ask "
            f"one question and this becomes a real measurement"
        )
        st.caption(
            f"{len(picked)} persona{'s' if len(picked) != 1 else ''} · "
            f"roughly **{seconds:.0f} seconds** of generation, one after "
            f"another. _({how}.)_"
        )
    if len(picked) > 4:
        st.warning(
            "Above four, the columns get too narrow to read from the back of a "
            "room. Fine at your desk.",
            icon="📽️",
        )

    if st.button("⚡ Run them all", type="primary", use_container_width=True,
                 disabled=not picked):
        columns = st.columns(len(picked))
        placeholders = {}
        for col, name in zip(columns, picked):
            with col:
                placeholders[name] = st.empty()
                placeholders[name].html(
                    viz.persona_card_html(
                        name, "", system_prompt=PERSONAS[name],
                        accent=viz.PERSONA_COLORS[list(PERSONAS).index(name) % len(viz.PERSONA_COLORS)],
                        pending=True,
                    )
                )

        for name in picked:
            accent = viz.PERSONA_COLORS[list(PERSONAS).index(name) % len(viz.PERSONA_COLORS)]
            slot = placeholders[name]

            def render(text: str, done: bool, _n=name, _a=accent, _s=slot) -> None:
                _s.html(
                    viz.persona_card_html(
                        _n, text, system_prompt=PERSONAS[_n],
                        accent=_a, pending=not done,
                    )
                )

            stream_answer(
                model, tokenizer, PERSONAS[name],
                st.session_state["compare_question"], render,
                max_tokens=st.session_state["answer_tokens"],
            )

        st.success(
            "Same question, same weights, same machine. The only difference was "
            "one sentence of English.",
            icon="🎛️",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 3 — Section 5, where it runs out of plutonium
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.subheader("Now ask it something it cannot possibly know.")
    st.markdown(
        "The DeLorean needs plutonium to make the jump. Ask this model "
        "something outside what it was trained on and it has none — "
        "**it cannot get there.** What it does instead is the problem.\n\n"
        "Here is the part that surprises people. Ask a bare model a question "
        "outside what it was trained on and it will usually **tell you it does "
        "not know** — which is correct, honest, and useless.\n\n"
        "Then give it the confident expert persona from Section 4 and ask the "
        "**same question again.**"
    )

    probe_key = st.radio(
        "Pick a question",
        options=list(PROBES),
        horizontal=True,
        key="probe_key",
        on_change=lambda: set_field(
            "probe_question", PROBES[st.session_state["probe_key"]]["question"]
        ),
    )
    probe = PROBES[probe_key]

    st.text_input("The question", key="probe_question")

    if st.button("🔋 Ask it both ways", type="primary", use_container_width=True):
        c1, c2 = st.columns(2)
        slots = {}
        pairs = [
            ("🤖 No persona", PLAIN_PERSONA, viz.PERSONA_COLORS[3], c1),
            ("🏛️ Expert appraiser persona", EXPERT_PERSONA, viz.PERSONA_COLORS[1], c2),
        ]
        for label, _sys, accent, col in pairs:
            with col:
                slots[label] = st.empty()
                slots[label].html(
                    viz.persona_card_html(label, "", accent=accent, pending=True)
                )

        answers = {}
        for label, sys_p, accent, _col in pairs:
            slot = slots[label]

            def render(text: str, done: bool, _l=label, _a=accent, _s=slot) -> None:
                _s.html(viz.persona_card_html(_l, text, accent=_a, pending=not done))

            answers[label] = stream_answer(
                model, tokenizer, sys_p,
                st.session_state["probe_question"], render,
                max_tokens=st.session_state["answer_tokens"],
            )

        st.session_state["probe_results"] = {
            "key": probe_key,
            "expert": answers["🏛️ Expert appraiser persona"],
            "question": st.session_state["probe_question"],
        }

    results = st.session_state.get("probe_results")
    if results and results["key"] == probe_key:
        st.divider()
        st.markdown("### 🔍 So which one was right?")

        truth = probe["truth"]
        if truth is None:                      # the comps probe checks live
            truth = comps_truth()

        st.html(
            viz.verification_html(
                results["expert"] or "(no answer)",
                truth,
                source=probe["source"],
                question=results["question"],
                consequence=probe["consequence"],
            )
        )

        st.info(
            "The model is not broken and it is not lying. It has no clock, no "
            "internet, and no connection to your CAMA system. It was trained "
            "once and frozen — so when it does not know, it does the thing it "
            "was built to do: **predict a plausible next word.** A plausible "
            "answer and a correct one look identical from the outside.",
            icon="🔋",
        )

        st.success(
            "**That is the whole problem, and it is fixable.** Section 6 explains "
            "how an agent reasons about what it needs. Section 7 hands it real "
            "tools. Section 9 points those tools at this county's actual data — "
            "and this question gets a real answer.",
            icon="⏭️",
        )

    with st.expander("💡 Why the persona makes it worse"):
        st.markdown(
            "A model with no persona is free to say *“I don't know.”* Nothing in "
            "its instructions makes that embarrassing.\n\n"
            "A model told it is a **25-year expert who knows this county "
            "thoroughly** is being asked to produce text consistent with that "
            "description. *“I don't know”* is inconsistent with it. A confident, "
            "specific, official-sounding answer **is** consistent with it — so "
            "that is what gets generated, whether or not any of it is true.\n\n"
            "This matters because **the persona is the useful part.** Nobody "
            "deploys the bland one. Every office that puts a model in front of "
            "staff will write exactly the kind of confident expert prompt that "
            "causes this."
        )

    with st.expander("⚠️ One honest caveat"):
        st.markdown(
            "This failure is **not reliable**, and that is worse, not better.\n\n"
            "The same expert persona will invent three comparable sales complete "
            "with closing dates, and then correctly refuse to describe a sale "
            "that never happened. You cannot tell in advance which you are going "
            "to get, and the confident wrong answer looks exactly like the "
            "confident right one.\n\n"
            "If it failed every time, you could plan around it."
        )

    # ── The wrap-up, and the hand-off to Section 6 ──────────────────────────
    #
    # INSIDE tab 3, not under the whole app. The app covers two sections in
    # sequence, so a footer below the tab strip would sit there announcing
    # §5's punchline while the room is still watching §4 build up to it. It
    # belongs at the end of the story, which is the end of this tab.
    st.divider()
    st.markdown(
        "#### 🎬 That's Sections 4 and 5\n"
        "**Section 4 — the good news.** A system prompt is not a setting or a "
        "configuration screen. It is **text, glued to the front of your "
        "question**, and it turns the same weights into a different colleague. "
        "That is most of what people mean by *customising* an assistant.\n\n"
        "**Section 5 — the bill.** The same confident persona will invent an "
        "ordinance, a filing deadline, or three comparable sales with closing "
        "dates — and none of it will look like a mistake. **A plausible answer "
        "and a correct one are indistinguishable from the outside.**\n\n"
        "The model is not short of intelligence here. It is short of "
        "**access** — to your roll, your sales file, your policy manual, "
        "today's date. **That is what Section 6 starts to fix.**"
    )
    st.caption(
        "🕰️ *“Roads? Where we're going, we don't need roads.”* — Dr. Emmett "
        "Brown … but we are going to need some tools."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Reference 3 of 3 — let go of the weights before this run ends
#
#  This one line is what makes the release at the top of the file work, and it
#  is the least obvious thing in this app.
#
#  Streamlit runs this script again on every click, in a brand new module
#  namespace each time. You would expect the previous run's `model` variable to
#  disappear with that namespace -- but Streamlit keeps THIS run's widget
#  callbacks alive into the NEXT run. It stores them in widget metadata that is
#  only replaced when each widget re-registers, which happens far below the
#  release code at the top. And a Python function carries its entire module
#  namespace with it, in `__globals__`.
#
#  So every `on_click=` button above is quietly holding this whole script --
#  including `model` -- at the exact moment the next run is trying to free it.
#  No amount of cache clearing or garbage collecting can touch a model that is
#  still reachable.
#
#  Dropping our own two names here means that by the time the next run asks for
#  the memory back, nothing is holding it. Measured: without this line a switch
#  frees 0.00 GB and both models sit on the card; with it, the old model is
#  gone before the new one loads.
# ─────────────────────────────────────────────────────────────────────────────

del model, tokenizer
