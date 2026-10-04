"""
02_Flux_Capacitor_local_Streamlit.py
====================================

The Section 2 demo — "What *is* a Large Language Model?" — as a clickable app
instead of a notebook.

    uv run streamlit run app/02_Flux_Capacitor_local_Streamlit.py

Everything runs on your own machine. No API key, nothing leaves the building.

Why this exists alongside the notebook
--------------------------------------
The notebook is the *explanation*: read it top to bottom and it teaches you.
This is the *instrument*: someone in the audience says "what about MY parcel
number?" and you type it in and everybody watches what happens.

It shows the same three things, in the same order:

    1. Tokens       your words stop being words
    2. Prediction   one guess, out of every token the model knows
    3. Streaming    do it again, and again -- that's generation

The one thing this can do that the notebook cannot is the "Take the winner"
button in tab 2. Press it and the winning token is glued onto the sentence and
the model predicts again, so you can walk the generation loop a single step at
a time and watch the odds move. That is the whole of tab 3, in slow motion.
"""

from __future__ import annotations

import queue
import sys
from pathlib import Path
from threading import Thread

import streamlit as st

# Allow running this file directly from anywhere in the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from delorean import llm, models, viz  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
#  Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="The Flux Capacitor",
    page_icon="⚡",
    layout="wide",
)

PRESET_TOKENS = [
    "appraisal",
    "The homestead exemption",
    "parcel 04-127-3891",
    "$412,500",
]

PRESET_PROMPTS = [
    "Location, location,",
    "An appraiser estimates the fair market",
    "The homestead exemption reduces a homeowner's property",
    "Property taxes are based on the assessed",
]

PRESET_QUESTIONS = [
    "What are the three approaches to value in real property appraisal?",
    "In one sentence, what does a property appraiser do?",
    "Explain the difference between market value and assessed value.",
]

DEFAULT_PREDICT = "Location, location,"

# §2 runs on the 0.8B on purpose: it is fast, it loads in seconds, and nothing
# in this section needs a bigger model. Named rather than `index=0` so that
# adding a model to the registry cannot silently make §2 load a 4B — the other
# four apps all pin their default the same way.
DEFAULT_MODEL = "tiny"

# How long tab 3 waits for the next token before deciding something is wrong.
# Generous: this is the gap between *consecutive* tokens, and the 0.8B produces
# about fourteen a second. See the comment at the streamer for why it exists.
STREAM_TIMEOUT = 60.0


def set_field(key: str, value: str) -> None:
    """Change the contents of a text box from a button press.

    Streamlit refuses to let you write to `st.session_state[key]` once the
    widget using that key has been created in this run. An `on_click` callback
    runs *before* the next run starts, when the write is still allowed — so
    every button that edits a text box goes through here.
    """
    st.session_state[key] = value


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
#
#  Streamlit re-runs this entire script every time you click anything. Without
#  the cache below, that would mean reloading gigabytes of model on every
#  keystroke. `cache_resource` keeps the loaded model in memory between runs.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    """Load a model once and keep it. Returns (model, tokenizer, vocab_size)."""
    model, tokenizer = llm.load_local(model_key, verbose=False)
    return model, tokenizer, llm.vocab_size(model)


def gpu_status() -> tuple[float, float] | None:
    """(free_gb, total_gb) or None when there is no GPU."""
    import torch

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return free / 1024**3, total / 1024**3


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ The Flux Capacitor")
    st.caption("Section 2 — what a Large Language Model actually is")
    st.divider()

    keys = models.selectable()
    choice = st.selectbox(
        "Model",
        options=keys,
        format_func=lambda k: f"{models.REGISTRY[k].label}  ({models.REGISTRY[k].params})",
        index=keys.index(DEFAULT_MODEL) if DEFAULT_MODEL in keys else 0,
        help="All of these run entirely on your own machine.",
    )
    spec = models.REGISTRY[choice]

    st.caption(f"`{spec.hf_id}`")
    st.caption(f"**License:** {spec.license} · **Precision:** {spec.precision}")

    if spec.role != "tiny":
        st.info(
            f"{spec.label} is a bigger download (several GB) and is meant for the "
            "agent demos in Sections 7-11. The small model is plenty for this one.",
            icon="ℹ️",
        )

    st.divider()

    status = gpu_status()
    if status:
        free_gb, total_gb = status
        st.metric("GPU memory free", f"{free_gb:.1f} GB", f"of {total_gb:.0f} GB")

        # Reports what it actually did. This button can only hand back blocks
        # PyTorch is holding but not using -- it cannot unload a loaded model.
        # Saying so is better than appearing to do nothing.
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
        "The notebook version of this demo is in `notebooks/` and explains "
        "everything in more depth."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("⚡ What *is* a Large Language Model?")

st.markdown(
    "Everybody has watched a chatbot type an answer back at them. Almost nobody "
    "has been shown **what is actually happening** while it does that.\n\n"
    "It is not looking anything up. There is no answer stored inside it waiting "
    "to be found. It is doing one thing, very fast, over and over: "
    "**guessing the next small piece of text.**\n\n"
    "Doc Brown built one component and called it the thing that *makes time "
    "travel possible*. This is that component. Agents, tools, document search, "
    "the three county offices MCP — **all of it sits on top of what "
    "happens in these three tabs.**"
)

# ── Switching models: let go of the old one before loading the new one ──────
#
# Three separate things hold a loaded model, and all three have to let go or
# nothing is freed and the card fills up after two or three switches:
#
#   1. Streamlit's @st.cache_resource     ->  load_model.clear()
#   2. delorean's own cache, llm._CACHE   ->  llm.unload()
#   3. this script's own `model` variable ->  dropped at the BOTTOM of this
#      file, where the comment explains why it cannot be done here
previous = st.session_state.get("loaded_model")
if previous is not None and previous != choice:
    load_model.clear()      # 1
    llm.unload()            # 2 — also collects garbage and empties the CUDA cache
st.session_state["loaded_model"] = choice

with st.spinner(f"Loading {spec.label} …"):
    model, tokenizer, VOCAB = load_model(choice)

st.success(
    f"**{spec.label}** is loaded and running on this machine. "
    f"It chooses between **{VOCAB:,}** different tokens.",
    icon="✅",
)

tab_tokens, tab_predict, tab_stream = st.tabs([
    "🔢 &nbsp;1. Words become numbers",
    "🎲 &nbsp;2. One prediction",
    "🔁 &nbsp;3. Generation",
])


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 1 — Tokens
# ─────────────────────────────────────────────────────────────────────────────

with tab_tokens:
    st.subheader("Your words stop being words")
    st.markdown(
        "Before the model sees your text, it gets chopped into **tokens** — "
        "pieces roughly the size of a syllable — and each piece is swapped for a "
        "number. The model only ever sees the numbers."
    )

    if "token_text" not in st.session_state:
        st.session_state.token_text = "appraisal"

    st.caption("Try one of these, or type anything you like:")
    cols = st.columns(len(PRESET_TOKENS))
    for col, preset in zip(cols, PRESET_TOKENS):
        col.button(
            preset,
            use_container_width=True,
            key=f"tok_{preset}",
            on_click=set_field,
            args=("token_text", preset),
        )

    st.text_input("Text to break apart", key="token_text", label_visibility="collapsed")
    text = st.session_state.token_text

    if text.strip():
        ids = tokenizer.encode(text, add_special_tokens=False)
        a, b, c = st.columns(3)
        a.metric("Words", len(text.split()))
        b.metric("Tokens", len(ids))
        c.metric("Characters", len(text))

        st.html(viz.tokens_html(tokenizer, text, show_source=False))

        st.caption(
            "The small number under each chip is that token's id — the actual "
            "integer handed to the model. The faint dots are spaces."
        )

        digits = sum(ch.isdigit() for ch in text)
        if digits >= 4:
            # Same wording as the notebook's Act 1 callout, including the count,
            # so the twins say the same thing in the same voice.
            st.warning(
                f"**Great Scott — {len(ids)} tokens.** Every digit became its own "
                f"piece. The model does not see `{text.strip()}` as an identifier — "
                f"it sees {len(ids)} unrelated fragments in a row.\n\n"
                f"**Remember this.** It is why language models fumble parcel numbers, "
                f"account numbers and arithmetic. The fix is not a smarter model — it "
                f"is giving it a calculator, which is what a *tool* is (Section 7).",
                icon="⚠️",
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 2 — Next-token prediction
# ─────────────────────────────────────────────────────────────────────────────

def next_token_odds(text: str, k: int = 5) -> list[tuple[str, float]]:
    """Ask the model what comes next. Returns the top k (token, probability)."""
    import torch

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        scores = model(**inputs).logits[0, -1, :]   # a score for EVERY token

    odds = torch.softmax(scores.float(), dim=-1)    # scores -> percentages
    top_odds, top_ids = torch.topk(odds, k)

    return [(tokenizer.decode([i]), p)
            for p, i in zip(top_odds.tolist(), top_ids.tolist())]


with tab_predict:
    # Read from the loaded model, never written as a literal: the five models
    # in the picker have vocabularies from 128,256 to 262,144, and a hardcoded
    # figure here contradicted both the banner above and the chart below the
    # moment anyone switched.
    st.subheader(f"One guess, out of {VOCAB:,} options")
    st.markdown(
        "Give the model some text and it produces a score for **every single "
        "token it knows**, saying how likely each one is to come next. "
        "Here are the front-runners."
    )

    if "predict_text" not in st.session_state:
        st.session_state.predict_text = DEFAULT_PREDICT

    st.caption("Try one of these, or write your own half-finished sentence:")
    for row in range(0, len(PRESET_PROMPTS), 2):
        cols = st.columns(2)
        for col, preset in zip(cols, PRESET_PROMPTS[row:row + 2]):
            col.button(
                preset,
                use_container_width=True,
                key=f"pred_{preset}",
                on_click=set_field,
                args=("predict_text", preset),
            )

    st.text_area(
        "Unfinished sentence",
        key="predict_text",
        height=80,
        label_visibility="collapsed",
    )
    prompt = st.session_state.predict_text

    k = st.slider("How many candidates to show", 3, 12, 5)

    if prompt.strip():
        pairs = next_token_odds(prompt, k)
        st.html(viz.probabilities_html(pairs, prompt=prompt, vocab_size=VOCAB))

        winner, winner_p = pairs[0]

        # Three buttons, not one. Taking the winner shows the loop; being able
        # to take second or third shows that the winner was a *choice* -- which
        # is the whole idea behind temperature, and the reason every demo in
        # this repository pins `do_sample=False`. The expander below says so.
        #
        # ⚠️ Explicit `key=` on each. Two different token ids can decode to the
        # same text (" Location" and "Location", say), and two buttons with the
        # same label and no key are the same widget as far as Streamlit is
        # concerned -- a DuplicateWidgetID error, mid-demo.
        st.markdown("**Take one step by hand — and notice you have a choice:**")
        places = ("▶ Take the winner", "② Take 2nd place", "③ Take 3rd place")
        for n, (col, (tok, prob)) in enumerate(zip(st.columns(3), pairs[:3])):
            col.button(
                f"{places[n]}  ·  `{tok}`",
                use_container_width=True,
                type="primary" if n == 0 else "secondary",
                key=f"take_{n}",
                help=f"Append `{tok}` ({prob * 100:.1f}%) and predict again.",
                on_click=set_field,
                args=("predict_text", prompt + tok),
            )

        if winner_p > 0.80:
            st.success(
                f"The model is **very confident** here — {winner_p*100:.1f}% on a "
                "single token. It has seen this pattern many, many times.",
                icon="🎯",
            )
        elif winner_p < 0.35:
            st.info(
                f"The model is **genuinely torn**. The favourite is only "
                f"{winner_p*100:.1f}%, and the runners-up are all plausible. "
                "You are watching it weigh the evidence.",
                icon="🤔",
            )
        else:
            st.info(
                f"A clear favourite at {winner_p*100:.1f}%, but not a certainty. "
                "Every answer is a chain of bets like this one.",
                icon="🎲",
            )

        st.caption(
            "**Press ▶ Take the winner a few times.** Each press glues that "
            "token onto the sentence and asks again — which is exactly what "
            "generation is, one step at a time. Tab 3 does this a few hundred "
            "times a second."
        )

        with st.expander("🎲 So why is there ever a second place? — this is *temperature*"):
            st.markdown(
                "Every demo in this repository always takes **first place**. That "
                "is called **greedy** decoding, and it is why you get the same "
                "answer every time — which is what you want on a stage, and what "
                "you want in an assessment office.\n\n"
                "But a model does not have to. It can be told to **sometimes take "
                "the runner-up instead**, and the dial controlling how willing it "
                "is to wander down the list is called **temperature**. Turn it up "
                "and the writing feels less robotic. Turn it up too far and it "
                "starts inventing.\n\n"
                "**Pressing ② or ③ above is you doing by hand what temperature "
                "does with dice.** Try it a few times and watch the sentence go "
                "somewhere the winning path would never have taken it — then "
                "press ↺ and walk the greedy path instead. Same model, same "
                "weights, different road.\n\n"
                "⚙️ In code this is one argument, `do_sample=False`, and it is set "
                "deliberately in every demo here so that what you rehearse is "
                "what you get."
            )

        st.button(
            "↺ Reset the sentence",
            on_click=set_field,
            args=("predict_text", DEFAULT_PREDICT),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 3 — Generation
# ─────────────────────────────────────────────────────────────────────────────

with tab_stream:
    st.subheader("Do it again. And again. And again.")
    st.markdown(
        "If the model only predicts *one* token, how do you get a paragraph? "
        "You take the winner, glue it on the end, and **ask the same question "
        "again** — until it decides to stop. That loop is called *generation*."
    )

    if "question" not in st.session_state:
        st.session_state.question = PRESET_QUESTIONS[0]

    st.caption("Pick a question, or ask your own:")
    for preset in PRESET_QUESTIONS:
        st.button(
            preset,
            use_container_width=True,
            key=f"q_{preset}",
            on_click=set_field,
            args=("question", preset),
        )

    st.text_area("Your question", key="question", height=80, label_visibility="collapsed")

    opt1, opt2 = st.columns([1, 2])
    with opt1:
        boundaries = st.toggle(
            "Show token boundaries",
            value=True,
            help="Shade each token separately so you can see the pieces arriving.",
        )
    with opt2:
        max_tokens = st.slider("Maximum length", 20, 400, 120, step=20)

    if st.button("⚡ Generate", type="primary", use_container_width=True):
        from transformers import TextIteratorStreamer

        messages = [{"role": "user", "content": st.session_state.question}]
        text_prompt = llm.chat_prompt(tokenizer, messages)
        inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)

        # The streamer hands us each token the moment it is produced, rather
        # than making us wait for the whole answer.
        #
        # ⚠️ `timeout` is not optional. Without it the queue below blocks
        # FOREVER. generate() runs on the thread that follows, and if it
        # raises -- an out-of-memory error from a very long question, a CUDA
        # fault -- nothing ever calls `streamer.end()`. The loop then waits for
        # a token that will never arrive, with no error and no spinner, and the
        # only way out is restarting the server. In front of a room, that is
        # the worst failure available. A timeout turns it into a message.
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True,
            timeout=STREAM_TIMEOUT,
        )

        # generate() blocks, so it runs on a background thread while we read
        # the stream here and redraw after every piece. `daemon=True` so a
        # wedged generation can never stop the interpreter exiting.
        Thread(
            target=model.generate,
            kwargs=dict(**inputs, streamer=streamer,
                        max_new_tokens=max_tokens, do_sample=False),
            daemon=True,
        ).start()

        placeholder = st.empty()
        pieces: list[str] = []
        stalled = False
        try:
            for piece in streamer:
                pieces.append(piece)
                placeholder.html(
                    viz.canvas_html(pieces, title=spec.label, boundaries=boundaries)
                )
        except queue.Empty:
            stalled = True

        if stalled:
            st.error(
                f"The model stopped producing text for {STREAM_TIMEOUT:.0f} seconds, "
                f"so this gave up rather than freezing. The usual cause is a very "
                f"long question filling the graphics card — try a shorter one, or "
                f"the 0.8B model.",
                icon="⏱️",
            )
        else:
            st.caption(
                f"That answer took **{len(pieces)} passes** through the model. Each "
                "one produced a fresh set of odds like the ones in tab 2, and the "
                "winner was appended before going round again."
            )
            st.caption(
                f"⚡ The DeLorean needed 1.21 gigawatts to make **one** jump. This "
                f"made **{len(pieces)} complete passes** through all {spec.params} "
                f"of the model's numbers to write one paragraph — which is why "
                f"Section 3 cares so much about how big the model is."
            )
    else:
        st.info("Press **⚡ Generate** to watch it happen.", icon="👆")

    # ── The wrap-up ─────────────────────────────────────────────────────────
    #
    # INSIDE tab 3, not under the whole app. A footer below the tab strip sits
    # there announcing "that's Section 2" while the room is still on tab 1
    # watching words turn into numbers — the conclusion arriving before the
    # argument. It belongs at the end of the story, which is the end of the
    # last tab.
    st.divider()
    st.markdown(
        "#### 🎬 That's Section 2\n"
        "A language model is a **next-token guesser** — a very sophisticated "
        "one, but a guesser. Two things to carry forward:\n\n"
        "1. **It sees tokens, not words**, which is why identifiers and "
        "arithmetic are a weak spot *(Section 5)*\n"
        "2. **Every answer is a chain of bets**, which is why *confident* does "
        "not mean *correct*\n\n"
        "And notice what it never did: it had no idea who it was, or that it "
        "was talking to an appraiser. **That's Section 4.**"
    )
    st.caption(
        '"Your future hasn\'t been written yet. No one\'s has. '
        'Your future is whatever you make it. So make it a good one." '
        "— Dr. Emmett Brown"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Reference 3 of 3 — let go of the weights before this run ends
#
#  Streamlit runs this script again on every click, in a brand new namespace
#  each time, so you would expect the previous run's `model` variable to vanish
#  with it. It does not: Streamlit keeps THIS run's widget callbacks alive into
#  the NEXT run, and a Python function carries its whole module namespace with
#  it in `__globals__`. Every `on_click=` above is therefore still holding this
#  script -- and `model` with it -- at the moment the next run tries to free
#  the memory. Nothing can release a model that is still reachable.
#
#  Dropping our own names here is what makes the release at the top work.
#  Full measurements: Plans/vram_model_switching.md.
# ─────────────────────────────────────────────────────────────────────────────

del model, tokenizer
