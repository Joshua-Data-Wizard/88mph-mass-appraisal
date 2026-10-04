"""
07_08_Office_Assistant_local_Streamlit.py
=========================================

Sections 7 and 8, in one app.

    uv run streamlit run app/07_08_Office_Assistant_local_Streamlit.py

    Section 7  "The Hoverboard"          -> tab 1, ONE tool
    Section 8  "Grays Sports Almanac"    -> tab 2, the SAME agent, one more tool

Everything runs on your own machine. No API key, nothing leaves the building.

Why these two sections share one app
------------------------------------
Because the point of Section 8 is that RAG is **not** a different kind of magic.
It is a tool. Same agent, same Reason -> Act -> Observe loop, same code path --
one more entry in a list. Building it as a second tab on the same agent makes
that argument visible instead of asserting it from a slide. Section 9 sits
between them in the talk, so you will reopen this app after it.

The system prompt is the third demo hiding in here
--------------------------------------------------
Section 4 taught that a system prompt changes how a model *sounds*. Here it does
something harder: it decides when the agent should reach for a tool and when it
should just answer. That is not a stylistic claim, it is measured. With three
tools offered and no guardrail, this model searched the policy manual to answer
"what is the difference between market value and assessed value?". With the
guardrail prompt it answers directly -- and still calls the right tool every
time it genuinely needs one.

    toolbelt          restraint     recall
    1 tool            6/6           1/1
    2 tools           6/6           2/2
    3 tools           5/6  <- slips 4/4
    3 tools + guard   6/6           4/4

So the system prompt panel is not decoration. Open it, switch it off, and watch
the agent get worse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from delorean import agent, data, llm, models, tools, viz  # noqa: E402

st.set_page_config(page_title="The Office Assistant", page_icon="🛹",
                   layout="wide")

DEFAULT_MODEL = "qwen"

# ─────────────────────────────────────────────────────────────────────────────
#  The system prompts — Section 4's lesson, doing real work
# ─────────────────────────────────────────────────────────────────────────────

GUARDRAIL = (
    "You are the front counter of the Hill Valley County Assessor's Office.\n"
    "\n"
    "Use lookup_parcel or search_parcels ONLY when the question is about a "
    "specific property, owner or address.\n"
    "Use search_policy ONLY for questions about THIS office's own rules, "
    "deadlines, standards or procedures.\n"
    "Answer general appraisal questions, definitions and small talk directly "
    "from your own knowledge. Do not use a tool for those.\n"
    "\n"
    "When you use a tool, answer from what it returned. When you quote office "
    "policy, name the section you took it from."
)

OVERCONFIDENT = (
    "You are a senior property appraiser with 25 years of experience in the "
    "Hill Valley County assessor's office. You know the county's parcels, "
    "policies, and sales history thoroughly. Answer confidently and concisely."
)

SYSTEM_PRESETS: dict[str, str] = {
    "🎛️ Front counter (the guardrail)": GUARDRAIL,
    "🤖 No system prompt at all": "",
    "🏛️ The Section 5 expert": OVERCONFIDENT,
}

DEFAULT_PRESET = "🎛️ Front counter (the guardrail)"

# ── The questions ────────────────────────────────────────────────────────────

HOVERBOARD_QUESTIONS = [
    "Who owns parcel 04-127-3891 and what is it assessed at?",
    "Pull up 4-127-3891.",
    "How big is the lot at parcel 04-127-3891?",
    "What is the difference between market value and assessed value?",
]

ALMANAC_QUESTIONS = [
    "Can a parcel be split if the owner owes back taxes?",
    "In Hill Valley County, what is the minimum lot size to split an R-1 parcel?",
    "When is the deadline to file an assessment appeal?",
    "Which deed types does this office treat as qualified sales?",
]


def set_field(key: str, value: str) -> None:
    """Change a text box from a button press.

    Streamlit forbids writing to `st.session_state[key]` once the widget with
    that key exists in the same run. An `on_click` callback runs before the
    next run, when the write is still allowed.
    """
    st.session_state[key] = value


def load_preset(name: str) -> None:
    st.session_state["preset_name"] = name
    st.session_state["system_text"] = SYSTEM_PRESETS[name]


@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    return llm.load_local(model_key, verbose=False)


@st.cache_resource(show_spinner=False)
def build(model_key: str, tool_names: tuple[str, ...]):
    """One compiled graph per toolbelt. Cached so a click does not rebuild it."""
    model, tokenizer = load_model(model_key)
    responder = agent.local_responder(model, tokenizer)
    return agent.build_agent(responder, list(tool_names)), responder


def gpu_status() -> tuple[float, float] | None:
    import torch

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return free / 1024**3, total / 1024**3


def run_agent(question: str, system: str, tool_names: list[str], slot,
              *, model_key: str) -> None:
    """Run one question, drawing each step of the loop as it lands.

    The graph is streamed rather than awaited so the audience watches Reason,
    then Act, then Observe appear one at a time. Waiting seven seconds and then
    showing a finished diagram teaches much less.

    The graph is built in here rather than passed in so that no variable in the
    script's own namespace ends up holding the model -- which matters for
    releasing it on a switch. See the comment at the bottom of this file.
    """
    graph, responder = build(model_key, tuple(tool_names))

    messages: list[dict] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    state: dict = {"messages": messages, "trace": [], "steps": 0}

    # The loop calls the model more than once, and `last_raw` only ever holds
    # the most recent call -- which is the one that wrote the final ANSWER, not
    # the one that wrote the tool call. Keep the first output that actually
    # contained a tool call, because that is the text worth showing.
    raw_with_call = ""

    for chunk in graph.stream(state, stream_mode="updates"):
        for node, delta in chunk.items():
            state.update(delta)
            if node == "reason" and not raw_with_call:
                candidate = getattr(responder, "last_raw", "") or ""
                if tools.attempted_tool_call(candidate):
                    raw_with_call = candidate
            slot.html(viz.trace_html(state["trace"], question=question))

    answer = agent.final_answer(state)
    slot.html(viz.trace_html(state["trace"], answer=answer, question=question))

    used = [e["tool"] for e in state["trace"] if e["step"] == "act"]
    # Fall back to the last output when nothing asked for a tool -- then the
    # plain answer IS what the model literally wrote.
    st.session_state["last_raw"] = raw_with_call or getattr(responder, "last_raw", "")
    st.session_state["last_used"] = used
    st.session_state["last_offered"] = tool_names


# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────

st.session_state.setdefault("preset_name", DEFAULT_PRESET)
st.session_state.setdefault("system_text", SYSTEM_PRESETS[DEFAULT_PRESET])
st.session_state.setdefault("hover_q", HOVERBOARD_QUESTIONS[0])
st.session_state.setdefault("almanac_q", ALMANAC_QUESTIONS[0])
st.session_state.setdefault("last_raw", "")
st.session_state.setdefault("last_used", [])
st.session_state.setdefault("last_offered", [])

# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛹 The Office Assistant")
    st.caption("Sections 7 & 8 — tools, and then RAG as one more tool")
    st.divider()

    keys = models.selectable()
    choice = st.selectbox(
        "Model", options=keys,
        index=keys.index(DEFAULT_MODEL) if DEFAULT_MODEL in keys else 0,
        format_func=lambda k: f"{models.REGISTRY[k].label}  ({models.REGISTRY[k].params})",
    )
    spec = models.REGISTRY[choice]
    st.caption(f"`{spec.hf_id}`")

    if spec.role == "tiny":
        st.warning(
            "The 0.8B model is not built for tool calling. Useful to try once, "
            "then switch back — the agent demos need a model that can think.",
            icon="⚠️",
        )

    # Say what is missing from that list and why. A shorter dropdown with no
    # explanation looks like we never tried the other model; we did, and the
    # reason it is not here is worth more than the entry would have been.
    for missing in models.blocked():
        st.caption(f"⚠️ **{missing.label}** is not offered — {missing.blocked_reason}")

    # ── Switching models: let go of the old one first ───────────────────────
    #
    # Three separate things hold a loaded model, and all three have to let go
    # or nothing is freed and the card fills up after two or three switches:
    #
    #   1. Streamlit's @st.cache_resource    ->  load_model.clear()
    #      ...and `build`, whose compiled graph closes over the same weights
    #   2. delorean's own cache, llm._CACHE  ->  llm.unload()
    #   3. this script's own variables       ->  dropped at the BOTTOM of this
    #      file, where the comment explains why it cannot be done here
    previous = st.session_state.get("loaded_model")
    if previous is not None and previous != choice:
        load_model.clear()      # 1
        build.clear()           # 1 — the graph holds the responder holds the model
        llm.unload()            # 2 — also collects garbage, empties the CUDA cache
    st.session_state["loaded_model"] = choice

    st.divider()

    # Show where this is actually running. Half demo feature, half diagnostic:
    # the audience should see that nothing is leaving the building, and anyone
    # who notices their CPU fan should be able to check the weights are on the
    # graphics card where they belong.
    st.markdown("**Where this is running**")
    loaded_model, _tokenizer = load_model(choice)
    where = llm.where_it_runs(loaded_model)

    if where["cuda_available"]:
        st.success(f"🖥️ {where['device_name']}", icon="✅")
        st.caption(
            f"**{where['on_gpu']} of {where['on_gpu'] + where['on_cpu']}** weight "
            f"tensors on the GPU · {where['allocated_gb']:.2f} GB in use"
        )
        st.metric("GPU memory free", f"{where['free_gb']:.1f} GB",
                  f"of {where['total_gb']:.0f} GB")
        if where["on_cpu"]:
            st.warning(f"{where['on_cpu']} tensors fell back to the CPU.",
                       icon="⚠️")
    else:
        st.warning("No GPU found — this will run on the CPU and be slow.",
                   icon="🐢")

    with st.expander("Why is my CPU busy?"):
        st.markdown(
            "Both things are true at once. The **weights and the maths are on "
            "the graphics card**; Python is busy launching one small kernel per "
            "token, which is ordinary for generating a word at a time.\n\n"
            "Two other things here are on the CPU **on purpose**:\n\n"
            "- **The embedding model** for `search_policy` (`bge-small`, 33M "
            "parameters). It runs in about a second and leaves the VRAM free "
            "for the language model — and it means an office with no graphics "
            "card can still do retrieval.\n"
            "- Tokenizing your question, which is pure text handling.\n\n"
            "`.streamlit/config.toml` also switches off Streamlit's file "
            "watcher. Without that it walked ~5,500 `transformers` submodules "
            "on **every click**, which burned real CPU and printed a wall of "
            "harmless `torchvision` errors."
        )

    st.divider()
    st.caption(
        "The notebook versions (`07_hoverboard`, `08_almanac`) build the tool "
        "up from a plain Python function and go deeper."
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("🛹 Give it a tool and it stops guessing")
st.markdown(
    "In Section 5 we asked this model about a parcel it had never seen, and it "
    "**invented a zoning ordinance**. Nothing about the model has changed since "
    "then. The only difference now is that it is allowed to go and look."
)

with st.spinner(f"Loading {spec.label} …"):
    load_model(choice)
data.check()

# ── The system prompt panel — Section 4, still earning its keep ──────────────

n_tokens = len(llm.load_tokenizer(choice).encode(st.session_state["system_text"])) \
    if st.session_state["system_text"].strip() else 0

with st.expander(
    f"🎛️  **The system prompt driving this agent**  ·  "
    f"{st.session_state['preset_name']}  ·  {n_tokens} tokens, re-sent every call",
    expanded=False,
):
    st.markdown(
        "Section 4 showed that a system prompt changes how a model **sounds**. "
        "Here it does something harder: it decides **when the agent reaches for "
        "a tool** and when it should simply answer.\n\n"
        "That is measured, not asserted. With three tools available and no "
        "guardrail, this model searched the policy manual to answer *“what is "
        "the difference between market value and assessed value?”* — a question "
        "it knew perfectly well. Restraint came back the moment the prompt told "
        "it when not to look."
    )

    cols = st.columns(len(SYSTEM_PRESETS))
    for col, name in zip(cols, SYSTEM_PRESETS):
        col.button(name, use_container_width=True, on_click=load_preset,
                   args=(name,), key=f"sysbtn_{name}")

    st.text_area(
        "Edit it and watch the agent's behaviour change",
        key="system_text", height=210,
    )

    st.caption(
        "**Try this on stage:** load *No system prompt at all*, go to the "
        "Almanac tab, and ask a general question. Then put the guardrail back."
    )
    st.markdown(
        "| toolbelt | restraint | recall |\n|---|---|---|\n"
        "| 1 tool | 6/6 | 1/1 |\n| 2 tools | 6/6 | 2/2 |\n"
        "| **3 tools** | **5/6** ❌ | 4/4 |\n"
        "| **3 tools + guardrail** | **6/6** ✅ | **4/4** ✅ |"
    )
    st.caption(
        "*restraint* = answered directly when it should have. "
        "*recall* = called the right tool when it genuinely needed one."
    )

tab_hover, tab_almanac = st.tabs([
    "🛹  §7 · The Hoverboard",
    "📖  §8 · Grays Sports Almanac",
])


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 1 — Section 6. One tool.
# ─────────────────────────────────────────────────────────────────────────────

with tab_hover:
    HOVER_TOOLS = ["lookup_parcel"]

    st.subheader("A tool is a Python function you gave it permission to call")
    st.html(viz.toolbelt_html(HOVER_TOOLS))

    st.markdown(
        "`lookup_parcel` reads one row out of the Appraiser's database. That is "
        "all it is. The model cannot see the database — it can only ask for a "
        "parcel and read what comes back."
    )

    st.text_input("Ask the assistant", key="hover_q")
    cols = st.columns(len(HOVERBOARD_QUESTIONS))
    for n, (col, q) in enumerate(zip(cols, HOVERBOARD_QUESTIONS)):
        col.button(f"❓ {q[:30]}…", use_container_width=True, key=f"hq{n}",
                   on_click=set_field, args=("hover_q", q))

    if st.button("🛹 Ask", type="primary", use_container_width=True,
                 key="hover_go"):
        slot = st.empty()
        run_agent(st.session_state["hover_q"],
                  st.session_state["system_text"], HOVER_TOOLS, slot,
                  model_key=choice)

        if st.session_state["last_raw"]:
            with st.expander("🔍 What the model **literally** wrote"):
                st.html(viz.raw_call_html(st.session_state["last_raw"]))
                st.markdown(
                    "No magic happened. The model produced **text**, in a format "
                    "it was trained to use, and about fifteen lines of our Python "
                    "noticed and ran the function. It never stopped being "
                    "next-word prediction — Section 2 was the whole story."
                )

    with st.expander("💡 Two things to try"):
        st.markdown(
            "**Give it a mangled parcel number.** *Pull up 4-127-3891* — with "
            "the leading zero missing. The model passes it through exactly as "
            "you typed it, and the tool repairs it. That is Section 2's lesson "
            "collecting its debt: the model sees that number as thirteen "
            "fragments, so it cannot be trusted to reassemble it.\n\n"
            "**Ask something it should not need a tool for.** *What is the "
            "difference between market value and assessed value?* A good "
            "assistant answers that one itself."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 2 — Section 8. The same agent, one more tool.
# ─────────────────────────────────────────────────────────────────────────────

with tab_almanac:
    ALMANAC_TOOLS = ["lookup_parcel", "search_policy"]

    st.subheader("RAG is not a different kind of magic. It is a tool.")
    st.html(viz.toolbelt_html(ALMANAC_TOOLS, active="search_policy"))

    st.markdown(
        "Same agent. Same loop. Same code. **One more entry in the list** — a "
        "tool that reads the office's policy manual instead of its database.\n\n"
        "Grays Sports Almanac was a book of facts you could look up instead of "
        "guessing. This is the office's binder, and now the assistant can read it."
    )

    st.text_input("Ask the assistant", key="almanac_q")
    cols = st.columns(2)
    for n, q in enumerate(ALMANAC_QUESTIONS):
        cols[n % 2].button(f"❓ {q[:44]}…", use_container_width=True,
                           key=f"aq{n}", on_click=set_field,
                           args=("almanac_q", q))

    if st.button("📖 Ask", type="primary", use_container_width=True,
                 key="almanac_go"):
        slot = st.empty()
        run_agent(st.session_state["almanac_q"],
                  st.session_state["system_text"], ALMANAC_TOOLS, slot,
                  model_key=choice)

        used = st.session_state["last_used"]
        if "search_policy" in used:
            st.success(
                "It read the manual and told you which section it came from. "
                "In Section 5 the same model answered this from memory and "
                "invented an ordinance — and cited a Planning Commission that "
                "does not exist. **This answer you can go and check.**",
                icon="📖",
            )
        elif not used:
            st.info(
                "It answered without reaching for a tool. Whether that is the "
                "right call is exactly the judgement the system prompt above "
                "controls — open it and see what it was told.",
                icon="🎛️",
            )

    with st.expander("💡 The question this is really setting up"):
        st.markdown(
            "*Can a parcel be split if the owner owes back taxes?* is not "
            "trivia. It is **Policy 4.2**, and it is the rule that decides "
            "whether Biff Tannen may split parcel 04-127-3891.\n\n"
            "**Section 10 asks that question for real** — and answering it needs "
            "the rule from this manual *plus* facts from three different county "
            "offices that do not talk to each other. That is what MCP is for."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Reference 3 of 3 — let go of the weights before this run ends
#
#  Streamlit runs this script again on every click, in a brand new namespace
#  each time, so you would expect the previous run's variables to vanish with
#  it. They do not: Streamlit keeps THIS run's widget callbacks alive into the
#  NEXT run, and a Python function carries its whole module namespace with it
#  in `__globals__`. Every `on_click=` above is therefore still holding this
#  script -- and the model with it -- at the moment the next run tries to free
#  the memory. Nothing can release a model that is still reachable.
#
#  This is also why `run_agent()` builds its own graph instead of taking one:
#  a `graph, responder = build(...)` line out here would leave the model
#  reachable in exactly the same way, and only on the runs where somebody had
#  pressed Ask -- which is the worst kind of bug to find on stage.
#
#  Full measurements: Plans/vram_model_switching.md.
# ─────────────────────────────────────────────────────────────────────────────

del loaded_model, _tokenizer
