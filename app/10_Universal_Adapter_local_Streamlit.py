"""
The Universal Adapter   ·   Section 10, Live Demos 7 and 8

    uv run streamlit run app/10_Universal_Adapter_local_Streamlit.py

The finale. Three county offices, three separate programs, one question that
none of them can answer alone.

WHAT MAKES THIS DIFFERENT FROM EVERY APP BEFORE IT
--------------------------------------------------
Section 7 built a tool: we wrote a Python function, hand-wrote a JSON schema
describing it, and handed both to the model. Two pieces of code per tool,
every time, forever.

Here we wrote neither. Three servers start up, the agent asks each one what it
can do, and the answers become its toolbelt. Add a fourth office next year and
nothing in this file changes.

    Section 7    a tool is a function YOU wrote
    Section 10   an MCP server is a program SOMEBODY ELSE published

That is the difference between a charger built for one phone and a USB-C
socket, and it is the whole of this section.

ORDER ON SCREEN, AND IT IS DELIBERATE
-------------------------------------
    1. how it got there      the trace, VISIBLE -- every office it asked,
                             every call, and what came back
    2. the decision          the rendered answer
    3. what that costs you   the sync impact, when a deed was corrected

This section has the strongest claim on an open trace of any app here. The
question is whether a machine may tell a man he cannot split his own land,
and the only thing that makes that answer inspectable is watching it ask four
separate programs and reading what each one said. Fold the trace away and a
county official is being asked to take a confident paragraph on trust.

🩹 **Two earlier versions of this were wrong in opposite directions.** The
first rendered the answer at the very TOP, which gave away the verdict before
the room had seen which offices were asked. The second moved it last but left
the trace in a collapsed expander -- still hiding the evidence, just further
down the page. **The trace is not supporting material. It is the argument.**
`scripts/test_app_adapter.py` compares caption positions and will fail if the
order moves. **Do not collapse it again.**

THE TWO DEMOS
-------------
    7   "May Biff Tannen split his lot?"
        Lot size lives with the Appraiser. Ownership lives with the Clerk.
        The tax balance lives with the Tax Collector. The rule lives in our
        own policy manual. Today that is three phone calls and a week.

    8   "Is our sales file still true?"
        The Appraiser works from a feed pulled on one date. The Clerk kept
        correcting deeds afterwards. Nobody sends a note. Section 9's ratio
        study was run on a sale that had already been reclassified.

EVERYTHING HERE IS SYNTHETIC. See data/README.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from delorean import agent, data, llm, models, prompts, ratio, viz  # noqa: E402
from delorean.offices import HILL_VALLEY, Offices                   # noqa: E402

st.set_page_config(page_title="The Universal Adapter", page_icon="🔌",
                   layout="wide")

DEFAULT_MODEL = "qwen"

#: The policy manual is OURS. It stays a local tool on purpose -- MCP is for
#: reaching somebody else's system, and the app should show that difference
#: rather than blur it.
LOCAL_TOOLS = ["search_policy"]

# Keys are written out rather than derived from the label: a key built from
# display text throws away widget state the moment somebody renames a button.
QUESTIONS = [
    ("q_split", "7️⃣ May Biff split his lot?",
     "Biff Tannen wants to split parcel 04-127-3891 into two lots. May he? "
     "Check everything the county requires."),
    ("q_sync", "8️⃣ Is our sales file still true?",
     "Our sales file was pulled from the Clerk on {pulled}. Has the Clerk "
     "corrected any deeds since then that we would still be using?"),
]

ALSO_TRY = [
    ("q_counter", "As a taxpayer would ask",
     "I want to split my lot at 1955 Hilldale Ct and sell half. Can I?"),
    ("q_owner", "Who owns it, and what did they pay?",
     "Who owns parcel 04-127-3891 and what did they pay for it?"),
    ("q_cure", "What would Biff have to do?",
     "What exactly would Biff Tannen have to do to get his split approved?"),
]


def ask_this(question: str) -> None:
    st.session_state["question"] = question
    st.session_state["pending"] = question


@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    return llm.load_local(model_key, verbose=False)


@st.cache_resource(show_spinner=False)
def connect_offices():
    """Start the three MCP servers once and keep them running.

    `cache_resource` is doing real work here. Without it Streamlit would
    restart three subprocesses on every click, which costs about five seconds
    and -- worse for the demo -- throws away the thing worth showing, which is
    three servers sitting there running.

    The connection is safe to call from Streamlit's script thread: `Offices`
    owns its own event loop and takes work off a queue.
    """
    return Offices(HILL_VALLEY, also_local=LOCAL_TOOLS).open()


@st.cache_resource(show_spinner=False)
def build(model_key: str):
    """One compiled graph, using the offices as its toolbox."""
    model, tokenizer = load_model(model_key)
    responder = agent.local_responder(model, tokenizer, max_new_tokens=460)
    offices = connect_offices()
    return agent.build_agent(responder, max_steps=9, toolbox=offices), offices


@st.cache_data(show_spinner=False)
def cached(model_key: str) -> bool:
    return models.is_cached(model_key)


def corrected_parcels(trace: list[dict]) -> list[dict]:
    """Any deeds `corrections_since` reported, pulled back out of the trace."""
    found = []
    for step in trace:
        if step.get("step") != "act" or step.get("tool") != "corrections_since":
            continue
        payload = step.get("result") or {}
        if not payload.get("ok"):
            continue
        body = payload.get("data")
        # The MCP result arrives as a JSON string inside our envelope.
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except ValueError:
                continue
        if isinstance(body, dict):
            body = body.get("data", [])
        for row in body or []:
            if isinstance(row, dict) and row.get("parcel_id"):
                found.append(row)
    return found


def render_sync_impact(rows: list[dict]) -> None:
    """What the correction actually costs the ratio study.

    The agent finds the problem; this panel measures it. That split is
    deliberate and it is the same rule Section 9 runs on -- the agent decides
    what to ask, and the arithmetic lives somewhere it cannot be got wrong.
    """
    feed = data.sales_feed_df()
    for row in rows:
        parcel = row["parcel_id"]
        match = feed[feed.parcel_id == parcel]
        if match.empty:
            continue
        hood = match.iloc[0]["neighborhood"]

        # `######`, not `####`. The section heading above this is `#####`,
        # and a bigger heading underneath it reads as a new section starting.
        st.markdown(f"###### What that one deed does to **{hood}**")
        st.caption(
            f"Deed {row.get('instrument_number', '?')} on parcel {parcel} was "
            f"booked as a {row.get('deed_type', '?')} and corrected on "
            f"{row.get('corrected_on', '?')}. Section 9 studied it as a valid "
            f"sale. These are the same statistics with and without it."
        )
        stale = feed[feed.neighborhood == hood]
        live = stale[stale.parcel_id != parcel]

        left, right = st.columns(2)
        with left:
            st.html(viz.scorecard_html(
                ratio.ratio_stats(stale),
                title=f"{hood} — our sales file"))
        with right:
            st.html(viz.scorecard_html(
                ratio.ratio_stats(live),
                title=f"{hood} — with the Clerk's correction"))

        before = ratio.ratio_stats(stale)
        after = ratio.ratio_stats(live)
        gained = [t for t in after["failed_tests"]
                  if t not in before["failed_tests"]]
        if gained:
            st.error(
                f"**Removing a bad sale did not make it look better.** "
                f"{hood} now fails **{', '.join(gained)}** as well — a second "
                f"problem the stale file was hiding. That feed was not just "
                f"out of date, it was flattering us.",
                icon="🚨")


# ─────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────

st.session_state.setdefault("question", "")
st.session_state.setdefault("pending", "")
st.session_state.setdefault("state", None)

# ─────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔌 The Universal Adapter")
    st.caption("Section 10 — three offices, one question")
    st.divider()

    keys = models.selectable()
    model_key = st.selectbox(
        "Model", options=keys,
        index=keys.index(DEFAULT_MODEL) if DEFAULT_MODEL in keys else 0,
        format_func=lambda k: (f"{'✅' if cached(k) else '⬇️'} "
                               f"{models.REGISTRY[k].label}  "
                               f"({models.REGISTRY[k].params})"))
    spec = models.REGISTRY[model_key]
    st.caption(f"`{spec.hf_id}`")

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
    #
    # ⚠️ NAMED caches only. A blanket `st.cache_resource.clear()` would also
    # drop `connect_offices` WITHOUT calling close() on it, leaving three
    # orphaned MCP subprocesses holding the GPU. `build.clear()` is safe
    # because rebuilding the graph calls `connect_offices()` again and gets
    # the same three servers back from its own cache.
    previous = st.session_state.get("loaded_model")
    if previous is not None and previous != model_key:
        load_model.clear()      # 1
        build.clear()           # 1 — the graph holds the responder holds the model
        llm.unload()            # 2 — also collects garbage, empties the CUDA cache
    st.session_state["loaded_model"] = model_key

    if not cached(model_key):
        st.error("**Not on this machine yet.** Choosing this downloads it "
                 "first — several GB, with no progress shown.", icon="⬇️")
    elif spec.role != "workhorse":
        st.warning(
            "Four chained tool calls across three servers is the hardest "
            "thing in this repository. Measured: the 4B does it; SmolLM3 "
            "reaches one office of four and then answers from memory.",
            icon="⚠️")

    st.divider()
    st.markdown("**The three servers**")
    for office in HILL_VALLEY:
        st.caption(f"{office.icon} **{office.label}** — `{office.script}`")
    st.caption("Each is a separate program. Start one yourself with "
               "`uv run python mcp_server/clerk_server.py` and it will sit "
               "there waiting to be asked.")

    st.divider()
    st.markdown("**Where this is running**")
    loaded, _tok = load_model(model_key)
    where = llm.where_it_runs(loaded)
    if where["cuda_available"]:
        st.success(f"🖥️ {where['device_name']}", icon="✅")
        st.caption(f"**{where['on_gpu']} of "
                   f"{where['on_gpu'] + where['on_cpu']}** weight tensors on "
                   f"the GPU · {where['allocated_gb']:.2f} GB in use")
    else:
        st.warning("No GPU found — this will be slow.", icon="🐢")
    st.caption("The model, the three servers and all three databases are on "
               "this machine. Nothing leaves the building.")

# ─────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────

st.title("🔌 The Universal Adapter")
st.markdown(
    "*If it has the adapter, it just works.*  \n"
    "Three county offices. Three separate programs that were never built to "
    "talk to each other. **We did not write a single tool for any of them** — "
    "we asked each one what it could do."
)

with st.spinner("Starting the three county offices…"):
    graph, offices = build(model_key)

catalogue = offices.catalogue()
office_slot = st.empty()
state = st.session_state.get("state")
office_slot.html(viz.offices_html(
    catalogue,
    trace=(state or {}).get("trace"),
    local=LOCAL_TOOLS))

# ─────────────────────────────────────────────────────────────────────────
#  Ask
# ─────────────────────────────────────────────────────────────────────────

pulled = data.feed_pulled_on()

st.text_input("Ask the county", key="question",
              placeholder="e.g. may the owner of 1955 Hilldale Ct split it?")
go = st.button("Ask  ▶", type="primary")

st.caption("**The demo** — the two questions no single office can answer")
cols = st.columns(len(QUESTIONS))
for col, (key, label, template) in zip(cols, QUESTIONS):
    col.button(label, width="stretch", key=key, on_click=ask_this,
               args=(template.format(pulled=pulled),))

st.caption("**Also try**")
cols = st.columns(len(ALSO_TRY))
for col, (key, label, question) in zip(cols, ALSO_TRY):
    col.button(label, width="stretch", key=key, on_click=ask_this,
               args=(question,))

if go and st.session_state["question"].strip():
    st.session_state["pending"] = st.session_state["question"]

st.divider()

# ─────────────────────────────────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────────────────────────────────

pending = st.session_state.pop("pending", "")

if pending:
    messages = [{"role": "system", "content": prompts.COUNTY_ASSISTANT},
                {"role": "user", "content": pending}]
    run_state: dict = {"messages": messages, "trace": [], "steps": 0}
    started = time.time()

    with st.status("Asking the county…", expanded=True) as status:
        detail = st.empty()
        # Streamed rather than awaited, so each office lights up as it
        # answers. Waiting twenty seconds and then showing a finished
        # picture teaches nothing; watching the Clerk answer after the
        # Appraiser is the demo.
        for chunk in graph.stream(run_state, stream_mode="updates"):
            for _node, delta in chunk.items():
                run_state.update(delta)
                office_slot.html(viz.offices_html(
                    catalogue, trace=run_state["trace"], local=LOCAL_TOOLS))
                detail.html(viz.trace_html(run_state["trace"],
                                           question=pending))

        run_state["elapsed"] = time.time() - started
        run_state["answer"] = agent.final_answer(run_state)
        asked = [s["tool"] for s in run_state["trace"] if s["step"] == "act"]
        # `expanded=True`, deliberately. With `expanded=False` the trace
        # folded shut at the exact moment the answer arrived -- the room
        # watched four offices' worth of evidence vanish as the verdict
        # appeared. The rerun below replaces it with the same trace rendered
        # as a permanent section.
        status.update(
            label=(f"{run_state['elapsed']:.0f}s · "
                   + (" → ".join(asked) if asked else "answered directly")),
            state="complete", expanded=True)

    st.session_state["state"] = run_state
    st.rerun()

state = st.session_state.get("state")

if state:
    # ── 1. How it got there ─────────────────────────────────────────────────
    #
    # NOT an expander, and NOT collapsed. See this file's docstring: this is
    # the section where a machine tells somebody they may not split their own
    # land, and the trace is the only thing that makes that inspectable.
    st.markdown("##### 🔁 How it got there")

    acts = [s for s in state["trace"] if s.get("step") == "act"]
    places = {(offices.owner_of(s["tool"]).label
               if offices.owner_of(s["tool"]) else "this office")
              for s in acts}
    if acts:
        st.caption(
            f"**{len(acts)} tool calls across {len(places)} separate "
            f"programs** to answer one question — and which building each one "
            f"went to. Not four functions over one database: four *programs*, "
            f"three of them belonging to somebody else."
        )

    for step in acts:
        owner = offices.owner_of(step["tool"])
        where = (f"{owner.icon} {owner.label}" if owner
                 else "📖 this office")
        st.caption(f"**{where}** · `{step['tool']}` · "
                   f"`{json.dumps(step['args'])}`")

    st.html(viz.trace_html(state["trace"],
                           answer=state.get("answer", ""), question=None))
    st.caption(f"{state['elapsed']:.0f}s · {state['steps']} reasoning steps")

    # ── 2. The decision ────────────────────────────────────────────────────
    #
    # ⚠️ `st.markdown`, never interpolated into an HTML string: the model
    # writes markdown, and model output is untrusted text that does not belong
    # concatenated into markup. The trace above holds this identical text
    # exactly as the model emitted it, escaped.
    if state.get("answer"):
        st.markdown("##### 🏁 The decision, with every office consulted")
        with st.container(border=True):
            st.markdown(state["answer"])
        st.caption(
            "Rendered from the identical characters the trace above shows raw "
            "— the model produced a **string**, not a formatted document."
        )

    # ── 3. What that costs you ─────────────────────────────────────────────
    #
    # Only the sync question produces these. The agent finds the problem; the
    # arithmetic lives where a model cannot get it wrong, which is the same
    # rule Section 9 runs on.
    corrections = corrected_parcels(state["trace"])
    if corrections:
        st.markdown("##### 📊 What that costs you")

        # The only theme reference in this app, and it is here because it is
        # the mechanism rather than a decoration: a stale snapshot of a record
        # that has been edited behind you is precisely Marty's photograph.
        st.info(
            f"📸 **Marty carried a photograph of his brother and sister, and "
            f"watched them fade out of it one at a time.** The photograph was "
            f"true when it was taken. What changed was the past behind it — "
            f"and nothing about the photograph told him so.\n\n"
            f"**Your sales file is that photograph.** It was pulled from the "
            f"Clerk on **{pulled}**, it was correct that morning, and the "
            f"Clerk has gone on correcting deeds ever since. Nobody sends a "
            f"note.",
            icon="🕰️")

        st.caption(
            "Below: the same IAAO statistics with and without the corrected "
            "deed. **`ratio.py` computes both and no language model goes "
            "anywhere near them.**"
        )
        render_sync_impact(corrections)
else:
    st.info(
        "Press **7️⃣ May Biff split his lot?** to begin. The agent will ask "
        "the Appraiser, the Clerk of Courts and the Tax Collector, then read "
        "our own policy manual before it answers.",
        icon="🔌")


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
#  `graph` goes too: it holds the responder, which holds the weights.
#  `offices` deliberately does NOT -- the three MCP servers are meant to stay
#  running, and `connect_offices()`'s own cache is what keeps them.
#
#  Full measurements: Plans/vram_model_switching.md.
# ─────────────────────────────────────────────────────────────────────────────

del graph, loaded, _tok
