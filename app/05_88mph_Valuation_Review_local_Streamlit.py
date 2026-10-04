"""
88 MPH — The Valuation Review Assistant   ·   Section 7, Live Demo 5

    uv run streamlit run app/05_88mph_Valuation_Review_local_Streamlit.py

The core IAAO payload, and the longest slot in the talk. An appraiser asks the
questions they would actually ask on a Monday morning, and an agent running
entirely on the machine under the desk goes and answers them.

WHAT MAKES THIS APP DIFFERENT FROM THE OTHER TWO
------------------------------------------------
Each app in this repository shows a different thing on purpose:

    Section 6/8   the LOOP is the show      -- the trace fills the screen
    Section 7     the FINDING is the show   -- results fill the screen,
                  (this one)                   the loop is one click below
    Section 11    only the ANSWER           -- the machinery is gone

By Section 7 the audience has already watched Reason -> Act -> Observe turn.
Showing it again competes with the thing they came for, which is what the
agent actually found. So the trace runs live while it works and folds itself
away when the answer lands.

THE STORY, IN THREE QUESTIONS
-----------------------------
    1. "How did the county do?"          every IAAO standard passes.
    2. "Break it down by neighborhood."  three of six fail -- and each one
                                         fails a DIFFERENT standard.
    3. "Show me Riverside Heights."      the under-assessment has a shape.

That is the whole argument. The county looks healthy in aggregate while half
of it needs work, and no one number was ever going to tell you that.

THREE TOOLS, ONE SQL SURFACE
----------------------------
`ratio_study` and `map_parcels` both take a QUERY rather than a list of
parcels, which is what makes this open-ended instead of a menu. "By
neighborhood", "homes built after 1980" and "where we came in under 80% of
price" are the same two tool calls with a different WHERE clause. Nobody had
to guess the question in advance.

EVERYTHING HERE IS SYNTHETIC. See data/README.md.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Run straight from a clone, without installing the package first.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from delorean import agent, data, llm, models, prompts, tools, viz  # noqa: E402
from delorean.tools import query as query_tool  # noqa: E402
from delorean.tools import review as review_tool  # noqa: E402

st.set_page_config(page_title="88 MPH — Valuation Review",
                   page_icon="⚡", layout="wide")

DEFAULT_MODEL = "qwen"

#: The live toolbelt. Three, which is the size we have actually measured, and
#: Section 6's two parcel tools are deliberately NOT here -- `run_sql` does
#: everything they did. Swap a tool out rather than pile another on.
TOOLBELT = ("run_sql", "ratio_study", "map_parcels")

#: The same agent with the ratio tool taken away, for the bonus probe.
TOOLBELT_NO_RATIO = ("run_sql", "map_parcels")

# ── The questions ────────────────────────────────────────────────────────────
#
# Three buttons for the story, so the presenter has the same three clicks
# every time, and a separate row of probes for when somebody in the audience
# asks something. The free text box takes anything.

# Each entry is (widget key, button label, question).
#
# The key is written out rather than derived from the label, which is how this
# started. A key built from display text means renaming a button silently
# throws away its widget state, and it puts punctuation and emoji into an
# identifier. Labels are for reading; keys are for Streamlit.
#
# The numbers are keycap emoji rather than the ①②③ characters. Those are
# ordinary glyphs, so they pick up whatever antialiasing the font gives them
# and go fuzzy on a projector; a keycap is rendered as an image and stays
# sharp at any size.
REVIEW_QUESTIONS = [
    ("rev_county", "1️⃣ How did the county do?",
     "How did the county do on the ratio study this year?"),
    ("rev_hoods", "2️⃣ By neighborhood",
     "Run the assessment ratio study broken down by neighborhood."),
    ("rev_map", "3️⃣ Map Riverside Heights",
     "Show me the sales in Riverside Heights on a map."),
]

ALSO_TRY = [
    ("try_top5", "Top 5 by value",
     "What are the five highest value parcels in the county?"),
    ("try_owner", "Who owns 04-127-3891?", "Who owns parcel 04-127-3891?"),
    ("try_1980", "Homes built after 1980",
     "Run a ratio study for homes built after 1980."),
    ("try_define", "Market vs assessed value",
     "What is the difference between market value and assessed value?"),
]

#: Off the story on purpose. This is a demonstration of what the ratio TOOL is
#: doing for you, not a demonstration that the model is untrustworthy -- the
#: model is given no better option, and the label says so.
NO_TOOL_PROBE = "Run the assessment ratio study broken down by neighborhood."


def svg_card(markup: str, height: int) -> None:
    """Render markup that contains an <svg>.

    ⚠️ `st.html()` cannot do this, and it fails in the most confusing way
    available: it sanitises with DOMPurify configured as `USE_PROFILES:
    {html: true}`, which allows HTML elements and strips **every SVG element**
    on the way through. So the card frame arrives intact -- title, counts,
    legend, border -- with a hole where the drawing was. Nothing errors.
    Nothing logs. You just get an empty box, and it looks like your own
    layout bug.

    A component iframe is not sanitised, so the picture actually arrives. The
    cost is that an iframe has no idea how tall its contents are, so the
    height has to be worked out in advance -- which is why the dial and the
    map both cap their width. See `viz.map_frame_height`.

    Notebooks are unaffected. `IPython.display.HTML` sanitises nothing, so
    `viz.show_*()` has always drawn these correctly.

    The height is passed as a number rather than left as `'content'`. Letting
    the frame measure itself is tidier right up until it measures zero, and
    the failure then looks exactly like the bug this function exists to fix.
    Both pictures cap their width, so their height is arithmetic.
    """
    st.iframe(
        "<style>body{margin:0;padding:0;background:transparent;}</style>"
        + markup,
        height=height)


def set_field(key: str, value: str) -> None:
    """Change a text box from a button press.

    Streamlit forbids writing to `st.session_state[key]` once the widget with
    that key exists in the same run. An `on_click` callback runs before the
    next run, while the write is still allowed.
    """
    st.session_state[key] = value


def ask_this(question: str, *, with_ratio_tool: bool = True) -> None:
    """Queue a question to run on the next rerun."""
    st.session_state["question"] = question
    st.session_state["pending"] = question
    st.session_state["use_ratio_tool"] = with_ratio_tool


@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    return llm.load_local(model_key, verbose=False)


@st.cache_resource(show_spinner=False)
def build(model_key: str, tool_names: tuple[str, ...]):
    """One compiled graph per toolbelt, cached so a click does not rebuild it."""
    model, tokenizer = load_model(model_key)
    responder = agent.local_responder(model, tokenizer, max_new_tokens=420)
    return agent.build_agent(responder, list(tool_names), max_steps=5), responder


@st.cache_data(show_spinner=False)
def cached(model_key: str) -> bool:
    """Is this model already downloaded? Cached so it is not a disk walk per rerun."""
    return models.is_cached(model_key)


@st.cache_data(show_spinner=False)
def county_parcels():
    """Every parcel, for drawing the county underneath a selection."""
    return data.parcels_df()


def run_agent(question: str, *, with_ratio_tool: bool, model_key: str,
              system: str, slot) -> dict:
    """Run one question, drawing the loop as it turns, then hand back the state."""
    belt = TOOLBELT if with_ratio_tool else TOOLBELT_NO_RATIO
    graph, _responder = build(model_key, belt)

    messages: list[dict] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    state: dict = {"messages": messages, "trace": [], "steps": 0}
    started = time.time()

    for chunk in graph.stream(state, stream_mode="updates"):
        for _node, delta in chunk.items():
            state.update(delta)
            slot.html(viz.trace_html(state["trace"], question=question))

    state["elapsed"] = time.time() - started
    state["answer"] = agent.final_answer(state)
    state["toolbelt"] = list(belt)
    state["had_ratio_tool"] = with_ratio_tool
    return state


def last_call(trace: list[dict], name: str) -> dict | None:
    """The most recent successful call to a given tool, if there was one."""
    for step in reversed(trace):
        if step.get("step") == "act" and step.get("tool") == name:
            result = step.get("result") or {}
            if result.get("ok"):
                return step
    return None


def render_findings(state: dict) -> None:
    """Draw whatever the agent actually produced.

    The app does not guess what the question was about -- it reads the trace
    and shows what the tools returned. Ask something that needs no tool and
    nothing is drawn, which is the restraint case working, not a bug.
    """
    trace = state.get("trace", [])

    ratio_step = last_call(trace, "ratio_study")
    if ratio_step:
        payload = ratio_step["result"]["data"]
        if payload.get("results"):
            st.html(viz.ratio_table_html(
                payload["results"],
                group_by=payload.get("grouped_by", "neighborhood"),
                title="Ratio study by "
                      + str(payload.get("grouped_by", "neighborhood"))))
        else:
            st.html(viz.scorecard_html(payload, title="IAAO ratio study"))

    map_step = last_call(trace, "map_parcels")
    if map_step:
        frame = review_tool.last_map()
        if frame is not None and len(frame):
            markup = viz.parcel_map_html(
                frame,
                color_by=map_step["result"]["data"].get("coloured_by", "ratio"),
                context=county_parcels(),
                title="Hill Valley County")
            svg_card(markup, height=viz.map_frame_height(markup))

    sql_step = last_call(trace, "run_sql")
    if sql_step and not ratio_step and not map_step:
        rows = sql_step["result"]["data"].get("rows", [])
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────

st.session_state.setdefault("question", REVIEW_QUESTIONS[0][2])
st.session_state.setdefault("pending", "")
st.session_state.setdefault("use_ratio_tool", True)
st.session_state.setdefault("state", None)
st.session_state.setdefault("dial", None)

# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ 88 MPH")
    st.caption("Section 7 — the valuation review assistant")
    st.divider()

    keys = models.selectable()
    model_key = st.selectbox(
        "Model", options=keys,
        index=keys.index(DEFAULT_MODEL) if DEFAULT_MODEL in keys else 0,
        format_func=lambda k: (
            f"{'✅' if cached(k) else '⬇️'} {models.REGISTRY[k].label}  "
            f"({models.REGISTRY[k].params})"),
    )
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
    previous = st.session_state.get("loaded_model")
    if previous is not None and previous != model_key:
        load_model.clear()      # 1
        build.clear()           # 1 — the graph holds the responder holds the model
        llm.unload()            # 2 — also collects garbage, empties the CUDA cache
    st.session_state["loaded_model"] = model_key

    # Picking a model that is not on this machine starts a multi-gigabyte
    # download the moment you press Run, with no progress bar anywhere the
    # audience can see. Say so before that happens rather than after.
    if not cached(model_key):
        st.error(
            f"**Not on this machine yet.** Choosing this will download it "
            f"first — several GB, and nothing on screen will show progress. "
            f"Fetch it before you present:\n\n"
            f"```\nuv run python -c \"from delorean import llm; "
            f"llm.load_local('{model_key}')\"\n```",
            icon="⬇️")
    elif spec.role == "tiny":
        st.warning(
            "The 0.8B model cannot drive three tools. Fine to try once, then "
            "switch back.", icon="⚠️")

    # This app is where a blocked model would have done the most damage --
    # Section 7's prompt is about 1,600 tokens, which is past where Nemotron
    # runs out of memory. Explain the absence rather than leaving a gap.
    for missing in models.blocked():
        st.caption(f"⚠️ **{missing.label}** is not offered — {missing.blocked_reason}")

    st.divider()
    st.markdown("**The toolbelt**")
    st.html(viz.toolbelt_html(list(TOOLBELT), title=""))
    st.caption(
        "Three tools, and `ratio_study` and `map_parcels` both take a SQL "
        "query — so the agent can study or map *any* set of parcels it can "
        "describe, not a fixed list of reports."
    )

    with st.expander("The system prompt"):
        st.caption(
            "Measured, not decorative. At three tools with no prompt the model "
            "reaches for one when it should just answer; naming when each tool "
            "applies fixed that. Telling it the figures are already on screen "
            "**halved the wall clock** — 39s to 12s on question 2."
        )
        st.code(prompts.REVIEW_ASSISTANT, language="text")

    with st.expander("What the model can see"):
        st.caption(
            "The whole schema, exactly as it is handed to the model. Two "
            "tables, eleven columns each, curated — `parcels` really has "
            "sixteen. Every extra column is one more thing to reach for and "
            "get wrong."
        )
        st.code(query_tool.SCHEMA_HELP, language="text")
        st.caption(
            f"The sales file was pulled from the Clerk of Courts on "
            f"**{data.feed_pulled_on()}**. It is a snapshot, and Section 10 is "
            f"about what that costs you."
        )

    st.divider()
    st.markdown("**Where this is running**")
    loaded_model, _tok = load_model(model_key)
    where = llm.where_it_runs(loaded_model)
    if where["cuda_available"]:
        st.success(f"🖥️ {where['device_name']}", icon="✅")
        st.caption(
            f"**{where['on_gpu']} of {where['on_gpu'] + where['on_cpu']}** "
            f"weight tensors on the GPU · {where['allocated_gb']:.2f} GB in use")
        st.metric("GPU memory free", f"{where['free_gb']:.1f} GB",
                  f"of {where['total_gb']:.0f} GB")
    else:
        st.warning("No GPU found — this will run on the CPU and be slow.",
                   icon="🐢")

    st.caption(
        "Nothing here leaves the building. The roll, the sales file, the "
        "model and the map are all on this machine."
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────

left, right = st.columns([3, 2], gap="large")

with left:
    st.title("⚡ 88 MPH — The Valuation Review Assistant")
    st.markdown(
        "*Where we're going, we don't need spreadsheets.*  \n"
        "Ask the questions you would actually ask on a Monday morning. "
        "The agent writes its own SQL, runs the IAAO statistics, and draws "
        "the map."
    )

with right:
    svg_card(viz.speedometer_html(
        st.session_state["dial"],
        caption="Median assessment ratio",
        subtitle=st.session_state.get("dial_note", "")),
        height=viz.speedometer_frame_height())

# ─────────────────────────────────────────────────────────────────────────────
#  Ask
# ─────────────────────────────────────────────────────────────────────────────

st.text_input(
    "Ask the review assistant",
    key="question",
    placeholder="e.g. which neighborhoods should I review this year?",
)

go = st.button("Run  ▶", type="primary")

st.caption("**The review** — the three questions the demo walks through")
cols = st.columns(len(REVIEW_QUESTIONS))
for col, (key, label, question) in zip(cols, REVIEW_QUESTIONS):
    col.button(label, width="stretch", on_click=ask_this,
               args=(question,), key=key)

st.caption("**Also try** — for when somebody in the room asks something")
cols = st.columns(len(ALSO_TRY) + 1)
for col, (key, label, question) in zip(cols, ALSO_TRY):
    col.button(label, width="stretch", on_click=ask_this,
               args=(question,), key=key)
cols[-1].button("🔧 With no ratio tool", width="stretch",
                on_click=ask_this, args=(NO_TOOL_PROBE,),
                kwargs={"with_ratio_tool": False},
                key="try_noratio",
                help="Same question, but the agent is given only run_sql and "
                     "map_parcels. Shows what the ratio tool is doing for you.")

if go and st.session_state["question"].strip():
    st.session_state["pending"] = st.session_state["question"]
    st.session_state["use_ratio_tool"] = True

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  Run and report
# ─────────────────────────────────────────────────────────────────────────────

pending = st.session_state.pop("pending", "")

if pending:
    with_ratio = st.session_state.get("use_ratio_tool", True)
    system = (prompts.REVIEW_ASSISTANT if with_ratio
              else prompts.REVIEW_ASSISTANT_NO_RATIO_TOOL)

    with st.status("Thinking…", expanded=True) as status:
        slot = st.empty()
        state = run_agent(pending, with_ratio_tool=with_ratio,
                          model_key=model_key, system=system, slot=slot)
        used = [s["tool"] for s in state["trace"] if s["step"] == "act"]
        status.update(
            label=(f"{state['elapsed']:.1f}s · "
                   + (" → ".join(used) if used
                      else "answered directly, no tool needed")),
            state="complete", expanded=False)

    st.session_state["state"] = state

    # Move the needle only when a single (ungrouped) study just ran. A grouped
    # study has six medians and no honest one-needle answer -- the table says
    # it better.
    step = last_call(state["trace"], "ratio_study")
    if step:
        payload = step["result"]["data"]
        if not payload.get("results"):
            for test in payload.get("tests", []):
                if test["metric"] == "Median ratio":
                    st.session_state["dial"] = test["value"]
                    st.session_state["dial_note"] = (
                        f"{payload.get('sales_used', 0)} qualified sales")
        else:
            # A grouped study has one median per group and no honest way to
            # put six of them on one needle, so the dial keeps whatever it
            # last showed -- and says so, rather than letting the reading look
            # like it belongs to the table underneath it.
            st.session_state["dial_note"] = (
                "still the countywide figure — the table below has one per "
                f"{payload.get('grouped_by', 'group')}"
                if st.session_state.get("dial") is not None
                else f"one per {payload.get('grouped_by', 'group')} "
                     "in the table below")
    st.rerun()

state = st.session_state.get("state")

if state:
    # This banner has to survive the rerun and sit WITH the results. Shown
    # only during the run it flashes past, and the audience is then looking
    # at a table of numbers with no idea why it is different.
    if not state.get("had_ratio_tool", True):
        st.warning(
            "**No `ratio_study` tool for this run** — the agent was given only "
            "`run_sql`, so it had to work the figures out in SQL itself. "
            "Compare what came back with question 2, which had the tool.",
            icon="🔧")

    if state.get("answer"):
        # Rendered as markdown in a plain bordered container rather than
        # pasted into an HTML string. Two reasons, and the second is the one
        # that bites: the model writes markdown -- given no ratio tool it
        # returns a full markdown TABLE -- which inside a <div> would show up
        # as raw pipes and dashes. And model output is untrusted text; it does
        # not belong concatenated into markup, in a repository other offices
        # are meant to copy from.
        with st.container(border=True):
            st.markdown(state["answer"])

    render_findings(state)

    with st.expander("How it got there — the loop, and the SQL it wrote"):
        for step in state["trace"]:
            if step.get("step") == "act" and step["args"].get("sql"):
                st.caption(f"`{step['tool']}` wrote this query itself:")
                st.code(step["args"]["sql"], language="sql")
        st.html(viz.trace_html(state["trace"], answer=state.get("answer", ""),
                               question=None))
        st.caption(
            f"Toolbelt for this run: "
            f"{', '.join('`' + t + '`' for t in state.get('toolbelt', []))}"
            f" · {state['elapsed']:.1f}s")
else:
    st.info(
        "Press **1️⃣ How did the county do?** to begin. The agent will write "
        "its own SQL, run the IAAO ratio study, and report back.",
        icon="⚡")


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
#  Dropping our own names here is what makes the release in the sidebar work.
#  Measured: without it a switch frees nothing and both models sit on the card.
#  Full measurements: Plans/vram_model_switching.md.
# ─────────────────────────────────────────────────────────────────────────────

del loaded_model, _tok
