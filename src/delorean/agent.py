"""
agent.py — the Reason → Act → Observe loop, built by hand.

Section 6 puts this on a slide. Section 9 runs it. They are the same three
boxes, and that is the entire reason we do not use a prebuilt agent:

    ┌──────────┐  wants a tool   ┌──────────┐
    │  REASON  │ ──────────────▶ │   ACT    │
    │ the model│                 │ run the  │
    │  thinks  │ ◀────────────── │   tool   │
    └──────────┘    OBSERVE      └──────────┘
         │ done
         ▼
       answer

`create_react_agent()` would build all of that in one line — and the audience
would never see it happen. Written out as two nodes and one conditional edge,
the picture on the slide and the code on screen are the same thing, and an
attendee can change one node without reverse-engineering anything.

The graph draws its own diagram:

    print(agent.get_graph().draw_mermaid())

No extra dependency, no network call.

Local and hosted models, one graph
----------------------------------
The two differ in exactly one place: how you get an answer out of them. A local
model emits tool calls as text we parse ourselves; the Anthropic API returns
them as structured blocks. Both are wrapped up as a **responder** — a function
that takes messages and tool schemas and hands back `(text, tool_calls)`.

Everything above that line is identical, which is what lets the `_local` and
`_api` notebooks share a graph and differ only in the cell that builds a model.
"""

from __future__ import annotations

import json
from typing import Any, Callable, TypedDict

from . import tools as toolkit
from .tools.base import ToolCall

#: A responder takes (messages, tool_schemas) and returns the model's prose
#: plus any tool calls it asked for.
Responder = Callable[[list[dict], list[dict]], tuple[str, list[ToolCall]]]


class AgentState(TypedDict):
    """What flows around the graph.

    `messages` is the conversation in plain dicts, the same shape the model
    APIs actually take. `trace` is purely for us -- it is what the notebook and
    the Streamlit app read to show the loop turning.
    """

    messages: list[dict]
    trace: list[dict]
    steps: int


# ─────────────────────────────────────────────────────────────────────────
#  Responders — the only part that knows what kind of model this is
# ─────────────────────────────────────────────────────────────────────────

def call_fields(call: dict) -> tuple[str, dict]:
    """Read (name, arguments) from a recorded tool call, in either shape.

    We write the OpenAI shape into `messages` because that is what chat
    templates are written against (see `reason`). This reader also accepts the
    older flat `{"name": ..., "args": ...}` so anything an attendee saved from
    a previous version of this repo still loads.
    """
    fn = call.get("function", call)
    args = fn.get("arguments")
    if args is None:
        args = fn.get("args")
    return str(fn.get("name", "")), dict(args or {})


def _unwrap_schemas(schemas: list[dict]) -> list[dict]:
    """Strip the OpenAI `{"type": "function", "function": {...}}` wrapper.

    That wrapper is an *API* convention. A model that reads its tool list as a
    raw JSON blob pasted into the system turn has no use for it, and at least
    one model is actively confused by it.

    Measured on `microsoft/Phi-4-mini-instruct`, same question, same three
    tools, only the declaration shape changed:

        wrapped    -> the model echoes the tool DEFINITION back at you,
                      verbatim, and calls nothing. Four times out of four.
        unwrapped  -> {"name": "ratio_study",
                       "arguments": {"sql": "...", "group_by": "neighborhood"}}

    The first result reads exactly like a model that cannot call tools. It can;
    it was handed a shape it did not recognise and did the most plausible
    next-token thing with it, which was to repeat it. Section 2, still true.

    Unwrapped output lands as Shape C, which `tools.parse_tool_calls` already
    handles, so no new parser was needed.
    """
    plain = []
    for schema in schemas:
        plain.append(schema.get("function", schema))
    return plain


def render_prompt(tokenizer: Any, messages: list[dict], schemas: list[dict],
                  *, thinking: bool = False) -> str:
    """Turn messages + tool schemas into the literal string a local model reads.

    This is more careful than it looks, because of one failure that is very
    hard to see.

    Most chat templates take the tool list through the standard
    `apply_chat_template(tools=...)` argument. **Some do not.** Phi-4-mini's
    looks for `message["tools"]` on the SYSTEM turn instead — and a template
    that ignores `tools=` does not complain about it. It renders a perfectly
    valid prompt with no tools in it and returns normally.

    Measured on `microsoft/Phi-4-mini-instruct`: the same call that produces
    643 characters of prompt with a full tool declaration produced **99
    characters with no tools at all** when the schemas were passed the usual
    way. Nothing raised. Nothing warned.

    The model then never learns the tool exists, never calls one, and looks
    exactly like a model too small to use tools. **We have already lost an
    afternoon to this exact class of bug once** — SmolLM3 scored 0/6 because
    its OUTPUT shape was unexpected (CLAUDE.md, Tool calling). This is the
    # same mistake
    at the input end, and it is worse, because there is no wrong-looking text
    to notice.

    So rather than keeping a list of which model needs which convention, we
    check: render the prompt with the tools and without them. If the two come
    out identical, the template ignored them, and the tools go on the system
    turn instead. That costs one extra string render — no tokenizing, no
    model — and it works for models released after this was written.
    """
    def render(msgs: list[dict], tool_list: list[dict] | None) -> str:
        try:
            return tokenizer.apply_chat_template(
                msgs, tools=tool_list, tokenize=False,
                add_generation_prompt=True, enable_thinking=thinking)
        except TypeError:
            # `enable_thinking` is a Qwen extension; not every template takes it.
            return tokenizer.apply_chat_template(
                msgs, tools=tool_list, tokenize=False,
                add_generation_prompt=True)

    prompt = render(messages, schemas)

    # No tools asked for, or the template visibly used them: nothing to fix.
    if not schemas or prompt != render(messages, None):
        return prompt

    # The template ignored `tools=`. Put them where Phi-4-mini looks, in the
    # dialect it was trained on -- see _unwrap_schemas for why that matters.
    declaration = json.dumps(_unwrap_schemas(schemas))
    patched = [dict(m) for m in messages]
    for message in patched:
        if message.get("role") == "system":
            message["tools"] = declaration
            break
    else:
        patched.insert(0, {"role": "system", "content": "",
                           "tools": declaration})
    return render(patched, schemas)


def local_responder(model: Any, tokenizer: Any, *, max_new_tokens: int = 320,
                    thinking: bool = False) -> Responder:
    """Talk to a model running on this machine, through `transformers`.

    320 tokens is not arbitrary. A tool call with thinking switched on runs to
    about 150; at 120 we measured it truncating mid-parameter, which the parser
    then sees as no tool call at all.
    """
    import torch

    def respond(messages: list[dict], schemas: list[dict]):
        prompt = render_prompt(tokenizer, messages, schemas, thinking=thinking)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        generated = out[0][inputs["input_ids"].shape[1]:]

        # Decoded TWICE, and the reason is Gemma 4.
        #
        # Its tool-call delimiters -- <|tool_call>, <|"|> and <tool_call|> --
        # are registered SPECIAL TOKENS, not ordinary text. So
        # `skip_special_tokens=True`, which every other model here wants,
        # silently deletes them and turns a perfectly good call into:
        #
        #     call:map_parcels{sql:SELECT parcel_id FROM sales_feed}
        #
        # The parser then sees nothing, quite correctly, and the model scores
        # 0/6 on recall while demonstrably emitting correct calls the whole
        # time. Measured -- that is exactly what happened.
        #
        # ⚠️ That is the THIRD time in this repository a model has looked
        # incapable because of how we READ it rather than what it wrote
        # (SmolLM3's missing tags, Phi-4-mini's dropped declaration, this).
        # When a model scores zero, suspect the harness before the model.
        raw_full = tokenizer.decode(generated, skip_special_tokens=False)
        raw = tokenizer.decode(generated, skip_special_tokens=True)

        # Keep the FULL output -- special tokens and all -- because Section 7
        # shows this verbatim to make the point that a "tool call" is just text
        # the model wrote. Reconstructing it from the parsed result would look
        # the same and quietly be a lie.
        #
        # It has to be the full decode, not the readable one. Gemma 4's
        # delimiters ARE special tokens, so the stripped version has no tool
        # call visible in it at all -- and the app, finding none, would fall
        # back to showing the model's final prose under a heading that says
        # "what the model literally wrote". A caption that is quietly wrong is
        # worse in a teaching repo than one that is ugly.
        respond.last_raw = raw_full

        calls = toolkit.parse_tool_calls(raw_full)
        if not calls and toolkit.attempted_tool_call(raw_full):
            # The call was cut off. Say so plainly rather than pretending the
            # model declined to use a tool.
            return ("(the tool call was cut off — raise max_new_tokens)", [])
        return toolkit.strip_tool_calls(raw), calls

    respond.last_raw = ""
    return respond


def api_responder(client: Any, model_id: str, *, max_tokens: int = 512,
                  system: str = "") -> Responder:
    """Talk to a hosted Claude model, through the Anthropic SDK.

    The API does tool calling natively: you get back structured `tool_use`
    blocks instead of text to parse. We translate them into the same
    `(text, calls)` shape the local responder produces, so the graph above
    cannot tell the difference.
    """
    def respond(messages: list[dict], schemas: list[dict]):
        # Anthropic wants its own message shape, so translate on the way out.
        #
        # The system prompt is the one that bites: the local models take it as
        # an ordinary message in the list, and this API rejects that outright --
        # it must be a top-level `system` parameter. So we lift any system
        # messages out here rather than making every caller know the difference.
        wire: list[dict] = []
        lifted: list[str] = []

        for m in messages:
            if m["role"] == "system":
                if str(m.get("content", "")).strip():
                    lifted.append(str(m["content"]))
                continue
            if m["role"] == "tool":
                wire.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_use_id", m.get("name", "tool")),
                    "content": m["content"],
                }]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for c in m["tool_calls"]:
                    name, args = call_fields(c)
                    blocks.append({"type": "tool_use", "id": c.get("id", name),
                                   "name": name, "input": args})
                wire.append({"role": "assistant", "content": blocks})
            else:
                wire.append({"role": m["role"], "content": m["content"]})

        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": wire,
            "tools": [{"name": s["function"]["name"],
                       "description": s["function"]["description"],
                       "input_schema": s["function"]["parameters"]}
                      for s in schemas],
        }
        combined = "\n\n".join(([system] if system.strip() else []) + lifted)
        if combined:
            kwargs["system"] = combined

        response = client.messages.create(**kwargs)

        text_parts, calls = [], []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(block.name, dict(block.input)))
        return "\n".join(text_parts).strip(), calls

    return respond


# ─────────────────────────────────────────────────────────────────────────
#  The graph
# ─────────────────────────────────────────────────────────────────────────

def build_agent(responder: Responder, tool_names: list[str] | None = None,
                *, max_steps: int = 4, toolbox: Any = None):
    """Wire up Reason → Act → Observe and hand back a runnable graph.

    Args:
        responder:  how to talk to the model. See above.
        tool_names: which tools the agent may use. Keep this list SHORT --
                    a bigger toolbelt measurably increases the chance the model
                    reaches for one when it should have just answered.
        max_steps:  how many times round the loop before we stop. Without this
                    a model that keeps asking for the same tool never finishes.
        toolbox:    where the tools come from. Anything with `.schemas(names)`
                    and `.dispatch(name, args)` will do; defaults to the local
                    registry in `delorean.tools`.

    **That last argument is Section 10 in one line.** Point it at
    `delorean.offices.Offices` and the identical graph runs against three MCP
    servers in three other county offices instead of functions in this
    repository. Reason → Act → Observe does not change, the responder does not
    change, the notebook does not change. Only where the tools live changes,
    which is the entire claim MCP makes.
    """
    from langgraph.graph import END, START, StateGraph

    box = toolkit if toolbox is None else toolbox
    schemas = box.schemas(tool_names)

    # ── Node 1: REASON ──────────────────────────────────────────────────
    def reason(state: AgentState) -> dict:
        """Ask the model what it wants to do next."""
        text, calls = responder(state["messages"], schemas)

        message: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            # The OpenAI shape, and not for fashion -- chat templates are
            # written against it. Measured across the registry with a
            # two-round conversation:
            #
            #   flat {"name", "args"}   Qwen renders the turn but DROPS the
            #                           arguments, so the model cannot see
            #                           what it just asked for; Gemma 4's
            #                           template raises UndefinedError
            #                           ('dict object' has no attribute
            #                           'function') and the run dies.
            #   {"type": "function",    Qwen keeps the arguments, Gemma
            #    "function": {...}}     renders, SmolLM3 and Phi unchanged.
            #
            # `call_fields()` reads either, so nothing downstream cares.
            message["tool_calls"] = [
                {"type": "function", "id": c.name,
                 "function": {"name": c.name, "arguments": c.args}}
                for c in calls]

        return {
            "messages": state["messages"] + [message],
            "trace": state["trace"] + [{
                "step": "reason",
                "thought": text,
                "wants": [c.name for c in calls],
            }],
            "steps": state["steps"] + 1,
        }

    # ── Node 2: ACT ─────────────────────────────────────────────────────
    def act(state: AgentState) -> dict:
        """Run whatever the model asked for, and hand back what happened.

        Note there is no error handling here. There does not need to be:
        every toolbox promises that `dispatch()` never raises, and a failure
        comes back as a readable result the model can act on. See
        tools/base.py -- and offices.py, which honours the same contract for a
        server that is switched off in another building.
        """
        last = state["messages"][-1]
        new_messages, new_trace = [], []

        for call in last.get("tool_calls", []):
            name, args = call_fields(call)
            result = box.dispatch(name, args)
            new_messages.append({"role": "tool", "name": name,
                                 "content": result})
            new_trace.append({
                "step": "act",
                "tool": name,
                "args": args,
                "result": json.loads(result),
            })

        return {
            "messages": state["messages"] + new_messages,
            "trace": state["trace"] + new_trace,
            "steps": state["steps"],
        }

    # ── The one decision the graph makes ────────────────────────────────
    def needs_a_tool(state: AgentState) -> str:
        """Did the model ask for a tool, and are we allowed another turn?"""
        if state["steps"] >= max_steps:
            return "stop"
        return "act" if state["messages"][-1].get("tool_calls") else "stop"

    graph = StateGraph(AgentState)
    graph.add_node("reason", reason)
    graph.add_node("act", act)
    graph.add_edge(START, "reason")
    graph.add_conditional_edges("reason", needs_a_tool,
                                {"act": "act", "stop": END})
    graph.add_edge("act", "reason")     # observe, then think again
    return graph.compile()


def ask(agent, question: str, system: str = "") -> AgentState:
    """Run one question all the way through the graph.

    Returns the finished state, so a notebook can print `state["trace"]` and
    show the audience every turn of the loop.
    """
    messages: list[dict] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    return agent.invoke({"messages": messages, "trace": [], "steps": 0})


def final_answer(state: AgentState) -> str:
    """The last thing the model actually said to the user."""
    for message in reversed(state["messages"]):
        if message["role"] == "assistant" and message.get("content"):
            return str(message["content"]).strip()
    return ""
