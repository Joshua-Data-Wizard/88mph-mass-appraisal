"""
agent.py — the Reason → Act → Observe loop, built by hand.

Section 5 puts this on a slide. Section 7 runs it. They are the same three
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

def local_responder(model: Any, tokenizer: Any, *, max_new_tokens: int = 320,
                    thinking: bool = False) -> Responder:
    """Talk to a model running on this machine, through `transformers`.

    320 tokens is not arbitrary. A tool call with thinking switched on runs to
    about 150; at 120 we measured it truncating mid-parameter, which the parser
    then sees as no tool call at all.
    """
    import torch

    def respond(messages: list[dict], schemas: list[dict]):
        prompt = tokenizer.apply_chat_template(
            messages, tools=schemas, tokenize=False,
            add_generation_prompt=True, enable_thinking=thinking,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)

        # Keep the untouched output so callers can show it verbatim. Section 6
        # reveals this to make the point that a "tool call" is just text the
        # model wrote -- reconstructing it from the parsed result would look
        # the same and quietly be a lie.
        respond.last_raw = raw

        calls = toolkit.parse_tool_calls(raw)
        if not calls and toolkit.attempted_tool_call(raw):
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
                    blocks.append({"type": "tool_use", "id": c.get("id", c["name"]),
                                   "name": c["name"], "input": c["args"]})
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
                *, max_steps: int = 4):
    """Wire up Reason → Act → Observe and hand back a runnable graph.

    Args:
        responder:  how to talk to the model. See above.
        tool_names: which tools the agent may use. Keep this list SHORT --
                    a bigger toolbelt measurably increases the chance the model
                    reaches for one when it should have just answered.
        max_steps:  how many times round the loop before we stop. Without this
                    a model that keeps asking for the same tool never finishes.
    """
    from langgraph.graph import END, START, StateGraph

    schemas = toolkit.schemas(tool_names)

    # ── Node 1: REASON ──────────────────────────────────────────────────
    def reason(state: AgentState) -> dict:
        """Ask the model what it wants to do next."""
        text, calls = responder(state["messages"], schemas)

        message: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            message["tool_calls"] = [{"name": c.name, "args": c.args}
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
        `dispatch()` never raises, and a failure comes back as a readable
        result the model can act on. See tools/base.py.
        """
        last = state["messages"][-1]
        new_messages, new_trace = [], []

        for call in last.get("tool_calls", []):
            result = toolkit.dispatch(call["name"], call["args"])
            new_messages.append({"role": "tool", "name": call["name"],
                                 "content": result})
            new_trace.append({
                "step": "act",
                "tool": call["name"],
                "args": call["args"],
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
