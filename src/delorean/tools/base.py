"""
base.py — the plumbing that makes a plain Python function safe for a small
language model to call.

The single most important idea in this file:

    ERRORS ARE PROMPTS.

When a tool fails, the message does not go to a log file. It goes straight back
into the model's context as the observation it reasons from next. A stack trace
teaches a model nothing. A sentence like "parcel numbers look like NN-NNN-NNNN,
did you mean 04-127-3891?" lets it fix its own mistake on the next turn.

So every failure in here produces a short, plain-English `hint` written to be
read BY THE MODEL, not by a developer.

Why any of this is needed
-------------------------
Small models are good at deciding WHICH tool to call and sloppy about HOW. We
measured this: across ten phrasings of the same question, Qwen3.5-4B picked the
right tool 10 out of 10 times, and emitted the parcel number three different
ways -- "04-127-3891", "04127-3891", and "4-127-3891". A plain
`WHERE parcel_id = ?` would have failed two of those ten. Silently.

That is a 20% failure rate on stage, and it looks exactly like a parcel that
does not exist.

This module absorbs that class of mistake so the demos do not have to.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

# ─────────────────────────────────────────────────────────────────────────
#  The result envelope
#
#  Everything a tool returns is a JSON *string*, because that is what the
#  chat templates want in a {"role": "tool"} message. Keeping one shape for
#  success and one for failure means the model learns the pattern quickly.
# ─────────────────────────────────────────────────────────────────────────

def ok(data: Any) -> str:
    """A successful tool result."""
    return json.dumps({"ok": True, "data": data}, default=str)


def fail(error: str, hint: str = "") -> str:
    """A failed tool result.

    `error` says what went wrong. `hint` says what to do about it, and it is
    the half that matters -- it is the model's next instruction.
    """
    payload: dict[str, Any] = {"ok": False, "error": error}
    if hint:
        payload["hint"] = hint
    return json.dumps(payload)


# ─────────────────────────────────────────────────────────────────────────
#  The registry
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Tool:
    """One tool: a function, the schema the model sees, and its nicknames."""

    name: str
    description: str
    parameters: dict           # JSON Schema, written by hand next to the function
    fn: Callable
    name_aliases: tuple[str, ...] = ()
    param_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def schema(self) -> dict:
        """The tool definition in the shape `apply_chat_template(tools=...)` wants."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


REGISTRY: dict[str, Tool] = {}


def tool(*, name: str, description: str, parameters: dict,
         name_aliases: tuple[str, ...] = (),
         param_aliases: dict[str, tuple[str, ...]] | None = None):
    """Register a function as a tool the model may call.

    The schema is written out by hand rather than derived from the signature.
    That is deliberate: an appraiser reading this repository should be able to
    see the function and the description the model gets, side by side, without
    working out what a decorator inferred.

    The decorated function returns a plain dict (or raises). Wrapping it in the
    JSON envelope is this module's job, not the tool author's.
    """
    def register(fn: Callable) -> Callable:
        REGISTRY[name] = Tool(
            name=name, description=description, parameters=parameters, fn=fn,
            name_aliases=name_aliases, param_aliases=param_aliases or {},
        )
        return fn

    return register


def schemas(names: list[str] | None = None) -> list[dict]:
    """Tool definitions to hand the model.

    Pass `names` to expose only some of them. Section 6 gives the agent exactly
    one tool on the live path, because watching it choose between five is a
    different lesson than watching it use one.
    """
    chosen = REGISTRY.values() if names is None else [
        REGISTRY[n] for n in names if n in REGISTRY
    ]
    return [t.schema() for t in chosen]


# ─────────────────────────────────────────────────────────────────────────
#  Forgiving lookup — tool names and parameter names
# ─────────────────────────────────────────────────────────────────────────

def _squash(text: str) -> str:
    """Reduce a name to letters and digits, lowercased.

    Turns `lookup-parcel`, `lookupParcel` and `Lookup_Parcel` into the same
    string, so a model that guesses the punctuation still gets through.
    """
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def resolve_tool(name: str) -> Tool | None:
    """Find a tool by name, tolerating punctuation, case and known nicknames."""
    if name in REGISTRY:
        return REGISTRY[name]

    target = _squash(name)
    for t in REGISTRY.values():
        if _squash(t.name) == target:
            return t
        if any(_squash(a) == target for a in t.name_aliases):
            return t

    # Last resort: a singular/plural slip, e.g. lookup_parcels -> lookup_parcel
    for t in REGISTRY.values():
        squashed = _squash(t.name)
        if target.rstrip("s") == squashed.rstrip("s"):
            return t
    return None


def _resolve_params(t: Tool, args: dict) -> dict:
    """Map whatever the model called the arguments onto the real parameter names.

    A small model asked for `parcel_id` may well send `parcel`, `id`, or
    `parcelId`. All of those are obvious to a human and fatal to a dict lookup.
    """
    known = list(t.parameters.get("properties", {}))
    out: dict[str, Any] = {}

    for given, value in args.items():
        if given in known:
            out[given] = value
            continue

        squashed = _squash(given)
        match = None

        # 1. an alias we declared on purpose
        for real, aliases in t.param_aliases.items():
            if squashed in {_squash(a) for a in aliases}:
                match = real
                break

        # 2. same name, different punctuation ("parcelId" -> "parcel_id")
        if match is None:
            for real in known:
                if _squash(real) == squashed:
                    match = real
                    break

        # 3. a fragment of the real name ("parcel" or "id" -> "parcel_id"),
        #    but only when it is unambiguous
        if match is None:
            candidates = [r for r in known
                          if squashed and squashed in _squash(r)]
            if len(candidates) == 1:
                match = candidates[0]

        if match is not None:
            out[match] = value
        # anything still unmatched is dropped rather than raising -- an extra
        # argument the model invented should not kill the call.

    return out


# ─────────────────────────────────────────────────────────────────────────
#  Coercion
#
#  Tool calls arrive as text. Qwen3.5 and Nemotron use an XML format with no
#  types at all:
#
#      <parameter=year_built>1971</parameter>   ->   the string "1971"
#
#  So every value has to be talked back into the type the schema asked for.
# ─────────────────────────────────────────────────────────────────────────

_NULLISH = {"", "null", "none", "nil", "n/a", "na", "undefined"}


def _coerce(value: Any, spec: dict) -> Any:
    """Turn one argument into the type its schema asks for."""
    wanted = spec.get("type", "string")

    if isinstance(value, str):
        value = value.strip()
        # Models often wrap values in quotes that were never meant literally.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1].strip()
        if value.lower() in _NULLISH:
            return None

    if wanted == "string":
        return str(value)

    if wanted in ("integer", "number"):
        if isinstance(value, (int, float)):
            return int(value) if wanted == "integer" else float(value)
        # "$412,500" and "1,640 sqft" both show up in practice.
        digits = re.sub(r"[^0-9.\-]", "", str(value))
        if digits in ("", "-", "."):
            raise ValueError(f"expected a number, got {value!r}")
        return int(float(digits)) if wanted == "integer" else float(digits)

    if wanted == "boolean":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "yes", "y", "1"}

    if wanted == "array":
        if isinstance(value, list):
            return value
        text = str(value).strip()
        if text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        return [p.strip() for p in text.split(",") if p.strip()]

    return value


# ─────────────────────────────────────────────────────────────────────────
#  Dispatch — the one function the agent node calls
# ─────────────────────────────────────────────────────────────────────────

class ToolCall(NamedTuple):
    """One tool call the model asked for."""

    name: str
    args: dict


# The registry emits TWO different formats, and both wrap in <tool_call>, so a
# tag check alone will fool you. Measured 2026-08-16:
#
#   Shape A -- Qwen3.5 and Nemotron        Shape B -- SmolLM3
#   <tool_call>                            <tool_call>
#   <function=lookup_parcel>               {"name": "lookup_parcel",
#   <parameter=parcel_id>                   "arguments": {"parcel_id": "..."}}
#   04-127-3891                            </tool_call>
#   </parameter>
#   </function>
#   </tool_call>
#
# A parser that only does json.loads() on the tag body -- the obvious
# implementation -- fails on our own default model.
_SHAPE_A = re.compile(r"<tool_call>\s*<function=([^>\s]+)\s*>(.*?)</function>", re.S)
_SHAPE_A_PARAM = re.compile(r"<parameter=([^>\s]+)\s*>\s*(.*?)\s*</parameter>", re.S)
_SHAPE_B = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Pull every tool call out of what the model generated.

    Returns a list because Qwen3.5 is known to be over-eager about asking for
    several tools at once. Callers that only want one take the first.

    Note that Shape A has no types at all -- every value arrives as a string.
    `dispatch()` coerces them against the schema.
    """
    calls: list[ToolCall] = []

    for name, body in _SHAPE_A.findall(text or ""):
        calls.append(ToolCall(name.strip(), dict(_SHAPE_A_PARAM.findall(body))))

    if not calls:
        for blob in _SHAPE_B.findall(text or ""):
            try:
                payload = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("name"):
                args = payload.get("arguments") or payload.get("parameters") or {}
                calls.append(ToolCall(str(payload["name"]), dict(args)))

    return calls


def attempted_tool_call(text: str) -> bool:
    """Did the model start a tool call, whether or not it finished one?

    True with an empty `parse_tool_calls()` means the call was **cut off** --
    almost always `max_new_tokens` set too low. Budget at least 320 when
    thinking is on, or you get a truncated parameter and no way to tell why.
    """
    return "<tool_call>" in (text or "")


def strip_tool_calls(text: str) -> str:
    """The prose the model wrote around its tool call, for showing a human."""
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", text or "", flags=re.S)
    cleaned = re.sub(r"<tool_call>.*", "", cleaned, flags=re.S)   # truncated tail
    return cleaned.strip()


def dispatch(name: str, args: dict | None = None) -> str:
    """Run a tool by name and return its JSON result string.

    This never raises. An agent that sees a traceback has no way to recover;
    an agent that sees {"ok": false, "hint": "..."} usually does.
    """
    args = dict(args or {})

    t = resolve_tool(name)
    if t is None:
        return fail(
            f"There is no tool called {name!r}.",
            "Available tools: " + ", ".join(sorted(REGISTRY)) + ".",
        )

    args = _resolve_params(t, args)
    props: dict = t.parameters.get("properties", {})
    required: list = t.parameters.get("required", [])

    # Coerce what we were given.
    clean: dict[str, Any] = {}
    for key, value in args.items():
        try:
            coerced = _coerce(value, props.get(key, {}))
        except (ValueError, TypeError):
            return fail(
                f"The value for {key!r} was {value!r}, which is not a "
                f"{props.get(key, {}).get('type', 'valid value')}.",
                f"Call {t.name} again with a valid {key}.",
            )
        if coerced is not None:
            clean[key] = coerced

    missing = [r for r in required if r not in clean]
    if missing:
        listed = ", ".join(missing)
        return fail(
            f"{t.name} needs {listed}, which was not provided.",
            f"Call {t.name} again and include {listed}. "
            + "; ".join(
                f"{m}: {props.get(m, {}).get('description', '')}" for m in missing
            ),
        )

    # Run it. Anything the tool throws becomes a readable failure.
    try:
        result = t.fn(**clean)
    except TypeError as e:
        return fail(f"{t.name} could not be called: {e}",
                    f"Expected parameters: {', '.join(props) or 'none'}.")
    except Exception as e:  # noqa: BLE001 - deliberately broad; see docstring
        return fail(f"{t.name} failed: {type(e).__name__}: {e}",
                    "Try a different value, or a different tool.")

    # A tool may return a finished envelope itself (for a rich not-found
    # message); otherwise wrap whatever it gave us.
    if isinstance(result, str):
        return result
    return ok(result)
