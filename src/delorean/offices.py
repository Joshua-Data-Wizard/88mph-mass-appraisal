"""
offices.py — talking to three county offices over MCP.

Section 10. This is the client side: the agent's end of the wire.

    from delorean.offices import Offices, HILL_VALLEY

    with Offices(HILL_VALLEY, also_local=["search_policy"]) as offices:
        offices.catalogue()                 # what each office publishes
        offices.schemas()                   # the same list, model-ready
        offices.dispatch("get_deeds", {"parcel_id": "04-127-3891"})

WHAT IS ACTUALLY DIFFERENT FROM SECTION 6
-----------------------------------------
In Section 6 we wrote a Python function, wrote a JSON schema describing it by
hand, and handed both to the model. Every tool cost us two pieces of code.

Here we write **neither**. `Offices` starts three programs it did not write,
asks each one what it can do, and turns the answers into the same schema
format the agent already understands. Add a fourth office next year and this
file does not change — you add a line to a dictionary.

That is the whole claim of MCP, and `catalogue()` exists so it can be shown
rather than asserted.

WHY THERE IS A THREAD IN HERE
-----------------------------
The MCP SDK is async. Our agent graph is synchronous, and deliberately so —
Section 5 puts Reason → Act → Observe on a slide and the code has to match it.
Rather than colour the whole repository async for one section, `Offices` owns
a background thread running an event loop, and hands the rest of the codebase
ordinary blocking methods.

The loop runs **one long-lived task** that holds every connection open and
takes work off a queue. The obvious alternative — open a session per call —
was rejected twice over: it spawns a subprocess per question (about a second
each), and it throws away the thing worth showing, which is three servers
sitting there running.

`dispatch()` never raises. Same contract as `tools.dispatch()`, and for the
same reason: an agent that sees an exception cannot recover, but an agent that
reads *"there is no tool called that, here are the ones there are"* can.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import queue
import sys
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SERVER_DIR = REPO_ROOT / "mcp_server"


@dataclass(frozen=True)
class Office:
    """One MCP server, and who owns it.

    `label` and `holds` exist for the audience, not for the protocol. Section
    10 draws three panels and a room full of appraisers should be able to tell
    at a glance which elected official is being asked what.
    """

    key: str
    label: str
    script: str
    holds: str
    icon: str = "🏛️"

    @property
    def path(self) -> Path:
        return SERVER_DIR / self.script


#: The three offices of Hill Valley County. Three elected officials, three
#: vendors, three databases that have never spoken to one another.
HILL_VALLEY: tuple[Office, ...] = (
    Office("appraiser", "Appraiser", "appraiser_server.py",
           "the roll — characteristics, lot size, assessed value", "🏛️"),
    Office("clerk", "Clerk of Courts", "clerk_server.py",
           "recorded deeds — who owns it, and how", "📜"),
    Office("tax", "Tax Collector", "tax_server.py",
           "the ledger — levies, payments, what is still owed", "💰"),
)


class Offices:
    """A live connection to several MCP servers, usable from ordinary code."""

    def __init__(self, offices: tuple[Office, ...] = HILL_VALLEY, *,
                 also_local: list[str] | None = None, quiet: bool = True):
        """
        Args:
            offices:    which servers to start.
            also_local: names of tools from `delorean.tools` to offer
                        alongside them. The policy manual belongs to THIS
                        office, so it stays a local tool — MCP is for reaching
                        somebody else's system.
            quiet:      send each server's stderr to devnull. FastMCP logs
                        every request at INFO, which is a wall of text in
                        front of an audience. **Set this False first when
                        something is not working** — a server that fails to
                        start says so on stderr and nowhere else.
        """
        self.offices = offices
        self.also_local = list(also_local or [])
        self.quiet = quiet

        self._tools: dict[str, tuple[Office, dict]] = {}
        #: Filled in by the loop thread once it is running. Work is handed
        #: over with `call_soon_threadsafe`, never by blocking a thread.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._inbox: asyncio.Queue | None = None
        self._ready = threading.Event()
        self._failed: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._errlogs: list[Any] = []

    # ── lifecycle ────────────────────────────────────────────────────────

    def __enter__(self) -> "Offices":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> "Offices":
        """Start every server and ask it what it can do."""
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="mcp-offices")
        self._thread.start()
        self._ready.wait(timeout=60)
        if self._failed is not None:
            raise RuntimeError(
                f"Could not connect to the county offices: {self._failed}\n"
                f"Try Offices(..., quiet=False) to see what the server said."
            ) from self._failed
        if not self._ready.is_set():
            raise RuntimeError("The MCP servers did not start within 60s.")

        # A caller who forgets to close() should not leave three server
        # processes running. The loop thread is a daemon so it cannot hold
        # the interpreter open, but the SUBPROCESSES are not ours to abandon:
        # on Windows a child is not reaped when its parent exits.
        atexit.register(self.close)
        return self

    def close(self) -> None:
        """Shut the servers down. Safe to call twice, and at exit."""
        if self._thread is None:
            return
        loop, inbox = self._loop, self._inbox
        if loop is not None and inbox is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(inbox.put_nowait, None)
            except RuntimeError:
                pass            # the loop is already going down
        self._thread.join(timeout=20)
        self._thread = None
        for handle in self._errlogs:
            try:
                handle.close()
            except OSError:
                pass
        self._errlogs.clear()

    # ── what the offices offer ───────────────────────────────────────────

    def catalogue(self) -> list[dict]:
        """One row per tool, with the office that published it.

        Nothing in this list was written by us. It is what the three servers
        said when asked.
        """
        rows = []
        for name, (office, spec) in self._tools.items():
            description = (spec.get("description") or "").strip()
            rows.append({
                "office": office.label,
                "icon": office.icon,
                "tool": name,
                "takes": list(spec.get("inputSchema", {})
                              .get("properties", {})),
                "summary": description.splitlines()[0] if description else "",
            })
        return rows

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        """Tool schemas in the shape the responders already expect.

        MCP hands back JSON Schema, which is what the OpenAI tool format wraps
        anyway, so this is an envelope change and nothing more.
        """
        out = []
        for name, (_office, spec) in self._tools.items():
            if names is not None and name not in names:
                continue
            out.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec.get("description") or "",
                    "parameters": spec.get("inputSchema")
                    or {"type": "object", "properties": {}},
                },
            })
        if self.also_local:
            from . import tools as toolkit
            out.extend(toolkit.schemas(self.also_local))
        return out

    def owner_of(self, tool_name: str) -> Office | None:
        """Which office publishes this tool, if any."""
        found = self._tools.get(tool_name)
        return found[0] if found else None

    # ── running a tool ───────────────────────────────────────────────────

    def dispatch(self, name: str, args: dict | None = None) -> str:
        """Run one tool and return a JSON string. Never raises."""
        args = dict(args or {})

        if name not in self._tools:
            # Not one of theirs. It might still be one of ours.
            if self.also_local:
                from . import tools as toolkit
                if toolkit.resolve_tool(name) is not None:
                    return toolkit.dispatch(name, args)
            known = ", ".join(sorted(self._tools) + self.also_local)
            return json.dumps({
                "ok": False,
                "error": f"No county office publishes a tool called {name!r}.",
                "hint": f"Available: {known}.",
            })

        try:
            payload = self._submit(name, args)
        except Exception as exc:                        # noqa: BLE001
            office = self._tools[name][0]
            return json.dumps({
                "ok": False,
                "error": f"The {office.label} did not answer: {exc}",
                "hint": "The office may be offline. Try a different source, "
                        "or answer from what you already have and say which "
                        "office you could not reach.",
            })
        return payload if isinstance(payload, str) else json.dumps(payload,
                                                                   default=str)

    def _submit(self, name: str, args: dict) -> str:
        """Hand one call to the loop thread and wait for its answer.

        The reply still comes back on an ordinary `queue.Queue`, because THIS
        thread is allowed to block -- it is the caller's own. What must never
        block is a thread we did not create.
        """
        if self._loop is None or self._inbox is None:
            raise RuntimeError("The county offices are not connected.")
        done: queue.Queue = queue.Queue(maxsize=1)
        self._loop.call_soon_threadsafe(
            self._inbox.put_nowait, (name, args, done))
        kind, value = done.get(timeout=120)
        if kind == "error":
            raise value
        return value

    # ── the background loop ──────────────────────────────────────────────

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:                    # noqa: BLE001
            self._failed = exc
            self._ready.set()

    async def _serve(self) -> None:
        """Hold every connection open on one task and work through the queue.

        Everything happens in this single task on purpose. anyio's cancel
        scopes are tied to the task that created them, so opening a session in
        one task and closing it in another is a runtime error rather than a
        style question.
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._loop = asyncio.get_running_loop()
        self._inbox = asyncio.Queue()
        sessions: dict[str, Any] = {}

        async with AsyncExitStack() as stack:
            for office in self.offices:
                if not office.path.exists():
                    raise FileNotFoundError(
                        f"{office.path} is missing — is this the repo root?")

                errlog = sys.stderr
                if self.quiet:
                    errlog = open(os.devnull, "w")      # noqa: SIM115
                    self._errlogs.append(errlog)

                params = StdioServerParameters(
                    command=sys.executable,
                    args=[str(office.path)],
                    cwd=str(REPO_ROOT),
                )
                read, write = await stack.enter_async_context(
                    stdio_client(params, errlog=errlog))
                session = await stack.enter_async_context(
                    ClientSession(read, write))
                await session.initialize()
                sessions[office.key] = session

                # Ask it what it has. This is the line that replaces every
                # hand-written schema in Section 6.
                listed = await session.list_tools()
                for spec in listed.tools:
                    if spec.name in self._tools:
                        other = self._tools[spec.name][0].label
                        raise RuntimeError(
                            f"Both {other} and {office.label} publish a tool "
                            f"called {spec.name!r}. Rename one — the agent "
                            f"has no way to tell them apart.")
                    self._tools[spec.name] = (office, {
                        "description": spec.description,
                        "inputSchema": spec.inputSchema,
                    })

            self._ready.set()

            while True:
                # ⚠️ Awaited natively, NOT through run_in_executor.
                #
                # This used to be `await loop.run_in_executor(None,
                # self._jobs.get)`, which parks a worker of the default
                # ThreadPoolExecutor on a blocking get that never returns.
                # Those workers are non-daemon and `concurrent.futures`
                # joins every one of them at interpreter shutdown, so the
                # process could never exit. Two finished test runs sat there
                # for twenty minutes each, holding several GB of VRAM, long
                # after printing their results.
                #
                # The tell, if it ever happens again: wall-clock climbing
                # while CPU time stays flat. A working process burns CPU; a
                # process stuck in atexit does not.
                job = await self._inbox.get()
                if job is None:
                    return
                name, args, done = job
                try:
                    office, _spec = self._tools[name]
                    result = await sessions[office.key].call_tool(name, args)
                    parts = [c.text for c in result.content
                             if getattr(c, "text", None)]
                    done.put(("ok", "\n".join(parts) if parts else "{}"))
                except Exception as exc:                # noqa: BLE001
                    done.put(("error", exc))
