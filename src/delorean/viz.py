"""
viz.py — making what the model is doing visible.

None of this changes what the model does. It exists so that a room full of
people can see it from the back row.

Everything is plain HTML with inline styles, which means the same visuals work
in three different places:

    Jupyter     viz.show_tokens(tok, "text")        <- displays it
    Streamlit   st.html(viz.tokens_html(tok, "text")) <- returns the string
    GitHub      committed notebook outputs render as-is

The `*_html()` functions build the markup. The `show_*()` functions display it
in a notebook. Same markup either way, so the notebook and the app never drift
apart.
"""

from __future__ import annotations

import html
import re
from typing import Any, Iterable, Sequence

# A light card, so it looks the same on a dark JupyterLab theme, a light
# GitHub page, and a projector that washes out subtle colors.
CARD = (
    "background:#ffffff; color:#0f172a; border:1px solid #e2e8f0;"
    "border-radius:10px; padding:18px 20px; margin:6px 0;"
    "font-family:-apple-system,Segoe UI,Roboto,sans-serif;"
)
MONO = "ui-monospace,SFMono-Regular,Consolas,'Courier New',monospace"

# Token chips alternate between these so you can see where one ends and the
# next begins. That boundary is the entire point of the tokenizer view.
CHIP_COLORS = ("#dbeafe", "#fef3c7")
CHIP_BORDERS = ("#93c5fd", "#fcd34d")


def show_html(markup: str) -> None:
    """Display any markup built by the *_html() functions in a notebook.

    Every builder in this file returns a string, so a notebook can compose or
    loop over them and then hand the result here. Streamlit calls
    `st.html(...)` on the same string instead.
    """
    from IPython.display import HTML, display

    display(HTML(markup))


def _visible(text: str) -> str:
    """Make whitespace visible.

    A token is very often " value" with a leading space, not "value". People
    should see that, because it explains a lot about how models behave.
    """
    escaped = html.escape(text)
    return (
        escaped.replace(" ", "<span style='opacity:.35'>&middot;</span>")
        .replace("\n", "<span style='opacity:.35'>&#8617;</span>")
        .replace("\t", "<span style='opacity:.35'>&#8677;</span>")
    )


# ── Tokens: words become numbers ─────────────────────────────────────────────

def tokens_html(tokenizer: Any, text: str, *, show_ids: bool = True,
                show_source: bool = True) -> str:
    """Build the markup showing how text is chopped into tokens.

    Each chip is one token — one indivisible piece as far as the model is
    concerned. The number underneath is that token's id: the actual integer
    the model receives. It never sees letters.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    chips = []
    for n, tid in enumerate(ids):
        piece = tokenizer.decode([tid])
        bg, border = CHIP_COLORS[n % 2], CHIP_BORDERS[n % 2]
        idline = (
            f"<div style='font-size:10px;opacity:.55;text-align:center;margin-top:3px'>{tid}</div>"
            if show_ids
            else ""
        )
        chips.append(
            f"<div style='display:inline-block;margin:3px 2px;vertical-align:top'>"
            f"<div style='background:{bg};border:1px solid {border};border-radius:5px;"
            f"padding:5px 8px;font-family:{MONO};font-size:15px;white-space:pre'>"
            f"{_visible(piece)}</div>{idline}</div>"
        )

    source = (
        f"<div style='font-size:13px;opacity:.6;margin-bottom:8px'>"
        f"&ldquo;{html.escape(text)}&rdquo;</div>"
        if show_source
        else ""
    )
    words = len(text.split())
    return (
        f"<div style='{CARD}'>{source}"
        f"<div>{''.join(chips)}</div>"
        f"<div style='margin-top:12px;font-size:15px'>"
        f"<b>{len(ids)}</b> token{'s' if len(ids) != 1 else ''} "
        f"&mdash; not {words} word{'s' if words != 1 else ''}</div>"
        f"</div>"
    )


def show_tokens(tokenizer: Any, text: str, *, show_ids: bool = True) -> None:
    """Display the token breakdown in a notebook."""
    from IPython.display import HTML, display

    display(HTML(tokens_html(tokenizer, text, show_ids=show_ids)))


# ── Probabilities: one prediction, out of many ───────────────────────────────

def probabilities_html(
    pairs: Iterable[tuple[str, float]],
    *,
    prompt: str | None = None,
    vocab_size: int | None = None,
) -> str:
    """Build the markup for the model's top candidates for the next token.

    `pairs` is (token_text, probability), highest first. The winner is green;
    everything else is grey. The bars are the point — people read shape faster
    than they read numbers.
    """
    pairs = list(pairs)
    if not pairs:
        return ""

    head = ""
    if prompt is not None:
        head = (
            f"<div style='font-family:{MONO};font-size:16px;background:#f8fafc;"
            f"border:1px dashed #cbd5e1;border-radius:6px;padding:10px 12px;margin-bottom:14px'>"
            f"{html.escape(prompt)}"
            f"<span style='background:#fde68a;padding:1px 8px;border-radius:4px;"
            f"margin-left:4px'>?</span></div>"
        )

    rows = []
    for n, (token, prob) in enumerate(pairs):
        winner = n == 0
        color = "#16a34a" if winner else "#94a3b8"
        weight = "700" if winner else "400"
        # Scale bars against the top candidate so a 99% winner does not make
        # every runner-up an invisible sliver.
        width = max(prob / pairs[0][1] * 100, 0.7) if pairs[0][1] else 0.7
        rows.append(
            f"<tr>"
            f"<td style='text-align:right;padding:5px 10px 5px 0;font-family:{MONO};"
            f"font-size:16px;font-weight:{weight};white-space:pre'>{_visible(token)}</td>"
            f"<td style='width:100%;padding:5px 0'>"
            f"<div style='background:{color};height:20px;width:{width:.1f}%;"
            f"border-radius:3px;min-width:2px'></div></td>"
            f"<td style='padding:5px 0 5px 12px;font-family:{MONO};font-size:16px;"
            f"font-weight:{weight};color:{color};white-space:nowrap'>{prob*100:5.1f}%</td>"
            f"</tr>"
        )

    foot = ""
    if vocab_size:
        foot = (
            f"<div style='margin-top:12px;font-size:13px;opacity:.6'>"
            f"chosen from {vocab_size:,} possible tokens</div>"
        )

    return (
        f"<div style='{CARD}'>{head}"
        f"<table style='width:100%;border-collapse:collapse'>{''.join(rows)}</table>"
        f"{foot}</div>"
    )


def show_probabilities(
    pairs: Iterable[tuple[str, float]],
    *,
    prompt: str | None = None,
    vocab_size: int | None = None,
) -> None:
    """Display the next-token odds in a notebook."""
    from IPython.display import HTML, display

    display(HTML(probabilities_html(pairs, prompt=prompt, vocab_size=vocab_size)))


# ── Generation: do it again, and again ───────────────────────────────────────

def canvas_html(
    tokens: Sequence[str],
    *,
    title: str | None = None,
    boundaries: bool = True,
    show_count: bool = True,
) -> str:
    """Build the markup for generated text, optionally with token boundaries.

    With boundaries on, each token gets its own shaded block. Ordinary
    streaming looks like someone typing; this looks like what is actually
    happening — one discrete piece arriving at a time.
    """
    if boundaries:
        body = "".join(
            f"<span style='background:{CHIP_COLORS[n % 2]};"
            f"border-radius:3px;padding:1px 1px;white-space:pre-wrap'>"
            f"{html.escape(tok)}</span>"
            for n, tok in enumerate(tokens)
        )
    else:
        body = f"<span style='white-space:pre-wrap'>{html.escape(''.join(tokens))}</span>"

    head = (
        f"<div style='font-size:13px;opacity:.6;margin-bottom:10px'>{html.escape(title)}</div>"
        if title
        else ""
    )
    count = (
        f"<div style='margin-top:12px;font-size:13px;opacity:.6'>"
        f"{len(tokens)} piece{'s' if len(tokens) != 1 else ''} so far</div>"
        if show_count
        else ""
    )
    return (
        f"<div style='{CARD}'>{head}"
        f"<div style='font-family:{MONO};font-size:16px;line-height:2'>{body}</div>"
        f"{count}</div>"
    )


class TokenCanvas:
    """A live text area that shows each token as a separate block.

        canvas = TokenCanvas()
        for token in stream:
            canvas.add(token)
    """

    _counter = 0

    def __init__(self, *, title: str | None = None, boundaries: bool = True):
        from IPython.display import HTML, display

        TokenCanvas._counter += 1
        self.display_id = f"delorean-canvas-{TokenCanvas._counter}"
        self.title = title
        self.boundaries = boundaries
        self.tokens: list[str] = []
        display(HTML(self._render()), display_id=self.display_id)

    def add(self, token: str) -> None:
        """Append one token and redraw."""
        from IPython.display import HTML, display

        self.tokens.append(token)
        display(HTML(self._render()), display_id=self.display_id, update=True)

    def _render(self) -> str:
        return canvas_html(self.tokens, title=self.title, boundaries=self.boundaries)


# ── A plain note box, for the callouts ───────────────────────────────────────

def note_html(title: str, body: str, *, kind: str = "info") -> str:
    """Build a colored callout box. kind: info | warn | good."""
    palette = {
        "info": ("#eff6ff", "#3b82f6"),
        "warn": ("#fffbeb", "#f59e0b"),
        "good": ("#f0fdf4", "#22c55e"),
    }
    bg, accent = palette.get(kind, palette["info"])
    return (
        f"<div style='{CARD} background:{bg}; border-left:5px solid {accent}'>"
        f"<div style='font-weight:700;font-size:15px;margin-bottom:6px'>{html.escape(title)}</div>"
        f"<div style='font-size:15px;line-height:1.6'>{body}</div></div>"
    )


def note(title: str, body: str, *, kind: str = "info") -> None:
    """Display a callout box in a notebook."""
    from IPython.display import HTML, display

    display(HTML(note_html(title, body, kind=kind)))


# ─────────────────────────────────────────────────────────────────────────────
#  Section 3 — the time circuits
#
#  The DeLorean's dashboard has three rows of LEDs: where you are going, when
#  you are now, and where you were last. It maps onto a chat call almost
#  exactly, which is too good a coincidence to waste:
#
#      DESTINATION TIME   -> the system prompt. Who the model is this time.
#      PRESENT TIME       -> the question being asked right now.
#      LAST TIME DEPARTED -> the persona used before this one.
#
#  Deliberately dark, unlike everything else in this file. It is a piece of
#  hardware on screen, and it needs to read from the back of a conference room.
# ─────────────────────────────────────────────────────────────────────────────

# (label, text colour, glow colour) for each of the three rows.
_CIRCUIT_ROWS = (
    ("Destination Time", "#ff3b30", "rgba(255,59,48,.80)"),
    ("Present Time", "#34e05a", "rgba(52,224,90,.75)"),
    ("Last Time Departed", "#ffb000", "rgba(255,176,0,.70)"),
)


def _clip(text: str, limit: int) -> str:
    """Shorten to fit one LED row, with an ellipsis if we cut it."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def time_circuits_html(
    destination: str,
    present: str = "",
    last_departed: str = "",
    *,
    limit: int = 62,
) -> str:
    """Build the three-row time circuit display.

    Args:
        destination:   the persona / system prompt now loaded.
        present:       the question being asked.
        last_departed: the persona used before this one.
        limit:         characters per row before truncation.
    """
    values = (destination, present, last_departed)
    rows = []
    for (label, color, glow), value in zip(_CIRCUIT_ROWS, values):
        shown = _clip(value, limit) or "—"
        dim = "" if value else "opacity:.35;"
        rows.append(
            f"<div style='display:flex;align-items:center;gap:12px;padding:5px 0'>"
            f"<div style='width:132px;flex:none;text-align:right;font-size:9px;"
            f"letter-spacing:.16em;text-transform:uppercase;color:#7b7b7b;"
            f"font-family:-apple-system,Segoe UI,sans-serif'>{label}</div>"
            f"<div style='flex:1;min-width:0;background:#0d0d0d;border:1px solid #242424;"
            f"border-radius:4px;padding:9px 13px;font-family:{MONO};font-size:16px;"
            f"font-weight:600;color:{color};text-shadow:0 0 9px {glow};{dim}"
            f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>"
            f"{html.escape(shown)}</div>"
            f"</div>"
        )

    # Two layers, and the outer one is the point.
    #
    # The display itself has to stay dark -- it is a DeLorean instrument, and
    # the glowing readouts only work against black. But a near-black rectangle
    # dropped straight onto a light page reads as a hole punched through it,
    # which is exactly how this looked once the Streamlit theme was pinned to
    # light. Wrapping it in a lit metal bezel and letting it cast a real shadow
    # turns the same panel into something that is *mounted on* the page: the
    # contrast becomes deliberate rather than jarring.
    display = (
        f"<div style='background:#0a0b0d;border-radius:7px;padding:12px 15px;"
        f"box-shadow:inset 0 2px 16px rgba(0,0,0,.85), inset 0 0 0 1px #1b1e23'>"
        f"{''.join(rows)}</div>"
    )

    return (
        f"<div style='background:linear-gradient(180deg,#5b616b 0%,#3d424a 45%,"
        f"#2c3037 100%);border-radius:12px;padding:9px;margin:8px 0;"
        f"box-shadow:0 12px 26px -14px rgba(15,23,42,.55),"
        f"0 2px 5px rgba(15,23,42,.16),"
        f"inset 0 1px 0 rgba(255,255,255,.22)'>{display}</div>"
    )


def show_time_circuits(destination: str, present: str = "",
                       last_departed: str = "") -> None:
    """Display the time circuit panel in a notebook."""
    from IPython.display import HTML, display

    display(HTML(time_circuits_html(destination, present, last_departed)))


# ─────────────────────────────────────────────────────────────────────────────
#  Section 3 — one persona's answer
# ─────────────────────────────────────────────────────────────────────────────

#: Accent colours for personas shown side by side. Picked to stay
#: distinguishable on a washed-out projector.
PERSONA_COLORS = ("#2563eb", "#dc2626", "#7c3aed", "#0d9488", "#c2410c")


def persona_card_html(
    name: str,
    answer: str,
    *,
    system_prompt: str | None = None,
    accent: str = PERSONA_COLORS[0],
    pending: bool = False,
) -> str:
    """Build one persona's answer card, for the side-by-side comparison.

    The system prompt is shown above the answer on purpose. The whole lesson
    is that this text -- and only this text -- produced that difference.
    """
    head = (
        f"<div style='background:{accent};color:#fff;padding:7px 13px;"
        f"border-radius:7px 7px 0 0;font-size:13px;font-weight:700;"
        f"letter-spacing:.03em'>{html.escape(name)}</div>"
    )

    prompt_block = ""
    if system_prompt is not None:
        prompt_block = (
            f"<div style='background:#f8fafc;border-bottom:1px solid #e2e8f0;"
            f"padding:9px 13px;font-family:{MONO};font-size:11.5px;"
            f"line-height:1.5;color:#475569'>"
            f"<span style='opacity:.55'>system:</span> "
            f"{html.escape(system_prompt)}</div>"
        )

    body_text = html.escape(answer) if answer else (
        "<span style='opacity:.4'>waiting…</span>" if pending else ""
    )
    cursor = (
        "<span style='display:inline-block;width:8px;height:15px;"
        f"background:{accent};vertical-align:-2px;margin-left:2px'></span>"
        if pending and answer else ""
    )

    return (
        f"<div style='border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;"
        f"margin:6px 0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;"
        f"background:#fff;color:#0f172a'>"
        f"{head}{prompt_block}"
        f"<div style='padding:13px;font-size:14.5px;line-height:1.62;"
        f"white-space:pre-wrap'>{body_text}{cursor}</div></div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Section 4 — what it said, against what is true
#
#  This is the component the whole of Section 4 rests on. A hallucination is
#  only alarming once you put the real record beside it, because on its own an
#  invented answer looks exactly like a correct one. That is the point.
# ─────────────────────────────────────────────────────────────────────────────

def verification_html(
    claim: str,
    truth: str,
    *,
    source: str,
    question: str | None = None,
    consequence: str | None = None,
) -> str:
    """Build the side-by-side 'model said / county record says' panel.

    Args:
        claim:       what the model asserted.
        truth:       what the record actually says.
        source:      where the truth came from, e.g. "Policy 4.3".
        question:    the question that was asked, shown above.
        consequence: why being wrong here matters. This is the line that
                     turns a curiosity into a problem, so it is worth filling in.
    """
    q_block = ""
    if question:
        q_block = (
            f"<div style='font-family:{MONO};font-size:13.5px;background:#f8fafc;"
            f"border:1px dashed #cbd5e1;border-radius:6px;padding:10px 12px;"
            f"margin-bottom:14px'>{html.escape(question)}</div>"
        )

    def column(title: str, body: str, note_line: str, color: str, bg: str) -> str:
        return (
            f"<div style='flex:1;min-width:0;background:{bg};border:1px solid {color}40;"
            f"border-top:3px solid {color};border-radius:7px;padding:12px 14px'>"
            f"<div style='font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;"
            f"color:{color};font-weight:700;margin-bottom:8px'>{title}</div>"
            f"<div style='font-size:15px;line-height:1.6;white-space:pre-wrap'>"
            f"{html.escape(body)}</div>"
            f"<div style='margin-top:9px;font-size:11.5px;opacity:.65'>{note_line}</div>"
            f"</div>"
        )

    cols = (
        f"<div style='display:flex;gap:12px;flex-wrap:wrap'>"
        + column("The model said", claim, "generated, not retrieved",
                 "#dc2626", "#fef2f2")
        + column("The county record says", truth, html.escape(source),
                 "#16a34a", "#f0fdf4")
        + "</div>"
    )

    tail = ""
    if consequence:
        tail = (
            f"<div style='margin-top:13px;background:#fffbeb;border-left:4px solid #f59e0b;"
            f"border-radius:0 6px 6px 0;padding:11px 14px;font-size:14px;line-height:1.6'>"
            f"<b>Why this matters.</b> {consequence}</div>"
        )

    return f"<div style='{CARD}'>{q_block}{cols}{tail}</div>"


def show_verification(claim: str, truth: str, *, source: str,
                      question: str | None = None,
                      consequence: str | None = None) -> None:
    """Display the truth-check panel in a notebook."""
    from IPython.display import HTML, display

    display(HTML(verification_html(claim, truth, source=source,
                                   question=question, consequence=consequence)))


# ─────────────────────────────────────────────────────────────────────────────
#  Sections 6-8 — watching the agent think
#
#  Section 5 draws Reason -> Act -> Observe on a slide. These functions draw the
#  same three boxes from a real run, so the concept and the machinery are
#  visibly the same thing.
#
#  The audience for this is a property appraisal director, not an engineer. The
#  design rule throughout: a "tool call" must stop sounding like machinery and
#  start looking like the model asking someone a question and getting an answer.
# ─────────────────────────────────────────────────────────────────────────────

# One colour per step of the loop. Used for the badge, the label and the rule
# down the left of each card, so the eye can follow the sequence at a glance.
STEP_COLORS = {
    "reason": "#4f46e5",    # indigo -- thinking
    "act": "#b45309",       # amber  -- doing
    "observe": "#15803d",   # green  -- what came back
    "error": "#b91c1c",     # red    -- what came back, badly
    "answer": "#0f172a",    # near-black -- the thing the human reads
}


def toolbelt_html(names: Sequence[str], *, active: str | None = None,
                  title: str = "Tools this agent may use") -> str:
    """Show the agent's tools as chips.

    Section 8's whole argument is that RAG is not special -- it is one more
    entry in this list. Showing the list grow from one chip to two makes that
    argument without anyone having to assert it.
    """
    chips = []
    for name in names:
        live = name == active
        bg = "#4f46e5" if live else "#eef2ff"
        fg = "#ffffff" if live else "#3730a3"
        border = "#4f46e5" if live else "#c7d2fe"
        weight = "700" if live else "500"
        chips.append(
            f"<span style='display:inline-block;background:{bg};color:{fg};"
            f"border:1px solid {border};border-radius:999px;padding:5px 13px;"
            f"margin:3px 4px 3px 0;font-family:{MONO};font-size:13px;"
            f"font-weight:{weight}'>{html.escape(name)}</span>"
        )

    body = "".join(chips) if chips else "<i>none</i>"
    return (
        f"<div style='{CARD}'>"
        f"<div style='font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;"
        f"color:#64748b;font-weight:700;margin-bottom:9px'>{html.escape(title)}"
        f" &middot; {len(names)}</div>"
        f"<div>{body}</div></div>"
    )


def raw_call_html(text: str, *, title: str = "What the model actually wrote") -> str:
    """Show the literal text the model produced, tool-call markup and all.

    This is the demystifying moment of Section 6. "Tool calling" sounds like
    plumbing somebody clever built into the model. It is not. The model wrote
    some text, in a format it was trained to use, and about fifteen lines of
    our Python noticed and acted on it.

    Anyone who sat through Section 2 already knows the punchline: this is still
    next-token prediction. It never stopped being next-token prediction.
    """
    escaped = html.escape(text or "")

    # Light up the tool-call markup so the eye lands on it immediately.
    escaped = re.sub(
        r"(&lt;/?(?:tool_call|function|parameter)[^&]*?&gt;)",
        r"<span style='color:#fbbf24;font-weight:700'>\1</span>",
        escaped,
    )

    # Same reasoning as the time circuits: a dark block on a light page needs a
    # shadow to sit on the page rather than float above a hole in it. A dark
    # code panel is a familiar sight, so this one needs far less help.
    return (
        f"<div style='background:#0b1020;border:1px solid #1e293b;border-radius:9px;"
        f"padding:14px 16px;margin:6px 0;"
        f"box-shadow:0 8px 20px -14px rgba(15,23,42,.5)'>"
        f"<div style='font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;"
        f"color:#64748b;font-weight:700;margin-bottom:9px'>{html.escape(title)}</div>"
        f"<pre style='margin:0;font-family:{MONO};font-size:13.5px;line-height:1.62;"
        f"color:#e2e8f0;white-space:pre-wrap;word-break:break-word'>{escaped}</pre>"
        f"</div>"
    )


def _call_signature(tool: str, args: dict) -> str:
    """Render a tool call the way a person would write it: name(arg=value)."""
    parts = []
    for key, value in (args or {}).items():
        shown = str(value)
        if len(shown) > 60:
            shown = shown[:57] + "..."
        parts.append(f"{html.escape(str(key))}=<b>{html.escape(shown)}</b>")
    return f"{html.escape(tool)}({', '.join(parts)})"


def _observe_html(result: dict) -> str:
    """Render whatever a tool handed back, in a shape a human can scan.

    A parcel record becomes a small table. Retrieved policy becomes citations.
    A failure shows its hint, because the hint is the interesting part -- it is
    the sentence the model reads next and uses to correct itself.
    """
    if not isinstance(result, dict):
        return (f"<div style='font-family:{MONO};font-size:13px'>"
                f"{html.escape(str(result))}</div>")

    if not result.get("ok", True):
        hint = result.get("hint", "")
        hint_block = ""
        if hint:
            hint_block = (
                f"<div style='margin-top:9px;background:#fffbeb;"
                f"border-left:3px solid #f59e0b;padding:8px 11px;"
                f"border-radius:0 5px 5px 0;font-size:13.5px;line-height:1.55'>"
                f"<b>Hint handed back to the model:</b><br/>"
                f"{html.escape(hint)}</div>"
            )
        return (
            f"<div style='font-size:14.5px;line-height:1.55'>"
            f"<b style='color:#b91c1c'>Failed.</b> "
            f"{html.escape(str(result.get('error', '')))}</div>{hint_block}"
        )

    data = result.get("data", result)

    # Retrieved policy passages -- lead with the citation.
    if isinstance(data, dict) and isinstance(data.get("passages"), list):
        rows = []
        for p in data["passages"]:
            body = " ".join(str(p.get("text", "")).split())
            more = "&hellip;" if len(body) > 340 else ""
            rows.append(
                f"<div style='margin-bottom:11px'>"
                f"<div style='font-family:{MONO};font-size:12.5px;color:#15803d;"
                f"font-weight:700'>{html.escape(str(p.get('cite', '')))}"
                f"<span style='opacity:.55;font-weight:400'> &middot; relevance "
                f"{p.get('relevance', '')}</span></div>"
                f"<div style='font-size:13.5px;line-height:1.55;margin-top:3px'>"
                f"{html.escape(body[:340])}{more}</div></div>"
            )
        return (f"<div style='font-size:12px;opacity:.6;margin-bottom:8px'>"
                f"{len(data['passages'])} passage(s) retrieved</div>"
                + "".join(rows))

    # A list of matches.
    if isinstance(data, dict) and isinstance(data.get("matches"), list):
        rows = "".join(
            f"<tr><td style='padding:3px 12px 3px 0;font-family:{MONO};"
            f"font-size:13px'>{html.escape(str(m.get('parcel_id', '')))}</td>"
            f"<td style='padding:3px 12px 3px 0;font-size:13.5px'>"
            f"{html.escape(str(m.get('situs_address', '')))}</td>"
            f"<td style='padding:3px 0;font-size:13.5px'>"
            f"{html.escape(str(m.get('owner_name', '')))}</td></tr>"
            for m in data["matches"]
        )
        count = data.get("match_count", len(data["matches"]))
        return (f"<div style='font-size:12px;opacity:.6;margin-bottom:6px'>"
                f"{count} match(es)</div>"
                f"<table style='border-collapse:collapse'>{rows}</table>")

    # A flat record -- a small two-column table.
    if isinstance(data, dict):
        rows = "".join(
            f"<tr><td style='padding:2px 14px 2px 0;font-size:12.5px;opacity:.6;"
            f"white-space:nowrap'>{html.escape(str(k).replace('_', ' '))}</td>"
            f"<td style='padding:2px 0;font-family:{MONO};font-size:13.5px'>"
            f"{html.escape(str(v))}</td></tr>"
            for k, v in data.items() if not isinstance(v, (dict, list))
        )
        return f"<table style='border-collapse:collapse'>{rows}</table>"

    return (f"<div style='font-family:{MONO};font-size:13px;white-space:pre-wrap'>"
            f"{html.escape(str(data))}</div>")


def _step_html(index: int, kind: str, label: str, body: str,
               *, last: bool = False) -> str:
    """One step of the loop: a numbered badge, a rule, and a card."""
    color = STEP_COLORS.get(kind, "#475569")
    connector = ""
    if not last:
        connector = ("<div style='flex:1;width:2px;background:#e2e8f0;"
                     "margin-top:4px'></div>")
    pad = "2px" if last else "16px"
    return (
        f"<div style='display:flex;gap:13px;align-items:stretch'>"
        f"<div style='display:flex;flex-direction:column;align-items:center;"
        f"width:30px;flex:none'>"
        f"<div style='width:28px;height:28px;border-radius:50%;background:{color};"
        f"color:#fff;font-size:13px;font-weight:700;display:flex;"
        f"align-items:center;justify-content:center;font-family:{MONO}'>"
        f"{index}</div>{connector}</div>"
        f"<div style='flex:1;min-width:0;padding-bottom:{pad}'>"
        f"<div style='font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;"
        f"color:{color};font-weight:700;margin-bottom:5px'>{html.escape(label)}</div>"
        f"<div style='border:1px solid #e2e8f0;border-left:4px solid {color};"
        f"border-radius:0 7px 7px 0;padding:11px 14px;background:#fff'>{body}</div>"
        f"</div></div>"
    )


def trace_html(trace: Sequence[dict], *, answer: str = "",
               question: str | None = None) -> str:
    """Draw a finished agent run as Reason -> Act -> Observe -> Answer.

    `trace` is what `delorean.agent.ask()` leaves in `state["trace"]`.

    Each tool call becomes two steps rather than one, on purpose. **Act** and
    **Observe** are different events -- the agent asking, and the world
    answering -- and collapsing them into a single box loses the part that
    makes an agent different from a chatbot.
    """
    blocks: list[str] = []
    n = 0

    head = ""
    if question:
        head = (
            f"<div style='font-family:{MONO};font-size:14.5px;background:#f8fafc;"
            f"border:1px dashed #cbd5e1;border-radius:6px;padding:11px 13px;"
            f"margin-bottom:16px'>{html.escape(question)}</div>"
        )

    for entry in trace:
        if entry.get("step") == "reason":
            wants = entry.get("wants") or []
            thought = (entry.get("thought") or "").strip()

            if wants:
                named = html.escape(", ".join(wants))
                if thought:
                    body = (f"<div style='font-size:14.5px;line-height:1.6;"
                            f"white-space:pre-wrap'>{html.escape(thought)}</div>"
                            f"<div style='margin-top:8px;font-size:13.5px;"
                            f"opacity:.7'>Decided to use <b>{named}</b>.</div>")
                else:
                    # Thinking is off, so it went straight for the tool. Say so
                    # rather than showing an empty box.
                    body = (f"<div style='font-size:14.5px;line-height:1.6'>"
                            f"Went straight for <b>{named}</b> &mdash; no "
                            f"deliberation needed.</div>")
                n += 1
                blocks.append(_step_html(n, "reason", "Reason", body))
            elif thought and not answer:
                n += 1
                blocks.append(_step_html(
                    n, "reason", "Reason",
                    f"<div style='font-size:14.5px;line-height:1.6;"
                    f"white-space:pre-wrap'>{html.escape(thought)}</div>"))

        elif entry.get("step") == "act":
            n += 1
            blocks.append(_step_html(
                n, "act", "Act",
                f"<div style='font-family:{MONO};font-size:14px;line-height:1.6'>"
                f"{_call_signature(entry.get('tool', ''), entry.get('args', {}))}"
                f"</div>"))

            result = entry.get("result", {})
            good = isinstance(result, dict) and result.get("ok", True)
            n += 1
            blocks.append(_step_html(n, "observe" if good else "error",
                                     "Observe", _observe_html(result)))

    if answer:
        n += 1
        blocks.append(_step_html(
            n, "answer", "Answer",
            f"<div style='font-size:15.5px;line-height:1.65;white-space:pre-wrap'>"
            f"{html.escape(answer)}</div>", last=True))

    if not blocks:
        return f"<div style='{CARD}'><i>Nothing happened yet.</i></div>"

    return f"<div style='{CARD}'>{head}{''.join(blocks)}</div>"


def show_toolbelt(names: Sequence[str], *, active: str | None = None) -> None:
    """Display the agent's tool list in a notebook."""
    show_html(toolbelt_html(names, active=active))


def show_raw_call(text: str) -> None:
    """Display the model's literal output in a notebook."""
    show_html(raw_call_html(text))


def show_trace(trace: Sequence[dict], *, answer: str = "",
               question: str | None = None) -> None:
    """Display a finished agent run in a notebook."""
    show_html(trace_html(trace, answer=answer, question=question))
