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
# ⚠️ DOUBLE quotes around "Courier New", and they must stay double.
#
# Every style attribute in this file is delimited with SINGLE quotes
# (`style='...'`), so a single quote inside this string ends the attribute
# early and the browser throws away everything after it:
#
#     style='…font-family:…Consolas,'      <- attribute ends HERE
#     Courier New',monospace;color:#34e05a;…   <- silently discarded
#
# That is not hypothetical. It shipped, and it quietly dropped `color`,
# `font-size`, `font-weight`, `text-shadow`, padding and borders from **12 of
# this file's builders** — anything written after `font-family:` in the same
# attribute. The DeLorean readouts rendered in default grey for weeks because
# of it, while the row labels (whose font stack has no quotes) were fine.
MONO = 'ui-monospace,SFMono-Regular,Consolas,"Courier New",monospace'

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
#  Section 4 — the time circuits
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
        # The label wears its own row's colour, which is both the DeLorean's
        # own livery -- red destination, green present, amber last departed --
        # and a straight legibility win. It used to be #7b7b7b, a mid grey on
        # near-black: about 3.5:1, under the 4.5:1 needed for small text. Every
        # one of the three row colours beats that against #0a0b0d, the red
        # being the tightest at roughly 5.5:1. Bumped 9px -> 10px too, because
        # uppercase at this letter-spacing was the smallest text in the app.
        rows.append(
            f"<div style='display:flex;align-items:center;gap:12px;padding:5px 0'>"
            f"<div style='width:132px;flex:none;text-align:right;font-size:10px;"
            f"letter-spacing:.16em;text-transform:uppercase;color:{color};"
            f"font-weight:600;"
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
#  Section 4 — one persona's answer
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
#  Section 5 — what it said, against what is true
#
#  This is the component the whole of Section 5 rests on. A hallucination is
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
#  Sections 7-8 — watching the agent think
#
#  Section 6 draws Reason -> Act -> Observe on a slide. These functions draw the
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

    This is the demystifying moment of Section 7. "Tool calling" sounds like
    plumbing somebody clever built into the model. It is not. The model wrote
    some text, in a format it was trained to use, and about fifteen lines of
    our Python noticed and acted on it.

    Anyone who sat through Section 2 already knows the punchline: this is still
    next-token prediction. It never stopped being next-token prediction.
    """
    escaped = html.escape(text or "")

    # Light up the tool-call markup so the eye lands on it immediately.
    #
    # Three delimiter styles, because three models in the registry use three.
    # Qwen and Nemotron write <tool_call><function=...>; SmolLM3 and Phi-4-mini
    # write bare JSON with no markup to highlight; Gemma 4 puts the pipe INSIDE
    # the bracket -- <|tool_call>call:name{...}<tool_call|> -- and fences its
    # strings with <|"|>. Matching only the first style left Gemma's call
    # rendered as flat grey text, which is the one thing this panel exists to
    # prevent.
    escaped = re.sub(
        r"(&lt;/?\|?(?:tool_call|tool_response|function|parameter)[^&]*?&gt;"
        r"|&lt;\|&quot;\|&gt;)",
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


# ═════════════════════════════════════════════════════════════════════════
#  Section 9 — the valuation review
# ═════════════════════════════════════════════════════════════════════════
#
# The colours below are measured, not chosen by eye, and one of them is a
# deliberate departure from the palette used elsewhere in this file.
#
# A pass/fail badge in green and red is the obvious design and it is the
# wrong one. Run the repo's existing green (#15803d) against its red
# (#b91c1c) through a deuteranopia simulation and they land 4.2 apart in
# OKLab -- which is to say a red-green colourblind reader, roughly one man in
# twelve in that Calgary audience, cannot tell PASS from FAIL. On the one
# visual Section 9 rests on.
#
# Swapping the green for teal takes that separation to 11.9, comfortably past
# the 8.0 target, and costs nothing an appraiser would notice.
#
# The colour is still only ever the SECOND signal. Every badge carries a tick
# or a cross and the word itself, so the meaning survives a monochrome
# printout, a washed-out projector and any kind of colour vision.

#: Verdict colours. Teal rather than green -- see above.
STATUS_PASS = "#0f766e"
STATUS_FAIL = "#b91c1c"

#: The map's three bands, which are the IAAO bands and not a smooth gradient.
#: A ratio study asks a categorical question -- is this parcel under the
#: standard, inside it, or over it -- so the map answers that question rather
#: than shading a continuum nobody can read off a legend.
#:
#: Two hues around a neutral grey midpoint, which is what a diverging scale
#: is supposed to be. The grey is deliberately low-chroma so the parcels that
#: are FINE recede and the problems come forward. Worst-pair separation under
#: simulated colourblindness is 20.7 -- these three are never confusable.
RATIO_UNDER = "#1d4ed8"    # assessed below the IAAO range -- we are low
RATIO_WITHIN = "#94a3b8"   # inside 0.90 - 1.10, nothing to see
RATIO_OVER = "#c2410c"     # assessed above the IAAO range -- we are high

#: Where the IAAO acceptance range sits, for the dial and the map legend.
IAAO_LOW, IAAO_HIGH = 0.90, 1.10


def ratio_band(value: float | None) -> tuple[str, str]:
    """Which of the three bands a ratio falls in: (colour, plain English)."""
    if value is None:
        return "#cbd5e1", "no sale on file"
    if value < IAAO_LOW:
        return RATIO_UNDER, "assessed below the IAAO range"
    if value > IAAO_HIGH:
        return RATIO_OVER, "assessed above the IAAO range"
    return RATIO_WITHIN, "within the IAAO range"


def _verdict_style(verdict: str) -> tuple[str, str]:
    """(colour, glyph) for a verdict. The glyph is not decoration."""
    if verdict == "PASS":
        return STATUS_PASS, "&#10003;"      # tick
    if verdict == "FAIL":
        return STATUS_FAIL, "&#10007;"      # cross
    return "#64748b", "&#8212;"             # em dash -- not judged


def verdict_badge_html(verdict: str, *, small: bool = False) -> str:
    """A verdict as colour AND glyph AND word. Never fewer than three."""
    colour, glyph = _verdict_style(verdict)
    pad = "1px 7px" if small else "3px 11px"
    size = "11px" if small else "13px"
    return (
        f"<span style='display:inline-block; background:{colour}; "
        f"color:#ffffff; border-radius:999px; padding:{pad}; "
        f"font-size:{size}; font-weight:700; letter-spacing:.03em; "
        f"white-space:nowrap;'>{glyph}&nbsp;{html.escape(verdict)}</span>")


# ─────────────────────────────────────────────────────────────────────────
#  The speedometer
# ─────────────────────────────────────────────────────────────────────────
#
# One dial, showing one number: the median assessment ratio.
#
# It is a real instrument, not a mascot. An appraiser reads a median ratio
# the way a driver reads a speedometer -- what matters is not the digits but
# whether the needle is inside the safe band, and you want to know that
# without doing any arithmetic. The IAAO acceptance range is painted onto the
# dial, so "are we alright?" is answered by where the needle points.
#
# The theme earns its keep here rather than decorating: a perfect ratio of
# 1.00 is marked 88, because 1.00 is exactly where you want to be.
#
# Only the median goes on the dial. COD and PRD are equally important and
# measure completely different things, and putting three needles on one arc
# would imply they share a scale. They get the scorecard instead.

_DIAL_LOW, _DIAL_HIGH = 0.70, 1.30


def _polar(cx: float, cy: float, r: float, degrees: float) -> tuple[float, float]:
    """A point on a circle. 0 degrees is due east, 180 due west."""
    import math

    angle = math.radians(degrees)
    return cx + r * math.cos(angle), cy - r * math.sin(angle)


def _arc_path(cx: float, cy: float, r: float, deg_from: float,
              deg_to: float) -> str:
    """An SVG arc between two angles on a circle."""
    x1, y1 = _polar(cx, cy, r, deg_from)
    x2, y2 = _polar(cx, cy, r, deg_to)
    large = 1 if abs(deg_to - deg_from) > 180 else 0
    sweep = 1 if deg_to < deg_from else 0
    return (f"M {x1:.2f} {y1:.2f} A {r:.2f} {r:.2f} 0 {large} {sweep} "
            f"{x2:.2f} {y2:.2f}")


def _dial_angle(value: float) -> float:
    """Where on the dial a ratio sits. Values off the scale peg at the end."""
    clamped = min(max(value, _DIAL_LOW), _DIAL_HIGH)
    span = _DIAL_HIGH - _DIAL_LOW
    return 180.0 - (clamped - _DIAL_LOW) / span * 180.0


#: How wide the dial is allowed to get. Fixed rather than fluid because
#: Streamlit has to be told the exact height of the frame it goes in -- see
#: `svg_card()` in the Section 9 app.
SPEEDOMETER_MAX_WIDTH = 430

# Wide enough for the dial, tall enough for the reading to sit UNDERNEATH it.
# The first version put the number in the middle of the face, which is where
# every real speedometer puts it -- and on a real speedometer the needle is
# not free to sweep through the digits. Here it is, and at any ratio near 1.00
# it drew straight over them.
SPEEDOMETER_VIEWBOX = (460, 312)


def speedometer_frame_height(max_width: int = SPEEDOMETER_MAX_WIDTH) -> int:
    """How tall a rendered dial card is, in pixels.

    Streamlit has to be told the height of the frame these go in, and getting
    it wrong is not cosmetic: too small and the caption is behind a scrollbar,
    which is how this was first shipped.

    So it is computed rather than typed in. The svg keeps its viewBox aspect
    once the width is capped, and the rest is the card around it -- padding,
    the caption, the subtitle line, borders and margins. Space for the
    subtitle is always reserved even when there is not one, so the header does
    not jump every time a question is asked.
    """
    box_w, box_h = SPEEDOMETER_VIEWBOX
    dial = max_width * box_h / box_w
    chrome = 36 + 20 + 20 + 12 + 2      # padding, caption, subtitle, margin, border
    return int(dial + chrome) + 12      # a little headroom for font differences


def speedometer_html(value: float | None = None, *,
                     caption: str = "Median assessment ratio",
                     subtitle: str = "") -> str:
    """The header instrument: one needle against the IAAO acceptance range.

    Pass `None` before anything has been asked, and the dial sits dark with
    the needle at rest -- an instrument waiting, rather than an empty box.
    """
    # The hub sits high enough to leave the bottom third of the box free for
    # the reading. Nothing the needle can reach goes below y = cy.
    cx, cy, r = 230.0, 208.0, 148.0
    live = value is not None

    # The face
    parts = [
        f"<path d='{_arc_path(cx, cy, r, 180, 0)}' fill='none' "
        f"stroke='#e2e8f0' stroke-width='26' stroke-linecap='round'/>",
        # The IAAO acceptance range, painted onto the dial.
        f"<path d='{_arc_path(cx, cy, r, _dial_angle(IAAO_LOW), _dial_angle(IAAO_HIGH))}' "
        f"fill='none' stroke='{STATUS_PASS}' stroke-width='26' "
        f"opacity='{0.85 if live else 0.35}'/>",
    ]

    # Ticks and their labels.
    for tick in (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30):
        angle = _dial_angle(tick)
        inner = _polar(cx, cy, r - 17, angle)
        outer = _polar(cx, cy, r + 15, angle)
        edge = tick in (IAAO_LOW, IAAO_HIGH)
        parts.append(
            f"<line x1='{inner[0]:.1f}' y1='{inner[1]:.1f}' "
            f"x2='{outer[0]:.1f}' y2='{outer[1]:.1f}' "
            f"stroke='{'#334155' if edge else '#94a3b8'}' "
            f"stroke-width='{2.5 if edge else 1.5}'/>")
        label = _polar(cx, cy, r + 31, angle)

        # 1.00 is 88 miles per hour -- a ratio of exactly one is the reading
        # you are aiming for, so the dial is marked there rather than putting
        # the joke somewhere it has to be explained. Everything inside the arc
        # belongs to the needle; anything parked in there gets driven over.
        if tick == 1.00:
            parts.append(
                f"<text x='{label[0]:.1f}' y='{label[1]:.1f}' "
                f"text-anchor='middle' dominant-baseline='middle' "
                f"font-size='21' font-weight='800' font-style='italic' "
                f"fill='{STATUS_PASS}' "
                f"opacity='{0.95 if live else 0.32}'>88</text>")
            continue

        parts.append(
            f"<text x='{label[0]:.1f}' y='{label[1]:.1f}' text-anchor='middle' "
            f"dominant-baseline='middle' font-size='13' "
            f"fill='{'#334155' if edge else '#94a3b8'}' "
            f"font-weight='{700 if edge else 500}'>{tick:.2f}</text>")

    # The needle.
    if live:
        angle = _dial_angle(float(value))
        tip = _polar(cx, cy, r - 30, angle)
        left = _polar(cx, cy, 11, angle + 90)
        right = _polar(cx, cy, 11, angle - 90)
        colour, _ = ratio_band(float(value))
        parts.append(
            f"<polygon points='{tip[0]:.1f},{tip[1]:.1f} "
            f"{left[0]:.1f},{left[1]:.1f} {right[0]:.1f},{right[1]:.1f}' "
            f"fill='#0f172a'/>")
        parts.append(f"<circle cx='{cx}' cy='{cy}' r='13' fill='#0f172a'/>")
        parts.append(f"<circle cx='{cx}' cy='{cy}' r='5.5' fill='{colour}'/>")
    else:
        parts.append(f"<circle cx='{cx}' cy='{cy}' r='13' fill='#cbd5e1'/>")

    # The number itself, which is what anybody actually reads. It lives INSIDE
    # the svg so it scales with the dial -- positioned with a pixel margin it
    # drifted off the face the moment the container changed width.
    # Below the dial, clear of everything the needle can reach.
    if live:
        colour, meaning = ratio_band(float(value))
        parts.append(
            f"<text x='{cx}' y='262' text-anchor='middle' "
            f"dominant-baseline='middle' font-size='48' font-weight='800' "
            f"fill='#0f172a' letter-spacing='-1'>{float(value):.3f}</text>")
        parts.append(
            f"<text x='{cx}' y='294' text-anchor='middle' font-size='13' "
            f"font-weight='700' fill='{colour}'>{html.escape(meaning)}</text>")
    else:
        parts.append(
            f"<text x='{cx}' y='268' text-anchor='middle' "
            f"dominant-baseline='middle' font-size='21' font-weight='600' "
            f"fill='#94a3b8'>waiting for a question</text>")

    reading_label = "no reading yet" if not live else f"{float(value):.3f}"
    body = (
        f"<div style='text-align:center;'>"
        f"<svg viewBox='0 0 {SPEEDOMETER_VIEWBOX[0]} {SPEEDOMETER_VIEWBOX[1]}' "
        f"width='100%' style='max-width:{SPEEDOMETER_MAX_WIDTH}px;' "
        f"role='img' aria-label='{html.escape(caption)}: {reading_label}'>"
        f"{''.join(parts)}</svg>"
        f"<div style='font-size:12px; color:#64748b; margin-top:4px; "
        f"letter-spacing:.06em; text-transform:uppercase;'>"
        f"{html.escape(caption)}</div>"
        + (f"<div style='font-size:12.5px; color:#475569; margin-top:3px;'>"
           f"{html.escape(subtitle)}</div>" if subtitle else "")
        + "</div>")

    return f"<div style='{CARD}'>{body}</div>"


def show_speedometer(value: float | None = None, **kwargs) -> None:
    """Display the dial in a notebook."""
    show_html(speedometer_html(value, **kwargs))


# ─────────────────────────────────────────────────────────────────────────
#  The scorecard — one study, three tests
# ─────────────────────────────────────────────────────────────────────────

def scorecard_html(result: dict, *, title: str = "IAAO ratio study") -> str:
    """The three IAAO tests as three cards, each with its own verdict.

    Three separate cards rather than one table, because the three tests are
    genuinely independent and the layout should say so. Level, uniformity and
    equity can pass and fail in any combination, and an office that reads them
    as a single score has thrown away the only useful thing about them: WHICH
    one failed tells you what to go and fix.
    """
    tests = result.get("tests") or []
    if not tests:
        return note_html(
            "Not enough sales",
            html.escape(str(result.get("note", "Nothing to report."))),
            kind="warn")

    cards = []
    for test in tests:
        colour, glyph = _verdict_style(test["verdict"])
        failed = test["verdict"] == "FAIL"
        cards.append(
            f"<div style='flex:1 1 190px; min-width:190px; "
            f"border:1px solid {colour if failed else '#e2e8f0'}; "
            f"border-top:4px solid {colour}; border-radius:9px; "
            f"padding:14px 16px; background:{'#fffafa' if failed else '#ffffff'};'>"
            f"<div style='font-size:11px; letter-spacing:.07em; "
            f"text-transform:uppercase; color:#64748b; font-weight:700;'>"
            f"{html.escape(test['test'])}</div>"
            f"<div style='font-size:13px; color:#475569; margin:2px 0 9px;'>"
            f"{html.escape(test['metric'])}</div>"
            f"<div style='font-size:33px; font-weight:800; color:#0f172a; "
            f"line-height:1;'>{test['value']}</div>"
            f"<div style='font-size:11.5px; color:#64748b; margin:7px 0 10px;'>"
            f"standard {html.escape(test['standard'])}</div>"
            f"{verdict_badge_html(test['verdict'], small=True)}"
            + (f"<div style='font-size:11.5px; color:{colour}; margin-top:9px; "
               f"font-weight:600; line-height:1.35;'>"
               f"{html.escape(test.get('meaning', ''))}</div>" if failed else "")
            + f"<div style='font-size:11px; color:#94a3b8; margin-top:9px; "
              f"line-height:1.35;'>{html.escape(test.get('asks', ''))}</div>"
            f"</div>")

    overall = result.get("verdict", "")
    used = result.get("sales_used", 0)
    excluded = result.get("sales_excluded_not_arms_length", 0)

    head = (
        f"<div style='display:flex; align-items:center; gap:12px; "
        f"flex-wrap:wrap; margin-bottom:14px;'>"
        f"<span style='font-weight:800; font-size:16px; color:#0f172a;'>"
        f"{html.escape(title)}</span>"
        f"{verdict_badge_html(overall)}"
        f"<span style='font-size:12.5px; color:#64748b;'>"
        f"{used} qualified sales</span></div>")

    # The exclusion line is not housekeeping. It is the tool applying a
    # professional standard the model never had to know about, and saying so.
    footer = ""
    if excluded:
        footer = (
            f"<div style='margin-top:14px; padding:10px 13px; "
            f"background:#f8fafc; border-left:3px solid #94a3b8; "
            f"border-radius:0 6px 6px 0; font-size:12.5px; color:#475569; "
            f"line-height:1.45;'><b>{excluded} sales excluded</b> as not "
            f"arm's length &mdash; foreclosures and transfers between family "
            f"members are not evidence of market value, so IAAO leaves them "
            f"out.</div>")

    if result.get("stratification_note"):
        footer += (
            f"<div style='margin-top:8px; padding:10px 13px; "
            f"background:#fffbeb; border-left:3px solid #f59e0b; "
            f"border-radius:0 6px 6px 0; font-size:12.5px; color:#78350f; "
            f"line-height:1.45;'>"
            f"{html.escape(result['stratification_note'])}</div>")

    return (f"<div style='{CARD}'>{head}"
            f"<div style='display:flex; gap:11px; flex-wrap:wrap;'>"
            f"{''.join(cards)}</div>{footer}</div>")


def show_scorecard(result: dict, **kwargs) -> None:
    """Display an IAAO scorecard in a notebook."""
    show_html(scorecard_html(result, **kwargs))


# ─────────────────────────────────────────────────────────────────────────
#  The neighborhood table — Section 9's whole argument, in one picture
# ─────────────────────────────────────────────────────────────────────────

def ratio_table_html(groups: Sequence[dict], *, group_by: str = "neighborhood",
                     title: str = "") -> str:
    """One ratio study per group, with the FAILING CELL marked.

    The design decision here is small and it is the entire point.

    The obvious thing is to shade the failing ROW red: three red rows, three
    problems, job done. It is also the wrong thing, because it throws away the
    finding. Hill Valley County's three troubled neighborhoods do not share a
    problem -- one is assessed too low, one is inconsistent, one is unfair
    between cheap and expensive homes. Three different afternoons of work.

    Mark the failing CELL instead and the red lands in three DIFFERENT columns.
    Nobody has to say "and notice they fail for different reasons" -- the table
    has already said it, and an appraiser reads it in about two seconds.

    The county line goes at the bottom, passing everything, which is the twist:
    the aggregate is healthy while half the neighborhoods are not.
    """
    groups = list(groups)
    if not groups:
        return note_html("No results", "Nothing to show.", kind="warn")

    metrics = [m for m in ("Median ratio", "COD", "PRD") if m in groups[0]]

    header = (
        f"<th style='text-align:left; padding:9px 12px; font-size:11px; "
        f"letter-spacing:.07em; text-transform:uppercase; color:#64748b;'>"
        f"{html.escape(group_by)}</th>"
        f"<th style='text-align:right; padding:9px 8px; font-size:11px; "
        f"letter-spacing:.07em; text-transform:uppercase; color:#64748b;'>"
        f"sales</th>")
    for metric in metrics:
        header += (
            f"<th style='text-align:right; padding:9px 12px; font-size:11px; "
            f"letter-spacing:.07em; text-transform:uppercase; color:#64748b; "
            f"white-space:nowrap;'>{html.escape(metric)}</th>")
    header += (
        "<th style='text-align:center; padding:9px 12px; font-size:11px; "
        "letter-spacing:.07em; text-transform:uppercase; color:#64748b;'>"
        "verdict</th>")

    rows = []
    for group in groups:
        cells = (
            f"<td style='padding:11px 12px; font-weight:700; font-size:14px; "
            f"color:#0f172a; white-space:nowrap;'>"
            f"{html.escape(str(group.get(group_by, '')))}</td>"
            f"<td style='padding:11px 8px; text-align:right; font-size:13px; "
            f"color:#64748b; font-family:{MONO};'>"
            f"{group.get('sales_used', '')}</td>")

        for metric in metrics:
            failed = group.get(metric + " verdict") == "FAIL"
            value = group.get(metric)
            shown = f"{value:.3f}" if metric != "COD" else f"{value:.2f}"
            if failed:
                cells += (
                    f"<td style='padding:11px 12px; text-align:right; "
                    f"font-family:{MONO}; font-size:14.5px; font-weight:800; "
                    f"color:#ffffff; background:{STATUS_FAIL};' "
                    f"title='fails the IAAO standard for {html.escape(metric)}'>"
                    f"&#10007;&nbsp;{shown}</td>")
            else:
                cells += (
                    f"<td style='padding:11px 12px; text-align:right; "
                    f"font-family:{MONO}; font-size:14px; color:#334155;'>"
                    f"{shown}</td>")

        cells += (f"<td style='padding:11px 12px; text-align:center;'>"
                  f"{verdict_badge_html(group.get('verdict', ''), small=True)}"
                  f"</td>")
        rows.append(f"<tr style='border-top:1px solid #e2e8f0;'>{cells}</tr>")

    failing = [g for g in groups if g.get("verdict") == "FAIL"]
    head = ""
    if title:
        head = (f"<div style='font-weight:800; font-size:16px; color:#0f172a; "
                f"margin-bottom:4px;'>{html.escape(title)}</div>")
    head += (
        f"<div style='font-size:13px; color:#475569; margin-bottom:13px;'>"
        f"<b>{len(failing)} of {len(groups)}</b> fail at least one IAAO "
        f"standard &mdash; and they do not fail the same one.</div>")

    legend = (
        f"<div style='margin-top:12px; font-size:11.5px; color:#64748b; "
        f"display:flex; gap:16px; flex-wrap:wrap; align-items:center;'>"
        f"<span><span style='display:inline-block; width:11px; height:11px; "
        f"background:{STATUS_FAIL}; border-radius:2px; "
        f"vertical-align:-1px;'></span> &#10007; outside the IAAO standard"
        f"</span>"
        f"<span>Level 0.90&ndash;1.10 &nbsp;&middot;&nbsp; COD 5&ndash;15 "
        f"&nbsp;&middot;&nbsp; PRD 0.98&ndash;1.03</span></div>")

    return (f"<div style='{CARD}'>{head}"
            f"<div style='overflow-x:auto;'>"
            f"<table style='width:100%; border-collapse:collapse; "
            f"min-width:520px;'>"
            f"<thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody>"
            f"</table></div>{legend}</div>")


def show_ratio_table(groups: Sequence[dict], **kwargs) -> None:
    """Display the grouped ratio study in a notebook."""
    show_html(ratio_table_html(groups, **kwargs))


# ─────────────────────────────────────────────────────────────────────────
#  The county map — drawn by hand, on purpose
# ─────────────────────────────────────────────────────────────────────────
#
# This started as folium, and folium turned out to be the wrong tool here for
# a reason worth writing down, because it will catch somebody else.
#
# A folium map pulls in ten external resources from CDNs -- and not just the
# background tiles. Leaflet ITSELF is one of them. On conference wifi that has
# given up you do not get a map missing its streets; you get no map at all. It
# would also mean the committed notebook outputs could not draw on GitHub,
# which is most of the reason we commit them.
#
# So the map is inline SVG, like everything else in this file. No network, no
# dependency, and it renders identically in JupyterLab, in Streamlit and on
# GitHub. What that costs is pan and zoom, which nobody wants in the middle of
# a fifteen-minute slot anyway.
#
# There is a second reason and it is the better one. These parcels are
# invented. Drawing invented parcels on top of real Calgary streets quietly
# claims they are real properties at real addresses, which is precisely what
# this repository promises not to do. An abstract map reads as Hill Valley
# County, which is what it is.


def _convex_hull(points):
    """The outline enclosing a set of points -- Andrew's monotone chain.

    Fifteen lines and no scipy. Used to sketch each neighborhood's boundary so
    the county has a shape instead of being a cloud of loose dots.
    """
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


#: Same reason as the dial: a host that has to reserve a frame needs to know
#: how tall the picture will be, and that is only knowable if the width stops
#: somewhere.
MAP_MAX_WIDTH = 780


def map_frame_height(markup: str, max_width: int = MAP_MAX_WIDTH) -> int:
    """How tall a rendered map card will be, in pixels.

    The svg keeps its viewBox aspect, so once the width is capped the height
    follows. The constant on the end is the card itself -- title, count line,
    legend, padding.
    """
    found = re.search(r"viewBox='0 0 (\d+) (\d+)'", markup)
    if not found:
        return 700
    box_w, box_h = int(found.group(1)), int(found.group(2))
    # Title, the count line (which wraps to two lines on a narrow window),
    # the legend, card padding, margins and borders. Generous on purpose:
    # a frame that is slightly too tall costs a little whitespace, one that
    # is slightly too short puts the legend behind a scrollbar.
    chrome = 26 + 40 + 30 + 36 + 12 + 2
    return int(max_width * box_h / box_w) + chrome


def parcel_map_html(frame, *, color_by: str = "ratio", context=None,
                    title: str = "", width: int = 720,
                    height: int | None = None) -> str:
    """Draw the selected parcels on a map, zoomed to fit them.

    `frame` is what the agent asked to see. `context` is every parcel in the
    county; whatever part of it falls inside the frame is drawn faintly
    underneath, so a selection sits among its neighbours rather than floating.

    **The frame fits the SELECTION, not the county.** That is a deliberate
    reversal of how this first worked, and the reason is worth keeping.

    The first version drew the whole county every time and outlined each
    neighborhood with a convex hull. Because parcels are sampled inside a
    disc, those hulls came out as six near-perfect circles -- which is not
    what a county looks like to anybody who values property for a living. The
    outline was doing the damage: a scatter of dots reads as a cluster, but a
    boundary drawn around it reads as a CLAIM about a shape, and that claim
    was false.

    So there are no neighborhood outlines. Asked to map the sales in Riverside
    Heights, this shows the sales in Riverside Heights, filling the frame,
    which is both what was asked for and what an appraiser expects to get.
    Nothing is asserted about where the neighborhood ends, because nothing
    needs to be.

    What replaces the outlines is a plain grid, a scale bar and a north arrow
    -- the grid says "coordinate space" without inventing streets that do not
    exist, and the scale bar is not decoration: once the frame fits whatever
    was selected, 36 sales across 800 metres and 244 across five kilometres
    fill it identically. Without a scale they are indistinguishable.
    """
    import math

    if frame is None or len(frame) == 0:
        return note_html("Nothing to map",
                         "That selection had no parcels with coordinates "
                         "on file.", kind="warn")

    pad = 52

    mid_lat = sum(frame["latitude"]) / len(frame)
    # Longitude degrees get narrower the further north you go. Without this
    # correction the map comes out stretched sideways.
    squeeze = math.cos(math.radians(mid_lat))

    xs = [lon * squeeze for lon in frame["longitude"]]
    ys = [-lat for lat in frame["latitude"]]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)

    # A single parcel, or a row of them along one street, would otherwise ask
    # for infinite magnification. Never show less than about 250 metres.
    floor = 0.0022
    if x_hi - x_lo < floor:
        middle = (x_hi + x_lo) / 2
        x_lo, x_hi = middle - floor / 2, middle + floor / 2
    if y_hi - y_lo < floor:
        middle = (y_hi + y_lo) / 2
        y_lo, y_hi = middle - floor / 2, middle + floor / 2

    # A little air around the edge, so nothing sits on the border.
    breathe = 0.08
    x_lo -= (x_hi - x_lo) * breathe
    x_hi += (x_hi - x_lo) * breathe
    y_lo -= (y_hi - y_lo) * breathe
    y_hi += (y_hi - y_lo) * breathe

    span_x = (x_hi - x_lo) or 1e-6
    span_y = (y_hi - y_lo) or 1e-6

    # Shape the canvas to what is being shown, within reason.
    if height is None:
        ideal = (width - 2 * pad) * (span_y / span_x) + 2 * pad
        height = int(min(max(ideal, 340), 620))

    # One scale for both axes, so nothing is distorted.
    scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y)
    off_x = (width - span_x * scale) / 2
    off_y = (height - span_y * scale) / 2

    def project(lat, lon):
        return (off_x + (lon * squeeze - x_lo) * scale,
                off_y + (-lat - y_lo) * scale)

    layers = []

    # -- A plain grid. Not streets; we are not inventing streets. ------------
    layers.append(f"<rect x='0' y='0' width='{width}' height='{height}' "
                  f"fill='#fbfcfd'/>")
    for gx in range(0, width + 1, 64):
        layers.append(f"<line x1='{gx}' y1='0' x2='{gx}' y2='{height}' "
                      f"stroke='#eef2f6' stroke-width='1'/>")
    for gy in range(0, height + 1, 64):
        layers.append(f"<line x1='0' y1='{gy}' x2='{width}' y2='{gy}' "
                      f"stroke='#eef2f6' stroke-width='1'/>")

    # -- Neighbouring parcels that happen to be in shot ----------------------
    visible_hoods: set[str] = set()
    if context is not None and len(context):
        chosen = set(frame["parcel_id"]) if "parcel_id" in frame else set()
        for pid, la, lo, hood in zip(context["parcel_id"], context["latitude"],
                                     context["longitude"],
                                     context["neighborhood"]):
            if pid in chosen:
                continue
            x, y = project(la, lo)
            if not (0 <= x <= width and 0 <= y <= height):
                continue        # outside the frame; clipped rather than drawn
            visible_hoods.add(str(hood))
            layers.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='1.6' "
                          f"fill='#cbd5e1'/>")

    # -- The selection, on top ----------------------------------------------
    values = None
    if color_by == "assessed_value" and "assessed_value" in frame:
        values = [v for v in frame["assessed_value"] if v is not None]
    lo_v = min(values) if values else 0
    hi_v = max(values) if values else 1

    for row in frame.to_dict(orient="records"):
        x, y = project(row["latitude"], row["longitude"])

        if color_by == "assessed_value":
            # Sequential: one hue, light to dark. Never a rainbow.
            value = row.get("assessed_value") or lo_v
            t = (value - lo_v) / ((hi_v - lo_v) or 1)
            fill = (f"rgb({int(219 - 189 * t)},{int(234 - 156 * t)},"
                    f"{int(254 - 138 * t)})")
            meaning = f"assessed ${value:,.0f}"
        else:
            fill, meaning = ratio_band(row.get("ratio"))

        address = row.get("situs_address")
        tip = str(row.get("parcel_id", ""))
        if address:
            tip += f" - {address}"
        tip += f"\n{row.get('neighborhood', '')}"
        tip += f"\nassessed ${row.get('assessed_value') or 0:,.0f}"
        if row.get("sale_price"):
            tip += f"\nsold ${row['sale_price']:,.0f}"
        if row.get("ratio") is not None:
            tip += f"\nratio {row['ratio']:.3f} - {meaning}"

        layers.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5.2' fill='{fill}' "
            f"stroke='#ffffff' stroke-width='1.6'/>")

        # A second, invisible circle twice the size, carrying the tooltip.
        # The visible dot is about ten pixels across, which is a fiddly thing
        # to keep a pointer on -- and hovering a ten-pixel target is not a
        # thing anybody is going to do from the back of a conference room. The
        # bigger hit area costs nothing and makes it work when someone does
        # try it on their own laptop later.
        layers.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='11' fill='transparent'>"
            f"<title>{html.escape(tip)}</title></circle>")

    # -- Scale bar. Not decoration: see the docstring. -----------------------
    #
    # `scale` is pixels per degree of latitude, and a degree of latitude is
    # 111.32 km anywhere on earth. Pick the largest tidy distance that fits in
    # a third of the frame, so the bar is always a round number somebody can
    # actually reason with rather than "0.37 km".
    km_per_px = 111.32 / scale
    for nice_km in (10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05):
        bar_px = nice_km / km_per_px
        if bar_px <= (width - 2 * pad) / 3:
            break
    bar_label = (f"{nice_km:g} km" if nice_km >= 1
                 else f"{int(nice_km * 1000)} m")
    bar_x, bar_y = pad, height - 26
    layers.append(
        f"<line x1='{bar_x}' y1='{bar_y}' x2='{bar_x + bar_px:.1f}' "
        f"y2='{bar_y}' stroke='#475569' stroke-width='2.5'/>")
    for end in (bar_x, bar_x + bar_px):
        layers.append(f"<line x1='{end:.1f}' y1='{bar_y - 4}' "
                      f"x2='{end:.1f}' y2='{bar_y + 4}' "
                      f"stroke='#475569' stroke-width='2.5'/>")
    layers.append(
        f"<text x='{bar_x + bar_px / 2:.1f}' y='{bar_y - 9}' "
        f"text-anchor='middle' font-size='11.5' font-weight='700' "
        f"fill='#475569'>{bar_label}</text>")

    # -- North arrow. Cheap, and it reads as "map" instantly. ----------------
    n_x, n_y = width - pad + 6, height - 30
    layers.append(
        f"<polygon points='{n_x},{n_y - 20} {n_x - 6},{n_y - 2} "
        f"{n_x},{n_y - 7} {n_x + 6},{n_y - 2}' fill='#475569'/>")
    layers.append(
        f"<text x='{n_x}' y='{n_y + 12}' text-anchor='middle' font-size='11' "
        f"font-weight='700' fill='#475569'>N</text>")

    # -- Legend. The grey band is low contrast on purpose, so it is labelled --
    if color_by == "assessed_value":
        keys = [("#dbeafe", f"${lo_v:,.0f}"), ("#1e40af", f"${hi_v:,.0f}")]
    else:
        keys = [(RATIO_UNDER, "below 0.90 - assessed low"),
                (RATIO_WITHIN, "0.90 to 1.10 - within standard"),
                (RATIO_OVER, "above 1.10 - assessed high")]

    legend = "".join(
        f"<span style='display:inline-flex; align-items:center; gap:6px;'>"
        f"<span style='width:11px; height:11px; border-radius:50%; "
        f"background:{colour}; border:1.5px solid #ffffff; "
        f"box-shadow:0 0 0 1px #cbd5e1;'></span>{html.escape(label)}</span>"
        for colour, label in keys)

    head = ""
    if title:
        head = (f"<div style='font-weight:800; font-size:16px; color:#0f172a; "
                f"margin-bottom:3px;'>{html.escape(title)}</div>")

    # Deliberately does NOT say "hover for details". Every dot does carry a
    # tooltip, but a caption is a promise, and a promise that only pays out
    # when you land a pointer on a small circle is not one worth making to a
    # room. The colours, the legend and the scale bar always work.
    hoods_shown = sorted(frame["neighborhood"].dropna().unique().tolist())
    where = (hoods_shown[0] if len(hoods_shown) == 1
             else f"{len(hoods_shown)} neighborhoods")
    nearby = ""
    if visible_hoods:
        nearby = " Grey dots are other parcels in the same area."
    head += (f"<div style='font-size:12.5px; color:#64748b; "
             f"margin-bottom:10px;'>{len(frame)} parcels in "
             f"{html.escape(where)}, coloured by how their assessment "
             f"compares with what they sold for.{nearby}</div>")

    return (
        f"<div style='{CARD}'>{head}"
        f"<svg viewBox='0 0 {width} {height}' width='100%' "
        f"style='max-width:{MAP_MAX_WIDTH}px; background:#ffffff; "
        f"border:1px solid #e2e8f0; border-radius:8px;' role='img' "
        f"aria-label='Map of {len(frame)} parcels in Hill Valley County'>"
        f"{''.join(layers)}</svg>"
        f"<div style='margin-top:11px; font-size:12px; color:#475569; "
        f"display:flex; gap:18px; flex-wrap:wrap;'>{legend}</div>"
        f"</div>")


def show_parcel_map(frame, **kwargs) -> None:
    """Display the county map in a notebook."""
    show_html(parcel_map_html(frame, **kwargs))


# ─────────────────────────────────────────────────────────────────────────
#  Section 10 — three offices, three servers
#
#  Deliberately built from divs and text, with NO <svg> anywhere. The map and
#  the speedometer both learned that the hard way: `st.html()` sanitises with
#  DOMPurify configured `USE_PROFILES: {html: true}`, which allows HTML
#  elements and strips every SVG element silently, leaving a card frame with a
#  hole in it. This panel is the centrepiece of Section 10 and has to render
#  in a notebook, on GitHub and in Streamlit without a special case.
#
#  Status is never carried by colour alone. Every panel states its condition
#  in a word and a glyph as well, which is the rule for anyone reading from
#  the back of a room or through a colour vision deficiency.
# ─────────────────────────────────────────────────────────────────────────

#: One panel per office. "Answered" is teal, NOT green -- green against our
#: red measures a CVD colour distance of 4.2, which a deuteranope cannot
#: separate. Teal against the same red measures 11.9.
_OFFICE_STATES = {
    "idle":     ("#f8fafc", "#e2e8f0", "#64748b", "\u25cb", "not asked"),
    "answered": ("#f0fdfa", STATUS_PASS, STATUS_PASS, "\u2713", "answered"),
    "failed":   ("#fef2f2", STATUS_FAIL, STATUS_FAIL, "\u2715", "did not answer"),
}


def _office_chip(tool: str, used: bool, ink: str) -> str:
    """One tool name inside an office panel."""
    return (f"<div style='font-family:{MONO};font-size:11.5px;"
            f"color:{ink if used else '#94a3b8'};"
            f"font-weight:{'700' if used else '400'};margin-top:3px'>"
            f"{'\u25b8' if used else '\u00b7'} {html.escape(tool)}</div>")


def offices_html(catalogue: Sequence[dict], *,
                 trace: Sequence[dict] | None = None,
                 local: Sequence[str] | None = None,
                 title: str = "The county offices this agent can reach") -> str:
    """Draw one panel per office, lit up by what the agent actually called.

    Section 10's argument is not that MCP is a clever protocol. It is that a
    question an ordinary taxpayer asks at a counter -- *may I split my lot?* --
    cannot be answered by any single office in the county, and that the only
    thing joining them today is somebody making three phone calls.

    So the picture is three buildings, not a list of tools. Watching the
    Appraiser, then the Clerk, then the Tax Collector each answer in turn is
    the demo; the protocol underneath is an implementation detail the room
    does not need.

    Args:
        catalogue: rows from `Offices.catalogue()` -- office, icon, tool.
                   Every one of them was published by the server that owns it,
                   not written here.
        trace:     the agent's trace so far, so panels light as calls land.
                   Pass a partial trace while streaming and it animates.
        local:     tools belonging to THIS office that did not come over MCP.
                   Drawn in a separate dashed panel on purpose -- the
                   difference between "my filing cabinet" and "somebody else's
                   system" is the distinction the whole section rests on.
        title:     heading above the row.
    """
    called: dict[str, bool] = {}
    for step in (trace or []):
        if step.get("step") == "act":
            ok = bool((step.get("result") or {}).get("ok"))
            # A tool asked twice counts as answered if it ever succeeded.
            called[step["tool"]] = called.get(step["tool"], False) or ok

    # Group by office, keeping the order the servers were listed in.
    order: list[str] = []
    by_office: dict[str, dict] = {}
    for row in catalogue:
        name = row.get("office", "?")
        if name not in by_office:
            order.append(name)
            by_office[name] = {"icon": row.get("icon", "\U0001f3db\ufe0f"),
                               "holds": row.get("holds", ""), "tools": []}
        by_office[name]["tools"].append(row.get("tool", ""))

    panels = []
    for name in order:
        office = by_office[name]
        hits = [t for t in office["tools"] if t in called]
        if not hits:
            state = "idle"
        elif any(called[t] for t in hits):
            state = "answered"
        else:
            state = "failed"
        bg, border, ink, glyph, word = _OFFICE_STATES[state]

        chips = "".join(_office_chip(t, t in called, ink)
                        for t in office["tools"])
        panels.append(
            f"<div style='flex:1 1 190px;min-width:190px;background:{bg};"
            f"border:2px solid {border};border-radius:10px;padding:13px 15px'>"
            f"<div style='font-size:22px;line-height:1'>{office['icon']}</div>"
            f"<div style='font-weight:700;font-size:14px;margin-top:6px;"
            f"color:#0f172a'>{html.escape(name)}</div>"
            f"<div style='font-size:11px;color:#64748b;margin-top:2px;"
            f"min-height:30px'>{html.escape(office['holds'])}</div>"
            f"<div style='font-size:11px;font-weight:700;color:{ink};"
            f"margin-top:7px'>{glyph} {word}</div>"
            f"<div style='margin-top:6px'>{chips}</div>"
            f"</div>")

    if local:
        used_any = any(t in called for t in local)
        ink = STATUS_PASS if used_any else "#64748b"
        chips = "".join(_office_chip(t, t in called, ink) for t in local)
        panels.append(
            f"<div style='flex:1 1 190px;min-width:190px;background:#ffffff;"
            f"border:2px dashed {STATUS_PASS if used_any else '#cbd5e1'};"
            f"border-radius:10px;padding:13px 15px'>"
            f"<div style='font-size:22px;line-height:1'>\U0001f4d6</div>"
            f"<div style='font-weight:700;font-size:14px;margin-top:6px;"
            f"color:#0f172a'>This office</div>"
            f"<div style='font-size:11px;color:#64748b;margin-top:2px;"
            f"min-height:30px'>our own policy manual &mdash; a local tool, "
            f"not MCP</div>"
            f"<div style='font-size:11px;font-weight:700;color:{ink};"
            f"margin-top:7px'>"
            f"{'\u2713 answered' if used_any else '\u25cb not asked'}</div>"
            f"<div style='margin-top:6px'>{chips}</div></div>")

    n_tools = len(catalogue) + len(local or [])
    return (
        f"<div style='{CARD}'>"
        f"<div style='font-size:10.5px;letter-spacing:.13em;"
        f"text-transform:uppercase;color:#64748b;font-weight:700;"
        f"margin-bottom:11px'>{html.escape(title)} &middot; "
        f"{len(order)} offices &middot; {n_tools} tools</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:10px'>"
        f"{''.join(panels)}</div>"
        f"<div style='font-size:11px;color:#64748b;margin-top:11px'>"
        f"Every tool above was <b>published by the office that owns it</b>. "
        f"Nothing in this list was written by the agent.</div>"
        f"</div>")


def show_offices(catalogue: Sequence[dict], **kwargs) -> None:
    """Display the three-office panel in a notebook."""
    show_html(offices_html(catalogue, **kwargs))
