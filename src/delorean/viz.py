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
