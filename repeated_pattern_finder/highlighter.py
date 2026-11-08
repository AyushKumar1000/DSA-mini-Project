"""
highlighter.py
==============
Two highlighting backends:

1. ``ansi_highlight``  – wraps repeated-pattern spans with ANSI 256-colour
   escape codes for terminal output.
2. ``html_highlight``  – wraps spans with ``<mark>`` tags styled with
   distinct background colours for browser / Streamlit rendering.

Both functions accept the original text and a list of PatternResult objects
(already computed by finder.py) and return a coloured string.
"""

from __future__ import annotations

from typing import List, Tuple

from finder import PatternResult


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

# ANSI 256-colour codes that are visually distinct and work on dark/light bg
_ANSI_BG_CODES: List[int] = [
    196, 202, 208, 214, 220,  # reds → oranges → yellows
    46,  82, 118, 154, 190,   # greens
    51,  87, 123, 159, 195,   # cyans
    21,  57,  93, 129, 165,   # blues → purples
    201, 207, 213, 219, 225,  # magentas → pinks
]

# HTML/CSS hex colours for <mark> tags
_HTML_COLORS: List[str] = [
    "#FF6B6B", "#FF9F43", "#FECA57", "#48DBFB", "#1DD1A1",
    "#54A0FF", "#5F27CD", "#FF9FF3", "#01CBC6", "#10AC84",
    "#EE5A24", "#0652DD", "#C4E538", "#FDA7DF", "#D980FA",
    "#9980FA", "#58B19F", "#F8EFBA", "#FEA47F", "#25CCF7",
]


def _build_span_map(
    text: str,
    results: List[PatternResult],
    max_patterns: int = 20,
) -> List[int]:
    """
    Build a character-level colour index array for *text*.

    For each position in text, store the index of the pattern that covers it
    (the longest one wins), or -1 if uncovered.

    Time  : O(n * min(m, max_patterns) * L)
             n = len(text), m = len(results), L = average pattern length
    Space : O(n)

    Parameters
    ----------
    text        : str
    results     : List[PatternResult]
    max_patterns: int   Cap how many patterns we colour (avoids palette exhaustion)

    Returns
    -------
    List[int]  colour_index[i] ∈ {-1, 0, 1, …, max_patterns-1}
    """
    n = len(text)
    color_map: List[int] = [-1] * n

    for ci, pr in enumerate(results[:max_patterns]):
        for pos in pr.positions:
            end = pos + pr.length
            if end > n:
                continue
            for k in range(pos, end):
                # Only colour if unset or current pattern is longer
                if color_map[k] == -1:
                    color_map[k] = ci

    return color_map


# ---------------------------------------------------------------------------
# 1. ANSI highlight
# ---------------------------------------------------------------------------

_ANSI_RESET = "\033[0m"


def _ansi_bg(code: int) -> str:
    return f"\033[48;5;{code}m\033[38;5;0m"  # background + black foreground


def ansi_highlight(
    text: str,
    results: List[PatternResult],
    max_patterns: int = 20,
) -> str:
    """
    Return *text* with ANSI 256-colour background codes highlighting each
    repeated pattern in a unique colour.

    Overlapping highlights: the longest pattern wins at each character.
    If more than *max_patterns* patterns are provided, only the first
    *max_patterns* (by length) are coloured.

    Time  : O(n * m * L)   (see _build_span_map)
    Space : O(n)

    Parameters
    ----------
    text         : str
    results      : List[PatternResult]   sorted by length (longest first)
    max_patterns : int

    Returns
    -------
    str  — text with embedded ANSI escape sequences
    """
    if not results:
        return text

    color_map = _build_span_map(text, results, max_patterns)
    n = len(text)
    out: List[str] = []
    current_color = -1

    for i in range(n):
        c = color_map[i]
        if c != current_color:
            if current_color != -1:
                out.append(_ANSI_RESET)
            if c != -1:
                bg_code = _ANSI_BG_CODES[c % len(_ANSI_BG_CODES)]
                out.append(_ansi_bg(bg_code))
            current_color = c
        out.append(text[i])

    if current_color != -1:
        out.append(_ANSI_RESET)

    return "".join(out)


def ansi_legend(results: List[PatternResult], max_patterns: int = 20) -> str:
    """
    Build a colour-keyed legend string for the ANSI output.

    Parameters
    ----------
    results      : List[PatternResult]
    max_patterns : int

    Returns
    -------
    str  — multi-line legend with coloured pattern samples
    """
    lines = ["\n--- Colour legend ---"]
    for ci, pr in enumerate(results[:max_patterns]):
        bg_code = _ANSI_BG_CODES[ci % len(_ANSI_BG_CODES)]
        coloured = f"{_ansi_bg(bg_code)} {pr.pattern!r} {_ANSI_RESET}"
        lines.append(
            f"  {coloured}  len={pr.length}  count={pr.count}"
            f"  positions={pr.positions[:5]}{'...' if len(pr.positions)>5 else ''}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. HTML highlight
# ---------------------------------------------------------------------------

def html_highlight(
    text: str,
    results: List[PatternResult],
    max_patterns: int = 20,
    newline_to_br: bool = True,
) -> str:
    """
    Return an HTML snippet where each repeated pattern is wrapped in a
    ``<mark>`` tag with a unique background colour.

    Overlapping highlights: the longest pattern wins.

    Time  : O(n * m * L)
    Space : O(n)

    Parameters
    ----------
    text          : str
    results       : List[PatternResult]
    max_patterns  : int
    newline_to_br : bool   Replace ``\\n`` with ``<br>`` for display.

    Returns
    -------
    str  — HTML string ready for embedding in a page or Streamlit ``st.markdown``.
    """
    if not results:
        escaped = _html_escape(text)
        if newline_to_br:
            escaped = escaped.replace("\n", "<br>")
        return f"<pre style='white-space:pre-wrap;'>{escaped}</pre>"

    color_map = _build_span_map(text, results, max_patterns)
    n = len(text)
    out: List[str] = []
    current_color = -1

    for i in range(n):
        c = color_map[i]
        if c != current_color:
            if current_color != -1:
                out.append("</mark>")
            if c != -1:
                colour = _HTML_COLORS[c % len(_HTML_COLORS)]
                out.append(
                    f"<mark style='background:{colour};border-radius:3px;"
                    f"padding:1px 2px;'>"
                )
            current_color = c
        ch = _html_escape(text[i])
        out.append(ch)

    if current_color != -1:
        out.append("</mark>")

    body = "".join(out)
    if newline_to_br:
        body = body.replace("\n", "<br>")

    return f"<pre style='white-space:pre-wrap;font-family:monospace;'>{body}</pre>"


def html_legend(results: List[PatternResult], max_patterns: int = 20) -> str:
    """
    Build an HTML colour-keyed legend table for the patterns.

    Returns
    -------
    str  — HTML string with a legend table.
    """
    rows = []
    for ci, pr in enumerate(results[:max_patterns]):
        colour = _HTML_COLORS[ci % len(_HTML_COLORS)]
        sample = _html_escape(pr.pattern[:30])
        if len(pr.pattern) > 30:
            sample += "…"
        pos_str = ", ".join(str(p) for p in pr.positions[:8])
        if len(pr.positions) > 8:
            pos_str += ", …"
        rows.append(
            f"<tr>"
            f"<td><mark style='background:{colour};padding:2px 6px;"
            f"border-radius:3px;'>{sample}</mark></td>"
            f"<td align='center'>{pr.length}</td>"
            f"<td align='center'>{pr.count}</td>"
            f"<td>{pos_str}</td>"
            f"</tr>"
        )

    header = (
        "<tr><th>Pattern</th><th>Length</th>"
        "<th>Count</th><th>Positions</th></tr>"
    )
    table_style = (
        "border-collapse:collapse;width:100%;"
        "font-family:monospace;font-size:13px;"
    )
    return (
        f"<table style='{table_style}'>"
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table>"
    )


def _html_escape(s: str) -> str:
    """Minimal HTML escaping for text content."""
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )
