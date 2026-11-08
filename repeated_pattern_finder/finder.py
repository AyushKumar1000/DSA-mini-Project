"""
finder.py
=========
High-level pattern-grouping and reporting logic built on top of the
suffix array + LCP array.

Key functions
-------------
* ``find_repeated_patterns``  – group adjacent SA entries by shared LCP,
  deduplicate, and return maximal repeats meeting the thresholds.
* ``longest_k_repeat``        – convenience wrapper returning the single
  longest substring that occurs ≥ k times.
* ``PatternResult``           – named dataclass for a single repeated pattern.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

from suffix_array import build_suffix_array, build_rank_array
from lcp import build_lcp_array
from sliding_window import longest_repeat_at_least_k


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclasses.dataclass(order=True)
class PatternResult:
    """
    Container for one repeated pattern.

    Attributes
    ----------
    pattern    : str    The repeated substring.
    length     : int    ``len(pattern)``.
    count      : int    Number of occurrences (≥ min_occurrences).
    positions  : List[int]
        Sorted list of all start indices in the original text
        where this pattern occurs.
    """

    pattern: str
    length: int
    count: int
    positions: List[int]

    def __repr__(self) -> str:
        pos_str = str(self.positions[:10])
        if len(self.positions) > 10:
            pos_str = pos_str[:-1] + ", ...]"
        return (
            f"PatternResult(pattern={self.pattern!r}, length={self.length}, "
            f"count={self.count}, positions={pos_str})"
        )


# ---------------------------------------------------------------------------
# Core grouping logic
# ---------------------------------------------------------------------------

def _extract_all_lcp_groups(
    sa: List[int],
    lcp: List[int],
    min_length: int,
    min_occurrences: int,
) -> List[Tuple[int, int, int]]:
    """
    Use a monotone stack to enumerate ALL (start, end, lcp_value) triples
    where:
      - [start, end] is a maximal range of SA indices,
      - every adjacent pair in that range has LCP >= lcp_value,
      - end - start + 1 >= min_occurrences,
      - lcp_value >= min_length.

    This is the standard "LCP interval tree" enumeration that correctly
    produces both "a" (block [0,2], lcp=1) AND "ana" (sub-block [0,2], lcp=3)
    for "banana".

    Time  : O(n)   — each index pushed/popped once
    Space : O(n)   — stack

    Returns
    -------
    List of (start, end, actual_lcp) tuples.
    """
    n = len(sa)
    if n == 0:
        return []

    results: List[Tuple[int, int, int]] = []
    # Stack holds (lcp_value, start_index)
    stack: List[Tuple[int, int]] = [(0, 0)]

    for i in range(1, n + 1):
        # Sentinel: lcp[n] = 0 to flush the stack
        cur_lcp = lcp[i] if i < n else 0
        last_start = i - 1

        while stack and stack[-1][0] > cur_lcp:
            top_lcp, top_start = stack.pop()
            end = i - 1
            count = end - top_start + 1
            if top_lcp >= min_length and count >= min_occurrences:
                results.append((top_start, end, top_lcp))
            last_start = top_start

        if cur_lcp > stack[-1][0]:
            stack.append((cur_lcp, last_start))

    return results


def find_repeated_patterns(
    text: str,
    min_length: int = 3,
    min_occurrences: int = 2,
    case_sensitive: bool = True,
) -> List[PatternResult]:
    """
    Find all repeated substrings in *text* that:
      * have length ≥ min_length, and
      * appear ≥ min_occurrences times.

    Only **maximal repeats** are reported: if pattern P is a substring of a
    longer pattern Q that occurs the same number of times, P is suppressed.
    Overlapping occurrences are counted.

    Algorithm
    ---------
    1. Build suffix array (O(n log² n)).
    2. Build LCP array via Kasai (O(n)).
    3. Group adjacent SA entries by LCP ≥ min_length.
    4. For each group compute the actual LCP of the whole group (stack-min)
       and collect positions.
    5. Deduplicate: if two patterns differ only in length (one is a prefix
       of the other with the same position set), keep only the longer one.

    Time  : O(n log² n)  (suffix array dominates)
    Space : O(n)

    Parameters
    ----------
    text            : str
        Input text.
    min_length      : int
        Minimum pattern length (must be ≥ 1).
    min_occurrences : int
        Minimum number of occurrences required (must be ≥ 2).
    case_sensitive  : bool
        If False, text is lowercased before analysis but positions still
        refer to the original text.

    Returns
    -------
    List[PatternResult]
        Maximal repeated patterns sorted by (length DESC, count DESC).

    Raises
    ------
    ValueError
        If min_length < 1 or min_occurrences < 2.
    """
    if min_length < 1:
        raise ValueError(f"min_length must be ≥ 1, got {min_length}")
    if min_occurrences < 2:
        raise ValueError(f"min_occurrences must be ≥ 2, got {min_occurrences}")

    if not text:
        return []

    working = text if case_sensitive else text.lower()
    n = len(working)

    if min_length > n:
        return []

    # --- Step 1 & 2: Build SA and LCP ---
    sa = build_suffix_array(working)
    rank = build_rank_array(sa)
    lcp = build_lcp_array(working, sa, rank)

    # --- Step 3: Extract all LCP intervals using monotone stack ---
    intervals = _extract_all_lcp_groups(sa, lcp, min_length, min_occurrences)

    # --- Step 4: Build PatternResult objects ---
    # Key: (pattern, positions_tuple) → keep best (longest) per position-set
    seen: Dict[Tuple[int, ...], PatternResult] = {}

    for start_idx, end_idx, actual_len in intervals:
        actual_len = min(actual_len, n)
        if actual_len < min_length:
            continue

        # Starting positions
        positions = sorted(sa[i] for i in range(start_idx, end_idx + 1))
        count = len(positions)

        # Extract pattern from the working (case-adjusted) text
        anchor = sa[start_idx]
        pattern = working[anchor: anchor + actual_len]

        key = tuple(positions)
        if key not in seen or seen[key].length < actual_len:
            seen[key] = PatternResult(
                pattern=pattern,
                length=actual_len,
                count=count,
                positions=positions,
            )

    # --- Step 5: Remove sub-patterns (maximality filter) ---
    results = list(seen.values())
    results = _filter_maximal(results)

    # Sort: longest first, then most frequent, then earliest position
    results.sort(key=lambda r: (-r.length, -r.count, r.positions[0]))
    return results


def _filter_maximal(results: List[PatternResult]) -> List[PatternResult]:
    """
    Remove patterns that are strict sub-strings of a longer pattern
    with the same or higher occurrence count.

    A pattern P is *non-maximal* if there exists another pattern Q such that:
      • P is a contiguous substring of Q (P ≠ Q), and
      • Q.count >= P.count.

    Time  : O(m²) in m = number of distinct patterns; m << n in practice.
    Space : O(m)
    """
    # Group by occurrence count because a longer pattern can only suppress
    # a shorter pattern if longer.count >= shorter.count. But any substring P
    # must appear at least as often as the containing string Q, so
    # count(Q) >= count(P) is only possible when count(Q) == count(P).
    import collections
    by_count: collections.defaultdict[int, List[PatternResult]] = collections.defaultdict(list)
    for r in results:
        by_count[r.count].append(r)

    kept: List[PatternResult] = []
    for cnt, group in by_count.items():
        # Sort by length descending within the same count bucket
        sorted_res = sorted(group, key=lambda r: -r.length)
        keep = [True] * len(sorted_res)
        for i in range(len(sorted_res)):
            if not keep[i]:
                continue
            longer_pat = sorted_res[i].pattern
            for j in range(i + 1, len(sorted_res)):
                if not keep[j]:
                    continue
                if sorted_res[j].pattern in longer_pat:
                    keep[j] = False
        kept.extend([r for r, k in zip(sorted_res, keep) if k])

    return kept


# ---------------------------------------------------------------------------
# Convenience: single longest repeat ≥ k
# ---------------------------------------------------------------------------

def longest_k_repeat(
    text: str,
    k: int = 2,
    case_sensitive: bool = True,
) -> Optional[Tuple[str, int, List[int]]]:
    """
    Return the single longest substring that occurs at least *k* times.

    Uses the sliding-window minimum over the LCP array (O(n) after the
    O(n log² n) suffix array construction).

    Parameters
    ----------
    text           : str  Input text.
    k              : int  Minimum occurrences (≥ 2).
    case_sensitive : bool If False, compare case-insensitively.

    Returns
    -------
    (pattern, length, positions) if a repeat of length ≥ 1 exists, else None.
    """
    if not text or k < 2:
        return None

    working = text if case_sensitive else text.lower()
    n = len(working)

    sa = build_suffix_array(working)
    rank = build_rank_array(sa)
    lcp = build_lcp_array(working, sa, rank)

    length, start = longest_repeat_at_least_k(lcp, sa, k)
    if length == 0 or start is None:
        return None

    pattern = working[start: start + length]

    # Collect all occurrences via linear scan of the SA+LCP
    positions = _collect_positions(working, sa, lcp, pattern, length)
    return pattern, length, sorted(positions)


def _collect_positions(
    text: str,
    sa: List[int],
    lcp: List[int],
    pattern: str,
    pat_len: int,
) -> List[int]:
    """
    Collect all starting positions of *pattern* (length *pat_len*) in *text*
    using the suffix array and LCP array.

    Runs a linear scan: adjacent SA entries with LCP ≥ pat_len share the
    same prefix of length pat_len.

    Time  : O(n)
    Space : O(occ)  where occ is the number of occurrences
    """
    n = len(sa)
    # Find one anchor: SA entry whose suffix starts with pattern
    # (we already know 'start'; just walk left and right)
    # For generality, do a full O(n) scan.
    positions: List[int] = []
    block = False
    for i in range(n):
        if text[sa[i]: sa[i] + pat_len] == pattern:
            positions.append(sa[i])
            block = True
        elif block:
            break  # Suffixes are sorted; once we exit the block we're done
    # If the above early-exit missed some (edge case), fall back to full scan
    if not block:
        positions = [sa[i] for i in range(n)
                     if text[sa[i]: sa[i] + pat_len] == pattern]
    return positions


# ---------------------------------------------------------------------------
# Sorting helpers for the UI layer
# ---------------------------------------------------------------------------

def sort_results(
    results: List[PatternResult],
    by: str = "length",
) -> List[PatternResult]:
    """
    Return *results* sorted by one of three keys.

    Parameters
    ----------
    results : List[PatternResult]
    by      : str  One of ``"length"``, ``"frequency"``, ``"position"``.

    Returns
    -------
    List[PatternResult]  (new list, original unchanged)
    """
    if by == "length":
        return sorted(results, key=lambda r: (-r.length, -r.count, r.positions[0]))
    elif by == "frequency":
        return sorted(results, key=lambda r: (-r.count, -r.length, r.positions[0]))
    elif by == "position":
        return sorted(results, key=lambda r: (r.positions[0], -r.length))
    else:
        raise ValueError(f"Unknown sort key: {by!r}. Choose length/frequency/position")
