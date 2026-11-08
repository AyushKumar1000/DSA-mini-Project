"""
suffix_array.py
===============
Suffix Array construction algorithms.

Two implementations are provided:
  1. ``build_suffix_array_naive``  – O(n² log n) time, O(n²) space.
     Sorts all suffixes lexicographically as plain strings.
     Used as a reference for correctness tests and benchmarks.

  2. ``build_suffix_array``  – prefix-doubling (Manber & Myers style),
     O(n log² n) time, O(n) space (the inner sort is Python's TimSort,
     which on integer keys is effectively O(n log n) per round,
     giving O(n log² n) overall).

Both return a list of integers ``sa`` where ``sa[i]`` is the starting
index of the i-th suffix in sorted order.
"""

from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------------
# 1. Naive suffix array — O(n² log n) time, O(n²) space
# ---------------------------------------------------------------------------

def build_suffix_array_naive(text: str) -> List[int]:
    """
    Build a suffix array by explicitly sorting all suffixes as strings.

    Time  : O(n² log n)  — n suffixes, each comparison O(n), TimSort O(n log n)
    Space : O(n²)        — storing all suffix strings

    Parameters
    ----------
    text : str
        The input string.  May be empty.

    Returns
    -------
    List[int]
        Suffix array: ``sa[i]`` is the start index of the i-th
        lexicographically smallest suffix.
    """
    n = len(text)
    if n == 0:
        return []
    # Generate (suffix_string, start_index) pairs and sort
    suffixes = sorted(range(n), key=lambda i: text[i:])
    return suffixes


# ---------------------------------------------------------------------------
# 2. Optimised suffix array — prefix-doubling, O(n log² n)
# ---------------------------------------------------------------------------

def build_suffix_array(text: str) -> List[int]:
    """
    Build a suffix array using the prefix-doubling (Manber & Myers) approach.

    Algorithm outline
    -----------------
    Round 0 : rank every character by its ASCII value → initial rank array.
    Round k : for doubling length 2^k, re-rank every suffix by the pair
              (rank[i], rank[i + 2^(k-1)]) using Python's TimSort.
              Stop when all ranks are distinct (no ties remain).

    Time  : O(n log² n)  — O(log n) rounds, each O(n log n) due to sort
    Space : O(n)         — rank / temporary arrays are all length n

    Parameters
    ----------
    text : str
        The input string.  May be empty.

    Returns
    -------
    List[int]
        Suffix array: ``sa[i]`` is the start index of the i-th
        lexicographically smallest suffix.
    """
    n = len(text)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # --- Initial ranking by single characters ---
    # rank[i] = ordinal of text[i]
    rank: List[int] = [ord(c) for c in text]
    sa: List[int] = sorted(range(n), key=lambda i: rank[i])

    # Assign dense ranks 0,1,2,... to break any ties after round 0
    tmp: List[int] = [0] * n
    tmp[sa[0]] = 0
    for idx in range(1, n):
        prev, curr = sa[idx - 1], sa[idx]
        tmp[curr] = tmp[prev] + (1 if rank[curr] != rank[prev] else 0)
    rank = tmp

    # --- Doubling rounds ---
    gap = 1  # current half-length
    while gap < n:
        # Sort by (rank[i], rank[i+gap]) — out-of-bounds suffix gets -1
        sa = sorted(
            range(n),
            key=lambda i: (rank[i], rank[i + gap] if i + gap < n else -1),
        )

        # Re-rank
        tmp = [0] * n
        tmp[sa[0]] = 0
        for idx in range(1, n):
            prev, curr = sa[idx - 1], sa[idx]
            prev_key = (rank[prev], rank[prev + gap] if prev + gap < n else -1)
            curr_key = (rank[curr], rank[curr + gap] if curr + gap < n else -1)
            tmp[curr] = tmp[prev] + (1 if curr_key != prev_key else 0)
        rank = tmp

        # Early exit when all ranks are distinct
        if rank[sa[-1]] == n - 1:
            break

        gap <<= 1  # double the gap

    return sa


# ---------------------------------------------------------------------------
# Utility: inverse suffix array (rank array)
# ---------------------------------------------------------------------------

def build_rank_array(sa: List[int]) -> List[int]:
    """
    Compute the inverse of the suffix array (the rank / ISA array).

    ``rank[sa[i]] = i``  for all i.

    Time  : O(n)
    Space : O(n)

    Parameters
    ----------
    sa : List[int]
        A valid suffix array of length n.

    Returns
    -------
    List[int]
        rank array where ``rank[i]`` is the position of suffix i in ``sa``.
    """
    n = len(sa)
    rank: List[int] = [0] * n
    for i, s in enumerate(sa):
        rank[s] = i
    return rank
