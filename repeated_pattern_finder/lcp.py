"""
lcp.py
======
Kasai's algorithm for building the Longest Common Prefix (LCP) array
in O(n) time and O(n) space.

The LCP array ``lcp[i]`` stores the length of the longest common prefix
between the suffixes ``text[sa[i-1]:]`` and ``text[sa[i]:]``.
By convention, ``lcp[0] = 0``.

Reference
---------
Kasai, T. et al. "Linear-Time Longest-Common-Prefix Computation in Suffix
Arrays and Its Applications."  CPPM 2001.
"""

from __future__ import annotations

from typing import List


def build_lcp_array(text: str, sa: List[int], rank: List[int]) -> List[int]:
    """
    Kasai's algorithm: construct the LCP array in O(n) time.

    Key insight
    -----------
    If the LCP between suffix i and its predecessor in the SA is h > 0,
    then the LCP between suffix i+1 and *its* predecessor is at least h-1.
    This monotonicity lets us scan the string in suffix-index order and
    avoid redundant character comparisons.

    Time  : O(n)   — h never decreases by more than 1 per outer iteration
    Space : O(n)   — lcp array of length n

    Parameters
    ----------
    text : str
        The original input string of length n.
    sa   : List[int]
        The suffix array (length n).
    rank : List[int]
        The inverse suffix array / rank array (length n).
        ``rank[sa[i]] == i`` for all i.

    Returns
    -------
    List[int]
        LCP array of length n.  ``lcp[0]`` is always 0.
        ``lcp[i]`` = length of LCP between ``text[sa[i-1]:]``
        and ``text[sa[i]:]`` for i ≥ 1.

    Raises
    ------
    ValueError
        If the lengths of text, sa, and rank are inconsistent.
    """
    n = len(text)
    if n == 0:
        return []
    if len(sa) != n or len(rank) != n:
        raise ValueError(
            f"Inconsistent lengths: text={n}, sa={len(sa)}, rank={len(rank)}"
        )

    lcp: List[int] = [0] * n
    h = 0  # current LCP length (the "carry")

    for i in range(n):
        # Process suffix starting at position i (in text order)
        r = rank[i]
        if r == 0:
            # First suffix in sorted order has no predecessor; reset carry
            h = 0
            continue
        # Predecessor in the suffix array
        j = sa[r - 1]
        # Extend the common prefix by comparing characters
        while i + h < n and j + h < n and text[i + h] == text[j + h]:
            h += 1
        lcp[r] = h
        # Key step: decrement carry (next iteration's h is at least h-1)
        if h > 0:
            h -= 1

    return lcp
