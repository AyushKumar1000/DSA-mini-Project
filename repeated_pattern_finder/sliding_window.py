"""
sliding_window.py
=================
Monotonic-deque sliding-window minimum and its application to finding
the longest substring that occurs at least k times.

Theory
------
Given the LCP array, the longest substring occurring at least k times is the
maximum over all windows of size (k-1) in the LCP array of the minimum value
in that window.

This follows because:
  • A substring occurs at least k times ⟺ there exist k adjacent suffixes
    in the suffix array all sharing that prefix.
  • k adjacent suffixes span a window of k-1 consecutive LCP values.
  • The shortest common prefix within those k suffixes equals the minimum
    of the LCP values in the window.
  • We want the window whose minimum is largest → sliding-window maximum-of-minima.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple


def sliding_window_min(arr: List[int], window: int) -> List[int]:
    """
    Compute the sliding-window minimum of *arr* over every window of size
    *window* using a monotonic deque.

    Time  : O(n)   — each element enters and leaves the deque at most once
    Space : O(w)   — deque holds at most *window* indices

    Parameters
    ----------
    arr    : List[int]
        Input array of length n.
    window : int
        Window size w (must satisfy 1 ≤ w ≤ n).

    Returns
    -------
    List[int]
        Array of length ``n - w + 1`` where element i is
        ``min(arr[i], arr[i+1], …, arr[i+w-1])``.

    Raises
    ------
    ValueError
        If window < 1 or window > len(arr).
    """
    n = len(arr)
    if n == 0:
        return []
    if window < 1 or window > n:
        raise ValueError(
            f"window={window} is out of valid range [1, {n}]"
        )

    result: List[int] = []
    # dq stores *indices* into arr; front is always the index of the current min
    dq: deque[int] = deque()

    for i in range(n):
        # --- Evict indices that have fallen out of the window ---
        while dq and dq[0] < i - window + 1:
            dq.popleft()

        # --- Maintain monotone increasing (from back) property ---
        # Pop indices whose arr-values are >= arr[i]; they can never be
        # the minimum for any future window that includes i.
        while dq and arr[dq[-1]] >= arr[i]:
            dq.pop()

        dq.append(i)

        # --- Record result once first full window is complete ---
        if i >= window - 1:
            result.append(arr[dq[0]])

    return result


def longest_repeat_at_least_k(
    lcp: List[int],
    sa: List[int],
    k: int,
) -> Tuple[int, Optional[int]]:
    """
    Find the length of the longest substring occurring at least *k* times,
    and the starting position (index into the original text) of one occurrence.

    Uses sliding-window minimum over windows of size ``k-1`` in the LCP array
    and returns the maximum of those minima.

    Time  : O(n)   — single pass over the LCP array
    Space : O(k)   — deque size bounded by window = k-1

    Parameters
    ----------
    lcp : List[int]
        LCP array of length n (``lcp[0]`` is always 0).
    sa  : List[int]
        Suffix array of length n.
    k   : int
        Minimum number of occurrences required (k ≥ 2).

    Returns
    -------
    (length, start_position) : Tuple[int, Optional[int]]
        *length* – length of the longest repeated substring (0 if none).
        *start_position* – starting index in the original text of one
        occurrence, or None when length == 0.

    Raises
    ------
    ValueError
        If k < 2 or the lcp and sa arrays have different lengths.
    """
    if k < 2:
        raise ValueError(f"k must be ≥ 2, got {k}")
    n = len(lcp)
    if n == 0 or k > n:
        return 0, None
    if len(sa) != n:
        raise ValueError(f"lcp and sa must have equal length ({n} vs {len(sa)})")

    window = k - 1  # window of LCP values → k adjacent suffixes

    if window > n:
        return 0, None

    # Slide over lcp[1:] (lcp[0] is always 0 and not meaningful)
    # We work on lcp[1..n-1], length n-1
    effective = lcp[1:]  # length n-1
    if window > len(effective):
        return 0, None

    best_len = 0
    best_start: Optional[int] = None

    # Monotonic deque for minimum over window
    dq: deque[int] = deque()

    for i in range(len(effective)):
        # Evict out-of-window indices
        while dq and dq[0] < i - window + 1:
            dq.popleft()
        # Maintain increasing minimum property
        while dq and effective[dq[-1]] >= effective[i]:
            dq.pop()
        dq.append(i)

        if i >= window - 1:
            win_min = effective[dq[0]]
            if win_min > best_len:
                best_len = win_min
                # Window in effective[] starts at i-window+1; that maps to
                # lcp index i-window+2 (since effective = lcp[1:]).
                # The k suffixes span sa indices (i-window+1) .. (i+1) in sa
                # (accounting for the offset from lcp[1:]).
                sa_window_start = (i - window + 1) + 1  # +1 for lcp[1:] offset
                best_start = sa[sa_window_start]

    return best_len, best_start
