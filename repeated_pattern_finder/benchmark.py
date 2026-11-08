"""
benchmark.py
============
Benchmark script comparing the naive suffix array construction against the
optimised prefix-doubling implementation on inputs of 1K, 10K, and 100K
characters.

Run with:
    python benchmark.py

Output example:
    === Suffix Array Benchmark ===
    Input: random DNA (seed=42)

    n=1,000  |  naive:  0.0120s  |  optimised:  0.0032s  |  speedup: 3.8×
    n=10,000 |  naive:  1.3400s  |  optimised:  0.0421s  |  speedup: 31.8×
    n=100,000|  naive:  ...
"""

from __future__ import annotations

import random
import time
from typing import Callable, List

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from suffix_array import build_suffix_array_naive, build_suffix_array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_dna(n: int, seed: int = 42) -> str:
    rng = random.Random(seed)
    return "".join(rng.choices("ACGT", k=n))


def _time_it(fn: Callable, *args, repeats: int = 1) -> float:
    """Return average elapsed seconds over *repeats* calls."""
    total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        total += time.perf_counter() - t0
    return total / repeats


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    sizes: List[int] | None = None,
    seed: int = 42,
) -> None:
    """
    Run the benchmark and print a formatted table.

    Parameters
    ----------
    sizes : List[int]  Input sizes to test (default: [1_000, 10_000, 100_000]).
    seed  : int        Random seed for reproducibility.
    """
    if sizes is None:
        sizes = [1_000, 10_000, 100_000]

    print("=" * 65)
    print("  Suffix Array Construction Benchmark")
    print(f"  Input: random DNA (alphabet=ACGT, seed={seed})")
    print("=" * 65)
    print(f"{'n':>10}  {'Naive (s)':>12}  {'Optimised (s)':>14}  {'Speedup':>9}")
    print("-" * 65)

    for n in sizes:
        text = _generate_dna(n, seed=seed)

        # Decide how many repeats (naive is very slow for large n)
        repeats_naive = 3 if n <= 1_000 else (1 if n <= 10_000 else 1)
        repeats_opt = 3

        # --- Naive ---
        if n <= 10_000:
            t_naive = _time_it(build_suffix_array_naive, text, repeats=repeats_naive)
            naive_str = f"{t_naive:.4f}"
        else:
            # Skip naive for 100K — would take minutes
            t_naive = None
            naive_str = "  (skipped)"

        # --- Optimised ---
        t_opt = _time_it(build_suffix_array, text, repeats=repeats_opt)
        opt_str = f"{t_opt:.4f}"

        # --- Speedup ---
        if t_naive is not None:
            speedup_str = f"{t_naive / t_opt:.1f}×"
        else:
            speedup_str = "   N/A"

        print(f"{n:>10,}  {naive_str:>12}  {opt_str:>14}  {speedup_str:>9}")

    print("=" * 65)
    print()
    print("Notes:")
    print("  • Naive  : O(n² log n) — sorts all n suffix strings.")
    print("  • Optimised: O(n log² n) — prefix-doubling with Python TimSort.")
    print("  • Naive skipped for n=100,000 (estimated > 60 s).")
    print()

    # Benchmark sliding-window longest_k_repeat (O(n log² n) end-to-end)
    print("=" * 65)
    print("  Sliding Window Longest Repeat Benchmark (longest_k_repeat, k=2)")
    print("=" * 65)
    print(f"{'n':>10}  {'Time (s)':>12}  {'Longest Substring Found':>25}")
    print("-" * 65)

    from finder import longest_k_repeat, find_repeated_patterns

    for n in sizes:
        text = _generate_dna(n, seed=seed)
        t0 = time.perf_counter()
        res = longest_k_repeat(text, k=2)
        elapsed = time.perf_counter() - t0
        found = f"'{res[0]}' (len={res[1]})" if res else "None"
        print(f"{n:>10,}  {elapsed:>12.4f}  {found:>25}")

    print("=" * 65)
    print()

    # Benchmark full pattern finding
    print("=" * 65)
    print("  Full Pattern Grouping Benchmark (find_repeated_patterns, k=2)")
    print("=" * 65)
    print(f"{'n':>10}  {'min_len':>9}  {'Time (s)':>12}  {'Patterns':>10}")
    print("-" * 65)

    for n in sizes:
        min_l = 3 if n <= 10_000 else 10
        text = _generate_dna(n, seed=seed)
        t0 = time.perf_counter()
        patterns = find_repeated_patterns(text, min_length=min_l, min_occurrences=2)
        elapsed = time.perf_counter() - t0
        print(f"{n:>10,}  {min_l:>9}  {elapsed:>12.4f}  {len(patterns):>10,}")

    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()
