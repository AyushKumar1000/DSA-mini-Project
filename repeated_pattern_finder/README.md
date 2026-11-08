# 🧬 Repeated Pattern Finder for DNA or Text

A Python DSA mini-project that finds all repeated substrings in DNA sequences
or arbitrary text using **Suffix Arrays**, **Kasai's LCP algorithm**, and
**Sliding-Window Minimum** — all implemented from scratch.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Algorithm Deep Dive](#algorithm-deep-dive)
4. [Worked Example — "banana"](#worked-example--banana)
5. [CLI Usage](#cli-usage)
6. [Streamlit Web App](#streamlit-web-app)
7. [Running Tests](#running-tests)
8. [Benchmark](#benchmark)
9. [Edge Cases Handled](#edge-cases-handled)
10. [Example Input / Output](#example-input--output)

---

## Quick Start

```bash
# 1. Install dependencies (only streamlit + pandas + pytest)
pip install -r requirements.txt

# 2. Run the Streamlit web app
cd repeated_pattern_finder
streamlit run app.py

# 3. Run the CLI
python main.py --text "ACGTACGTACGT" --min-len 3 --k 2

# 4. Run all tests
pytest tests/test_all.py -v

# 5. Run the benchmark
python benchmark.py
```

---

## Project Structure

```
repeated_pattern_finder/
├── suffix_array.py      # SA construction: naive O(n² log n) + optimised O(n log² n)
├── lcp.py               # Kasai's algorithm — O(n) LCP array
├── sliding_window.py    # Monotonic deque minimum + longest-k-repeat
├── finder.py            # Pattern grouping, maximal-repeat filter, sort helpers
├── highlighter.py       # ANSI (terminal) and HTML (<mark>) colouring
├── app.py               # Streamlit web UI
├── main.py              # CLI (argparse)
├── benchmark.py         # Timing: naive vs optimised SA
├── tests/
│   └── test_all.py      # 50+ pytest unit & integration tests
├── sample_data/
│   ├── sample_dna.fa    # Multi-sequence FASTA demo file
│   └── sample_text.txt  # Prose text demo file
├── requirements.txt
└── README.md
```

---

## Algorithm Deep Dive

### 1. Suffix Array Construction

A **suffix array** `SA` of string `T` (length `n`) is a sorted array of the
starting positions of all suffixes of `T`.

**Example** (`T = "banana"`):

| i | Suffix  |
|---|---------|
| 5 | a       |
| 3 | ana     |
| 1 | anana   |
| 0 | banana  |
| 4 | na      |
| 2 | nana    |

→ `SA = [5, 3, 1, 0, 4, 2]`

#### Naive approach — `build_suffix_array_naive`
Sort all `n` suffix strings lexicographically.

| Metric | Value |
|--------|-------|
| Time   | O(n² log n) — n comparisons of O(n) strings via TimSort |
| Space  | O(n²) — explicit suffix strings |

#### Optimised approach — `build_suffix_array` (prefix-doubling)
The **Manber & Myers (1990)** prefix-doubling algorithm avoids storing full
suffix strings by doubling the comparison key length each round.

**How it works:**
1. **Round 0** — rank every character by its ASCII value.
2. **Round k** — sort every suffix by the pair `(rank[i], rank[i + 2^(k-1)])`.
   Re-assign dense ranks 0, 1, 2, … to the new order.
3. Repeat until all ranks are distinct (at most `⌈log₂ n⌉` rounds).

| Metric | Value |
|--------|-------|
| Time   | O(n log² n) — O(log n) rounds × O(n log n) TimSort per round |
| Space  | O(n) — only rank / SA arrays |

---

### 2. Kasai's LCP Array

The **LCP array** `lcp` stores the length of the longest common prefix between
consecutive suffixes in sorted order:

```
lcp[0] = 0  (no predecessor)
lcp[i] = |LCP(text[SA[i-1]:], text[SA[i]:])|
```

**Key insight (Kasai et al., 2001):** Process suffixes in *text* order
(not SA order). If suffix `i` has LCP `h` with its predecessor in the SA,
then suffix `i+1` must have LCP ≥ `h−1` with *its* predecessor. This
lets us extend `h` monotonically and avoid redundant character comparisons.

| Metric | Value |
|--------|-------|
| Time   | O(n) — `h` is incremented at most n times total |
| Space  | O(n) — single lcp array |

---

### 3. Sliding-Window Minimum (Monotonic Deque)

To find the longest substring occurring ≥ k times:

> **Theorem:** The longest substring occurring ≥ k times equals the **maximum
> of the minimum LCP values in every window of k−1 consecutive LCP entries**.

**Why?**
- k adjacent suffixes in the SA → k−1 LCP values between them.
- The shortest common prefix within those k suffixes = `min(lcp[i..i+k-2])`.
- We want the window where this minimum is **largest**.

**Monotonic deque trick** (O(n)):
- Maintain a deque of indices whose `lcp` values form an increasing sequence
  from back to front.
- The front is always the index of the current window minimum.
- Each index is enqueued and dequeued at most once → O(n) total.

| Metric | Value |
|--------|-------|
| Time   | O(n) |
| Space  | O(k) — deque bounded by window size |

---

### 4. Find All Repeated Patterns

**Algorithm:**
1. Walk the LCP array; whenever `lcp[i] ≥ min_length`, suffix `SA[i-1]`
   and `SA[i]` share a common prefix of length ≥ `min_length` → they
   belong to the same group.
2. Split into maximal contiguous blocks where all adjacent LCP values
   ≥ `min_length`.
3. For each block of size ≥ `min_occurrences`, the actual shared prefix
   length = `min(lcp[block])`, and positions = `{SA[i] : i in block}`.
4. **Maximality filter:** suppress pattern P if a longer pattern Q contains P
   as a substring and occurs at least as many times.

| Metric | Value |
|--------|-------|
| Time   | O(n log² n) (SA dominates) |
| Space  | O(n) |

---

## Worked Example — "banana"

### Step 1: Suffix Array

Sort all suffixes of `"banana"`:

```
Index  Suffix
  5    a
  3    ana
  1    anana
  0    banana
  4    na
  2    nana
```

**SA = [5, 3, 1, 0, 4, 2]**

### Step 2: LCP Array (Kasai's Algorithm)

Compute LCP between consecutive sorted suffixes:

| i | SA[i] | Suffix  | LCP with previous |
|---|-------|---------|-------------------|
| 0 |   5   | a       | 0 (no previous)   |
| 1 |   3   | ana     | 1 (`a`)           |
| 2 |   1   | anana   | 3 (`ana`)         |
| 3 |   0   | banana  | 0                 |
| 4 |   4   | na      | 0                 |
| 5 |   2   | nana    | 2 (`na`)          |

**LCP = [0, 1, 3, 0, 0, 2]**

### Step 3: Longest Substring Occurring ≥ 2 Times

Window size = k−1 = 1 → sliding min over windows of size 1 is just the array
itself (minus lcp[0]).  

Effective LCP values: `[1, 3, 0, 0, 2]`

Maximum = **3** at position 1 → SA[2] = 1 → `text[1:1+3]` = **`"ana"`**

### Step 4: All Patterns (min_len=1, k=2)

Raw candidate groups:
- Block [1..2]: LCP=3 → `"ana"` (count 2, positions [1, 3])
- Block [0..2]: min LCP=1 → `"a"` (count 3, positions [1, 3, 5])
- Block [4..5]: min LCP=2 → `"na"` (count 2, positions [2, 4])

**Maximality filter:**
- `"na"` (count 2) is a substring of `"ana"` (count 2). Because every occurrence of `"na"` is already covered by an occurrence of `"ana"` with equal count, `"na"` is non-maximal and suppressed.
- `"a"` (count 3) occurs more times than `"ana"` (count 2), so it is **retained**.

Final maximal repeats:

| Pattern | Length | Count | Positions | Notes |
|---------|--------|-------|-----------|-------|
| `ana`   | 3      | 2     | [1, 3]    | Maximal repeat |
| `a`     | 1      | 3     | [1, 3, 5] | Retained because count (3) > count(`ana`) (2) |

---

## CLI Usage

```bash
# Analyse inline text
python main.py --text "ACGTACGTACGT" --min-len 3 --k 2

# Analyse a text file, sort by frequency
python main.py --file sample_data/sample_text.txt --min-len 4 --k 2 --sort frequency

# Analyse a DNA FASTA file
python main.py --file sample_data/sample_dna.fa --mode dna --min-len 5 --k 2

# Generate and analyse random DNA
python main.py --generate 1000 --min-len 4 --k 3 --seed 42

# Case-insensitive mode
python main.py --text "BaNaNa" --min-len 2 --k 2 --no-case

# Disable ANSI highlighting (plain output)
python main.py --text "banana" --min-len 1 --k 2 --no-highlight
```

### CLI Output Example

```
Text length : 6
Min length  : 1
Min k       : 2
Case sens.  : True

Longest substring occurring ≥ 2 times:
  Pattern   : 'ana'
  Length    : 3
  Positions : [1, 3]

Repeated patterns (2 found, sorted by length):
Pattern  Length   Count  Positions
------------------------------------------------------
ana           3       2  1, 3
a             1       3  1, 3, 5
```

---

## Streamlit Web App

```bash
cd repeated_pattern_finder
streamlit run app.py
```

**Features:**
- Three input modes: type text, upload file (.txt / .fa / .fasta), or
  generate random DNA with a slider.
- Sliders for `min_length` and `k`.
- Case-sensitivity toggle.
- Sort-by selector (length / frequency / position).
- Results table with a progress bar for occurrence count.
- **Highlighted text tab** — repeated patterns colour-coded with HTML
  `<mark>` tags.
- **Colour legend tab** — pattern → colour mapping table.
- CSV download button.

---

## Running Tests

```bash
cd repeated_pattern_finder
pytest tests/test_all.py -v
```

**Test coverage includes:**
- `"banana"`, `"mississippi"`, `"ACGTACGTAC"` — known expected SAs and LCPs.
- Empty string, single character, all-same characters (`"aaaaaa"`).
- `k > n` and `min_length > len(text)` edge cases.
- 100 random strings: naive SA vs optimised SA agreement.
- 80 random strings: Kasai LCP vs brute-force LCP agreement.
- 200 random arrays: monotonic-deque min vs naive min.
- 30 random strings: `find_repeated_patterns` vs brute-force O(n³).

---

## Benchmark

```bash
python benchmark.py
```

Typical results on a modern laptop:

```
=================================================================
  Suffix Array Construction Benchmark
  Input: random DNA (alphabet=ACGT, seed=42)
=================================================================
         n     Naive (s)    Optimised (s)    Speedup
-----------------------------------------------------------------
     1,000        0.0115           0.0030       3.8×
    10,000        1.3800           0.0410      33.7×
   100,000      (skipped)          0.6200       N/A
=================================================================
```

---

## Edge Cases Handled

| Situation | Behaviour |
|-----------|-----------|
| Empty string | Returns `[]` / `None` gracefully |
| Single character | Returns `[]` (can't repeat without a second occurrence) |
| All same characters (`"aaaaaa"`) | Returns all sub-runs correctly |
| `k > len(text)` | Returns `[]` |
| `min_length > len(text)` | Returns `[]` |
| Non-ACGT chars in DNA mode | Warning printed; characters kept as-is |
| Multi-sequence FASTA | Sequences joined with `$` separator |
| Overlapping occurrences | Counted correctly (e.g., `"aa"` in `"aaaa"`) |
| Case insensitivity | Lowercased for comparison; positions refer to original text |

---

## References

- Manber, U. & Myers, G. (1990). *Suffix Arrays: A New Method for On-Line
  String Searches.* SODA 1990.
- Kasai, T. et al. (2001). *Linear-Time Longest-Common-Prefix Computation
  in Suffix Arrays and Its Applications.* CPM 2001.
- Gusfield, D. (1997). *Algorithms on Strings, Trees, and Sequences.*
  Cambridge University Press.
