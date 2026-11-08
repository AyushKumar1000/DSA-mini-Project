"""
main.py
=======
CLI entry point for the Repeated Pattern Finder.

Usage examples
--------------
# From a text file:
python main.py --file input.txt --min-len 3 --k 2

# From a FASTA DNA file:
python main.py --file genome.fa --mode dna --min-len 5 --k 3

# Directly from the command line:
python main.py --text "ACGTACGTACGT" --min-len 4 --k 2

# Generate random DNA and analyse:
python main.py --generate 500 --min-len 4 --k 2

# Case-insensitive mode:
python main.py --file story.txt --min-len 4 --k 3 --no-case

# Sort by frequency:
python main.py --text "banana" --min-len 1 --k 2 --sort frequency
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Optional

from finder import find_repeated_patterns, longest_k_repeat, sort_results
from highlighter import ansi_highlight, ansi_legend

# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

_DNA_ALPHABET = "ACGT"


def load_text_file(path: str) -> str:
    """Read a plain-text file and return its contents."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def load_fasta(path: str, warn_non_dna: bool = True) -> str:
    """
    Parse a FASTA file and return the concatenated sequence (uppercase).

    Multi-sequence FASTA files have their sequences concatenated with a
    separator character ``$`` so suffixes do not cross sequence boundaries.
    Non-ACGT characters are reported as a warning but kept as-is.
    """
    sequences = []
    current: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                current = []
            else:
                current.append(line.upper())
        if current:
            sequences.append("".join(current))

    combined = "$".join(sequences)

    if warn_non_dna:
        non_dna = set(combined) - set(_DNA_ALPHABET) - {"$"}
        if non_dna:
            print(
                f"[warning] Non-ACGT characters found: "
                f"{sorted(non_dna)}.  Keeping as-is.",
                file=sys.stderr,
            )
    return combined


def generate_dna(length: int, seed: Optional[int] = None) -> str:
    """Generate a random DNA string of the given length."""
    rng = random.Random(seed)
    return "".join(rng.choices(_DNA_ALPHABET, k=length))


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_table(results, max_rows: int = 50) -> None:
    """Print a formatted table of PatternResult objects."""
    if not results:
        print("  (no repeated patterns found)")
        return

    # Determine column widths
    pat_w = max(7, max(len(r.pattern) for r in results[:max_rows]))
    pat_w = min(pat_w, 40)

    header = (
        f"{'Pattern':<{pat_w}}  {'Length':>6}  {'Count':>6}  Positions"
    )
    print(header)
    print("-" * (len(header) + 20))

    for r in results[:max_rows]:
        pat_display = r.pattern[:pat_w]
        if len(r.pattern) > pat_w:
            pat_display = pat_display[:-1] + "…"
        pos_str = ", ".join(str(p) for p in r.positions[:8])
        if len(r.positions) > 8:
            pos_str += " …"
        print(f"{pat_display:<{pat_w}}  {r.length:>6}  {r.count:>6}  {pos_str}")

    if len(results) > max_rows:
        print(f"  … (showing first {max_rows} of {len(results)} patterns)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="repeated_pattern_finder",
        description="Find repeated patterns in text or DNA using suffix arrays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input source (mutually exclusive) ---
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", "-f", metavar="PATH",
                     help="Path to a .txt or .fa/.fasta file.")
    src.add_argument("--text", "-t", metavar="STRING",
                     help="Inline text string.")
    src.add_argument("--generate", "-g", type=int, metavar="LENGTH",
                     help="Generate random DNA of this length.")

    # --- Mode ---
    p.add_argument(
        "--mode", "-m",
        choices=["text", "dna"],
        default="text",
        help="'dna' enables FASTA parsing and non-ACGT warnings (default: text).",
    )

    # --- Parameters ---
    p.add_argument("--min-len", "-l", type=int, default=3, metavar="N",
                   help="Minimum pattern length (default: 3).")
    p.add_argument("--k", type=int, default=2, metavar="K",
                   help="Minimum occurrences required (default: 2).")
    p.add_argument("--no-case", action="store_true",
                   help="Case-insensitive matching.")
    p.add_argument("--sort", choices=["length", "frequency", "position"],
                   default="length",
                   help="Sort results by length|frequency|position (default: length).")
    p.add_argument("--max-rows", type=int, default=50, metavar="N",
                   help="Maximum table rows to print (default: 50).")
    p.add_argument("--no-highlight", action="store_true",
                   help="Skip the ANSI-highlighted text view.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for --generate (default: random).")

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # --- Load text ---
    if args.file:
        if not os.path.isfile(args.file):
            print(f"[error] File not found: {args.file}", file=sys.stderr)
            return 1
        if args.mode == "dna":
            text = load_fasta(args.file)
        else:
            text = load_text_file(args.file)
    elif args.text is not None:
        text = args.text
    else:  # --generate
        text = generate_dna(args.generate, seed=args.seed)
        print(f"Generated DNA ({args.generate} bp): {text[:80]}"
              + ("…" if len(text) > 80 else ""))

    if not text.strip():
        print("[error] Input text is empty.", file=sys.stderr)
        return 1

    case_sensitive = not args.no_case
    min_len = args.min_len
    k = args.k

    print(f"\nText length : {len(text):,}")
    print(f"Min length  : {min_len}")
    print(f"Min k       : {k}")
    print(f"Case sens.  : {case_sensitive}")
    print()

    # --- Find patterns ---
    results = find_repeated_patterns(
        text,
        min_length=min_len,
        min_occurrences=k,
        case_sensitive=case_sensitive,
    )
    results = sort_results(results, by=args.sort)

    # --- Longest repeat ---
    lr = longest_k_repeat(text, k=k, case_sensitive=case_sensitive)
    if lr:
        pat, ln, pos = lr
        print(f"Longest substring occurring ≥ {k} times:")
        print(f"  Pattern   : {pat!r}")
        print(f"  Length    : {ln}")
        print(f"  Positions : {pos}")
    else:
        print(f"No substring occurs ≥ {k} times.")
    print()

    # --- Table ---
    print(f"Repeated patterns ({len(results)} found, sorted by {args.sort}):")
    print_table(results, max_rows=args.max_rows)
    print()

    # --- ANSI highlighted view ---
    if not args.no_highlight:
        print("Highlighted text (ANSI colours):")
        print(ansi_highlight(text[:2000], results))
        if len(text) > 2000:
            print(f"  … (showing first 2000 of {len(text)} characters)")
        print(ansi_legend(results))

    return 0


if __name__ == "__main__":
    sys.exit(main())
