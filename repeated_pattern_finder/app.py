"""
app.py
======
Streamlit web application for the Repeated Pattern Finder.

Run with:
    streamlit run app.py

Features
--------
* Text area, file upload (.txt / .fa / .fasta), or random DNA generator.
* Sliders for min_length and k (minimum occurrences).
* Case-sensitivity toggle.
* Sort-by selector (length / frequency / position).
* Results table rendered with st.dataframe.
* Highlighted text rendered with HTML <mark> tags and colour-coded per pattern.
* Colour legend table.
* "Longest repeat" highlighted call-out.
* DNA mode warning for non-ACGT characters.
"""

from __future__ import annotations

import io
import random
import sys
import os

# Ensure the package directory is on sys.path when running directly
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd

from finder import find_repeated_patterns, longest_k_repeat, sort_results
from highlighter import html_highlight, html_legend
from main import load_fasta, generate_dna

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Repeated Pattern Finder",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark-glass aesthetic
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
/* ---- Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Gradient background ---- */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05) !important;
    border-right: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(16px);
}

/* ---- Metric cards ---- */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px 16px;
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(10px);
}

/* ---- Headers ---- */
h1 { 
    font-weight: 700; 
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
h2, h3 { font-weight: 600; color: #e2e8f0; }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
    color: #94a3b8;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.4);
}

/* ---- Text area ---- */
textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}

/* ---- Highlighted text container ---- */
.highlight-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 20px;
    max-height: 400px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.8;
    color: #e2e8f0;
}

/* ---- Info / warning boxes ---- */
.info-card {
    background: rgba(99,102,241,0.15);
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #c7d2fe;
}
.success-card {
    background: rgba(52,211,153,0.15);
    border-left: 4px solid #34d399;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #a7f3d0;
}

/* ---- Dataframe ---- */
[data-testid="stDataFrameContainer"] {
    border-radius: 10px;
    overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("# 🧬 Repeated Pattern Finder")
st.markdown(
    "<p style='color:#94a3b8;margin-top:-10px;'>Suffix Array · Kasai LCP · "
    "Sliding-Window Minimum · DNA & Text Analysis</p>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Parameters")

    min_len = st.slider("Minimum Pattern Length", min_value=1, max_value=50,
                        value=3, step=1)
    k = st.slider("Minimum Occurrences (k)", min_value=2, max_value=20,
                  value=2, step=1)
    case_sensitive = st.toggle("Case Sensitive", value=True)
    sort_by = st.selectbox("Sort Results By",
                            ["length", "frequency", "position"],
                            index=0)
    max_patterns_color = st.slider("Max Patterns to Colour", min_value=5,
                                   max_value=20, value=15, step=1)

    st.divider()
    st.markdown("## 🎲 Random DNA Generator")
    dna_len = st.number_input("Length", min_value=10, max_value=100_000,
                               value=200, step=10)
    dna_seed = st.number_input("Seed (0 = random)", min_value=0,
                                max_value=99999, value=42)
    gen_btn = st.button("🎲 Generate DNA", use_container_width=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "text" not in st.session_state:
    st.session_state["text"] = ""

# ---------------------------------------------------------------------------
# Input tabs
# ---------------------------------------------------------------------------

tab_type, tab_file, tab_dna = st.tabs(
    ["✏️ Type / Paste Text", "📁 Upload File", "🧬 Generated DNA"]
)

with tab_type:
    typed = st.text_area(
        "Enter your text here:",
        value=st.session_state.get("text", ""),
        height=180,
        placeholder="e.g. banana  |  ACGTACGTACGT  |  To be or not to be…",
        key="typed_text",
    )
    if typed:
        st.session_state["text"] = typed

with tab_file:
    uploaded = st.file_uploader(
        "Upload a .txt, .fa, or .fasta file",
        type=["txt", "fa", "fasta"],
        key="uploaded_file",
    )
    if uploaded:
        raw_bytes = uploaded.read()
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        if uploaded.name.lower().endswith((".fa", ".fasta")):
            # Parse FASTA in-memory
            lines = raw_text.splitlines()
            seqs = []
            cur = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur:
                        seqs.append("".join(cur))
                    cur = []
                else:
                    cur.append(line.upper())
            if cur:
                seqs.append("".join(cur))
            file_text = "$".join(seqs)
            non_dna = set(file_text) - set("ACGT$")
            if non_dna:
                st.warning(
                    f"Non-ACGT characters detected: {sorted(non_dna)}. "
                    "They are kept as-is."
                )
        else:
            file_text = raw_text
        st.session_state["text"] = file_text
        st.success(f"Loaded **{uploaded.name}** — {len(file_text):,} characters")

with tab_dna:
    if gen_btn:
        seed_val = int(dna_seed) if dna_seed else None
        generated = generate_dna(int(dna_len), seed=seed_val)
        st.session_state["text"] = generated

    if st.session_state.get("text", "").replace("A", "").replace("C", "").replace("G", "").replace("T", "") == "":
        # looks like pure DNA
        preview = st.session_state.get("text", "")
        if preview:
            st.code(preview[:300] + ("…" if len(preview) > 300 else ""),
                    language=None)

# ---------------------------------------------------------------------------
# Analyse button
# ---------------------------------------------------------------------------

st.divider()
col_btn, col_info = st.columns([1, 4])
with col_btn:
    analyse = st.button("🔍 Analyse", use_container_width=True, type="primary")

text = st.session_state.get("text", "").strip()

with col_info:
    if text:
        st.markdown(
            f"<div class='info-card'>Text length: <b>{len(text):,}</b> characters</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Analysis & results
# ---------------------------------------------------------------------------

if analyse:
    if not text:
        st.error("⚠️ Please provide some text first.")
        st.stop()

    with st.spinner("Building suffix array and LCP array…"):
        try:
            results = find_repeated_patterns(
                text,
                min_length=min_len,
                min_occurrences=k,
                case_sensitive=case_sensitive,
            )
            results = sort_results(results, by=sort_by)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    # --- Metrics row ---
    st.markdown("## 📊 Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Text Length", f"{len(text):,}")
    m2.metric("Patterns Found", len(results))
    m3.metric("Min Length", min_len)
    m4.metric("Min Occurrences (k)", k)

    # --- Longest repeat ---
    with st.spinner("Computing longest k-repeat…"):
        lr = longest_k_repeat(text, k=k, case_sensitive=case_sensitive)

    if lr:
        pat, ln, pos = lr
        st.markdown(
            f"<div class='success-card'>"
            f"🏆 <b>Longest substring occurring ≥ {k} times:</b> "
            f"<code>{pat}</code> &nbsp;|&nbsp; "
            f"Length <b>{ln}</b> &nbsp;|&nbsp; "
            f"Positions: {pos[:10]}{'…' if len(pos)>10 else ''}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info(f"No substring occurs ≥ {k} times with length ≥ 1.")

    if not results:
        st.warning(
            f"No repeated patterns found with length ≥ {min_len} "
            f"occurring ≥ {k} times."
        )
        st.stop()

    # --- Tabs: Table | Highlighted | Legend ---
    r_tab1, r_tab2, r_tab3 = st.tabs(
        ["📋 Pattern Table", "🎨 Highlighted Text", "🗂️ Colour Legend"]
    )

    with r_tab1:
        df = pd.DataFrame(
            {
                "Pattern": [r.pattern for r in results],
                "Length": [r.length for r in results],
                "Count": [r.count for r in results],
                "Positions": [str(r.positions[:10]) + ("…" if len(r.positions) > 10 else "")
                              for r in results],
                "First Position": [r.positions[0] for r in results],
            }
        )
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pattern": st.column_config.TextColumn("Pattern", width="medium"),
                "Length": st.column_config.NumberColumn("Length", format="%d"),
                "Count": st.column_config.ProgressColumn(
                    "Count",
                    min_value=0,
                    max_value=max(r.count for r in results),
                    format="%d",
                ),
                "First Position": st.column_config.NumberColumn("First Pos."),
            },
        )

        # CSV download
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv_data,
            file_name="repeated_patterns.csv",
            mime="text/csv",
        )

    with r_tab2:
        st.markdown("#### Original text with repeated patterns highlighted")
        display_text = text[:5000]
        if len(text) > 5000:
            st.caption(f"Showing first 5,000 of {len(text):,} characters.")
        highlighted_html = html_highlight(
            display_text,
            results,
            max_patterns=max_patterns_color,
        )
        st.markdown(
            f"<div class='highlight-box'>{highlighted_html}</div>",
            unsafe_allow_html=True,
        )

    with r_tab3:
        st.markdown("#### Colour legend")
        legend_html = html_legend(results, max_patterns=max_patterns_color)
        st.markdown(legend_html, unsafe_allow_html=True)

elif not text:
    st.markdown(
        "<div class='info-card' style='text-align:center;padding:30px;'>"
        "👆 Enter or upload text in the tabs above, then click <b>Analyse</b>."
        "</div>",
        unsafe_allow_html=True,
    )
