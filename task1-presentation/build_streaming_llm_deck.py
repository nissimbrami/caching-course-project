"""Build the StreamingLLM presentation deck.

Paper: Efficient Streaming Language Models with Attention Sinks
       Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis
       ICLR 2024 - arXiv:2309.17453

Style follows the reference decks under `reference-decks/`:
  - 16:9 widescreen (13.33 x 7.5 in), matching the prior student deck (SCALM.pptx)
  - Structure: Title -> Agenda -> Basics -> Paper -> Results -> Limits ->
    What I'd change -> Conclusion -> Thank you (mirrors SCALM.pptx)
  - Vocabulary anchored on course concepts (LRU/LFU/admission/eviction)
    the way Prof. Einziger's Chapter1 - Cache deck introduces them.

Every numeric claim on every slide is traceable to a specific page/table
of the paper via `papers/streaming-llm/deep-read.md`.

Author: Nissim Brami (nissimbrami@post.bgu.ac.il)
"""

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn


HERE = Path(__file__).resolve().parent
OUT = HERE / "StreamingLLM_Presentation.pptx"


# ---------- Colour palette (kept close to the Chapter1 deck) --------------

NAVY = RGBColor(0x0B, 0x1F, 0x3A)
BLUE = RGBColor(0x2E, 0x86, 0xAB)
ACCENT = RGBColor(0xE6, 0x7E, 0x22)
BG = RGBColor(0xFF, 0xFF, 0xFF)
TEXT = RGBColor(0x22, 0x22, 0x22)
MUTED = RGBColor(0x66, 0x66, 0x66)
HIGHLIGHT = RGBColor(0xC0, 0x39, 0x2B)
GREEN_OK = RGBColor(0x27, 0xAE, 0x60)
GREY = RGBColor(0xB3, 0xB3, 0xB3)
LIGHT = RGBColor(0xF4, 0xF6, 0xF8)


# ---------- Helpers -------------------------------------------------------


def _blank(prs: Presentation):
    """Blank layout - we place everything explicitly."""
    return prs.slides.add_slide(prs.slide_layouts[6])


def _title(slide, text: str, top_in: float = 0.35):
    left = Inches(0.5)
    top = Inches(top_in)
    width = Inches(12.33)
    height = Inches(0.9)
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = text
    r.font.name = "Calibri"
    r.font.size = Pt(34)
    r.font.bold = True
    r.font.color.rgb = NAVY
    # accent underline
    line = slide.shapes.add_connector(1, Inches(0.5), Inches(top_in + 0.95),
                                      Inches(12.83), Inches(top_in + 0.95))
    line.line.color.rgb = ACCENT
    line.line.width = Pt(2.25)
    return tb


def _footer(slide, page_num: int):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(7.05),
                                  Inches(12.33), Inches(0.3))
    tf = tb.text_frame
    tf.margin_left = 0
    tf.margin_top = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = "Nissim Brami · Caching in LLMs · BGU"
    r.font.name = "Calibri"
    r.font.size = Pt(9)
    r.font.color.rgb = MUTED
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.RIGHT
    r2 = p2.add_run()
    r2.text = f"{page_num}"
    r2.font.name = "Calibri"
    r2.font.size = Pt(9)
    r2.font.color.rgb = MUTED
    # Move page number to the right edge with a separate textbox for clarity
    tb2 = slide.shapes.add_textbox(Inches(12.4), Inches(7.05),
                                   Inches(0.4), Inches(0.3))
    tf2 = tb2.text_frame
    tf2.margin_left = 0
    tf2.margin_top = 0
    p3 = tf2.paragraphs[0]
    p3.alignment = PP_ALIGN.RIGHT
    r3 = p3.add_run()
    r3.text = f"{page_num}"
    r3.font.name = "Calibri"
    r3.font.size = Pt(9)
    r3.font.color.rgb = MUTED
    # Remove the duplicate page number in the left footer (leave only the credit line)
    tf.paragraphs[0].runs[0].text = "Nissim Brami · Caching in LLMs · BGU"
    # Clear the second paragraph on the left footer
    tf.paragraphs[1].runs[0].text = ""


def _bullets(slide, items, left_in=0.7, top_in=1.6, width_in=12.0,
             height_in=5.0, size=20, color=TEXT, spacing=8):
    """Add a bullet list. Items can be strings or (text, bold, colour) tuples."""
    tb = slide.shapes.add_textbox(Inches(left_in), Inches(top_in),
                                  Inches(width_in), Inches(height_in))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_top = 0
    for idx, it in enumerate(items):
        if isinstance(it, tuple):
            text, bold, col = it
        else:
            text, bold, col = it, False, color
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(spacing)
        r = p.add_run()
        r.text = "• " + text
        r.font.name = "Calibri"
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = col
    return tb


def _plain_text(slide, text, left_in, top_in, width_in, height_in,
                size=18, bold=False, color=TEXT, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(left_in), Inches(top_in),
                                  Inches(width_in), Inches(height_in))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_top = 0
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.name = "Calibri"
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return tb


def _box(slide, left_in, top_in, width_in, height_in, fill=LIGHT,
         line=GREY, line_pt=0.75):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 Inches(left_in), Inches(top_in),
                                 Inches(width_in), Inches(height_in))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = line
    shp.line.width = Pt(line_pt)
    shp.shadow.inherit = False
    return shp


def _labelled_box(slide, left_in, top_in, width_in, height_in,
                  label, sub=None, fill=LIGHT, border=GREY,
                  label_size=16, sub_size=12,
                  label_color=NAVY, sub_color=MUTED):
    _box(slide, left_in, top_in, width_in, height_in, fill=fill, line=border)
    _plain_text(slide, label, left_in + 0.1, top_in + 0.1,
                width_in - 0.2, 0.4, size=label_size, bold=True,
                color=label_color)
    if sub is not None:
        _plain_text(slide, sub, left_in + 0.1, top_in + 0.55,
                    width_in - 0.2, height_in - 0.65,
                    size=sub_size, color=sub_color)


def _table(slide, headers, rows, left_in, top_in, col_widths_in,
           head_size=13, cell_size=12, row_height_in=0.36,
           head_fill=NAVY, head_color=BG, stripe=LIGHT):
    n_cols = len(headers)
    n_rows = len(rows) + 1
    total_w = sum(col_widths_in)
    tbl_shape = slide.shapes.add_table(n_rows, n_cols,
                                       Inches(left_in), Inches(top_in),
                                       Inches(total_w),
                                       Inches(row_height_in * n_rows))
    tbl = tbl_shape.table
    for i, w in enumerate(col_widths_in):
        tbl.columns[i].width = Inches(w)
    for i in range(n_rows):
        tbl.rows[i].height = Inches(row_height_in)
    for j, h in enumerate(headers):
        c = tbl.cell(0, j)
        c.fill.solid()
        c.fill.fore_color.rgb = head_fill
        tf = c.text_frame
        tf.margin_left = Inches(0.05)
        tf.margin_right = Inches(0.05)
        tf.margin_top = Inches(0.02)
        tf.margin_bottom = Inches(0.02)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = h
        r.font.name = "Calibri"
        r.font.size = Pt(head_size)
        r.font.bold = True
        r.font.color.rgb = head_color
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = stripe if (i % 2 == 0) else BG
            tf = c.text_frame
            tf.margin_left = Inches(0.05)
            tf.margin_right = Inches(0.05)
            tf.margin_top = Inches(0.02)
            tf.margin_bottom = Inches(0.02)
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            r.font.name = "Calibri"
            r.font.size = Pt(cell_size)
            r.font.color.rgb = TEXT
    return tbl


# ---------- Slide builders ------------------------------------------------


def s_title(prs):
    s = _blank(prs)
    # Accent bar
    _box(s, 0, 0, 13.33, 0.4, fill=NAVY, line=NAVY)
    _plain_text(s, "Efficient Streaming Language Models",
                0.7, 1.7, 12.0, 0.9, size=40, bold=True, color=NAVY)
    _plain_text(s, "with Attention Sinks",
                0.7, 2.55, 12.0, 0.8, size=32, bold=True, color=BLUE)
    _plain_text(s,
                "Guangxuan Xiao · Yuandong Tian · Beidi Chen · Song Han · Mike Lewis",
                0.7, 3.6, 12.0, 0.4, size=16, color=TEXT)
    _plain_text(s, "MIT · Meta AI · CMU · NVIDIA · ICLR 2024",
                0.7, 3.95, 12.0, 0.4, size=14, color=MUTED)
    _plain_text(s, "arXiv:2309.17453  ·  github.com/mit-han-lab/streaming-llm",
                0.7, 4.35, 12.0, 0.4, size=13, color=MUTED)
    _box(s, 0.7, 5.35, 12.0, 0.03, fill=ACCENT, line=ACCENT)
    _plain_text(s, "Presented by Nissim Brami",
                0.7, 5.55, 12.0, 0.4, size=16, bold=True, color=NAVY)
    _plain_text(s, "Caching in LLMs · Ben-Gurion University · Prof. Gil Einziger",
                0.7, 5.95, 12.0, 0.4, size=13, color=MUTED)
    return s


def s_agenda(prs):
    s = _blank(prs)
    _title(s, "Agenda")
    items = [
        "The problem: LLMs cannot stream past their training window",
        "Course anchor: this is a cache eviction policy",
        "The observation: attention sinks",
        "The mechanism: sinks + rolling KV cache",
        "The pre-training variant: a learnable sink token",
        "Results: perplexity, throughput, streaming QA",
        "Limitations: cache size is still the ceiling",
        "What I would change (and how it bridges to my final project)",
    ]
    _bullets(s, items, top_in=1.7, size=22, spacing=10)
    return s


# --- Basics section ------------------------------------------------------


def s_section_basics(prs):
    s = _blank(prs)
    _box(s, 0, 0, 13.33, 7.5, fill=NAVY, line=NAVY)
    _plain_text(s, "Basics", 0.7, 3.0, 12.0, 1.2,
                size=54, bold=True, color=BG, align=PP_ALIGN.LEFT)
    _box(s, 0.7, 4.1, 6.0, 0.05, fill=ACCENT, line=ACCENT)
    _plain_text(s, "Streaming LLMs · KV cache · Softmax",
                0.7, 4.3, 12.0, 0.5, size=20, color=BG)
    return s


def s_llm_decoding(prs):
    s = _blank(prs)
    _title(s, "The scene: decoding a large language model")
    _bullets(s, [
        "Generate one token at a time (autoregressive decoding).",
        "Every new token attends back to every previous token.",
        "To avoid recomputing, we cache each previous token’s "
        "Key and Value vectors: the KV cache.",
        "KV cache size grows linearly with sequence length.",
        "For a 7B model at 4K tokens, KV cache is ~2 GB (FP16 estimate).",
    ], top_in=1.6, size=20)
    _labelled_box(s, 8.0, 4.9, 5.0, 1.9,
                  "The bottleneck at decode time",
                  "Memory bandwidth, not compute.\n"
                  "The KV cache is what we cache.",
                  fill=LIGHT, border=BLUE,
                  label_size=15, sub_size=13, sub_color=TEXT)
    return s


def s_course_anchor(prs):
    s = _blank(prs)
    _title(s, "This is a caching problem: course vocabulary")
    _bullets(s, [
        "Cache: the KV cache in GPU memory.",
        "Capacity: bounded by GPU RAM.",
        "Requests: attention lookups from each new query token.",
        "Miss: no cache = recompute the whole context (quadratic).",
        "Eviction policy: which past tokens to keep, which to drop?",
        "Baseline eviction: window attention = LRU by position.",
    ], top_in=1.55, size=20)
    _labelled_box(s, 8.2, 4.6, 4.8, 2.2,
                  "Chapter 1 recap",
                  "LRU · LFU · ARC · LIRS · Hyperbolic · FRD · W-TinyLFU\n"
                  "Admission vs. eviction.\n"
                  "StreamingLLM chooses a positional eviction rule.",
                  fill=LIGHT, border=BLUE,
                  label_size=15, sub_size=13, sub_color=TEXT)
    return s


def s_naive_options(prs):
    s = _blank(prs)
    _title(s, "Three obvious KV-cache strategies, all three break")
    headers = ["Strategy", "Cost", "Behaviour on long streams", "PPL"]
    rows = [
        ["(a) Dense attention",     "O(T²)",   "OOM past training window; degrades on long text", "5641"],
        ["(b) Window attention",    "O(T·L)",  "Collapses when first tokens are evicted", "5158"],
        ["(c) Window + re-compute", "O(T·L²)", "Correct but slow (per-token rebuild)",   "5.43"],
    ]
    _table(s, headers, rows, left_in=0.7, top_in=1.6,
           col_widths_in=[3.3, 1.2, 5.6, 1.3],
           head_size=14, cell_size=13, row_height_in=0.55)
    _plain_text(s,
                "Numbers: Llama-2-13B, first book of PG19 (65K tokens) "
                "[paper Fig. 1, p. 2].",
                0.7, 4.2, 12.0, 0.4, size=13, color=MUTED)
    _labelled_box(s, 0.7, 4.9, 12.0, 1.9,
                  "The puzzle",
                  "Window attention has the right cost but the wrong "
                  "answer. Evicting a single token, the first one, "
                  "turns a fluent model into gibberish (PPL 5158). "
                  "Why?",
                  fill=LIGHT, border=HIGHLIGHT,
                  label_size=16, sub_size=15, sub_color=TEXT,
                  label_color=HIGHLIGHT)
    return s


# --- The observation -----------------------------------------------------


def s_section_observation(prs):
    s = _blank(prs)
    _box(s, 0, 0, 13.33, 7.5, fill=NAVY, line=NAVY)
    _plain_text(s, "The observation", 0.7, 3.0, 12.0, 1.2,
                size=54, bold=True, color=BG)
    _box(s, 0.7, 4.1, 6.0, 0.05, fill=ACCENT, line=ACCENT)
    _plain_text(s, "Attention Sinks",
                0.7, 4.3, 12.0, 0.5, size=22, color=BG)
    return s


def s_attention_sink_phenomenon(prs):
    s = _blank(prs)
    _title(s, "Attention concentrates on the first few tokens, everywhere")
    _bullets(s, [
        "Look at the attention map of Llama-2-7B, across every "
        "layer and every head.",
        "Beyond the bottom two layers, the model heavily attends "
        "to the initial tokens, regardless of what those tokens say.",
        "The phenomenon holds in Llama-2, MPT, Falcon, Pythia; "
        "and in BERT (on the [SEP] token); and in ViTs (register tokens).",
        "The authors call these tokens attention sinks.",
    ], top_in=1.55, size=20)
    _labelled_box(s, 0.7, 5.15, 12.0, 1.7,
                  "Paper's exact wording (Fig. 2, p. 3)",
                  "\"Beyond the bottom two layers, the model heavily attends "
                  "to the initial token across all layers and heads.\" "
                  "Visualised over 256 sentences of length 16 in Llama-2-7B.",
                  fill=LIGHT, border=BLUE,
                  label_size=16, sub_size=15, sub_color=TEXT)
    return s


def s_softmax_argument(prs):
    s = _blank(prs)
    _title(s, "Why? The softmax must sum to one")
    # Softmax equation card
    _box(s, 0.7, 1.6, 12.0, 1.4, fill=LIGHT, line=BLUE, line_pt=1.0)
    _plain_text(s,
                "softmax(x)_i  =  exp(x_i) / (exp(x_1) + Σ_{j≥2} exp(x_j))",
                0.9, 1.75, 11.5, 0.5, size=22, bold=True, color=NAVY,
                align=PP_ALIGN.CENTER)
    _plain_text(s,
                "If x_1 ≫ x_j for j ≥ 2, then exp(x_1) dominates the denominator.",
                0.9, 2.35, 11.5, 0.4, size=15, color=TEXT,
                align=PP_ALIGN.CENTER)
    _bullets(s, [
        "Softmax forces the model to allocate attention mass, "
        "even when nothing in the context deserves it.",
        "That excess mass has to land somewhere.",
        "In a causal LM, initial tokens are visible to every "
        "subsequent position; they are the natural dumping ground.",
        "Evicting them removes a huge chunk of the softmax "
        "denominator → the attention distribution warps → PPL explodes.",
    ], top_in=3.35, size=19)
    return s


def s_semantics_or_position(prs):
    s = _blank(prs)
    _title(s, "Is it the tokens, or is it the position?")
    _bullets(s, [
        "Hypothesis A: the first tokens are semantically special.",
        "Hypothesis B: the first positions are structurally special.",
        "Test: replace the first four tokens with newline characters "
        "and re-run.",
    ], top_in=1.55, size=20)
    headers = ["Cache configuration (Llama-2-13B, PG19)", "PPL ↓"]
    rows = [
        ["0 + 1024  (window attention, no sinks)",  "5158.07"],
        ["4 + 1020  (StreamingLLM, real sinks)",    "5.40"],
        ["4×\"\\n\" + 1020  (linebreak sinks)",     "5.60"],
    ]
    _table(s, headers, rows, left_in=1.5, top_in=3.55,
           col_widths_in=[7.0, 2.0], head_size=13, cell_size=13,
           row_height_in=0.5)
    _labelled_box(s, 1.5, 5.75, 9.0, 1.1,
                  "Verdict",
                  "The first four positions do the work, not their content. "
                  "Table 1, p. 5.",
                  fill=LIGHT, border=GREEN_OK,
                  label_size=15, sub_size=14, sub_color=TEXT,
                  label_color=GREEN_OK)
    return s


# --- The mechanism -------------------------------------------------------


def s_section_mechanism(prs):
    s = _blank(prs)
    _box(s, 0, 0, 13.33, 7.5, fill=NAVY, line=NAVY)
    _plain_text(s, "The mechanism", 0.7, 3.0, 12.0, 1.2,
                size=54, bold=True, color=BG)
    _box(s, 0.7, 4.1, 6.0, 0.05, fill=ACCENT, line=ACCENT)
    _plain_text(s, "Sinks + Rolling KV cache",
                0.7, 4.3, 12.0, 0.5, size=22, color=BG)
    return s


def s_cache_layout(prs):
    s = _blank(prs)
    _title(s, "The KV cache: 4 sink tokens + rolling window")
    # 4 sink cells
    left = 0.9
    top = 2.1
    cell = 0.7
    for i in range(4):
        x = left + i * (cell + 0.05)
        _box(s, x, top, cell, 0.9, fill=ACCENT, line=ACCENT)
        _plain_text(s, str(i), x, top + 0.25, cell, 0.5,
                    size=22, bold=True, color=BG, align=PP_ALIGN.CENTER)
    _plain_text(s, "SINKS", left + 0.05, top + 1.0, 2.5, 0.35,
                size=12, bold=True, color=ACCENT)
    # gap
    _plain_text(s, "…", left + 4 * (cell + 0.05) + 0.1,
                top + 0.15, 0.6, 0.6,
                size=32, bold=True, color=MUTED, align=PP_ALIGN.CENTER)
    _plain_text(s, "(evicted)", left + 4 * (cell + 0.05) + 0.1,
                top + 1.0, 1.0, 0.35, size=12, color=MUTED)
    # rolling window cells
    left2 = left + 4 * (cell + 0.05) + 1.0
    labels = ["T-6", "T-5", "T-4", "T-3", "T-2", "T-1", "T"]
    for i, lbl in enumerate(labels):
        x = left2 + i * (cell + 0.05)
        _box(s, x, top, cell, 0.9, fill=BLUE, line=BLUE)
        _plain_text(s, lbl, x, top + 0.3, cell, 0.5,
                    size=13, bold=True, color=BG, align=PP_ALIGN.CENTER)
    _plain_text(s, "ROLLING WINDOW (last L tokens)",
                left2, top + 1.0, 6.0, 0.35,
                size=12, bold=True, color=BLUE)
    _bullets(s, [
        "Keep the first S tokens (S = 4 works everywhere) as sinks.",
        "Keep the last L tokens as the rolling window.",
        "Everything in between is evicted.",
        "Cache size = S + L; independent of stream length.",
    ], top_in=3.85, size=20)
    return s


def s_position_reindex(prs):
    s = _blank(prs)
    _title(s, "The trick: positions are indexed inside the cache")
    _bullets(s, [
        "Naive attempt: use each token’s position in the original text.",
        "That does not work: the sinks and the rolling window are "
        "far apart in the text but adjacent in the cache.",
        "StreamingLLM re-indexes position IDs inside the cache.",
        "Compatible with RoPE and ALiBi, the two dominant positional "
        "encodings in 2024.",
    ], top_in=1.55, size=20)
    _labelled_box(s, 0.7, 4.7, 12.0, 2.2,
                  "Concrete example (paper §3.2, p. 5)",
                  "Cache contains tokens whose original positions are "
                  "[0, 1, 2, 3, 6, 7, 8].\n"
                  "The model is asked to predict position 9.\n"
                  "It receives cache-local positions [0, 1, 2, 3, 4, 5, 6, 7], "
                  "not [0, 1, 2, 3, 6, 7, 8, 9].\n"
                  "RoPE keys are cached pre-rotation, then rotated at decode "
                  "using the cache-local index.",
                  fill=LIGHT, border=BLUE,
                  label_size=15, sub_size=14, sub_color=TEXT)
    return s


def s_sink_token_pretraining(prs):
    s = _blank(prs)
    _title(s, "Refinement: pre-train with a dedicated sink token")
    _bullets(s, [
        "If we can control pre-training, we can train the model to "
        "concentrate its excess attention on one specific token.",
        "Prepend a single learnable placeholder to every pre-training "
        "sample. Call it the sink token.",
        "One sink token now suffices at inference, instead of the "
        "four accidental ones.",
    ], top_in=1.55, size=19)
    headers = ["Cache config (160M model, PG19)", "0+1024", "1+1023",
               "2+1022", "4+1020"]
    rows = [
        ["Vanilla",         "27.87", "18.49", "18.05", "18.05"],
        ["Zero Sink",       "29214", "19.90", "18.27", "18.01"],
        ["Learnable Sink",  "1235",  "18.01", "18.01", "18.02"],
    ]
    _table(s, headers, rows, left_in=1.2, top_in=4.35,
           col_widths_in=[3.9, 1.6, 1.6, 1.6, 1.6],
           head_size=13, cell_size=13, row_height_in=0.4)
    _plain_text(s,
                "Table 3, p. 6.  \"Zero Sink\" = SoftMax-off-by-One "
                "(Miller, 2023). ",
                1.2, 6.15, 11.0, 0.35, size=13, color=MUTED)
    return s


# --- Results -------------------------------------------------------------


def s_section_results(prs):
    s = _blank(prs)
    _box(s, 0, 0, 13.33, 7.5, fill=NAVY, line=NAVY)
    _plain_text(s, "Results", 0.7, 3.0, 12.0, 1.2,
                size=54, bold=True, color=BG)
    _box(s, 0.7, 4.1, 6.0, 0.05, fill=ACCENT, line=ACCENT)
    _plain_text(s, "Perplexity · Throughput · Streaming QA",
                0.7, 4.3, 12.0, 0.5, size=22, color=BG)
    return s


def s_perplexity_result(prs):
    s = _blank(prs)
    _title(s, "Result 1: perplexity stays flat past 4 million tokens")
    _bullets(s, [
        "Concatenated PG19 test set (100 long books).",
        "Cache = 2048 for Llama-2 · 1024 for MPT, Falcon, Pythia.",
        "Perplexity stable up to 4M+ tokens across every family "
        "(Llama-2-{7,13,70}B · MPT-{7,30}B · Falcon-{7,40}B · "
        "Pythia-{2.8,6.9,12}B).",
        "Matches the sliding-window-with-recomputation baseline; "
        "beats dense (OOMs past training window) and window "
        "attention (PPL explodes).",
    ], top_in=1.55, size=20)
    _labelled_box(s, 0.7, 4.9, 12.0, 1.9,
                  "The single strongest table",
                  "Llama-2-13B, PG19, 65K tokens (Table 1, p. 5):\n"
                  "  Window attention        →  PPL 5158.07\n"
                  "  StreamingLLM (4+1020)   →  PPL 5.40\n"
                  "  Linebreak sinks         →  PPL 5.60  (sinks are positional!)",
                  fill=LIGHT, border=GREEN_OK,
                  label_size=15, sub_size=14, sub_color=TEXT,
                  label_color=GREEN_OK)
    return s


def s_throughput_result(prs):
    s = _blank(prs)
    _title(s, "Result 2: up to 22.2× faster than window-with-recomputation")
    _bullets(s, [
        "Single NVIDIA A6000 · Hugging Face Transformers · batch 1.",
        "The sliding-window-with-recomputation baseline is quadratic; "
        "StreamingLLM is linear.",
    ], top_in=1.55, size=20)
    headers = ["Cache size", "Sliding+Recompute (ms)", "StreamingLLM (ms)",
               "Speed-up"]
    rows = [
        ["256",  "2355", "106", "22.2×"],
        ["512",  "860",  "75",  "11.5×"],
        ["1024", "361",  "60",  "6.0×"],
        ["2048", "169",  "52",  "3.3×"],
        ["4096", "99",   "48",  "2.1×"],
    ]
    _table(s, headers, rows, left_in=1.5, top_in=3.6,
           col_widths_in=[2.0, 3.4, 2.6, 1.6],
           head_size=13, cell_size=13, row_height_in=0.42)
    _plain_text(s,
                "Llama-2-13B, per-token decode latency; Fig. 10, p. 9. "
                "Memory footprint essentially unchanged.",
                1.5, 6.35, 11.0, 0.4, size=12, color=MUTED)
    return s


def s_streaming_qa(prs):
    s = _blank(prs)
    _title(s, "Result 3: streaming QA is now actually usable")
    _bullets(s, [
        "Setup: concatenate all ARC-{Easy,Challenge} questions and "
        "feed them as one long stream. Score exact-match on each answer.",
    ], top_in=1.55, size=19)
    headers = ["Model / policy", "ARC-Easy", "ARC-Challenge"]
    rows = [
        ["Llama-2-7B-Chat  one-shot",       "71.25", "53.16"],
        ["Llama-2-7B-Chat  window",         "3.58",  "1.39"],
        ["Llama-2-7B-Chat  StreamingLLM",   "71.34", "55.03"],
        ["Llama-2-13B-Chat one-shot",       "78.16", "63.31"],
        ["Llama-2-13B-Chat window",         "0.25",  "0.34"],
        ["Llama-2-13B-Chat StreamingLLM",   "80.89", "65.61"],
        ["Llama-2-70B-Chat one-shot",       "91.29", "78.50"],
        ["Llama-2-70B-Chat window",         "0.12",  "0.32"],
        ["Llama-2-70B-Chat StreamingLLM",   "91.37", "80.20"],
    ]
    _table(s, headers, rows, left_in=2.7, top_in=2.55,
           col_widths_in=[5.0, 1.6, 2.2],
           head_size=13, cell_size=12, row_height_in=0.32)
    _plain_text(s,
                "Cache 1024.  Dense OOMs.  Table 5, p. 8.",
                2.7, 5.85, 10.0, 0.35, size=13, color=MUTED)
    return s


# --- Limits --------------------------------------------------------------


def s_limits(prs):
    s = _blank(prs)
    _title(s, "What StreamingLLM does not do")
    _bullets(s, [
        ("It does not extend the context window. If the answer is "
         "older than the rolling window, it is gone.", True, HIGHLIGHT),
        ("On LongBench, StreamingLLM 4+3496 underperforms the default "
         "truncation baseline on all six tasks, because it loses the "
         "initial prompt. Table 8, p. 17.", False, TEXT),
        ("Bigger cache does not always mean lower perplexity. For "
         "Llama-2-7B, 4+2044 gives 9.08 PPL but 4+4092 gives 9.59. "
         "Table 6, p. 9.", False, TEXT),
        ("No comparison with attention-score-based eviction "
         "(H2O, SnapKV, FastGen, Keyformer). Positional only.",
         False, TEXT),
        ("Learnable-sink pre-training was validated at 160 M parameters, "
         "not at 7 B, 13 B, or 70 B.", False, TEXT),
    ], top_in=1.55, size=18)
    return s


# --- What I would change / bridge to Task 2 -----------------------------


def s_what_id_change(prs):
    s = _blank(prs)
    _title(s, "What I would change")
    _bullets(s, [
        "Layer-specific sink budgets. The first two layers barely "
        "need sinks; deeper layers need more.",
        "Attention-score-based eviction inside the rolling window: "
        "combine StreamingLLM’s positional sinks with H2O-style "
        "content-aware eviction.",
        "Test on real streaming traffic (LMSYS-Chat, ShareGPT). "
        "The paper only tests on PG19 books.",
        "Sink-token pre-training at 7B+. The strongest §3.3 claim "
        "still awaits validation at scale.",
    ], top_in=1.55, size=20)
    _labelled_box(s, 0.7, 5.2, 12.0, 1.7,
                  "Same idea, one level up the stack",
                  "StreamingLLM chooses positional KV-cache eviction. "
                  "My final project (Task 2) chooses cost-aware "
                  "response-cache eviction (GDSF on GPTCache). Both "
                  "attack the same underlying problem, dumb eviction "
                  "wastes capacity, at different layers.",
                  fill=LIGHT, border=BLUE,
                  label_size=16, sub_size=15, sub_color=TEXT)
    return s


# --- Conclusion + Thank you ---------------------------------------------


def s_conclusion(prs):
    s = _blank(prs)
    _title(s, "Conclusion")
    _bullets(s, [
        "Attention sinks are real, universal, and structural, not semantic.",
        "Keeping 4 initial tokens + a rolling window is enough to make "
        "LLMs stream stably to 4M+ tokens.",
        "22.2× faster than the only sane baseline, with the same memory.",
        "Adopted upstream: NVIDIA TensorRT-LLM, HF Transformers, "
        "Intel Extension for Transformers, MLC LLM.",
        "But: it does not extend context, and it does not compete with "
        "attention-score-based eviction. Those are the frontiers.",
    ], top_in=1.55, size=20)
    _labelled_box(s, 0.7, 5.5, 12.0, 1.35,
                  "One-line take",
                  "A robust empirical fix for a softmax quirk: small, "
                  "easy to implement, and honest about its own ceiling.",
                  fill=LIGHT, border=ACCENT,
                  label_size=16, sub_size=15, sub_color=TEXT,
                  label_color=ACCENT)
    return s


def s_thanks(prs):
    s = _blank(prs)
    _box(s, 0, 0, 13.33, 7.5, fill=NAVY, line=NAVY)
    _plain_text(s, "Thank you", 0.7, 2.7, 12.0, 1.5,
                size=64, bold=True, color=BG, align=PP_ALIGN.CENTER)
    _box(s, 4.7, 4.4, 3.9, 0.05, fill=ACCENT, line=ACCENT)
    _plain_text(s, "Questions welcome.", 0.7, 4.6, 12.0, 0.5,
                size=22, color=BG, align=PP_ALIGN.CENTER)
    _plain_text(s, "Nissim Brami  ·  nissimbrami@post.bgu.ac.il",
                0.7, 6.4, 12.0, 0.4, size=14, color=BG,
                align=PP_ALIGN.CENTER)
    _plain_text(s, "Caching in LLMs  ·  Ben-Gurion University",
                0.7, 6.75, 12.0, 0.4, size=12, color=BG,
                align=PP_ALIGN.CENTER)
    return s


# ---------- Assemble ------------------------------------------------------


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    builders = [
        s_title,
        s_agenda,
        s_section_basics,
        s_llm_decoding,
        s_course_anchor,
        s_naive_options,
        s_section_observation,
        s_attention_sink_phenomenon,
        s_softmax_argument,
        s_semantics_or_position,
        s_section_mechanism,
        s_cache_layout,
        s_position_reindex,
        s_sink_token_pretraining,
        s_section_results,
        s_perplexity_result,
        s_throughput_result,
        s_streaming_qa,
        s_limits,
        s_what_id_change,
        s_conclusion,
        s_thanks,
    ]

    for i, b in enumerate(builders, start=1):
        slide = b(prs)
        # Page numbers on all but the two full-navy dividers/title/thanks
        if b not in (s_title, s_section_basics, s_section_observation,
                     s_section_mechanism, s_section_results, s_thanks):
            tb = slide.shapes.add_textbox(Inches(12.7), Inches(7.05),
                                          Inches(0.6), Inches(0.3))
            tf = tb.text_frame
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT
            r = p.add_run()
            r.text = f"{i}"
            r.font.name = "Calibri"
            r.font.size = Pt(9)
            r.font.color.rgb = MUTED
            tb2 = slide.shapes.add_textbox(Inches(0.5), Inches(7.05),
                                           Inches(6.0), Inches(0.3))
            tf2 = tb2.text_frame
            p2 = tf2.paragraphs[0]
            p2.alignment = PP_ALIGN.LEFT
            r2 = p2.add_run()
            r2.text = "Nissim Brami · Caching in LLMs · BGU"
            r2.font.name = "Calibri"
            r2.font.size = Pt(9)
            r2.font.color.rgb = MUTED

    prs.save(str(OUT))
    print(f"Wrote {OUT}  ·  {len(prs.slides)} slides")


if __name__ == "__main__":
    build()
