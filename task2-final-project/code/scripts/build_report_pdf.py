"""Compile docs/report-draft.md into docs/report.pdf using reportlab.

Handles a subset of Markdown sufficient for the report:
  - # / ## / ### headings
  - Paragraphs (blank-line separated)
  - Unordered / ordered lists
  - `inline code`, **bold**, *italic*
  - Fenced code blocks (```)
  - Tables (| ... | ... |)
  - Image references of the form:
        [Figure N: caption ...]
    which are replaced by the actual PNG in results/plots/figN_*.png,
    followed by the caption.

Author: Nissim Brami
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, KeepTogether,
    Table, TableStyle, Preformatted,
)


HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
PLOTS = HERE.parent / "results" / "plots"

REPORT_MD = DOCS / "report-draft.md"
REPORT_PDF = DOCS / "report.pdf"


# ---------- Styles -------------------------------------------------------

styles = getSampleStyleSheet()

BODY = ParagraphStyle(
    "Body", parent=styles["BodyText"],
    fontName="Helvetica", fontSize=10, leading=13,
    alignment=TA_JUSTIFY, spaceAfter=6,
)
H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"],
    fontName="Helvetica-Bold", fontSize=18, leading=22,
    spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#0B1F3A"),
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontName="Helvetica-Bold", fontSize=13, leading=17,
    spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#0B1F3A"),
)
H3 = ParagraphStyle(
    "H3", parent=styles["Heading3"],
    fontName="Helvetica-Bold", fontSize=11, leading=14,
    spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#2E86AB"),
)
CAPTION = ParagraphStyle(
    "Caption", parent=BODY,
    fontSize=9, leading=11, alignment=TA_LEFT, textColor=colors.HexColor("#333333"),
    spaceBefore=2, spaceAfter=10, fontName="Helvetica-Oblique",
)
CODE = ParagraphStyle(
    "Code", parent=BODY,
    fontName="Courier", fontSize=8, leading=10,
    backColor=colors.HexColor("#F4F6F8"), leftIndent=8, rightIndent=8,
    spaceBefore=4, spaceAfter=6, textColor=colors.HexColor("#111111"),
)


# ---------- Inline markdown ----------------------------------------------

_bold = re.compile(r"\*\*(.+?)\*\*")
_italic = re.compile(r"(?<!\*)\*(?!\*)(.+?)\*(?!\*)")
_code = re.compile(r"`([^`]+)`")


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = _bold.sub(r"<b>\1</b>", text)
    text = _italic.sub(r"<i>\1</i>", text)
    text = _code.sub(r'<font face="Courier" size="9">\1</font>', text)
    # Strip LaTeX math delimiters, keep the content readable.
    text = text.replace("$", "")
    text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1) / (\2)", text)
    text = re.sub(r"\\cdot", "·", text)
    text = re.sub(r"\\alpha", "α", text)
    text = re.sub(r"\\beta", "β", text)
    text = re.sub(r"\\lambda", "λ", text)
    text = re.sub(r"\\tau", "τ", text)
    return text


# ---------- Figure replacement -------------------------------------------

FIGURE_MAP = {
    "1": "fig1_hit_rate_vs_cache_size.png",
    "2": "fig2_cwhr_vs_cache_size.png",
    "3": "fig3_dollar_savings.png",
    "4": "fig4_latency_cdf.png",
    "5": "fig5_ablation_heatmap.png",
    "6": "fig6_workload_sensitivity.png",
    "7": "fig7_memory_overhead.png",
    "8": "fig8_parameter_sensitivity.png",
}


def figure_flowables(num: str, caption: str) -> List:
    path = PLOTS / FIGURE_MAP.get(num, "")
    if not path.exists():
        return [Paragraph(f"<i>[Figure {num} missing: {path}]</i>", BODY)]
    img = Image(str(path))
    # Fit width to page (A4 minus margins ~ 16 cm)
    max_w = 15.5 * cm
    if img.imageWidth > 0:
        scale = max_w / img.imageWidth
        img.drawWidth = max_w
        img.drawHeight = img.imageHeight * scale
        # Cap height to keep on-page
        max_h = 9 * cm
        if img.drawHeight > max_h:
            r = max_h / img.drawHeight
            img.drawHeight = max_h
            img.drawWidth *= r
    cap = Paragraph(f"<b>Figure {num}.</b> {inline(caption)}", CAPTION)
    return [KeepTogether([img, cap])]


FIG_RE = re.compile(r"^\[Figure\s+(\d+)\s*[:.]?\s*(.*?)\]\s*$")


# ---------- Table parsing ------------------------------------------------

def parse_table_block(lines: List[str], start: int):
    """Return (Table flowable, next_index) or (None, start)."""
    if start >= len(lines) or "|" not in lines[start]:
        return None, start
    if start + 1 >= len(lines) or not re.match(r"^\|?\s*[:-]+", lines[start + 1]):
        return None, start

    header = [c.strip() for c in lines[start].strip().strip("|").split("|")]
    i = start + 2
    rows = []
    while i < len(lines) and "|" in lines[i] and lines[i].strip():
        rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
        i += 1

    data = [header] + rows
    # Sanitize cells
    data = [[inline(c) for c in row] for row in data]
    styled = [[Paragraph(c, BODY) for c in row] for row in data]

    # column widths auto
    n_cols = len(header)
    total = 16 * cm
    col_w = [total / n_cols] * n_cols
    tbl = Table(styled, colWidths=col_w, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1F3A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#888888")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F4F6F8")]),
    ]))
    return tbl, i


# ---------- Main parser --------------------------------------------------

def render(md: str) -> list:
    lines = md.splitlines()
    flow: list = []
    i = 0
    in_code = False
    code_buf: list = []

    while i < len(lines):
        line = lines[i]

        # Fenced code
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True; code_buf = []
            else:
                flow.append(Preformatted("\n".join(code_buf), CODE))
                in_code = False
            i += 1; continue

        if in_code:
            code_buf.append(line)
            i += 1; continue

        # Figure reference
        fig_match = FIG_RE.match(line.strip())
        if fig_match:
            flow.extend(figure_flowables(fig_match.group(1), fig_match.group(2)))
            i += 1; continue

        # Horizontal rule
        if line.strip() in ("---", "***"):
            flow.append(Spacer(1, 0.15 * cm))
            i += 1; continue

        # Headings
        if line.startswith("### "):
            flow.append(Paragraph(inline(line[4:].strip()), H3))
            i += 1; continue
        if line.startswith("## "):
            flow.append(Paragraph(inline(line[3:].strip()), H2))
            i += 1; continue
        if line.startswith("# "):
            flow.append(Paragraph(inline(line[2:].strip()), H1))
            i += 1; continue

        # Table
        if "|" in line and line.strip().startswith("|"):
            tbl, nxt = parse_table_block(lines, i)
            if tbl is not None:
                flow.append(Spacer(1, 0.1 * cm))
                flow.append(tbl)
                flow.append(Spacer(1, 0.25 * cm))
                i = nxt; continue

        # Blank line
        if not line.strip():
            i += 1; continue

        # Lists
        if re.match(r"^\s*[-*]\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                i += 1
            for it in items:
                flow.append(Paragraph("• " + inline(it), BODY))
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            items = []
            n = 1
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append((n, re.sub(r"^\s*\d+\.\s+", "", lines[i])))
                n += 1; i += 1
            for num, it in items:
                flow.append(Paragraph(f"{num}. " + inline(it), BODY))
            continue

        # Paragraph — gather until blank
        para_lines = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and \
                not lines[i].startswith(("#", "```", "|")) and \
                not re.match(r"^\s*[-*]\s+", lines[i]) and \
                not re.match(r"^\s*\d+\.\s+", lines[i]) and \
                not FIG_RE.match(lines[i].strip()):
            para_lines.append(lines[i])
            i += 1
        text = " ".join(para_lines).strip()
        if text:
            flow.append(Paragraph(inline(text), BODY))

    return flow


def compile_report():
    md = REPORT_MD.read_text(encoding="utf-8")
    flow = render(md)

    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=A4,
        leftMargin=2.0 * cm, rightMargin=2.0 * cm,
        topMargin=2.0 * cm, bottomMargin=2.0 * cm,
        title="Cost-Aware GDSF Eviction for GPTCache",
        author="Nissim Brami",
    )

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(2 * cm, 1.2 * cm,
                          "Nissim Brami · Caching in LLMs · BGU")
        canvas.drawRightString(doc_.pagesize[0] - 2 * cm, 1.2 * cm,
                               f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)
    print(f"Wrote {REPORT_PDF}")


if __name__ == "__main__":
    compile_report()
