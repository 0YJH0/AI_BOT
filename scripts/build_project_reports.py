"""Build the four audited Markdown reports as Chinese PDFs; no simulator imports.

Example: python scripts/build_project_reports.py --render
Dependencies: reportlab, pypdf, pymupdf, Pillow. Poppler is preferred if available.
Use --font to supply a distributable Chinese TrueType font on another OS.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "docs/reports_20261013"
STEMS = ["01_project_challenges", "02_project_issue_report", "03_project_overview", "04_project_one_hour_talk"]
NAVY = colors.HexColor("#193B58")
BLUE = colors.HexColor("#246788")
GRAY = colors.HexColor("#536572")
LIGHT = colors.HexColor("#EEF4F7")
WIDTH = A4[0] - 38 * mm


def clean_text(text: str) -> str:
    # Keep typographic ranges portable in the embedded Windows Chinese font.
    return text.translate(str.maketrans({"–": "-", "—": "-", "‑": "-", "−": "-"}))


def rich(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r'<font color="#246788">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r'<font color="#193B58">\1</font>', text)
    return text


class ReportDoc(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in {"H1", "H2", "H3"}:
            level = {"H1": 0, "H2": 1, "H3": 2}[flowable.style.name]
            key = f"heading-{getattr(self, '_heading_count', 0)}"
            self._heading_count = getattr(self, "_heading_count", 0) + 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=level, closed=False)


def styles():
    base = dict(fontName="ReportCN", fontSize=10, leading=14.5, wordWrap="CJK", alignment=TA_LEFT,
                textColor=colors.HexColor("#253746"), splitLongWords=True)
    return {
        "body": ParagraphStyle("Body", **base, spaceAfter=5),
        "reference": ParagraphStyle("Reference", **{**base, "fontSize": 8.5, "leading": 12}, spaceAfter=4),
        "h1": ParagraphStyle("H1", **{**base, "fontSize": 19, "leading": 27}, textTransform=None,
                             spaceBefore=8, spaceAfter=16, keepWithNext=True),
        "h2": ParagraphStyle("H2", **{**base, "fontSize": 14, "leading": 21, "textColor": NAVY},
                             spaceBefore=11, spaceAfter=7, keepWithNext=True),
        "h3": ParagraphStyle("H3", **{**base, "fontSize": 11, "leading": 17, "textColor": BLUE},
                             spaceBefore=7, spaceAfter=4, keepWithNext=True),
        "cell": ParagraphStyle("Cell", **{**base, "fontSize": 8, "leading": 11}, spaceAfter=0),
        "headcell": ParagraphStyle("HeadCell", **{**base, "fontSize": 8, "leading": 11, "textColor": colors.white}),
        "code": ParagraphStyle("Code", **{**base, "fontSize": 8, "leading": 12},
                               backColor=LIGHT, borderPadding=7, spaceBefore=4, spaceAfter=10),
    }


def markdown_flow(text: str):
    ss = styles()
    lines = text.splitlines()
    flow = []
    i = 0
    references = False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith(("```", "~~~")):
            fence = line[:3]
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith(fence):
                code.append(html.escape(clean_text(lines[i])).replace(" ", "&#160;"))
                i += 1
            flow.append(Paragraph("<br/>".join(code), ss["code"]))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells):
                    rows.append(cells)
                i += 1
            n = len(rows[0])
            # Give code-path columns enough room while preserving readable result columns.
            ratios = [1.0] * n
            if n == 3:
                ratios = [1.0, 1.8, 2.1]
            elif n == 4 and rows[0][0] == "编号":
                ratios = [.85, 1.7, .55, 2.3]
            elif n == 6:
                ratios = [1.6, .75, .9, 1.2, 1.0, .9]
            elif n == 5:
                ratios = [1.5, 1, 1, 1, 1.1]
            matrix = [[Paragraph(rich(c), ss["headcell" if j == 0 else "cell"]) for c in row] for j, row in enumerate(rows)]
            table = Table(matrix, colWidths=[WIDTH * r / sum(ratios) for r in ratios], repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW", (0, 0), (-1, 0), .5, BLUE),
            ]))
            flow.extend([KeepTogether([table]) if len(rows) <= 5 else table, Spacer(1, 9)])
            continue
        match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if match:
            if "证据索引" in match.group(2):
                references = True
            flow.append(Paragraph(rich(match.group(2)), ss[f"h{len(match.group(1))}"]))
            i += 1
            continue
        if re.match(r"^[-*]\s", line) or re.match(r"^\d+\.\s", line):
            flow.append(Paragraph(rich(line), ss["reference" if references else "body"]))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||```|~~~|[-*]\s|\d+\.\s)", lines[i].strip()):
            paragraph.append(lines[i].strip())
            i += 1
        flow.append(Paragraph(rich(" ".join(paragraph)), ss["reference" if references else "body"]))
    return flow


def build_pdf(source: Path, dest: Path):
    title = source.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
    short_title = title.split("：")[0]

    def page_header(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(BLUE)
        canvas.setLineWidth(1.0)
        canvas.line(19 * mm, A4[1] - 16 * mm, A4[0] - 19 * mm, A4[1] - 16 * mm)
        canvas.setFont("ReportCN", 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(19 * mm, A4[1] - 12 * mm, f"ai_BOT  /  {short_title}")
        canvas.drawRightString(A4[0] - 19 * mm, A4[1] - 12 * mm, "技术报告 · 2026-10-13")
        canvas.setFont("ReportCN", 8)
        canvas.drawString(19 * mm, 12 * mm, "基于代码与已保存结果；未验证项单独说明")
        canvas.drawRightString(A4[0] - 19 * mm, 12 * mm, f"{doc.page}")
        canvas.restoreState()

    doc = ReportDoc(str(dest), pagesize=A4, title=title, author="ai_BOT Project",
                    leftMargin=19 * mm, rightMargin=19 * mm, topMargin=22 * mm, bottomMargin=21 * mm)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0)
    doc.addPageTemplates(PageTemplate(id="Report", frames=[frame], onPage=page_header))
    doc.build(markdown_flow(source.read_text(encoding="utf-8")))


def snapshot():
    paths = [
        "results/benchmark/benchmark_report.json",
        "results/ppo_evaluation_dagger2_ppo_bc_128step_100iter_final_256/benchmark_report.json",
        "results/ppo/ik_abs_dagger2_ppo_bc_128step_1024env_100iter/training_summary.json",
        "dataset/vla_execution_aligned_full_20261011_truth_v9/collection_summary.json",
        *[f"exports/lerobot/pick_lift_execution_aligned_20261011_truth_v9/{split}/meta/info.json" for split in ("train", "validation", "test")],
        "results/vla/medium64_parallel16_20261013_160323_916/comparison_summary.json",
        "results/vla/parallel_capacity_20261013_155045_631_standard16/capacity_summary.json",
        "results/vla/medium64_parallel4_20261013_153842_856/v18_r32_chunk4_execute1/camera_wave0.json",
        "results/vla/medium64_parallel4_20261013_154409_905/v18_r32_chunk4_execute1/camera_wave0.json",
    ]

    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            marker = "/ai_BOT/"
            if marker.lower() in normalized.lower():
                index = normalized.lower().index(marker.lower())
                return "<PROJECT_ROOT>/" + normalized[index + len(marker):]
        return value

    records = []
    for name in paths:
        data = (ROOT / name).read_bytes()
        records.append({"source": name, "source_sha256": hashlib.sha256(data).hexdigest(),
                        "snapshot": portable(json.loads(data.decode("utf-8-sig")))})
    payload = {"report_date": "2026-10-13", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "note": "Original JSON SHA256 hashes; snapshot project-local absolute paths are normalized. No new experiments.",
               "success_criteria_caveat": "Historical state benchmarks use center height above table; latest VLA adds actual lift >= 0.10 m.",
               "sources": records}
    (SOURCES / "evidence_snapshot.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_and_render(pdf: Path, render_root: Path, render: bool):
    from pypdf import PdfReader
    reader = PdfReader(pdf)
    texts = [page.extract_text() for page in reader.pages]
    assert texts and all(len(t.strip()) > 50 for t in texts), f"Empty/near-empty page: {pdf}"
    assert all("\ufffd" not in t for t in texts), f"Replacement glyph: {pdf}"
    assert "项目" in "".join(texts), f"Chinese extraction failed: {pdf}"
    result = {"pdf": str(pdf.relative_to(ROOT)), "pages": len(texts), "text_chars": sum(map(len, texts)),
              "bytes": pdf.stat().st_size, "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}
    if not render:
        return result
    from PIL import Image, ImageDraw
    import fitz
    folder = render_root / pdf.stem
    folder.mkdir(parents=True, exist_ok=True)
    poppler = shutil.which("pdftoppm")
    if poppler:
        subprocess.run([poppler, "-r", "100", "-png", str(pdf), str(folder / "page")], check=True)
        pages = sorted(folder.glob("page-*.png"))
        result["renderer"] = "Poppler pdftoppm"
    else:
        doc = fitz.open(pdf)
        pages = []
        for index, page in enumerate(doc):
            path = folder / f"page-{index + 1:03}.png"
            page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False).save(path)
            pages.append(path)
        result["renderer"] = "PyMuPDF (Poppler unavailable)"
    # Small sheets for page-by-page layout review; full page PNGs remain available.
    sheets = []
    for start in range(0, len(pages), 6):
        batch = pages[start:start + 6]
        tile_w, tile_h = 420, 620
        sheet = Image.new("RGB", (tile_w * 3, tile_h * 2), "#DCE4EA")
        draw = ImageDraw.Draw(sheet)
        for offset, path in enumerate(batch):
            im = Image.open(path).convert("RGB")
            im.thumbnail((400, 576))
            x, y = (offset % 3) * tile_w + 10, (offset // 3) * tile_h + 28
            sheet.paste(im, (x, y))
            draw.text((x, y - 20), f"Page {start + offset + 1}", fill="black")
        target = folder / f"contact-{start // 6 + 1:02}.png"
        sheet.save(target)
        sheets.append(str(target.relative_to(ROOT)))
    result["contact_sheets"] = sheets
    # A geometric diagnostic complements (does not replace) visual review.
    doc = fitz.open(pdf)
    violations = []
    for i, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                x0, y0, x1, y1 = line["bbox"]
                if x0 < 17 * mm or x1 > A4[0] - 17 * mm or y0 < 5 or y1 > A4[1] - 5:
                    violations.append({"page": i + 1, "bbox": line["bbox"]})
    result["out_of_bounds_text"] = violations
    assert not violations, f"Out-of-bounds text: {pdf}: {violations[:2]}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/simhei.ttf"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/pdf")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--snapshot", action="store_true", help="Refresh evidence from original local results")
    args = parser.parse_args()
    if not args.font.is_file():
        raise FileNotFoundError("Pass --font with a Chinese TrueType font")
    pdfmetrics.registerFont(TTFont("ReportCN", str(args.font)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_root = ROOT / "tmp/pdfs/reports_20261013"
    render_root.mkdir(parents=True, exist_ok=True)
    if args.snapshot:
        snapshot()
    records = []
    for stem in STEMS:
        dest = args.output_dir / f"{stem}.pdf"
        build_pdf(SOURCES / f"{stem}.md", dest)
        records.append(validate_and_render(dest, render_root, args.render))
        print(json.dumps(records[-1], ensure_ascii=False))
    (render_root / "qa_report.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
