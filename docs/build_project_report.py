"""Documentation-only detailed project PDF; no application imports/dependencies.

Run with the supplied document runtime (ReportLab/pypdf), after Docker gates and
schema/report exports. Never used by catan, rl or a rollout worker.
"""

from hashlib import sha256
import html
import json
from pathlib import Path
import re
import textwrap

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    Preformatted,
    Flowable,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/Catan_Project_Decisions_2026-10-11.pdf"
INK = colors.HexColor("#17324d")
TEAL = colors.HexColor("#007e83")
MUTED = colors.HexColor("#596675")
PALE = colors.HexColor("#eef4f7")
styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        "BodyReport",
        fontName="Helvetica",
        fontSize=9.4,
        leading=13.6,
        spaceAfter=7,
        textColor=INK,
        splitLongWords=True,
    )
)
styles.add(
    ParagraphStyle(
        "Chapter",
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=24,
        spaceAfter=15,
        textColor=INK,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        "SubReport",
        fontName="Helvetica-Bold",
        fontSize=11.7,
        leading=15.3,
        spaceBefore=10,
        spaceAfter=7,
        textColor=TEAL,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        "SmallReport",
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        spaceAfter=5,
        textColor=MUTED,
        splitLongWords=True,
    )
)
styles.add(
    ParagraphStyle(
        "CellReport",
        fontName="Helvetica",
        fontSize=8,
        leading=10.6,
        textColor=INK,
        splitLongWords=True,
    )
)
styles.add(
    ParagraphStyle(
        "HeaderReport",
        parent=styles["CellReport"],
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )
)
styles.add(
    ParagraphStyle(
        "CodeReport",
        fontName="Courier",
        fontSize=7.8,
        leading=10.5,
        backColor=PALE,
        borderPadding=8,
        spaceBefore=5,
        spaceAfter=10,
    )
)
styles.add(
    ParagraphStyle(
        "CoverTitle",
        fontName="Helvetica-Bold",
        fontSize=34,
        leading=40,
        textColor=INK,
        spaceAfter=22,
    )
)
styles.add(
    ParagraphStyle(
        "CoverSub",
        fontName="Helvetica",
        fontSize=15,
        leading=21,
        textColor=TEAL,
        spaceAfter=20,
    )
)
styles.add(
    ParagraphStyle(
        "TOCReport",
        fontName="Helvetica",
        fontSize=9.1,
        leading=11,
        leftIndent=0,
        firstLineIndent=0,
        textColor=INK,
        spaceBefore=0,
    )
)


def clean(text):
    return (
        text.replace("\ufeff", "")
        .replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2192", "->")
        .replace("\u00a0", " ")
    )


def rich(text):
    text = clean(text)
    parts = []
    pos = 0
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
        parts.append(html.escape(text[pos : match.start()]))
        label, target = match.groups()
        if target.startswith(("https://", "http://")):
            parts.append(
                '<link href="'
                + html.escape(target, quote=True)
                + '"><u>'
                + html.escape(label)
                + "</u></link>"
            )
        else:
            parts.append(html.escape(label))
        pos = match.end()
    parts.append(html.escape(text[pos:]))
    result = "".join(parts)
    result = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", result)
    result = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', result)
    return result


def para(text, style="BodyReport"):
    return Paragraph(rich(text), styles[style])


def table(rows):
    columns = len(rows[0])
    weights = [
        max(9, min(36, max(len(row[i]) for row in rows))) for i in range(columns)
    ]
    if columns > 5:
        weights[0] = max(weights[0], 20)
    widths = [512 * w / sum(weights) for w in weights]
    values = [
        [para(value, "HeaderReport" if row == 0 else "CellReport") for value in cells]
        for row, cells in enumerate(rows)
    ]
    result = Table(values, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, TEAL),
                ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor("#c8d4dc")),
            ]
        )
    )
    return result


class Architecture(Flowable):
    def __init__(self):
        super().__init__()
        self.width, self.height = 512, 95

    def draw(self):
        canvas = self.canv
        labels = (
            "Immutable\nGeometry/Board",
            "Owned\nGameState",
            "Validate +\ntransition",
            "Observe +\nlegal mask",
            "Caller\nchoice",
        )
        for i, label in enumerate(labels):
            x = i * 103
            canvas.setFillColor(PALE)
            canvas.setStrokeColor(TEAL)
            canvas.roundRect(x, 24, 96, 58, 6, stroke=1, fill=1)
            canvas.setFillColor(INK)
            canvas.setFont("Helvetica-Bold", 8)
            for j, line in enumerate(label.split("\n")):
                canvas.drawCentredString(x + 48, 60 - j * 13, line)
            if i < 4:
                canvas.setStrokeColor(TEAL)
                canvas.line(x + 96, 53, x + 103, 53)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(
            0,
            7,
            "Rules own true state; optional observations redact hidden information.",
        )


class Report(SimpleDocTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chapter_count = 0

    def beforeDocument(self):
        self.chapter_count = 0

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "Chapter":
            text = flowable.getPlainText()
            if text == "Contents":
                return
            key = "chapter" + str(self.chapter_count)
            self.chapter_count += 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, 0, False)
            self.notify("TOCEntry", (0, text, self.page, key))


def footer(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#d2dce3"))
        canvas.line(50, 754, 562, 754)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(
            50, 764, "CATAN ENGINE | ARCHITECTURE, DECISIONS AND PERFORMANCE"
        )
        canvas.line(50, 38, 562, 38)
        canvas.drawString(
            50, 25, "Verified project report | 2026-10-11 | Training deferred"
        )
        canvas.drawRightString(562, 25, str(doc.page))
    canvas.restoreState()


def chapter(story, title):
    if story and not isinstance(story[-1], PageBreak):
        story.append(PageBreak())
    story.append(para(title, "Chapter"))


def markdown(story, text, main=False):
    lines = clean(text).splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            i += 1
            continue
        if main and line.startswith("Version 2026-"):
            i += 1
            continue
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                for wrapped in textwrap.wrap(
                    lines[i], width=88, replace_whitespace=False, drop_whitespace=False
                ) or [""]:
                    code.append(wrapped)
                i += 1
            # Line-by-line paragraphs let large code blocks split safely.
            story.append(Preformatted("\n".join(code), styles["CodeReport"]))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells):
                    rows.append(cells)
                i += 1
            if rows and len({len(row) for row in rows}) == 1:
                story.extend((table(rows), Spacer(1, 10)))
            continue
        if line.startswith("## "):
            if main:
                chapter(story, line[3:])
            else:
                story.append(para(line[3:], "SubReport"))
            i += 1
            continue
        if line.startswith("### "):
            story.append(para(line[4:], "SubReport"))
            i += 1
            continue
        if re.match(r"^(- |\d+\. )", line):
            story.append(para(line))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not re.match(r"^(#|\||```|- |\d+\. )", lines[i].strip())
        ):
            paragraph.append(lines[i].strip())
            i += 1
        story.append(para(" ".join(paragraph)))


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    story = [
        Spacer(1, 72),
        para("Catan engine", "CoverTitle"),
        para("Architecture, decisions and performance", "CoverSub"),
    ]
    story.append(
        para(
            "A detailed audit of the implemented simulator, rule expansion, visibility contracts, measured optimizations and remaining roadmap."
        )
    )
    story.extend((Spacer(1, 20), Architecture(), Spacer(1, 24)))
    story.append(
        para(
            "71 documented material choices | Editable source + complete milestone appendices",
            "SubReport",
        )
    )
    story.append(
        para(
            "Prepared 2026-10-11 for the main Catan project. Current observation schema: v2, 1505 float32 values. Current action catalog: 254 slots plus conditional discards. Training remains on hold."
        )
    )
    story.append(
        para(
            "Historical records are explicitly dated. Performance measures simulator/representation work, not neural training. Original build-guide PDF is preserved.",
            "SmallReport",
        )
    )
    story.append(PageBreak())
    story.append(para("Contents", "Chapter"))
    toc = TableOfContents()
    toc.levelStyles = [styles["TOCReport"]]
    story.append(toc)
    story.append(PageBreak())
    core = ROOT / "docs/project-decisions.md"
    markdown(story, core.read_text(encoding="utf-8-sig"), main=True)
    chapter(story, "Appendix A - Exact current data contracts")
    schema = json.loads((ROOT / "output/full-review/schema.json").read_text())
    story.append(
        para(
            "Exported from the tested container source; stop offsets are exclusive. Integer arrays are independently owned by each state. Byte counts exclude object/RNG/board overhead."
        )
    )
    story.append(
        table(
            [["State array", "Shape", "Dtype", "Bytes (4 seats)"]]
            + [
                [
                    r["name"],
                    " x ".join(map(str, r["shape"])),
                    r["dtype"],
                    str(r["bytes"]),
                ]
                for r in schema["state_arrays"]
            ]
        )
    )
    story.extend(
        (
            Spacer(1, 12),
            para(
                "Observation v" + str(schema["observation_schema"]) + " groups",
                "SubReport",
            ),
        )
    )
    story.append(
        table(
            [["Group", "Shape", "Start", "Stop"]]
            + [
                [
                    r["name"],
                    " x ".join(map(str, r["shape"])),
                    str(r["start"]),
                    str(r["stop_exclusive"]),
                ]
                for r in schema["groups"]
            ]
        )
    )
    story.extend((Spacer(1, 12), para("Phases and timing", "SubReport")))
    story.append(
        table(
            [["Phase", "Value"]]
            + [[name, str(value)] for name, value in schema["phases"].items()]
        )
    )
    story.extend((Spacer(1, 12), para("Exact Year of Plenty catalog", "SubReport")))
    story.append(
        para(
            "Counts are wood/sheep/wheat/brick/ore. Legality masks filter vectors by exact available total and per-resource supply. Index order is frozen in the catalog."
        )
    )
    story.append(
        table(
            [["Index", "Selected five-resource counts"]]
            + [[str(r["index"]), str(r["resources"])] for r in schema["plenty_catalog"]]
        )
    )
    chapter(story, "Appendix B - Current rules and performance contract")
    markdown(story, (ROOT / "docs/expanded-engine.md").read_text(encoding="utf-8-sig"))
    chapter(story, "Appendix C - Evidence inventory and provenance")
    records = sorted((ROOT / "output/full-review").glob("*.json"))
    story.append(
        para(
            "Raw result files are local audit artifacts retained outside Git in output/full-review. Reports distinguish prototype, original minimal context, expanded baseline, paired optimization and final expanded verification. Profiling adds overhead; profile elapsed times are not benchmark rates."
        )
    )
    story.append(
        table(
            [["Report", "Purpose"]]
            + [
                [
                    p.name,
                    {
                        "improvement-prototypes.json": "Initial direct-mask/batch proposal comparison",
                        "original-system.json": "Previous minimal-rule workload context",
                        "expanded-before-optimization.json": "Expanded rules before holistic changes",
                        "paired-optimization.json": "Same-rule paired timing and exact replay",
                        "final-lookup.json": "Retained expanded-schema masks/batches",
                        "expanded-stress.json": "10000-game continuous expanded-rule check",
                        "schema.json": "Current container-exported addresses/storage",
                    }.get(p.name, "Retained verification evidence"),
                ]
                for p in records
            ]
        )
    )
    reports = {p.name: json.loads(p.read_text()) for p in records}
    if "expanded-stress.json" in reports:
        r = reports["expanded-stress.json"]
        story.append(para("Final expanded stress report", "SubReport"))
        selected = {k: v for k, v in r.items() if k not in ("capped_games", "failures")}
        story.append(Preformatted(json.dumps(selected, indent=2), styles["CodeReport"]))
    else:
        story.append(
            para(
                "Expanded stress report not yet present; this draft must be regenerated after completion."
            )
        )
    milestone_files = [
        ("Phase 3 state contract", "phase-three-plan.md"),
        ("Stages 1-3 audit", "stages-1-3-audit.md"),
        ("Manual discards", "manual-discards.md"),
        ("Owned-transition optimization", "engine-optimization.md"),
        ("Stage 4 minimal engine", "stage-four-engine.md"),
        ("Stages 1-4 audit", "stages-1-4-audit.md"),
        ("Stage 5 validation", "stage-five-validation.md"),
        ("Observation representation", "representation-encoding.md"),
    ]
    source_files = [
        core,
        ROOT / "docs/expanded-engine.md",
        ROOT / "docs/representation-encoding.md",
        ROOT / "Catan_RL_Guide.pdf",
        ROOT / "generate_guide.py",
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        Path(__file__),
        *sorted((ROOT / "catan").glob("*.py")),
        *sorted((ROOT / "rl").glob("*.py")),
    ]
    for _, name in milestone_files:
        source_files.append(ROOT / "docs" / name)
    hashes = {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p.read_bytes()).hexdigest()
        for p in dict.fromkeys(source_files)
    }
    story.extend((Spacer(1, 12), para("Inspected source hashes", "SubReport")))
    story.append(
        table(
            [["Source", "SHA-256"]]
            + [[name, digest] for name, digest in hashes.items()]
        )
    )
    chapter(story, "Appendix D - Sources and interpretation")
    story.append(
        para(
            "Original project roadmap: Catan_RL_Guide.pdf (37 pages), preserved without regeneration. The numbered register explains deviations: requested legal token/port setup early, realistic hidden observations early, owned mutation after profiling, and requested card/award expansion before training."
        )
    )
    story.append(
        para(
            "[Official CATAN base rules/almanac, 2020](https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf): purchases, card timing, robber, production, scoring and awards. Statements in this report are paraphrases."
        )
    )
    story.append(
        para(
            "[Official CATAN basegame FAQ](https://www.catan.com/faq/basegame): tie/blocking and development-card clarifications; consulted 2026-10-11."
        )
    )
    story.append(
        para(
            "Current source, tests, JSON reports and the workspace continuity file were reviewed. Historical documents reproduce their original workload/limitations and include a current-boundary notice. Future network, PPO, search and negotiation choices are plans, not existing implementations. Unrecorded motivations remain unconfirmed."
        )
    )
    for index, (title, name) in enumerate(milestone_files):
        chapter(story, "Appendix " + chr(ord("E") + index) + " - " + title)
        story.append(
            para(
                "Source: docs/"
                + name
                + ". Complete milestone text follows. Dated measurements describe that milestone; current contract and schema tables above supersede earlier implementation boundaries.",
                "SmallReport",
            )
        )
        markdown(story, (ROOT / "docs" / name).read_text(encoding="utf-8-sig"))
    document = Report(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=54,
        bottomMargin=52,
        title="Catan engine: architecture, decisions and performance",
        author="Catan project",
        subject="Detailed implementation decisions and verified performance review",
    )
    document.multiBuild(story, onFirstPage=footer, onLaterPages=footer)
    from pypdf import PdfReader

    reader = PdfReader(OUTPUT)
    texts = [page.extract_text() or "" for page in reader.pages]
    assert all(text.strip() for text in texts)
    combined = "\n".join(texts)
    for i in range(1, 72):
        assert f"D{i:02d} -" in combined, f"Missing decision {i}"
    normalized = " ".join(combined.split())
    for match in re.finditer(
        r"^### D\d{2}[^\n]*\n\s*\n([^\n]+)",
        core.read_text(encoding="utf-8-sig"),
        re.MULTILINE,
    ):
        prefix = " ".join(match.group(1).split()[:18])
        assert prefix in normalized, f"Missing decision body: {prefix}"
    for fragment in ("1505", "254", "Longest Road", "Largest Army", "Training"):
        assert fragment in combined, fragment
    metadata = {
        "path": str(OUTPUT),
        "pages": len(reader.pages),
        "decisions": 71,
        "words": len(combined.split()),
        "source_sha256": hashes,
        "pdf_sha256": sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    (ROOT / "output/full-review/pdf-verification.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in metadata.items() if k != "source_sha256"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
