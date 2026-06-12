"""Build the submission-ready manuscript DOCX from generated result artifacts.

This script intentionally reads the canonical manuscript tables and figures from
``results/manuscript`` so the Word draft is tied to the same regenerated outputs
used for the final figures and completion gate.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence

from docx import Document
from docx.enum.section import WD_ORIENTATION, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


TITLE = (
    "Representation choices for somatic mutation catalogues strongly influence "
    "prediction performance across mechanistic and clinical endpoints"
)

AUTHORS = "Aaron Ge1,2, Jeya Balaji Balasubramanian3, Jonas De Almeida3, Bradley A. Maron1,2*"

AFFILIATIONS = [
    "1 University of Maryland-Institute of Health Computing, University of Maryland School of Medicine, Baltimore, MD, USA",
    "2 University of Maryland School of Medicine, Baltimore, MD, USA",
    "3 Division of Cancer Epidemiology and Genetics, National Cancer Institute, National Institutes of Health, Bethesda, MD, USA",
]

CORRESPONDING = [
    "Correspondence: Aaron Ge",
    "6116 Executive Blvd, Bethesda, MD 20282, USA",
    "age1@som.umaryland.edu",
]

KEYWORDS = (
    "Somatic mutations; Mutational signatures; Mutation annotation format; "
    "Representation learning; Homologous recombination deficiency; Tumour type; "
    "Survival analysis; Cox proportional hazards; MuAt; XGBoost"
)

ABSTRACT = {
    "Background": (
        "Cancer sequencing produces variable-length catalogues of somatic mutation events. "
        "These catalogues can be summarized as mutation burdens, mutational spectra, event-level "
        "annotation features, or learned neural event bags, but it remains unclear how much "
        "prediction performance depends on representation choice rather than model family alone."
    ),
    "Methods": (
        "We regenerated a strict no-leakage benchmark spanning five main endpoints: Kucab DNA "
        "damage class (n = 259), TCGA-BRCA HRD score (n = 772), HRD-high status at threshold 33 "
        "(n = 772), TCGA cancer type among the fixed twenty-class registry (n = 8,800), and TCGA CDR "
        "overall survival (n = 9,986). Representations included mutational burden, SBS/DBS/ID "
        "mutational spectra, Bio MAF v4 features, and combined Signatures + Bio MAF v4 "
        "features. Tabular learners used five outer folds, an inner train/validation split for "
        "hyperparameter selection, final refitting on the full outer-training fold, and pooled "
        "out-of-fold metrics. Survival was modeled with Cox proportional hazards and Harrell's "
        "C-index. We also added a MuAt-compatible event-bag comparator on the same manuscript "
        "endpoints and outer folds where possible."
    ),
    "Results": (
        "Mutational signatures improved over burden for XGBoost on Kucab damage class, HRD "
        "score, HRD33 high/low status, and cancer-type prediction. The strongest tabular "
        "performance generally came from Signatures + MAF stack with XGBoost: HRD score reached "
        "0.749 Spearman r, HRD33 high/low reached 0.886 AUROC, and the fixed top-20 cancer-type "
        "endpoint was evaluated by balanced accuracy. Overall survival was best predicted by mutational signatures under Cox PH "
        "(C-index 0.607), while richer event-level MAF features did not improve this survival "
        "endpoint. The MuAt-compatible comparator was evaluated by balanced accuracy on "
        "the directly comparable fixed TCGA-WES top-20 task and compared with the tuned "
        "Signatures + MAF stack XGBoost baseline."
    ),
    "Conclusions": (
        "Representation choice is a first-order determinant of prediction performance for "
        "somatic-mutation benchmarks. Mutational signatures remain a strong and efficient "
        "baseline, event-level MAF features add endpoint-specific biology, and combining spectra "
        "with event-level annotations is the most consistent practical tabular default. A local "
        "MuAt-compatible reimplementation does not automatically exceed a tuned tabular baseline "
        "under a directly comparable TCGA-WES evaluation."
    ),
}


REFERENCES = [
    "Alexandrov LB, Kim J, Haradhvala NJ, et al. The repertoire of mutational signatures in human cancer. Nature. 2020;578:94-101. doi:10.1038/s41586-020-1943-3.",
    "Kucab JE, Zou X, Morganella S, et al. A compendium of mutational signatures of environmental agents. Cell. 2019;177:821-836.e16. doi:10.1016/j.cell.2019.03.001.",
    "Ellrott K, Bailey MH, Saksena G, et al. Scalable open science approach for mutation calling of tumor exomes using multiple genomic pipelines. Cell Systems. 2018;6:271-281.e7. doi:10.1016/j.cels.2018.03.002.",
    "Liu J, Lichtenberg T, Hoadley KA, et al. An integrated TCGA pan-cancer clinical data resource to drive high-quality survival outcome analytics. Cell. 2018;173:400-416.e11. doi:10.1016/j.cell.2018.02.052.",
    "Knijnenburg TA, Wang L, Zimmermann MT, et al. Genomic and molecular landscape of DNA damage repair deficiency across The Cancer Genome Atlas. Cell Reports. 2018;23:239-254.e6. doi:10.1016/j.celrep.2018.03.076.",
    "Davies H, Glodzik D, Morganella S, et al. HRDetect is a predictor of BRCA1 and BRCA2 deficiency based on mutational signatures. Nature Medicine. 2017;23:517-525. doi:10.1038/nm.4292.",
    "Chen T, Guestrin C. XGBoost: a scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016:785-794. doi:10.1145/2939672.2939785.",
    "Cox DR. Regression models and life-tables. Journal of the Royal Statistical Society: Series B. 1972;34:187-220.",
    "Harrell FE Jr, Lee KL, Mark DB. Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. Statistics in Medicine. 1996;15:361-387. doi:10.1002/(SICI)1097-0258(19960229)15:4<361::AID-SIM168>3.0.CO;2-4.",
    "Sanjaya P, Maljanen K, Katainen R, et al. Mutation-Attention (MuAt): deep representation learning of somatic mutations for tumour typing and subtyping. Genome Medicine. 2023;15:47. doi:10.1186/s13073-023-01204-4.",
]


FIGURE_CAPTIONS = [
    (
        "figure_1_conceptual_overview.png",
        "Figure 1. Conceptual overview of mutation-catalogue representations.",
        "Each tumour is represented as a catalogue of somatic mutation events that can be transformed into compact burden summaries, mutational spectra, event-level MAF-stack aggregates, combined tabular representations, or MuAt-compatible mutation bags. The benchmark evaluates these choices under a strict pooled out-of-fold design.",
    ),
    (
        "figure_2_signature_baselines.png",
        "Figure 2. Signature baselines compared with mutational burden.",
        "Nested five-fold out-of-fold performance is shown for burden and mutational signatures across the five main endpoints. Metrics are Spearman r for HRD score, AUROC for HRD33, macro-AUROC for Kucab damage class, balanced accuracy for top-20 cancer type, and Harrell C-index for survival.",
    ),
    (
        "figure_3_geometry_vs_signatures.png",
        "Figure 3. MuAt-compatible event-bag comparator.",
        "The MuAt-compatible reimplementation is evaluated on the manuscript endpoints rather than as a reproduction of the original MuAt TCGA-20 benchmark. For TCGA endpoints, the comparator uses the same held-out folds as the canonical tabular benchmark where configured.",
    ),
    (
        "figure_4_maf_stack_vs_signatures.png",
        "Figure 4. Event-level MAF-stack features and combined signature-plus-event representations.",
        "Mutational signatures, event-level MAF-stack features, and their concatenation are compared for each endpoint and tabular model family. The combined representation is strongest for HRD score, HRD33 status, and cancer type under XGBoost.",
    ),
    (
        "figure_5_cross_endpoint_summary.png",
        "Figure 5. Cross-endpoint representation summary.",
        "The summary heatmap reports the canonical main-panel scores. MuAt-compatible results are displayed as a separate event-bag comparator, and overall survival is reported as a Cox PH C-index rather than a binary event AUROC.",
    ),
]

SUPPLEMENTARY_FIGURES = [
    (
        "figure_s1_representation_construction.png",
        "Supplementary Figure S1. Representation construction and reproducibility workflow.",
        "Raw mutation catalogues are converted into spectra, geometry variants, MAF-stack aggregates, combined tabular matrices, and MuAt-compatible event bags with cached, restartable feature-generation steps.",
    ),
    (
        "figure_s2_calibration_thresholds.png",
        "Supplementary Figure S2. Calibration of selected classification models.",
        "Reliability curves are shown for selected out-of-fold classification models to assess whether predicted probabilities are broadly aligned with observed frequencies.",
    ),
    (
        "figure_s3_feature_importance.png",
        "Supplementary Figure S3. Supplementary measured representation panels.",
        "Measured supplementary geometry, exposure, and supplementary check results are shown; unsupported combinations are omitted and documented in Supplementary Table S3.",
    ),
]


def read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.reader(handle)]


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def set_cell_border(cell, **kwargs) -> None:
    """Set specific cell border edges using WordprocessingML attributes."""
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_borders = tc_pr.first_child_found_in("w:tcBorders")
    if tc_borders is None:
        tc_borders = OxmlElement("w:tcBorders")
        tc_pr.append(tc_borders)
    for edge in ("top", "left", "bottom", "right"):
        attrs = kwargs.get(edge)
        if attrs is None:
            continue
        tag = f"w:{edge}"
        element = tc_borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            tc_borders.append(element)
        for key, value in attrs.items():
            element.set(qn(f"w:{key}"), str(value))


def set_cell_text(
    cell,
    text: str,
    bold: bool = False,
    size: int = 8,
    align: WD_ALIGN_PARAGRAPH = WD_ALIGN_PARAGRAPH.LEFT,
) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = "Arial"
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP


def set_cell_width(cell, width_inches: float) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_w = tc_pr.first_child_found_in("w:tcW")
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(int(width_inches * 1440)))
    tc_w.set(qn("w:type"), "dxa")


def apply_minimal_table_rules(table, header_rows: int = 1) -> None:
    no_rule = {"val": "nil", "sz": "0", "space": "0", "color": "FFFFFF"}
    strong_rule = {"val": "single", "sz": "12", "space": "0", "color": "000000"}
    medium_rule = {"val": "single", "sz": "10", "space": "0", "color": "000000"}
    row_rule = {"val": "single", "sz": "4", "space": "0", "color": "D0D0D0"}
    n_rows = len(table.rows)
    for row in table.rows:
        for cell in row.cells:
            set_cell_border(cell, top=no_rule, bottom=row_rule, left=no_rule, right=no_rule)
    if n_rows == 0:
        return
    for cell in table.rows[0].cells:
        set_cell_border(cell, top=strong_rule)
    for idx in range(min(header_rows, n_rows)):
        for cell in table.rows[idx].cells:
            set_cell_border(cell, bottom=medium_rule)


def split_display_title(title: str) -> tuple[str, str]:
    if "." in title:
        number, rest = title.split(".", 1)
        return number.strip(), rest.strip().rstrip(".")
    return title, ""


def add_csv_table(
    doc: Document,
    csv_path: Path,
    title: str,
    caption: str,
    *,
    font_size: int = 8,
) -> None:
    rows = read_csv(csv_path)
    if not rows:
        return
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title_p.paragraph_format.space_after = Pt(4)
    title_run = title_p.add_run(title)
    title_run.bold = True
    title_run.font.name = "Arial"
    title_run.font.size = Pt(11)
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for r_idx, row in enumerate(rows):
        for c_idx, value in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            if c_idx == 0:
                align = WD_ALIGN_PARAGRAPH.LEFT
            else:
                align = WD_ALIGN_PARAGRAPH.RIGHT
            set_cell_text(cell, value, bold=(r_idx == 0), size=font_size, align=align)
    apply_minimal_table_rules(table)
    note = doc.add_paragraph()
    note.paragraph_format.space_before = Pt(4)
    note.paragraph_format.space_after = Pt(8)
    note_run = note.add_run("Note. ")
    note_run.italic = True
    note_run.font.name = "Arial"
    note_run.font.size = Pt(max(font_size, 8))
    body_run = note.add_run(caption)
    body_run.font.name = "Arial"
    body_run.font.size = Pt(max(font_size, 8))
    doc.add_paragraph("")


def add_booktabs_table(
    doc: Document,
    title: str,
    caption: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    group_rows: set[int] | None = None,
    widths: Sequence[float] | None = None,
    font_size: int = 10,
    header_font_size: int = 10,
    title_font_size: int = 11,
    note_font_size: int = 9,
) -> None:
    """Add a publication-style table with minimal horizontal rules."""
    group_rows = group_rows or set()
    table_number, table_title = split_display_title(title)
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title_p.paragraph_format.space_after = Pt(4)
    title_run = title_p.add_run(title)
    title_run.bold = True
    title_run.font.name = "Arial"
    title_run.font.size = Pt(title_font_size)

    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False if widths else True
    for c_idx, header in enumerate(headers):
        cell = table.cell(0, c_idx)
        if widths:
            set_cell_width(cell, widths[c_idx])
        header_align = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.RIGHT
        set_cell_text(cell, header, bold=True, size=header_font_size, align=header_align)

    for r_idx, row in enumerate(rows, start=1):
        is_group = (r_idx - 1) in group_rows
        for c_idx, value in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            if widths:
                set_cell_width(cell, widths[c_idx])
            if is_group:
                align = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.RIGHT
                set_cell_text(cell, value, bold=True if c_idx == 0 else False, size=font_size, align=align)
            else:
                align = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.RIGHT
                set_cell_text(cell, value, bold=False, size=font_size, align=align)
    apply_minimal_table_rules(table)

    note = doc.add_paragraph()
    note.paragraph_format.space_before = Pt(4)
    note.paragraph_format.space_after = Pt(10)
    note_run = note.add_run("Note. ")
    note_run.italic = True
    note_run.font.name = "Arial"
    note_run.font.size = Pt(note_font_size)
    body_run = note.add_run(caption)
    body_run.font.name = "Arial"
    body_run.font.size = Pt(note_font_size)
    body_run.font.color.rgb = RGBColor(51, 51, 51)
    doc.add_paragraph("")


def add_endpoint_table(doc: Document, table_dir: Path) -> None:
    source = read_csv_dicts(table_dir / "table_1_datasets_endpoints.csv")
    headers = ["Endpoint", "Cohort/source", "N", "Task", "Primary metric"]
    rows = [
        [
            row["Endpoint"],
            row["Cohort/source"],
            row["N"],
            row["Task"].replace("; ", "\n"),
            row["Primary metric"],
        ]
        for row in source
    ]
    add_booktabs_table(
        doc,
        "Table 1. Datasets, endpoints, and evaluation design.",
        "Label definitions are provided in the endpoint registry; survival is modeled as a time-to-event endpoint with censoring rather than as binary event status.",
        headers,
        rows,
        widths=[1.45, 2.3, 0.65, 1.55, 1.25],
        font_size=9,
        header_font_size=9,
    )


def split_score_pair(value: str) -> tuple[str, str]:
    if "/" in value:
        left, right = value.split("/", 1)
        return left.strip(), right.strip()
    value = value.strip()
    return value, value


def add_performance_table(doc: Document, table_dir: Path) -> None:
    source = read_csv_dicts(table_dir / "table_2_full_performance_metrics.csv")
    reps = [
        ("Burden", "Mutational burden"),
        ("Signatures", "Mutational signatures"),
        ("MAF stack", "Event-level MAF stack"),
        ("Signatures + MAF", "Signatures + MAF stack"),
    ]
    rows: list[list[str]] = []
    group_rows: set[int] = set()
    for row in source:
        group_rows.add(len(rows))
        rows.append([f"{row['Endpoint']} ({row['Metric']}; n={row['N']})", "", "", ""])
        for column, label in reps:
            en, xgb = split_score_pair(row[column])
            if row["Task"].lower() == "survival":
                rows.append([f"    {label}", "", xgb, ""])
            else:
                rows.append([f"    {label}", en, xgb, ""])
        rows.append(["    MuAt-compatible event bag", "", "", row["MuAt-compatible"]])
    add_booktabs_table(
        doc,
        "Table 2. Main-panel performance by endpoint and representation.",
        "Scores are pooled out-of-fold primary metrics. Elastic net and XGBoost columns apply to tabular representations; the Cox PH value is shown in the XGBoost/Cox PH column for survival. MuAt-compatible is shown as a separate event-bag comparator.",
        ["Endpoint / representation", "Elastic net", "XGBoost / Cox PH", "MuAt-compatible"],
        rows,
        group_rows=group_rows,
        widths=[4.2, 1.35, 1.45, 1.45],
        font_size=8,
        header_font_size=9,
        note_font_size=8,
    )


def add_representation_table(doc: Document, table_dir: Path) -> None:
    source = read_csv_dicts(table_dir / "table_3_hyperparameters_feature_dimensionality.csv")
    rows = [
        [
            row["Representation"],
            row["Input signal"],
            row["Feature dimensionality"] or "event bag",
            row["Evaluated models"],
        ]
        for row in source
    ]
    add_booktabs_table(
        doc,
        "Table 3. Representation summary and dimensionality.",
        "Feature dimensionality is reported for tabular matrices; MuAt-compatible uses padded mutation bags and learned embeddings rather than a fixed tabular feature count.",
        ["Representation", "Input signal", "Dimensionality", "Evaluated models"],
        rows,
        widths=[1.65, 4.1, 1.2, 1.75],
        font_size=8,
        header_font_size=9,
        note_font_size=8,
    )


def add_glossary_table(doc: Document, table_dir: Path) -> None:
    source = read_csv_dicts(table_dir / "table_4_label_mapping.csv")
    rows = [[row["Term"], row["Definition"], row["Used in"]] for row in source]
    add_booktabs_table(
        doc,
        "Table 4. Key terminology and abbreviations.",
        "Abbreviations and shorthand used in the manuscript figures and tables.",
        ["Term", "Definition", "Used in"],
        rows,
        widths=[1.55, 5.3, 1.4],
        font_size=7,
        header_font_size=8,
        note_font_size=8,
    )


def crop_image_for_docx(image_path: Path, cache_dir: Path) -> Path:
    """Crop near-white whitespace around figures for Word embedding."""
    from PIL import Image, ImageChops

    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / image_path.name
    image = Image.open(image_path).convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    diff = ImageChops.difference(image, background).convert("L")
    bbox = diff.point(lambda p: 255 if p > 10 else 0).getbbox()
    if bbox is None:
        image.save(output)
        return output
    left, top, right, bottom = bbox
    pad = 24
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(image.width, right + pad)
    bottom = min(image.height, bottom + pad)
    image.crop((left, top, right, bottom)).save(output)
    return output


def add_captioned_figure(
    doc: Document,
    image_path: Path,
    title: str,
    caption: str,
    max_width_inches: float,
    max_height_inches: float = 6.25,
) -> None:
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    prepared = crop_image_for_docx(image_path, image_path.parent / "_docx_cropped")
    from PIL import Image

    with Image.open(prepared) as image:
        width_px, height_px = image.size
    aspect = width_px / max(height_px, 1)
    width_inches = min(max_width_inches, max_height_inches * aspect)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(prepared), width=Inches(width_inches))
    doc.add_paragraph(title, style="CaptionTitle")
    doc.add_paragraph(caption, style="CaptionText")


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def add_paragraphs(doc: Document, paragraphs: Iterable[str]) -> None:
    for paragraph in paragraphs:
        doc.add_paragraph(paragraph)


def add_landscape_section(doc: Document) -> None:
    section = doc.add_section(WD_SECTION.NEW_PAGE)
    section.orientation = WD_ORIENTATION.LANDSCAPE
    section.page_width = Inches(11)
    section.page_height = Inches(8.5)
    section.left_margin = Inches(0.5)
    section.right_margin = Inches(0.5)
    section.top_margin = Inches(0.45)
    section.bottom_margin = Inches(0.45)


def add_portrait_section(doc: Document) -> None:
    section = doc.add_section(WD_SECTION.NEW_PAGE)
    section.orientation = WD_ORIENTATION.PORTRAIT
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08

    for name in ["Heading 1", "Heading 2", "Heading 3"]:
        style = styles[name]
        style.font.name = "Times New Roman"
        style.font.color.rgb = RGBColor(0, 0, 0)

    styles.add_style("CaptionTitle", 1)
    styles["CaptionTitle"].font.name = "Times New Roman"
    styles["CaptionTitle"].font.bold = True
    styles["CaptionTitle"].font.size = Pt(9)
    styles["CaptionTitle"].paragraph_format.space_after = Pt(2)

    styles.add_style("CaptionText", 1)
    styles["CaptionText"].font.name = "Times New Roman"
    styles["CaptionText"].font.size = Pt(8.5)
    styles["CaptionText"].paragraph_format.space_after = Pt(8)


def build_docx(repo_root: Path, output: Path) -> None:
    manuscript = repo_root / "results" / "manuscript"
    table_dir = manuscript / "tables"
    figure_dir = manuscript / "figures"
    supplement_dir = manuscript / "supplement"

    doc = Document()
    configure_styles(doc)
    section = doc.sections[0]
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)

    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title_p.add_run(TITLE)
    run.bold = True
    run.font.size = Pt(16)
    run.font.name = "Arial"
    doc.add_paragraph(AUTHORS).alignment = WD_ALIGN_PARAGRAPH.CENTER
    for affiliation in AFFILIATIONS:
        doc.add_paragraph(affiliation)
    for line in CORRESPONDING:
        doc.add_paragraph(line)

    add_heading(doc, "Abstract", 1)
    for key, text in ABSTRACT.items():
        p = doc.add_paragraph()
        p.add_run(f"{key}. ").bold = True
        p.add_run(text)
    doc.add_paragraph("Trial registration. Not applicable.")
    doc.add_paragraph("Keywords: " + KEYWORDS)

    add_heading(doc, "Background", 1)
    add_paragraphs(
        doc,
        [
            "Somatic mutation catalogues integrate the activity of endogenous and exogenous mutational processes, DNA repair defects, and clonal selection. Large pan-cancer resources have catalogued SBS, DBS, and indel signatures across whole genomes and exomes, while controlled mutagen-exposure experiments have shown that many environmental agents leave reproducible sequence-context spectra [1,2]. These observations make mutation catalogues a natural substrate for mechanistic attribution and clinical prediction.",
            "A practical prediction pipeline must decide how to transform a variable-length set of mutation events into model inputs. Common tabular choices include mutation burden, mutational spectra, and aggregated MAF-derived gene, locus, consequence, pathway, and variant-allele-fraction features. More recent neural approaches, including Mutation-Attention (MuAt), instead operate on mutation-level events and learn attention-based tumour representations [10].",
            "The manuscript question is therefore not whether mutation catalogues contain predictive information, but how strongly the chosen representation shapes performance when evaluation is held fixed. We address critiques of exome sparsity, insufficient tuning, survival modeling, and deep-learning comparability by regenerating the benchmark under a strict no-leakage, nested out-of-fold protocol with Cox survival endpoints and a directly comparable MuAt-compatible event-bag model.",
        ],
    )
    add_captioned_figure(doc, figure_dir / FIGURE_CAPTIONS[0][0], FIGURE_CAPTIONS[0][1], FIGURE_CAPTIONS[0][2], 6.7)

    add_heading(doc, "Methods", 1)
    add_heading(doc, "Study design and endpoint registry", 2)
    add_paragraphs(
        doc,
        [
            "All analyses were regenerated inside the project repository using an endpoint registry and strict completion gate. The main panel contains five endpoints chosen to span mechanism, DNA repair deficiency, tumour identity, and clinical outcome. Binary overall-survival classification was removed from the main benchmark and replaced by Cox proportional hazards modeling of TCGA CDR overall survival.",
            "The five primary endpoint families were Kucab DNA damage class; continuous HRD score; HRD-high versus HRD-low binary thresholds; TCGA cancer type among the fixed twenty-class registry; and TCGA CDR overall survival. Survival endpoints required positive follow-up time, at least 200 eligible samples, and at least 25 events.",
        ],
    )
    add_endpoint_table(doc, table_dir)

    add_heading(doc, "Representations", 2)
    add_paragraphs(
        doc,
        [
            "Mutational burden features provide the minimal count-based baseline, and mutational signatures summarize SBS, DBS, and indel spectra. Bio MAF v4 converts each tumour's variable-length MC3 Mutation Annotation Format table into fixed, named biological feature blocks grounded in MAF/VEP annotations and vendored external cancer-gene, driver-evidence, and hotspot resources.",
            "The strict Bio MAF v4 stack uses predeclared candidate blocks: compact biological annotations, external evidence-confidence controls, hotspot summaries, exact consensus driver-gene features, a full strict core, and optional mutation-load controls. Signatures are not counted as Bio MAF features; signatures + Bio MAF models concatenate the signature matrix with the Bio MAF block chosen inside the inner loop for that outer fold.",
            "Bio MAF v4 is therefore not one feature per mutation. It is one fixed-length vector per tumour. The feature builder normalizes gene symbols and standard MAF/VEP fields, then computes named counts, fractions, unique-gene summaries, hotspot summaries, driver-evidence summaries, and allele-fraction summaries. Count features are log1p-transformed after all mutations for that tumour are counted.",
            "Feature-block selection is nested: candidate Bio MAF v4 blocks are defined before cross-validation, selected only using each outer fold's inner-validation data, and then evaluated once on that outer fold. Held-out fold labels and held-out performance do not define the Bio MAF v4 feature schema.",
            "For the neural comparator, a MuAt-compatible preprocessing layer turns each mutation into three token streams: a mutation motif token, a 1-Mb chromosome-position-bin token, and an annotation token derived from available MAF fields. The active comparator uses fixed-size deterministic token dictionaries, preventing held-out test samples from changing the available token set.",
        ],
    )
    add_landscape_section(doc)
    add_representation_table(doc, table_dir)
    add_portrait_section(doc)

    add_heading(doc, "Nested out-of-fold evaluation", 2)
    add_paragraphs(
        doc,
        [
            "Each endpoint used five outer folds. For every outer fold, the 80% outer-training partition was split into inner-training and inner-validation partitions. Candidate models were trained only on the inner-training set, selected only on the inner-validation set, refit once on the full 80% outer-training set using the selected hyperparameters, and predicted once on the 20% held-out outer fold. Primary metrics were computed after pooling all out-of-fold predictions over the full cohort.",
            "Classification used stratified splits when feasible; survival used event-stratified splits; regression used shuffled K-fold splits; and Kucab analyses used grouped splitting to prevent treatment-agent leakage across folds. Logistic elastic-net models searched C values of 0.001, 0.01, 0.1, 1, and 10 with l1_ratio values of 0, 0.25, 0.5, 0.75, and 1. Regression elastic net and Cox PH models searched alpha or penalizer values from 1e-4 to 10 with the same l1_ratio grid. XGBoost used seeded candidate search with inner-validation early stopping; final refits used exactly the selected number of boosting rounds.",
            "Primary metrics were Spearman r for continuous endpoints, AUROC for binary endpoints, macro-AUROC for Kucab damage class, balanced accuracy for the fixed top-20 cancer-type endpoint, and Harrell C-index for survival endpoints. Multiclass Kucab robustness was further summarized with micro-AUROC, balanced accuracy, macro-F1, and Cohen's kappa. Statistical comparisons used paired tests over pooled out-of-fold predictions with Benjamini-Hochberg FDR correction.",
        ],
    )

    add_heading(doc, "MuAt-compatible comparator", 2)
    add_paragraphs(
        doc,
        [
            "The MuAt-compatible comparator follows the central structure reported for MuAt: separate motif, position, and annotation embeddings; concatenation of modality embeddings; Q/K/V self-attention; residual/normalization layers; a fully connected block; average pooling; and a 24-dimensional tumour feature layer before prediction. We label this model MuAt-compatible rather than MuAt because official MuAt package checkpoints were not used for the manuscript comparison.",
            "The primary manuscript comparison evaluates this neural event-bag model on the same TCGA-WES endpoints as the tabular benchmark. The event tensor allows up to 5,000 mutation events per tumour; samples above this ceiling are deterministically truncated after sorting events by chromosome, position, and motif. In practice, most TCGA-WES tumours were far below this ceiling; the directly comparable cancer-type task is now the fixed top-20 TCGA benchmark defined from matched MC3 feature support.",
            "The fixed top-20 TCGA cancer-type task is the active manuscript endpoint, so neural and tabular cancer-type comparisons share the same endpoint definition and fold structure.",
        ],
    )

    add_heading(doc, "Results", 1)
    add_heading(doc, "Mutational signatures improve over burden, but not uniformly", 2)
    add_paragraphs(
        doc,
        [
            "In the strict main panel, XGBoost models using mutational signatures outperformed burden on Kucab damage class (0.636 vs 0.505 macro-AUROC), HRD score (0.704 vs 0.596 Spearman r), HRD33 high/low status (0.845 vs 0.811 AUROC), and cancer-type prediction by balanced accuracy. Elastic-net models showed the same qualitative gain for Kucab damage class and cancer type but not for every clinical endpoint, underscoring that signatures are strong but not universally dominant.",
            "The survival result should be interpreted separately from the older binary-OS draft: overall survival is now a Cox PH endpoint evaluated by Harrell C-index. Under this survival analysis, mutational signatures achieved the strongest overall-survival C-index among the main tabular representations (0.607).",
        ],
    )

    add_landscape_section(doc)
    add_captioned_figure(doc, figure_dir / FIGURE_CAPTIONS[1][0], FIGURE_CAPTIONS[1][1], FIGURE_CAPTIONS[1][2], 9.7)
    add_portrait_section(doc)

    add_heading(doc, "Event-level MAF features add endpoint-specific biology", 2)
    add_paragraphs(
        doc,
        [
            "Event-level MAF-stack features were most useful for endpoints where gene, locus, consequence, or event-level biology carries direct label information. With XGBoost, MAF stack alone improved cancer-type prediction relative to signatures by balanced accuracy, but it underperformed signatures for the mechanistic Kucab damage-class task (0.529 vs 0.636 macro-AUROC). This pattern supports the expected distinction between mutagen-mechanism endpoints, which are strongly sequence-context driven, and clinical or tumour-identity endpoints, which can benefit from broader annotation features.",
            "The combined Signatures + MAF stack representation was the strongest practical tabular default. With XGBoost it reached 0.749 Spearman r for HRD score, 0.886 AUROC for HRD33 high/low, and the strongest balanced accuracy for cancer type. It did not improve overall survival, where signatures alone remained best under Cox PH.",
        ],
    )

    add_landscape_section(doc)
    add_captioned_figure(doc, figure_dir / FIGURE_CAPTIONS[3][0], FIGURE_CAPTIONS[3][1], FIGURE_CAPTIONS[3][2], 9.7)
    add_portrait_section(doc)

    add_heading(doc, "MuAt-compatible event-bag modeling is directly comparable but not dominant", 2)
    add_paragraphs(
        doc,
        [
            "The MuAt-compatible comparator was added to avoid treating deep event-level models only as a conceptual future baseline. On the fixed 20-class TCGA cancer-type endpoint, it is evaluated by balanced accuracy against the tuned Signatures + MAF stack XGBoost model. On HRD score, HRD33 high/low, Kucab damage class, and overall survival, the MuAt-compatible model also did not exceed the best tabular result.",
            "These results do not imply that the official pretrained MuAt model is ineffective. They show that, in this manuscript's strict TCGA-WES benchmark and local reimplementation, a mutation-attention architecture does not automatically outperform a leakage-safe, tuned tabular representation.",
        ],
    )

    add_landscape_section(doc)
    add_captioned_figure(doc, figure_dir / FIGURE_CAPTIONS[2][0], FIGURE_CAPTIONS[2][1], FIGURE_CAPTIONS[2][2], 9.7)
    doc.add_page_break()
    add_performance_table(doc, table_dir)
    add_portrait_section(doc)

    add_heading(doc, "Cross-endpoint synthesis", 2)
    add_paragraphs(
        doc,
        [
            "No representation wins across every endpoint. Mutational signatures are a strong and efficient baseline; event-level MAF features capture biology that spectra can miss; and combined signature-plus-MAF features offer the best overall tabular default for HRD and tumour-identity endpoints. Survival remains difficult, and richer annotation stacks did not improve over signatures in the current Cox PH analysis.",
            "The Kucab sparsity control downsampled WGS mutation inventories to budgets of 20, 41, 100, and 200 mutations per clone. These runs provide an exome-like stress test for the mechanistic endpoint and are reported in the supplementary tables. They make explicit that WGS-versus-WES mutation count differences can affect representation stability and should not be ignored when comparing mechanistic and clinical cohorts.",
        ],
    )

    add_landscape_section(doc)
    add_captioned_figure(doc, figure_dir / FIGURE_CAPTIONS[4][0], FIGURE_CAPTIONS[4][1], FIGURE_CAPTIONS[4][2], 9.7)
    add_portrait_section(doc)

    add_heading(doc, "Discussion", 1)
    add_paragraphs(
        doc,
        [
            "This regenerated benchmark changes the interpretation of the original draft in three important ways. First, survival is now evaluated as time-to-event data with censoring rather than as binary event status. Second, high-dimensional tabular baselines are tuned within a nested out-of-fold framework, reducing the risk that poor performance reflects inadequate regularization. Third, the neural event-bag comparator is measured directly rather than invoked as an untested future alternative.",
            "The results support a pragmatic view of mutation-catalogue modeling. For mechanistic mutagen attribution, mutational spectra remain highly competitive because the label is closely tied to sequence-context processes. For HRD and cancer-type endpoints, event-level annotations add useful biological structure, and the combined Signatures + MAF stack representation performs best with XGBoost. For survival, predictive signal from mutation catalogues alone remains modest and appears sensitive to representation and censoring-aware modeling assumptions.",
            "The MuAt-compatible model's weaker performance relative to tuned tabular models is scientifically plausible. Although the model permits 5,000 events per tumour, TCGA WES mutation bags are sparse in the present data, with median event counts near 100 for the primary TCGA comparison. The model was also trained locally rather than initialized from official MuAt checkpoints, and the manuscript endpoints include HRD and survival tasks that were not the original MuAt paper's primary tumour-typing benchmark. The result should therefore be read as a fair local comparator, not as a failure to reproduce the full MuAt PCAWG/TCGA/GEL study.",
        ],
    )

    add_heading(doc, "Limitations", 2)
    add_paragraphs(
        doc,
        [
            "The benchmark uses bundled MC3, CDR, HRD, and Kucab assets rather than downloading new GDC cohorts during the default run, so manuscript reproduction stays offline and reproducible from the bundled assets.",
            "The MuAt-compatible comparator implements the central mutation-attention architecture and token modalities but does not include official pretrained checkpoints, external WGS validation cohorts, structural-variant and mobile-element modalities unavailable in the bundled TCGA WES assets, or the full 150-epoch MuAt paper training regimen. Although the comparator uses a 5,000-event cap, this capacity is rarely used in TCGA-WES; most tumours contribute far fewer events than the cap, so the comparison should be interpreted as a sparse exome-facing benchmark rather than a high-mutation whole-genome MuAt reproduction.",
            "Some comparisons involving MuAt and simpler tabular representations share the same endpoint and out-of-fold pooling but are not all paired on identical fold manifests for every representation family. The strongest TCGA cancer-type comparison was aligned to the canonical tabular fold assignment, and this is the comparison emphasized for direct model interpretation.",
        ],
    )

    add_heading(doc, "Conclusions", 1)
    add_paragraphs(
        doc,
        [
            "Representation choice strongly influences somatic-mutation prediction performance. Mutational signatures are a robust baseline, event-level MAF features add complementary endpoint-specific signal, and their combination with XGBoost is the strongest practical default for several clinical and tumour-identity endpoints. Cox survival modeling and no-leakage feature generation materially strengthen the clinical interpretation of the benchmark. A MuAt-compatible event-bag comparator provides a direct neural-model comparison but does not supersede the tuned tabular baseline under the current TCGA-WES evaluation.",
        ],
    )

    add_heading(doc, "Abbreviations", 1)
    add_landscape_section(doc)
    add_glossary_table(doc, table_dir)
    add_portrait_section(doc)

    add_heading(doc, "Declarations", 1)
    add_heading(doc, "Ethics approval and consent to participate", 2)
    doc.add_paragraph("Not applicable. The analyses use public, de-identified research datasets.")
    add_heading(doc, "Consent for publication", 2)
    doc.add_paragraph("Not applicable.")
    add_heading(doc, "Availability of data and materials", 2)
    doc.add_paragraph(
        "All analyses were implemented in the GitHub-connected project repository. The default reproducibility run uses bundled MC3, TCGA CDR, TCGA-BRCA HRD, and Kucab assets; no active manuscript command downloads GDC data. Generated tables, figures, split manifests, and audit files are stored under the project results directory."
    )
    add_heading(doc, "Competing interests", 2)
    doc.add_paragraph("The authors declare that they have no competing interests.")
    add_heading(doc, "Funding", 2)
    doc.add_paragraph(
        "The research reported here was supported by the AIM-AHEAD Coordinating Center at the University of North Texas Health Science Center at Fort Worth and, in part, by the National Institutes of Health Agreement No. 1OT2OD032581. The views and conclusions are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the NIH."
    )
    add_heading(doc, "Authors' contributions", 2)
    doc.add_paragraph(
        "AG designed and implemented the benchmark, performed analyses, generated figures and tables, and drafted the manuscript. JBB and JDA contributed cancer-genomics and benchmarking expertise. BAM supervised the project and revised the manuscript. All authors reviewed and approved the manuscript."
    )
    add_heading(doc, "Acknowledgements", 2)
    doc.add_paragraph(
        "The authors thank the teams responsible for TCGA, MC3, TCGA CDR, HRD, and Kucab resources for making reusable cancer-genomics datasets available to the community."
    )

    add_heading(doc, "References", 1)
    for idx, reference in enumerate(REFERENCES, start=1):
        doc.add_paragraph(f"{idx}. {reference}")

    add_landscape_section(doc)
    add_heading(doc, "Supplementary Material", 1)
    add_csv_table(
        doc,
        table_dir / "table_s1_class_distribution_baselines.csv",
        "Supplementary Table S1. Supplementary endpoint inventory.",
        "Endpoint-level inventory for supplementary analyses outside the five-endpoint main panel.",
        font_size=5,
    )
    doc.add_page_break()
    add_csv_table(
        doc,
        table_dir / "table_s2_sensitivity_analyses.csv",
        "Supplementary Table S2. Headline supplementary results.",
        "Best baseline/event-level, geometry/sensitivity, and overall result for each supplementary endpoint.",
        font_size=5,
    )
    doc.add_page_break()
    add_csv_table(
        doc,
        table_dir / "table_s3_completeness_and_na_reasons.csv",
        "Supplementary Table S3. Non-applicability summary.",
        "Grouped explanation of unsupported, intentionally omitted, or not-applicable supplementary combinations.",
        font_size=6,
    )

    add_landscape_section(doc)
    for filename, title, caption in SUPPLEMENTARY_FIGURES:
        width = 8.8 if filename == "figure_s3_feature_importance.png" else 9.7
        add_captioned_figure(doc, supplement_dir / filename, title, caption, width)
        if filename != SUPPLEMENTARY_FIGURES[-1][0]:
            doc.add_page_break()

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build_docx(args.repo_root.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
