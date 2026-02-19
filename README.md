# PDF Remediation Tool

> Automatically transform any PDF into a **PDF/UA-1 compliant**, fully accessible document — passing PAC and veraPDF validation out of the box.

---

## Overview

**PdfRemediationTool** is a Python pipeline that takes ordinary PDFs and applies a four-stage remediation process to make them conform to the **PDF/UA-1** (ISO 14289-1) accessibility standard. It handles structure tagging, metadata injection, font embedding, link annotation wiring, and full validation — with both a command-line interface and a Streamlit web UI.

### What it fixes

| Issue | How it's handled |
|---|---|
| Missing structure tags | Injects `/Document`, `/P`, `/H1`–`/H6`, `/Figure`, `/Table`, `/TR`, `/TD`, `/L`, `/LI` |
| Untagged images | Adds `/Figure` with `/Alt` text and `/BBox` layout attributes |
| Missing XMP metadata | Writes `dc:title`, `dc:language`, `pdf:Producer`, `pdfuaid:part` |
| Unembedded fonts | Locates and embeds font files without disturbing existing metrics |
| Missing `MarkInfo` | Sets `/Marked true` and `/Suspects false` |
| Tab order & ViewerPrefs | Sets `/TabOrder /S` on all pages and `DisplayDocTitle true` |
| Broken link annotations | Wires `/Link` structure elements with both MCR (text) and OBJR (annotation ref) |
| Watermarks / headers / footers | Tagged as `/Artifact` so they are invisible to screen readers |

---

## Architecture

```
input PDF
    │
    ▼
┌──────────────────┐
│  pdf_extractor   │  Stage 1 — Content extraction & classification
│                  │  • Parses text blocks, font metrics, bounding boxes
│                  │  • Detects headings (by font-size ratio), lists, tables
│                  │  • Identifies watermarks, headers, footers as artifacts
└────────┬─────────┘
         │ DocumentContent (dataclass graph)
         ▼
┌──────────────────┐
│   pdf_tagger     │  Stage 2 — Structure tag injection
│                  │  • Writes PDF structure tree directly into the file
│                  │  • Marks content streams with MCID markers
│                  │  • Builds ParentTree, RoleMap, ClassMap
│                  │  • Handles Link annotation tagging (MCR + OBJR)
└────────┬─────────┘
         │ tagged PDF
         ▼
┌──────────────────┐
│ pdf_postprocess  │  Stage 3 — Metadata & font post-processing
│                  │  • XMP metadata (pdfuaid, dc, pdf namespaces)
│                  │  • MarkInfo, ViewerPreferences, TabOrder
│                  │  • Font embedding (preserves original descriptor metrics)
│                  │  • Type0 / CIDFont descendant handling
└────────┬─────────┘
         │ remediated PDF
         ▼
┌──────────────────┐
│   validator      │  Stage 4 — veraPDF / PAC validation
│                  │  • Runs veraPDF CLI with PDF/UA-1 profile
│                  │  • Parses JSON report, surfaces failed clauses
└──────────────────┘
         │
         ▼
    output PDF  ✓ PDF/UA-1 compliant
```

---

## Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 | Runtime |
| [pikepdf](https://pikepdf.readthedocs.io/) | ≥ 8 | Low-level PDF read/write |
| [pdfminer.six](https://pdfminersix.readthedocs.io/) | ≥ 20221105 | Text extraction |
| [Pillow](https://python-pillow.org/) | ≥ 10 | Image handling |
| [streamlit](https://streamlit.io/) | ≥ 1.30 | Web UI (optional) |
| [veraPDF](https://verapdf.org/software/) | ≥ 1.24 | PDF/UA-1 validation (Java) |

Install Python dependencies:

```bash
pip install pikepdf pdfminer.six Pillow streamlit
```

Install veraPDF (Java required):

```bash
# macOS (Homebrew)
brew install verapdf

# Linux / manual
# Download installer from https://verapdf.org/software/
java -jar verapdf-installer.jar
```

---

## Quick Start

### CLI

```bash
# Process all PDFs in the input/ directory
python main.py

# Process a single file
python main.py --input report.pdf

# Specify custom directories
python main.py --input-dir docs/ --output-dir accessible_docs/

# Skip veraPDF validation step
python main.py --skip-validation

# Enable verbose/debug logging
python main.py --verbose
```

Output files are written as `<original_name>_accessible.pdf` in the output directory.

### Web UI (Streamlit)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, drag and drop PDFs, and download the remediated files.

---

## Project Structure

```
PdfRemediationTool/
├── main.py            # CLI entry point & pipeline orchestrator
├── app.py             # Streamlit web UI
├── pdf_extractor.py   # Stage 1: content extraction & classification
├── pdf_tagger.py      # Stage 2: structure tag injection
├── pdf_postprocess.py # Stage 3: metadata, fonts, annotations
├── validator.py       # Stage 4: veraPDF integration
├── models.py          # Shared dataclasses (DocumentContent, TextBlock, …)
├── config.py          # Tunable constants (heading ratios, zone sizes, …)
├── input/             # Drop source PDFs here (CLI mode)
└── output/            # Remediated PDFs written here (CLI mode)
```

---

## Configuration

Edit `config.py` to tune detection thresholds:

```python
# Heading detection — ratio of element font size to body text size
HEADING_SIZE_RATIO_H1 = 1.8
HEADING_SIZE_RATIO_H2 = 1.5
HEADING_SIZE_RATIO_H3 = 1.25
HEADING_SIZE_RATIO_H4 = 1.1

# Header / footer zone (fraction of page height from top/bottom)
HEADER_ZONE_FRACTION = 0.08
FOOTER_ZONE_FRACTION = 0.08

# Watermark detection
WATERMARK_MIN_ROTATION   = 15.0
WATERMARK_MAX_ROTATION   = 75.0
WATERMARK_MIN_FONT_SIZE  = 36.0

# veraPDF profile
VERAPDF_PROFILE = "ua1"
```

---

## Compliance Standards

The tool targets **PDF/UA-1 (ISO 14289-1)** compliance as measured by:

- **PAC** (PDF Accessibility Checker) — Swiss PDF Association checker
- **veraPDF** — industry-standard open-source PDF/UA validator

Key Matterhorn Protocol checkpoints addressed:

- `01-004` — Tagged PDF flag set (`MarkInfo /Marked true`)
- `01-006` — Link elements contain both MCR and OBJR children
- `06-001` — Document language specified (`/Lang`)
- `07-001` — Natural language identified in metadata
- `09-004` — `/Figure` elements have `/Alt` text
- `14-002` — Artifacts correctly marked
- `28-002` — XMP metadata includes `pdfuaid:part = 1`

---

## How Heading Detection Works

The extractor computes a **body text size baseline** (median font size across the page) and classifies spans whose font size exceeds it by a configurable ratio:

| Ratio | Assigned tag |
|---|---|
| ≥ 1.8× body | `/H1` |
| ≥ 1.5× body | `/H2` |
| ≥ 1.25× body | `/H3` |
| ≥ 1.1× body | `/H4` |
| Bold at body size | `/H5` / `/H6` heuristic |
| Otherwise | `/P` |

Bold and italic font names are detected via common name suffixes (`Bold`, `Italic`, `Heavy`, `Light`, etc.).

---

## Limitations

- **Scanned PDFs** (image-only) are not supported — the pipeline requires selectable text.
- **Right-to-left scripts** (Arabic, Hebrew) are detected and language-tagged, but reading-order reversal is not applied.
- **Complex multi-column layouts** may produce suboptimal reading order; manual review is recommended.
- veraPDF must be installed separately (Java runtime required).

---

## License

MIT — see [LICENSE](LICENSE) for details.
