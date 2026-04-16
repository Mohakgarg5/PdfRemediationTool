# VAPT - PDF Accessibility Remediation Pipeline

[![GitHub](https://img.shields.io/badge/GitHub-Mohakgarg5%2FPdfRemediationTool-blue)](https://github.com/Mohakgarg5/PdfRemediationTool)

> Automatically transform any PDF into a **PDF/UA-1 (ISO 14289-1) compliant**, fully accessible document - passing **PAC 2024** and **veraPDF** validation out of the box.

---

## The Problem

Organizations are legally required to make their digital documents accessible (ADA, Section 508, EN 301 549, EAA 2025). Manually remediating PDFs for accessibility is expensive, slow, and error-prone - a single document can take hours of specialist work.

## The Solution

**VAPT** is an automated pipeline that takes ordinary PDFs and applies a four-stage remediation process to produce fully accessible PDF/UA-1 documents. It handles structure tagging, metadata injection, font embedding, link annotation wiring, and validation - with both a **CLI** and a **Streamlit web UI**.

### What It Fixes

| Accessibility Issue | How It's Handled |
|---|---|
| Missing structure tags | Injects `/Document`, `/P`, `/H1`-`/H6`, `/Figure`, `/Table`, `/TR`, `/TD`, `/L`, `/LI` |
| Untagged images | Adds `/Figure` elements with `/Alt` text, `/BBox`, and `/Placement` layout attributes |
| Missing XMP metadata | Writes `dc:title`, `dc:language`, `pdfuaid:part=1`, `pdf:Producer` |
| Unembedded fonts | Locates system TTF files, subsets to used glyphs, embeds without altering existing metrics |
| Missing MarkInfo | Sets `/Marked true` and `/Suspects false` |
| Tab order & ViewerPrefs | Sets `/Tabs /S` on all pages and `/DisplayDocTitle true` |
| Broken link annotations | Wires `/Link` structure elements with both MCR (text content ref) and OBJR (annotation ref) |
| Watermarks / headers / footers | Detected and tagged as `/Artifact` so screen readers skip them |
| Missing ToUnicode CMap | Generates Windows-1252 CMap for proper text extraction |
| CIDFont issues | Adds `/CIDToGIDMap /Identity`, removes invalid CIDSet streams |

---

## Architecture

```
Input PDF
    |
    v
+--------------------+
|  pdf_extractor.py  |  Stage 1 - Content Extraction & Classification
|                    |  Parses text, fonts, images, bounding boxes
|                    |  Detects headings, lists, tables, watermarks, headers/footers
+--------+-----------+
         | DocumentContent (dataclass graph)
         v
+--------------------+
|   pdf_tagger.py    |  Stage 2 - Structure Tag Injection
|                    |  Rewrites content streams with BDC/EMC markers
|                    |  Builds StructTreeRoot, ParentTree, RoleMap
|                    |  Handles link annotations (MCR + OBJR)
+--------+-----------+
         | Tagged PDF
         v
+--------------------+
| pdf_postprocess.py |  Stage 3 - Metadata & Font Post-Processing
|                    |  XMP metadata, MarkInfo, ViewerPreferences
|                    |  Font embedding with metric preservation
|                    |  CIDFont fixes, annotation tagging, cleanup
+--------+-----------+
         | Remediated PDF
         v
+--------------------+
|   validator.py     |  Stage 4 - veraPDF Validation (optional)
|                    |  Runs veraPDF CLI with PDF/UA-1 profile
|                    |  Parses JSON report, surfaces failing clauses
+--------------------+
         |
         v
    Output PDF  --> <original>_accessible.pdf
```

---

## Quick Start

### Prerequisites

| Dependency | Version | Required | Purpose |
|---|---|---|---|
| Python | >= 3.10 | Yes | Runtime |
| [pikepdf](https://pikepdf.readthedocs.io/) | >= 8 | Yes | Low-level PDF manipulation |
| [pdfminer.six](https://pdfminersix.readthedocs.io/) | >= 20221105 | Yes | Text extraction with font metrics |
| [Pillow](https://python-pillow.org/) | >= 10 | Yes | Image processing |
| [fonttools](https://github.com/fonttools/fonttools) | >= 4.0 | Yes | TTF subsetting for font embedding |
| [langdetect](https://github.com/Mimino666/langdetect) | >= 1.0.9 | Yes | Document language detection |
| [streamlit](https://streamlit.io/) | >= 1.30 | For Web UI | Web interface |
| [veraPDF](https://verapdf.org/software/) | >= 1.24 | For validation | PDF/UA-1 validation (Java) |

### Installation

```bash
# Clone the repo
git clone https://github.com/Mohakgarg5/PdfRemediationTool.git
cd PdfRemediationTool

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Optional - Install veraPDF** (for validation stage):
```bash
# macOS
brew install verapdf

# Linux - download from https://verapdf.org/software/
java -jar verapdf-installer.jar
```

### Usage

#### CLI

```bash
# Process all PDFs in input/ directory
python main.py

# Process a single file
python main.py --input report.pdf

# Custom directories
python main.py --input-dir docs/ --output-dir accessible_docs/

# Skip veraPDF validation (no Java needed)
python main.py --skip-validation

# Debug logging
python main.py --verbose
```

Output files are written as `<original_name>_accessible.pdf` in the output directory.

#### Web UI (Streamlit)

```bash
streamlit run app.py
```

Open `http://localhost:8501` - drag and drop PDFs, process, and download remediated files.

#### Shell Script

```bash
chmod +x run.sh
./run.sh                       # Process all PDFs in input/
./run.sh --input file.pdf      # Single file
./run.sh --skip-validation     # Skip veraPDF
```

---

## Project Structure

```
vapt/
|-- main.py              # CLI entry point & pipeline orchestrator
|-- app.py               # Streamlit web UI
|-- pdf_extractor.py     # Stage 1: content extraction & classification (839 lines)
|-- pdf_tagger.py        # Stage 2: structure tag injection (1,096 lines)
|-- pdf_postprocess.py   # Stage 3: metadata, fonts, annotations (1,002 lines)
|-- validator.py         # Stage 4: veraPDF integration (201 lines)
|-- models.py            # Shared dataclasses (DocumentContent, TextBlock, BBox, etc.)
|-- config.py            # Tunable constants (heading ratios, zone sizes, etc.)
|-- requirements.txt     # Python dependencies
|-- packages.txt         # System packages for Streamlit Cloud (fonts-liberation)
|-- run.sh               # CLI convenience wrapper
|-- input/               # Drop source PDFs here (CLI mode)
|-- output/              # Remediated PDFs appear here
|-- report/              # veraPDF validation reports
```

---

## Configuration

All tunable constants live in `config.py`:

```python
# Heading detection - ratio of font size to body text
HEADING_SIZE_RATIO_H1 = 1.8    # >= 1.8x body = H1
HEADING_SIZE_RATIO_H2 = 1.5    # >= 1.5x body = H2
HEADING_SIZE_RATIO_H3 = 1.25   # >= 1.25x body = H3
HEADING_SIZE_RATIO_H4 = 1.1    # >= 1.1x body = H4 (must also be bold)

# Header/footer zones (fraction of page height)
HEADER_ZONE_FRACTION = 0.08    # Top 8%
FOOTER_ZONE_FRACTION = 0.08    # Bottom 8%

# Watermark detection
WATERMARK_MIN_ROTATION = 15.0       # degrees
WATERMARK_MAX_ROTATION = 75.0       # degrees
WATERMARK_MIN_FONT_SIZE = 36.0      # points
WATERMARK_LIGHT_COLOR_THRESHOLD = 0.7  # 0=black, 1=white

# veraPDF profile
VERAPDF_PROFILE = "ua1"
```

---

## Compliance Standards

**Target:** PDF/UA-1 (ISO 14289-1)

**Validated by:**
- **PAC 2024** (PDF Accessibility Checker) - Swiss PDF Association
- **veraPDF** - Industry-standard open-source validator

**Matterhorn Protocol checkpoints addressed:**

| Clause | Requirement |
|---|---|
| 01-004 | Tagged PDF flag (`MarkInfo /Marked true`) |
| 01-006 | Link elements contain both MCR and OBJR |
| 06-001 | Document language specified (`/Lang`) |
| 07-001 | Natural language in metadata |
| 07-010 | OCProperties properly configured |
| 07-18.1 | Widget annotations tagged as `/Form` |
| 07-18.5 | Link annotations tagged as `/Link` |
| 07-21.3.2 | CIDFontType2 has CIDToGIDMap |
| 07-21.4.2 | CIDSet streams valid or removed |
| 09-004 | `/Figure` elements have `/Alt` text |
| 14-002 | Artifacts correctly marked |
| 28-002 | XMP metadata includes `pdfuaid:part = 1` |

---

## Limitations

- **Scanned PDFs** (image-only) are not supported - the pipeline requires selectable text. Run OCR first (e.g., `ocrmypdf`).
- **Right-to-left scripts** (Arabic, Hebrew) are detected and language-tagged, but reading-order reversal is not applied.
- **Complex multi-column layouts** may produce suboptimal reading order; manual review is recommended.
- **Encrypted PDFs** must be decrypted before processing.
- **Type1 fonts** - only TTF/OTF embedding is supported (not PostScript Type1).
- **veraPDF** requires Java 11+ and must be installed separately.

---

## Example Output

```
PDF Accessibility Remediation Pipeline
==================================================
Processing 1 file(s)

[1/1] Processing: annual_report.pdf
  [1/4] Extracting content from annual_report.pdf...
        Found 142 text blocks, 8 images across 12 pages
        Language: en, Title: Annual Report 2024
        Structure: 18 headings, 3 tables, 21 list items, 4 artifacts
  [2/4] Adding structure tags to original PDF...
        Tagged: headings, paragraphs, images, artifacts
  [3/4] Post-processing metadata with pikepdf...
        Metadata fixed: MarkInfo, Lang, ViewerPreferences, TabOrder, XMP
  [4/4] Validating with veraPDF...
        Validation: PASS (94 passed, 0 failed)

============================
SUMMARY
PASS   annual_report.pdf  (3.2s)
Total: 1 | Processed: 1 | Compliant: 1 | Failed: 0
```

---

## License

MIT
