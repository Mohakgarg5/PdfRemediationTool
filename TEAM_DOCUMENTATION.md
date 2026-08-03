# VAPT - Complete Technical Documentation

**For the team. Read this if Mohak is unavailable and you need to understand, maintain, debug, or extend this system.**

Last updated: April 2026

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [PDF/UA-1 Crash Course](#3-pdfua-1-crash-course)
4. [Pipeline Deep Dive](#4-pipeline-deep-dive)
5. [Data Model](#5-data-model)
6. [Key Algorithms](#6-key-algorithms)
7. [Configuration Reference](#7-configuration-reference)
8. [Troubleshooting Guide](#8-troubleshooting-guide)
9. [How to Extend](#9-how-to-extend)
10. [Validation & Testing](#10-validation--testing)
11. [Environment Setup](#11-environment-setup)
12. [Stakeholders & Review Process](#12-stakeholders--review-process)
13. [Glossary](#13-glossary)

---

## 1. Problem Statement

### Why this exists

Organizations are legally required to make digital documents accessible to people who use assistive technologies (screen readers, braille displays, etc.). The relevant mandates include:

- **ADA** (Americans with Disabilities Act)
- **Section 508** (U.S. federal agencies)
- **EN 301 549** (European public sector)
- **European Accessibility Act (EAA) 2025** (private sector in the EU)

PDF is the most common format for formal documents (reports, contracts, policies, invoices). Most PDFs are **not accessible** - they lack the internal structure that tells a screen reader what's a heading, what's a paragraph, what's an image, and what order to read them in.

### The cost of manual remediation

Manually remediating a PDF for accessibility in Adobe Acrobat Pro takes hours per document - a specialist must tag every element, set alt text on images, fix reading order, embed fonts, and verify with PAC. Organizations with thousands of documents face a backlog that is effectively unshippable by hand.

### Who the stakeholders are

- **Compliance teams** - need documents to pass PDF/UA-1 validation before publishing
- **Document authors** - produce PDFs from Word, InDesign, PowerPoint, etc. and need them made accessible without manual intervention
- **End users with disabilities** - consume the output with screen readers (JAWS, NVDA, VoiceOver)
- **Accessibility auditors** - validate the output using PAC 2024 and veraPDF
- **Katharine** - reviews the tool's output for quality and compliance; see [Section 12](#12-stakeholders--review-process)

---

## 2. Solution Overview

### What the tool does

Takes any ordinary PDF and outputs a **PDF/UA-1 (ISO 14289-1) compliant** version that passes PAC 2024 and veraPDF validators. The output file is named `<original>_accessible.pdf`.

### Design principles

1. **Non-destructive tagging** - We modify the original PDF's content streams in-place by inserting BDC/EMC markers. We never reconstruct the PDF from scratch. This preserves the exact visual layout - every glyph stays in exactly the same position.

2. **Preserve existing work** - If the PDF already has embedded fonts, we don't re-embed them. If font descriptor metrics exist (`/Ascent`, `/Descent`, `/FontBBox`, etc.), we never overwrite them. If XMP metadata already has a title, we keep it (unless it's a generic placeholder like "Untitled").

3. **InDesign-aware** - PDFs from InDesign often have Optional Content Groups (OCG) with watermark layers, RoleMaps with self-mappings, and Type0/CIDFont structures. The tool handles all of these explicitly.

4. **Fail gracefully** - If one page can't be parsed, the tool wraps it in a fallback `/P` tag and continues with the rest of the document. If veraPDF isn't installed, the pipeline still runs (Stages 1-3). If a font can't be found for embedding, the tool logs a warning and continues. No single failure mode kills the pipeline.

5. **Artifact-as-default** - The content stream tagger opens an artifact wrapper at the start of every page. Any content that doesn't match an extracted block stays wrapped as an artifact. This means **nothing is ever untagged** in the output - a hard PDF/UA-1 requirement.

### Tech stack

| Component | Version | Purpose |
|---|---|---|
| Python | >= 3.10 | Runtime |
| pikepdf | 10.3.0 | Low-level PDF read/write, structure tree, content streams |
| pdfminer.six | 20260107 | Text extraction with font metrics, bounding boxes, colors |
| Pillow | >= 10.0.0 | Image conversion (XObject -> PNG) |
| fonttools | >= 4.0.0 | TTF loading, subsetting, metric extraction for embedding |
| langdetect | >= 1.0.9 | Document language detection |
| streamlit | 1.54.0 | Web UI |
| veraPDF | >= 1.24 | PDF/UA-1 validation (Java, optional) |

**Total codebase:** ~4,100 lines of Python across 9 modules.

---

## 3. PDF/UA-1 Crash Course

If you've never worked with PDF accessibility, read this section first. It explains the concepts the pipeline manipulates.

### What makes a PDF "accessible"?

An accessible PDF has an internal **structure tree** that tells assistive technology what each piece of content is (heading, paragraph, image, table cell) and what order to read it in. Without this tree, a screen reader just reads raw text in whatever order it finds it - which is often wrong (columns get interleaved, footnotes get read mid-sentence, etc.).

### Structure tags

Every piece of visible content in a PDF/UA-1 document must be tagged as either:
- A **structure element** (content the user should hear): `/P`, `/H1`-`/H6`, `/Figure`, `/Table`, `/TR`, `/TD`, `/TH`, `/L`, `/LI`, `/Link`, `/Form`
- An **artifact** (decorative content the user should NOT hear): watermarks, page numbers, headers/footers, background images

### How tagging works in PDF internals

PDF content streams are sequences of operators (like assembly language for rendering):

```
BT                        % Begin text object
/F1 12 Tf                 % Set font F1 at 12pt
100 700 Td                % Move cursor to (100, 700)
(Hello World) Tj          % Draw the text "Hello World"
ET                        % End text object
```

To tag this as a paragraph, we wrap it in BDC/EMC markers:

```
/P <</MCID 0>> BDC       % Begin Marked Content: Paragraph, ID=0
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
EMC                       % End Marked Content
```

The `MCID 0` is a **Marked Content Identifier** - it links this piece of the content stream to a node in the structure tree.

### The structure tree

The structure tree lives at `/StructTreeRoot` in the PDF catalog:

```
/StructTreeRoot
  /K -> /Document
    /K -> [
      /H1  (MCID 0, page 0)       "Annual Report 2024"
      /P   (MCID 1, page 0)       "This report summarizes..."
      /Figure (MCID 2, /Alt "Chart showing revenue growth", /BBox [...])
      /L                            List container
        /LI (MCID 3)               "First item"
        /LI (MCID 4)               "Second item"
      /Table                        Table container
        /TR                         Row
          /TH (MCID 5)             "Name"
          /TH (MCID 6)             "Value"
        /TR
          /TD (MCID 7)             "Revenue"
          /TD (MCID 8)             "$1.2M"
      /Link                         Hyperlink
        K -> [MCR(MCID 9), OBJR(annotation)]
    ]
```

### ParentTree

The **ParentTree** is a reverse index: given an MCID on a page, it tells you which structure element owns it. veraPDF checks this aggressively - every MCID must map to exactly one structure element.

### MCR and OBJR

- **MCR** (Marked Content Reference): points from a structure element to tagged content in the content stream (via MCID + page reference)
- **OBJR** (Object Reference): points from a structure element to a PDF object like a link annotation

Link elements need **both**: the MCR references the visible link text, the OBJR references the annotation (which holds the URL). Without both, PAC warns "Possibly inappropriate use of Link structure element" (Matterhorn 01-006).

### Artifacts

Content tagged as `/Artifact` is invisible to screen readers. The pipeline marks these as artifacts:
- Watermarks (rotated, large, light-colored text)
- Running headers (text repeated at top of every page)
- Running footers (text repeated at bottom, page numbers)
- Decorative images

Artifact marking uses `BMC` (Begin Marked Content without dictionary) or `BDC` with `/Artifact` type:

```
/Artifact <</Type /Pagination>> BDC
... header content ...
EMC
```

### Metadata requirements

PDF/UA-1 requires:

| Requirement | PDF location | What we set |
|---|---|---|
| Tagged PDF flag | `/MarkInfo /Marked true` | `true` |
| Suspects flag | `/MarkInfo /Suspects false` | `false` |
| Document language | `/Lang` on catalog | Detected via langdetect (e.g., "en") |
| Title display | `/ViewerPreferences /DisplayDocTitle true` | `true` |
| Tab order | `/Tabs /S` on every page | Structure-based tab order |
| Conformance claim | XMP `pdfuaid:part = 1` | Written via pikepdf metadata API |
| Meaningful title | XMP `dc:title` | Detected from PDF metadata / largest font / first heading |
| Font embedding | `/FontFile2` on font descriptor | TTF subset embedded |
| ToUnicode CMap | `/ToUnicode` on font object | Generated for WinAnsiEncoding |
| PDF version | Header | Saved as minimum 1.7 |

### Nesting rules

PDF/UA-1 requires specific parent-child relationships:

- `/LI` (list item) must be inside `/L` (list) - never directly under `/Document`
- `/TD`, `/TH` (table cells) must be inside `/TR` (table row) inside `/Table`
- `/Link`, `/Span` are inline elements - must be wrapped in `/P` as children of `/Document`
- `/P`, `/H1`-`/H6`, `/Figure`, `/L`, `/Table` are valid direct children of `/Document`

---

## 4. Pipeline Deep Dive

### Overview

```
Input PDF
    |
    v  Stage 1: pdf_extractor.py    extract_document()
    |  Stage 2: pdf_tagger.py       tag_pdf()
    |  Stage 3: pdf_postprocess.py  postprocess_pdf()
    |  Stage 4: validator.py        validate_pdf()
    v
Output: <name>_accessible.pdf
```

### Orchestration (main.py)

**Entry point:** `python main.py [options]`

The CLI orchestrator is simple - it collects input files, runs each through `process_single_pdf()`, and prints a summary table.

| Function | Location | Purpose |
|---|---|---|
| `main()` | `main.py:113` | Parses CLI args, collects PDFs, runs pipeline, prints summary |
| `process_single_pdf()` | `main.py:36` | Runs Stages 1-4 for one file, returns `PipelineResult` |
| `_print_summary()` | `main.py:182` | Formats summary table with PASS/WARN/ERROR per file |

**CLI flags:**

| Flag | Default | Purpose |
|---|---|---|
| `--input` / `-i` | None | Single PDF file to process |
| `--input-dir` / `-d` | `input/` | Directory of PDFs |
| `--output-dir` / `-o` | `output/` | Where to write results |
| `--skip-validation` | False | Skip veraPDF (Stage 4) |
| `--verbose` / `-v` | False | DEBUG-level logging |

**Pre-check:** Before processing, reads the first 8 bytes and rejects files not starting with `%PDF`.

---

### Stage 1: Content Extraction (`pdf_extractor.py`, ~839 lines)

**Entry point:** `extract_document(pdf_path: str) -> DocumentContent`

Parses the PDF and produces a structured representation with classification labels. This is the "understanding" stage.

#### Function-by-function breakdown

| Function | Line | Purpose |
|---|---|---|
| `extract_document()` | 30 | **Main entry.** Orchestrates all 10 phases, returns `DocumentContent` |
| `_extract_text_blocks()` | 95 | Phase 1: pdfminer.six parses pages into text boxes with font metrics. Uses `LAParams(line_margin=0.5, word_margin=0.1, char_margin=2.0, boxes_flow=0.5, detect_vertical=True)` |
| `_process_layout_element()` | 122 | Recursively walks pdfminer layout tree. Splits mixed-size text boxes (max/min ratio > 1.3) |
| `_split_text_box_by_lines()` | 183 | Groups consecutive lines with similar font sizes (tolerance: 1.5pt), creates separate TextBlocks per group |
| `_merge_split_blocks()` | 234 | Phase 2: Merges pdfminer drop-cap fragments. Short block (<=3 chars) + adjacent block -> merged. Max X gap: 30pt. Three adjacency strategies: left-adjacent, same-column, contained |
| `_dominant_font()` | 327 | Determines font name (by frequency), average size, bold/italic (from name keywords), color (from `graphicstate.ncolor` - handles RGB, CMYK, grayscale) |
| `_calc_rotation()` | 384 | Extracts rotation angle from PDF transformation matrix using `atan2(b, a)` |
| `_is_list_item()` | 396 | Regex check for bullet chars, numbered/lettered/roman prefixes, parenthesized markers |
| `_is_page_number()` | 426 | Regex check for `"1"`, `"- 1 -"`, `"Page 1"`, `"page 1 of 10"`, `"i"`, `"ii"` |
| `_detect_body_font_size()` | 445 | Phase 4: Weighted median - rounds sizes to nearest 0.5pt, counts by character count, returns most common |
| `_detect_single_page_hf()` | 458 | Fallback for single-page docs: detects page numbers and small footer text (< 9pt) in header/footer zones |
| `_detect_header_footer_signatures()` | 494 | Phase 5: Finds text repeated on 2+ pages at similar positions in top/bottom 8% zones. Normalizes text (lowercase, page numbers -> `__page_number__`) |
| `_classify_elements()` | 538 | Phase 6: Labels each block - watermark, header/footer, list item, heading (H1-H6), or paragraph. See [Section 6](#6-key-algorithms) for details |
| `_normalize_heading_hierarchy()` | 609 | Phase 7: Remaps heading levels to sequential (H3,H5 -> H1,H2). No skipped levels |
| `_detect_tables()` | 637 | Phase 8: Groups paragraphs by Y-coordinate (tolerance: `max(3, min(8, body_size*0.4))`). Requires 2+ rows with 2+ columns and consistent column count (allows 1 missing cell) |
| `_extract_images()` | 699 | Phase 3: pikepdf extracts XObject images, converts via Pillow to PNG. Matches to pdfminer layout positions. Alt text: `"Figure N on page M"` |
| `_detect_title()` | 768 | Phase 10: Five strategies in order: XMP dc:title -> DocInfo /Title -> largest font on page 1 (> 1.2x body) -> first heading -> first paragraph -> filename |
| `_is_meaningful_title()` | 825 | Rejects generic placeholders: "title", "untitled", "document", "microsoft word", etc. Also rejects titles < 3 chars |

---

### Stage 2: Structure Tag Injection (`pdf_tagger.py`, ~1,096 lines)

**Entry point:** `tag_pdf(input_path: str, output_path: str, doc_content: DocumentContent) -> str`

The most complex stage. Directly rewrites PDF content streams to inject structure markers, then builds the complete structure tree.

#### Function-by-function breakdown

| Function | Line | Purpose |
|---|---|---|
| `tag_pdf()` | 30 | **Main entry.** Opens PDF, removes existing structure, tags each page, builds structure tree, saves |
| `_safe_float()` | 68 | Converts pikepdf operands to float with fallback to 0.0 |
| `_resolve_resources()` | 76 | Gets page Resources, walking up the page tree with circular reference protection |
| `_get_xobjects()` | 99 | Gets XObject dictionary (handles inherited Resources) |
| `_remove_existing_structure()` | 111 | Deletes `/StructTreeRoot` and all `/StructParents` (clean slate) |
| `_tag_page()` | 124 | Tags one page: parse content stream, strip old markers, detect watermarks, collect link annotations, insert new markers, rebuild stream |
| `_tag_page_fallback()` | 151 | Emergency fallback: wraps entire page content in single `/P` tag. Used when content stream parsing fails |
| `_build_block_index()` | 179 | Converts Stage 1 blocks into position-indexed lookup list for matching |
| `_strip_markers()` | 212 | Removes all existing BDC/BMC/EMC from content stream |
| `_collect_link_annots()` | 221 | Collects link annotation rects. **Skips annotations wider than `min(500pt, page_width * 0.8)`** to prevent false matches from page-wide link areas |
| `_find_link_annot()` | 269 | Position matching for links with **asymmetric tolerances**: left=`max(2, width*0.1)`, right=`0.5`, vertical=`max(3, height*0.5)`. Tight right tolerance because text position is glyph start - past x1 means next content |
| `_insert_markers()` | 305 | **Core function.** Walks every operator in the content stream. Tracks CTM (Current Transformation Matrix), text matrix, leading. Inserts BDC/EMC around text (`Tj`/`TJ`/`'`/`"`), images (`Do`), handles `q`/`Q` boundaries, inline images (`BI`/`ID`/`EI`). Handles page rotation (90/180/270). See detailed walkthrough below |
| `_find_block()` | 630 | Adaptive tolerance position matching for text: `tol_y = max(20, height*0.3)`, `tol_x = max(20, width*0.15)`. Returns closest matching block index |
| `_find_image_block_by_position()` | 660 | CTM-based image matching with `tol = max(50, max(w,h)*0.5)`. Falls back to next sequential unused image if position match fails |
| `_detect_watermark_forms()` | 714 | Detects Form XObject watermarks via: (1) Adobe marker in `/PieceInfo`, (2) OCG named "Watermark", (3) keyword scan of content stream (limited to 10KB). Multilingual keywords: English, French, German, Spanish |
| `_get_xobject_subtype()` | 780 | Returns "Image", "Form", or "" for named XObject |
| `_build_structure_tree()` | 809 | Builds StructTreeRoot -> Document -> children. Includes second-pass heading normalization (PDF/UA Clause 7.4.2: forward jumps can't skip levels). Handles /Link elements with OBJR+MCR. Builds ParentTree with annotation StructParent entries |
| `_group_and_add_children()` | 951 | Groups leaf elements into proper containers: consecutive `/LI` -> `/L`, consecutive `/TD`/`/TH` -> `/Table`/`/TR`, inline `/Link`/`/Span` -> wrapped in `/P` |

#### How `_insert_markers()` works (the core loop)

The function maintains state as it walks through every operator:

**State tracked:**
- `ctm` - Current Transformation Matrix (6-element array), initialized based on page `/Rotate`
- `ctm_stack` - Stack for `q`/`Q` save/restore
- `tm`, `tlm` - Text matrix and text line matrix (reset at `BT`)
- `leading` - Text leading value (for `T*`, `'`, `"` operators)
- `artifact_open` / `struct_open` - Which wrapper type is currently active
- `current_link_idx` - Which link annotation is currently being tagged
- `linked_annot_indices` - Set of annotation indices that already have OBJR (prevents duplicates on `q`/`Q` splits)

**Operator handling:**

| Operator | Action |
|---|---|
| `q` (save) | Close any open marker. Push CTM. Open artifact wrapper. **Critical: markers can't cross q/Q boundaries** |
| `Q` (restore) | Close any open marker. Pop CTM. Open artifact wrapper |
| `cm` | Matrix multiply with CTM |
| `BT` | Reset text matrices, enter text mode |
| `ET` | Close struct marker, ensure artifact wrapper, exit text mode |
| `Tm` | Set text matrix directly |
| `Td`/`TD` | Translate text line matrix. `TD` also sets leading |
| `T*` | Apply leading to text line matrix |
| `TL` | Set leading value |
| `Tj`/`TJ`/`'`/`"` | **Text drawing.** Compute position `(ux, uy)` from CTM * text matrix. Check link annotations first (priority). Then match to extracted blocks. Unmatched text gets `/P` tag |
| `Do` | **XObject.** If watermark form -> artifact. If image -> match by CTM position, tag as `/Figure` with bbox and alt text. Otherwise -> artifact |
| `BI`/`ID`/`EI` | Inline images stay in artifact wrapper |
| Everything else | Stays in current wrapper |

**Position computation for text:**
```python
ux = ctm[0] * tm[4] + ctm[2] * tm[5] + ctm[4]
uy = ctm[1] * tm[4] + ctm[3] * tm[5] + ctm[5]
```

**Link tagging with `q`/`Q` splits:**
When a `q` or `Q` operator appears in the middle of link text, the marker must close and reopen. The first opening creates a `/Link` with OBJR; subsequent reopenings after `q`/`Q` create `/Span` elements (MCR only, no duplicate OBJR).

---

### Stage 3: Post-Processing (`pdf_postprocess.py`, ~1,002 lines)

**Entry point:** `postprocess_pdf(pdf_path: str, title: str, language: str, source_path: str = None) -> str`

Applies all remaining PDF/UA-1 compliance fixes that aren't part of structure tagging. Modifies the tagged PDF in-place. **Saves with `min_version="1.7"`** (PDF/UA-1 requires PDF 1.7+).

#### Function-by-function breakdown

| Function | Line | Purpose |
|---|---|---|
| `postprocess_pdf()` | 23 | **Main entry.** Opens tagged PDF, runs all fixes in order, saves |
| `_ensure_mark_info()` | 59 | Sets `/MarkInfo /Marked true` (preserves existing keys) |
| `_ensure_language()` | 66 | Sets `/Lang` on catalog |
| `_ensure_viewer_preferences()` | 70 | Sets `/ViewerPreferences /DisplayDocTitle true` |
| `_ensure_tab_order()` | 76 | Sets `/Tabs /S` (structure order) on every page |
| `_ensure_xmp_metadata()` | 88 | Writes XMP: `dc:title` (skips known placeholders), `dc:language`, `pdfuaid:part=1`, `pdf:Producer`, `xmp:CreatorTool`. **Placeholder titles rejected:** "title", "untitled", "document", "microsoft word", "powerpoint presentation", etc. (line 81) |
| `_ensure_role_map()` | 118 | Removes self-mappings (`/P -> /P`) and standard-to-standard mappings from RoleMap. Deletes empty RoleMap. Standard types defined at line 105 |
| `_fix_optional_content()` | 149 | Ensures OCProperties configs have `/Name` (defaults to "Default"), removes forbidden `/AS` key. Fixes both default config (`/D`) and alternates (`/Configs` array) |
| `_cleanup_empty_markers()` | 186 | Iteratively removes empty BDC/BMC...EMC pairs that contain no content operators. Uses a set of known content operators (text drawing, path construction/painting, XObject, etc.) |
| `_fix_annotations()` | 244 | **Backstop** for untagged annotations. Maps `/Link` -> `/Link` struct, `/Widget` -> `/Form` struct. Creates OBJR, wraps inline types in `/P`, assigns StructParent, updates ParentTree. Skips already-tagged annotations via `_is_already_tagged()` |
| `_derive_annot_contents()` | 370 | Extracts descriptive text for annotation `/Contents`: URI for links, GoTo/GoToR/Named action descriptions, field name or tooltip for widgets |
| `_is_already_tagged()` | 409 | Checks if annotation's StructParent already maps to correct element type in ParentTree |
| `_fix_cidset_streams()` | 446 | Removes `/CIDSet` from CID font descriptors (optional per spec, but if present and incomplete, causes failure). Checks both top-level and Type0 descendants |
| `_fix_cid_to_gid_map()` | 499 | Adds `/CIDToGIDMap /Identity` to embedded CIDFontType2 fonts missing it. Only operates on Type0 -> DescendantFonts chain |
| `_fix_fonts()` | 657 | **Main font fix orchestrator.** Iterates all fonts across all pages (deduped by object generation). For Type0 fonts, checks DescendantFonts for descriptor/embedding status. Adds ToUnicode for WinAnsi fonts, embeds unembedded fonts |
| `_is_simple_winansi()` | 722 | Returns True for `/WinAnsiEncoding` without `/Differences`. Handles Name, Dictionary, and String encoding objects |
| `_add_tounicode_cmap()` | 748 | Attaches generated CMap stream to font object |
| `_generate_winansi_tounicode()` | 755 | Generates ToUnicode CMap for Windows-1252. Maps 0x20-0xFF to Unicode. **Special mappings for 0x80-0x9F** (Euro 0x20AC, smart quotes, dashes, etc. - see `_WIN1252_SPECIAL` dict at line 552). Chunks into blocks of 100 per PDF CMap spec limit |
| `_try_embed_font()` | 797 | Finds system TTF, loads with fontTools, subsets to used glyphs (FirstChar-LastChar + Win1252 specials), embeds as `/FontFile2`. **Uses `_set_if_missing()` guard** (line 886) to preserve existing metrics |
| `_find_system_font()` | 922 | Multi-strategy font location: (1) exact match from `_FONT_FILE_NAMES` dict, (2) `BaseFont.ttf` direct, (3) fuzzy match (strip PS suffixes, CamelCase splitting), (4) TTC map for macOS, (5) case-insensitive partial scan of font dirs, (6) `fc-match` on Linux |
| `_resolve_page_resources()` | 635 | Same as tagger's - walks page tree with circular reference protection |

#### Font name to file mapping (`_FONT_FILE_NAMES`, line 566)

Maps PostScript names to TTF filenames. Each entry has Windows, macOS, and Liberation fallbacks:

```
TimesNewRomanPSMT       -> Times New Roman.ttf / times.ttf / LiberationSerif-Regular.ttf
ArialMT                 -> Arial.ttf / arial.ttf / LiberationSans-Regular.ttf
CourierNewPSMT          -> Courier New.ttf / cour.ttf / LiberationMono-Regular.ttf
Helvetica               -> Arial.ttf / LiberationSans-Regular.ttf (Helvetica has no free TTF)
Verdana, Georgia, Tahoma, Calibri, Cambria -> direct TTF names
+ Bold, Italic, BoldItalic variants for each
```

#### Font directories searched (`_FONT_DIRS`, line 611)

- **macOS:** `/System/Library/Fonts/Supplemental`, `/System/Library/Fonts`, `/Library/Fonts`, `~/Library/Fonts`
- **Linux:** `/usr/share/fonts/truetype`, `/usr/share/fonts/truetype/msttcorefonts`, `/usr/share/fonts/truetype/liberation`, `/usr/share/fonts`, `/usr/local/share/fonts`, `~/.fonts`, `~/.local/share/fonts`
- **Windows:** `%WINDIR%\Fonts` (default `C:\Windows\Fonts`)

#### TTC map (`_FONT_TTC_MAP`, line 596)

macOS-only. Maps Helvetica variants to `/System/Library/Fonts/Helvetica.ttc` with font indices 0-3 (Regular, Bold, Italic, BoldItalic).

---

### Stage 4: Validation (`validator.py`, ~201 lines)

**Entry point:** `validate_pdf(pdf_path: str) -> ValidationResult`

#### Function-by-function breakdown

| Function | Line | Purpose |
|---|---|---|
| `validate_pdf()` | 48 | **Main entry.** Finds veraPDF, sets JAVA_HOME, runs CLI, parses results. 120-second timeout. Returns `ValidationResult` with graceful degradation |
| `_parse_verapdf_json()` | 114 | Parses `report.jobs[0].validationResult` from veraPDF JSON. Extracts `compliant`, `passedRules`, `failedRules`, per-rule clause/testNumber/description/context |
| `format_validation_report()` | 181 | Formats human-readable report with clause numbers and context snippets |

**veraPDF command:** `verapdf -f ua1 --format json <pdf_path>`

**veraPDF search order:** `$PATH` first, then:
- `~/verapdf/verapdf`
- `/usr/local/bin/verapdf`
- `/opt/verapdf/verapdf`

**Java auto-detection** (if `JAVA_HOME` not set):
- macOS Homebrew: `/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home`
- Ubuntu: `/usr/lib/jvm/java-11-openjdk-amd64`
- Debian: `/usr/lib/jvm/default-java`

---

### Web UI (`app.py`, ~372 lines)

**Entry point:** `streamlit run app.py`

Runs Stages 1-3 only (no veraPDF). Features:
- Drag-and-drop upload (single or batch), max 100 MB per file (line 22)
- Progress indicators per stage: 10% -> 45% -> 80% -> 100%
- Metrics cards: Pages, Text Blocks, Images, Tables, Processing Time
- Per-file download buttons + batch ZIP download
- Dark theme with custom CSS, sidebar showing all fixes
- No authentication, no persistent storage (files are temporary)

---

## 5. Data Model

**File:** `models.py` (96 lines)

### Hierarchy visualization

```
DocumentContent
|-- title: str                    "Annual Report 2024"
|-- language: str                 "en"
|-- source_path: str              "/path/to/input.pdf"
|-- pages: list[PageContent]
    |-- page_number: int          0, 1, 2, ...
    |-- width: float              612.0  (points, US Letter)
    |-- height: float             792.0
    |-- text_blocks: list[TextBlock]
    |   |-- text: str             "This is a paragraph..."
    |   |-- bbox: BBox
    |   |   |-- x0, y0: float    Lower-left corner (PDF coordinates)
    |   |   |-- x1, y1: float    Upper-right corner
    |   |   |-- width: property   x1 - x0
    |   |   |-- height: property  y1 - y0
    |   |-- font: FontInfo
    |   |   |-- name: str         "ArialMT" (PostScript name)
    |   |   |-- size: float       12.0 (points)
    |   |   |-- is_bold: bool     Detected from font name keywords
    |   |   |-- is_italic: bool   Detected from font name keywords
    |   |   |-- color: tuple      (R, G, B) each 0.0-1.0
    |   |-- element_type: ElementType
    |   |   HEADING               -> /H1-/H6 structure tag
    |   |   PARAGRAPH             -> /P structure tag
    |   |   LIST_ITEM             -> /LI (inside /L)
    |   |   TABLE_CELL            -> /TD (inside /TR -> /Table)
    |   |   TABLE_HEADER          -> /TH (inside /TR -> /Table)
    |   |   IMAGE                 -> /Figure
    |   |   WATERMARK             -> /Artifact
    |   |   HEADER_FOOTER         -> /Artifact
    |   |-- heading_level: int?   1-6 for headings, None otherwise
    |   |-- rotation_degrees: float  0.0 normally, 15-75 for watermarks
    |   |-- page_number: int
    |
    |-- images: list[ImageBlock]
    |   |-- image_bytes: bytes    PNG data
    |   |-- format: str           "png"
    |   |-- bbox: BBox
    |   |-- page_number: int
    |   |-- alt_text: str         "Figure 1 on page 1"
    |   |-- is_decorative: bool   False (always, currently)
    |
    |-- tables: list[TableBlock]
        |-- rows: list[list[str]] Grid of cell text strings
        |-- header_rows: int      1 (always)
        |-- bbox: BBox?
        |-- page_number: int
```

### ElementType -> Structure tag mapping

| ElementType | PDF Structure Tag | Wrapper | Notes |
|---|---|---|---|
| `HEADING` | `/H1` - `/H6` | Direct child of `/Document` | Level from heading_level field |
| `PARAGRAPH` | `/P` | Direct child of `/Document` | Default for unclassified text |
| `LIST_ITEM` | `/LI` | Inside `/L` (List) | Consecutive items grouped |
| `TABLE_CELL` | `/TD` | Inside `/TR` -> `/Table` | |
| `TABLE_HEADER` | `/TH` | Inside `/TR` -> `/Table` | First row of detected tables |
| `IMAGE` | `/Figure` | Direct child of `/Document` | Gets `/Alt` and `/BBox` attributes |
| `WATERMARK` | `/Artifact` (Pagination/Watermark) | N/A | Invisible to screen readers |
| `HEADER_FOOTER` | `/Artifact` (Pagination) | N/A | Invisible to screen readers |

---

## 6. Key Algorithms

### Heading Detection

**File:** `pdf_extractor.py:538-606`

**Step 1:** Compute body font size = most common font size, weighted by character count, rounded to 0.5pt (`_detect_body_font_size`, line 445).

**Step 2:** For each text block, compute ratio = `block_font_size / body_font_size`.

**Step 3:** Apply classification rules in order:

```
ratio >= 1.8                                    -> H1
ratio >= 1.5                                    -> H2
ratio >= 1.25                                   -> H3
ratio >= 1.1 AND bold                           -> H4
bold AND ratio >= 1.0 AND text < 200 chars      -> H5
ALL CAPS AND ratio >= 0.95 AND text < 200 chars AND has letters -> H6
everything else                                  -> Paragraph
```

**Step 4:** Normalize heading hierarchy (`_normalize_heading_hierarchy`, line 609): Collect all used levels, remap to sequential starting at 1. If document uses H3 and H5, they become H1 and H2.

**Step 5:** Second-pass normalization in `_build_structure_tree` (line 819): Walk headings in document order; if a heading jumps more than +1 from the previous, clamp it (e.g., H1 -> H4 becomes H1 -> H2).

**Tuning:** Change thresholds in `config.py`. If body text is getting tagged as headings, raise the ratios. If real headings are missed, lower them.

### Header/Footer Detection

**File:** `pdf_extractor.py:494-535` (multi-page), `pdf_extractor.py:458-491` (single-page)

**Multi-page algorithm:**
1. For each page, identify text blocks in the top 8% (header zone) or bottom 8% (footer zone)
2. Normalize text: strip whitespace, lowercase, replace page number patterns with `__page_number__`
3. Count how many pages each (normalized_text, zone) pair appears on
4. Keep only pairs appearing on 2+ pages

**Single-page fallback:**
1. Check header/footer zones
2. Mark page numbers (detected by regex) as header/footer
3. Mark small text (< 9pt) in the footer zone as header/footer (copyright lines, disclaimers)

**Additional catch** (line 573): Text in footer zone with font size < 85% of body text is also classified as header/footer.

### Form XObject Watermark Detection

**File:** `pdf_tagger.py:714-773`

Form XObjects are mini-PDFs embedded as reusable resources. Some watermarks are implemented as Form XObjects rather than inline text.

**Three detection strategies:**

1. **Adobe marker:** Check `/PieceInfo` -> `/ADBE_CompoundType` -> `/Private` == `/Watermark`. This is how Adobe Acrobat stores watermarks.

2. **OCG name:** Check `/OC` (Optional Content Group) for "Watermark" in the group name. InDesign and some tools create watermarks in a layer named "Watermark".

3. **Keyword scan:** Read the Form XObject's content stream (limited to 10KB to avoid processing huge forms). Extract text between parentheses (Tj/TJ operands). Search for watermark keywords in **English, French, German, Spanish** (full list at line 695).

### Alt Text Index Alignment

**File:** `pdf_extractor.py:699-765` (extraction), `pdf_tagger.py:574-606` (tagging)

**The problem:** pdfminer and pikepdf see images differently. pdfminer finds layout positions; pikepdf extracts actual image data. They need to be matched.

**Extraction phase:**
1. pdfminer detects `LTImage` elements with bounding boxes -> placeholder `ImageBlock` entries
2. pikepdf iterates XObject resources -> extracts actual PNG data
3. For each pikepdf image: if there's a pdfminer placeholder at the same index, inherit its bbox. Otherwise, append as new entry.
4. Alt text is generated as `"Figure {index+1} on page {page+1}"`

**Tagging phase:**
1. All image blocks are indexed by position
2. When a `Do` operator appears for an Image XObject, compute the center position from CTM: `img_x = ctm[4] + ctm[0]*0.5 + ctm[2]*0.5`
3. Search for the closest unused image block within tolerance: `max(50pt, max(width, height) * 0.5)`
4. **Fallback:** If no position match, use the next sequential unused image block
5. Compute `/BBox` from CTM: `[ctm[4], ctm[5], ctm[4]+ctm[0], ctm[5]+ctm[3]]`

This two-pass approach (position-based + sequential fallback) handles cases where the CTM position doesn't perfectly align with pdfminer's layout analysis.

### List Item Detection

**File:** `pdf_extractor.py:396-420`

Compiled regex (cached globally) matches these patterns at the start of text:

| Pattern | Examples |
|---|---|
| Unicode bullet chars | `\u2022` (bullet), `\u2013` (en-dash), `\u00b7` (middle dot), 15+ more |
| Dash/asterisk bullets | `-`, `*`, `\u2010` (hyphen), `\u2011` (non-breaking hyphen) |
| Numbered | `1.`, `2)`, `10.` (1-3 digits + period or paren) |
| Lettered | `a.`, `b)`, `A.` (single letter + period or paren) |
| Roman | `i.`, `ii)`, `IV.` (1-4 roman chars + period or paren) |
| Parenthesized number | `(1)`, `(2)` |
| Parenthesized letter | `(a)`, `(b)` |

### Text Block Merging (Drop-Cap Fix)

**File:** `pdf_extractor.py:234-324`

pdfminer sometimes splits the first character of a paragraph into a separate text box (common with drop-cap styling or slight position offsets).

**Algorithm:**
1. For each pair of blocks, check Y-overlap or proximity (< 5pt)
2. One block must be short (<=3 chars) - this is the "fragment"
3. Check X-adjacency using three strategies:
   - Left-adjacent: gap between -10pt and 30pt
   - Same column: `|x0 difference| < 5pt`
   - Contained: short block's X range within long block's +/- 5pt
4. Merge: prepend fragment text to long block, union bounding boxes, keep long block's font info

---

## 7. Configuration Reference

**File:** `config.py` (30 lines)

Every tunable constant is here. Most behavior adjustments can be made without touching the Python modules.

| Constant | Default | Purpose | When to change |
|---|---|---|---|
| `HEADING_SIZE_RATIO_H1` | 1.8 | Font >= 1.8x body -> H1 | Body text getting tagged as H1: raise to 2.0 |
| `HEADING_SIZE_RATIO_H2` | 1.5 | Font >= 1.5x body -> H2 | Too many false H2s: raise to 1.7 |
| `HEADING_SIZE_RATIO_H3` | 1.25 | Font >= 1.25x body -> H3 | Real H3s missed: lower to 1.15 |
| `HEADING_SIZE_RATIO_H4` | 1.1 | Font >= 1.1x body AND bold -> H4 | Bold body text becoming H4: raise to 1.2 |
| `HEADER_ZONE_FRACTION` | 0.08 | Top 8% of page = header zone | Headers missed: raise to 0.12 |
| `FOOTER_ZONE_FRACTION` | 0.08 | Bottom 8% of page = footer zone | Footers missed: raise to 0.12 |
| `WATERMARK_MIN_ROTATION` | 15.0 | Min rotation angle (degrees) | Near-horizontal watermarks missed: lower to 10.0 |
| `WATERMARK_MAX_ROTATION` | 75.0 | Max rotation angle (degrees) | Near-vertical watermarks missed: raise to 85.0 |
| `WATERMARK_MIN_FONT_SIZE` | 36.0 | Min font size (points) | Small watermarks missed: lower to 24.0 |
| `WATERMARK_LIGHT_COLOR_THRESHOLD` | 0.7 | Min RGB channel value | Dark watermarks missed: lower to 0.5 |
| `VERAPDF_PROFILE` | "ua1" | veraPDF validation profile | Only change if targeting different standard |
| `DEFAULT_IMAGE_ALT` | "Figure" | Fallback alt text prefix | Customize per-project if needed |
| `DEFAULT_INPUT_DIR` | `Path("input")` | CLI input directory | Change for custom project layout |
| `DEFAULT_OUTPUT_DIR` | `Path("output")` | CLI output directory | Change for custom project layout |

---

## 8. Troubleshooting Guide

### veraPDF: Clause 7.21.3.2 - CIDToGIDMap missing

**Error:** `"CIDFont of subtype CIDFontType2 does not have a CIDToGIDMap"`

**Fix:** Already handled by `_fix_cid_to_gid_map()` (postprocess.py:499). If still failing: enable `--verbose`, check if the font is being found by `_resolve_page_resources()`.

### veraPDF: Clause 7.18.5 - Link annotation not in structure tree

**Error:** `"An annotation is not an indirect child of a structure element of type Link"`

**Fix:** Already handled by `_fix_annotations()` backstop (postprocess.py:244). If still failing: the annotation may have unusual `/Subtype` or `_is_already_tagged()` is returning a false positive. Debug by adding print in `_fix_annotations`:
```python
print(f"Annotation subtype={subtype}, struct_type={struct_type}")
```

### veraPDF: Clause 28-002 - Missing pdfuaid:part

**Error:** `"XMP document metadata does not identify PDF/UA as the conformance standard"`

**Fix:** `_ensure_xmp_metadata()` (postprocess.py:88). If failing: the XMP packet may have syntax issues. Check with `--verbose`.

### PAC: "Structure elements" check fails - N failures

**Cause:** `/Figure` elements missing `/BBox` and `/Placement`. The tagger computes bbox from CTM at `Do` operator (`pdf_tagger.py:594-604`) and attaches it as `/A << /O /Layout /BBox [...] /Placement /Block >>` in `_build_structure_tree` (line 894). If still failing, the image was matched via the sequential fallback - run with `--verbose`.

### PAC: "Possibly inappropriate use of Link structure element"

**Cause:** `/Link` has OBJR but no MCR (Matterhorn 01-006). Links must be detected during Stage 2 tagging so both MCR (text content) and OBJR (annotation ref) are present. Check `_collect_link_annots` (tagger.py:221) and `_find_link_annot` (tagger.py:269). The text position tolerance may need widening, or the annotation may be wider than the `max_annot_width` threshold.

### Wrong heading levels (body text tagged as H1, real headings missed)

**Quick fix:** Adjust ratios in `config.py`:
```python
HEADING_SIZE_RATIO_H1 = 2.0    # raise from 1.8
HEADING_SIZE_RATIO_H2 = 1.7    # raise from 1.5
```

**Deep fix:** The bold/ALL_CAPS heuristics at `pdf_extractor.py:597-606` catch non-heading bold text. Tighten the character limit from `< 200` to `< 80`.

### Headers/footers appear as paragraphs in structure tree

**Quick fix:** Widen zones in `config.py`:
```python
HEADER_ZONE_FRACTION = 0.12    # from 0.08
FOOTER_ZONE_FRACTION = 0.12    # from 0.08
```

**Deep fix:** Multi-page detection requires repetition on 2+ pages (line 534). For 2-page documents, lower to `>= 1` (but risks false positives).

### Watermarks not tagged as artifacts

**Quick fix:** Lower thresholds in `config.py`:
```python
WATERMARK_MIN_FONT_SIZE = 24.0          # from 36
WATERMARK_LIGHT_COLOR_THRESHOLD = 0.5   # from 0.7
```

**Deep fix:** Add keywords to `_WATERMARK_KEYWORDS` (tagger.py:695) or add detection logic to `_detect_watermark_forms()` (tagger.py:714).

### Text visually corrupted after processing

**Cause:** Font descriptor metrics were overwritten during embedding. The `_set_if_missing()` guard at `postprocess.py:886` must check `if key not in desc` before setting any value. If using plain assignment instead of the guard, that's the bug. Never overwrite existing `/Ascent`, `/Descent`, `/CapHeight`, `/FontBBox`, `/StemV`, `/Flags` - the original metrics are calibrated to the document's layout.

### Specific font not being embedded

1. Check `_FONT_FILE_NAMES` dict (postprocess.py:566). Add the PostScript name if missing.
2. Check font directories `_FONT_DIRS` (postprocess.py:611). Add custom paths if needed.
3. For TTC files: add to `_FONT_TTC_MAP` (postprocess.py:596).
4. On Linux: install `fonts-liberation` and/or `ttf-mscorefonts-installer`.

### Language detected incorrectly

Force override: hardcode in `pdf_extractor.py:79` or set `dc:language` in the original PDF's XMP metadata.

### Title is wrong or missing

Check the 5-strategy detection in `_detect_title()` (extractor.py:768). If the original PDF has a generic title, add it to `_PLACEHOLDER_TITLES` (postprocess.py:81) or `generic_titles` in `_is_meaningful_title` (extractor.py:830).

### Processing crashes on a specific page

Run with `--verbose`. The per-page error handler at tagger.py:52 logs the error and falls back to wrapping the page as `/P`. Check `_tag_page()` for the specific operator causing the crash.

### Multi-column layout produces wrong reading order

Adjust pdfminer's layout analysis in `pdf_extractor.py:97`:
```python
laparams = LAParams(
    boxes_flow=0.5,    # increase toward 1.0 for better column detection
)
```

### Encrypted PDF fails

Decrypt first: `pikepdf --password="yourpassword" input.pdf decrypted.pdf`

### Scanned (image-only) PDF produces empty structure

Run OCR first: `ocrmypdf input_scanned.pdf ocr_output.pdf`, then process `ocr_output.pdf`.

---

## 9. How to Extend

### Adding a new element type

1. Add to `ElementType` enum in `models.py`
2. Add detection logic in `pdf_extractor.py:_classify_elements()` (before the heading checks if it should take priority)
3. Map to PDF structure tag in `pdf_tagger.py:_element_to_struct_type()`
4. Handle nesting in `_group_and_add_children()` if needed (e.g., if the type needs a parent container)
5. Test with both PAC and veraPDF

### Adding AI-powered alt text

Replace the placeholder in `pdf_extractor.py:749`:
```python
# Current:
alt_text=f"Figure {img_index + 1} on page {page_idx + 1}",

# With AI:
try:
    alt_text = your_ai_client.describe_image(buf.getvalue())
except Exception:
    alt_text = f"Figure {img_index + 1} on page {page_idx + 1}"
```

Add your AI client library to `requirements.txt`. Always include fallback.

### Supporting a new font

Add to `_FONT_FILE_NAMES` (postprocess.py:566):
```python
"YourFontPSName": ["YourFont.ttf", "yourfont.ttf"],
```

For TTC files, add to `_FONT_TTC_MAP` (postprocess.py:596):
```python
"YourFontName": ("/path/to/fonts.ttc", 0),  # 0 = index in collection
```

### Adding new image patterns (e.g., decorative detection)

In `_extract_images()` (extractor.py:699), after extracting each image:
```python
# Detect decorative images (very small, or single-color)
if pil_image.size[0] < 10 and pil_image.size[1] < 10:
    image_block.is_decorative = True
```

Decorative images are tagged as artifacts (the tagger checks `block.get("is_artifact")` which includes `is_decorative`).

### Adding a new post-processing fix

1. Add a function in `pdf_postprocess.py` following the pattern `_fix_yourname(pdf: pikepdf.Pdf)`
2. Call it from `postprocess_pdf()` (line 35-46) in the correct order
3. Use `_resolve_page_resources()` for per-page resource access
4. Deduplicate by `objgen` to avoid processing shared objects twice
5. Test with both PAC and veraPDF

### Adding a new validation backend

Create a function in `validator.py`:
```python
def validate_with_custom(pdf_path: str) -> ValidationResult:
    # Run your validator
    # Parse results
    # Return ValidationResult(pdf_path=..., is_compliant=..., ...)
```

Wire it into `main.py:process_single_pdf()` alongside or instead of veraPDF.

### Adding support for a new PDF source (e.g., new PDF generator)

Common issues per PDF generator:
- **InDesign:** OCG watermark layers, RoleMap self-mappings, CIDFont structures
- **Word:** Generic titles ("Microsoft Word Document"), WinAnsiEncoding fonts
- **PowerPoint:** Slide-oriented layout, many images
- **LaTeX:** Type1 fonts (not supported for embedding), custom encoding

For new generators: run a sample through the pipeline with `--verbose`, check what fails, and add fixes to the appropriate stage.

---

## 10. Validation & Testing

### Manual verification steps

1. **Run the pipeline:**
   ```bash
   python main.py --input test.pdf --verbose
   ```

2. **Check PAC:**
   - Open `output/test_accessible.pdf` in PAC 2024
   - All checks should show green checkmarks
   - Check "Structure Elements" section for /Figure BBox
   - Check "Alternative Descriptions" for /Figure Alt text
   - Check "Metadata" for pdfuaid:part

3. **Check veraPDF:**
   ```bash
   verapdf -f ua1 output/test_accessible.pdf
   ```

4. **Visual comparison:**
   - Open original and output side-by-side
   - Text should be in the exact same positions
   - No shifted/overlapping text (font metrics bug)
   - Images should be intact

5. **Screen reader test:**
   - Open in Adobe Acrobat with NVDA/JAWS/VoiceOver
   - Tab through structure
   - Verify heading hierarchy makes sense
   - Verify images read alt text
   - Verify links are navigable
   - Verify watermarks/headers/footers are NOT read

### End-to-end test cases

| Test case | Input | Expected |
|---|---|---|
| Simple document | 1-page text PDF | All text tagged as /P, title detected |
| Headings | PDF with H1-H4 by font size | Correct heading levels, no skips |
| Images | PDF with 3 images | All tagged as /Figure with /Alt and /BBox |
| Tables | PDF with 2x3 table | /Table -> /TR -> /TH + /TD structure |
| Lists | PDF with bullet/numbered list | /L -> /LI structure |
| Links | PDF with hyperlinks | /Link with MCR + OBJR |
| Watermarks | PDF with diagonal watermark | Watermark tagged as /Artifact |
| Headers/footers | Multi-page with repeating header | Headers tagged as /Artifact |
| Encrypted | Password-protected PDF | Raises pikepdf.PasswordError |
| Scanned | Image-only PDF | Minimal structure (no text blocks found) |
| Multi-language | German PDF | `/Lang (de)` detected |
| Large fonts | PDF where all text is 24pt | Body size = 24pt, no false H1s |
| Type0 fonts | CIDFont PDF | CIDToGIDMap added, ToUnicode preserved |
| InDesign PDF | OCG + RoleMap | OCProperties fixed, self-mappings removed |

### Automated testing strategy

Currently no automated test suite. To add one:

1. Create `tests/` directory with sample PDFs (tiny, synthetic)
2. Use pytest with pikepdf to verify output structure:
   ```python
   def test_structure_tree_exists():
       # Run pipeline
       postprocess_pdf(tagged_path, "Test", "en")
       pdf = pikepdf.Pdf.open(tagged_path)
       assert "/StructTreeRoot" in pdf.Root
       assert pdf.Root.MarkInfo.Marked == True
   ```
3. Use veraPDF in CI (requires Java in CI environment)

---

## 11. Environment Setup

### Local development

**Repository:** [https://github.com/Mohakgarg5/PdfRemediationTool](https://github.com/Mohakgarg5/PdfRemediationTool)

```bash
# Clone
git clone https://github.com/Mohakgarg5/PdfRemediationTool.git
cd PdfRemediationTool

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run CLI
python main.py --input test.pdf

# Run web UI
streamlit run app.py
```

### Streamlit Cloud deployment

The repo includes:
- `requirements.txt` - Python dependencies
- `packages.txt` - System packages (`fonts-liberation` for Linux font embedding)

Steps:
1. Push to GitHub
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Set `app.py` as main file
4. Deploy

Note: veraPDF is NOT available on Streamlit Cloud (no Java). Web UI runs Stages 1-3 only.

### Docker deployment (example)

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y fonts-liberation && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### CI environment requirements

- Python 3.10+
- `fonts-liberation` package (apt-get)
- Java 11+ if using veraPDF validation
- Memory: large PDFs with many images can use 500MB+

### Git workflow

- `main` branch is the primary branch
- No protected branches or PR requirements currently
- Commit messages: descriptive, one-line summary

### Utility scripts

**`run.sh`** - Convenience wrapper. Uses venv Python at `venv/bin/python3.12` (line 9). If your Python version differs, update the path. Checks venv existence and prints setup instructions if missing.

**`make_guide_pdf.py`** - Converts `GUIDE.md` to `GUIDE.pdf` using reportlab. Creates a styled PDF with cover page, colored headings, code blocks, and tables. Run: `python make_guide_pdf.py`. Note: both `GUIDE.md` and `make_guide_pdf.py` are gitignored.

---

## 12. Stakeholders & Review Process

### Key people

- **Mohak Garg** (mohakgarg2026@u.northwestern.edu) - Primary developer and maintainer. Built the entire pipeline. Contact for architecture questions, bug triage, and feature decisions.

- **Katharine** - Reviews the tool's output for accessibility quality and compliance. Her feedback drives priority fixes. When Katharine reports a PAC or veraPDF failure on a specific document, treat it as high priority.

### Review workflow

1. **Mohak processes sample PDFs** through the pipeline
2. **Output sent to Katharine** for PAC validation and manual screen reader testing
3. **Katharine reports** any failures with PAC screenshots and specific clause numbers
4. **Mohak diagnoses and fixes** the issue, typically in Stage 2 (tagging) or Stage 3 (post-processing)
5. **Re-test** the specific document plus regression testing on previous samples

### Communication notes

- Bug reports from Katharine typically reference **Matterhorn Protocol clause numbers** (e.g., "01-006") and **PAC check names** (e.g., "Structure elements"). These map directly to code:
  - Clause numbers -> look at the relevant `_fix_*` function in `pdf_postprocess.py`
  - PAC check names -> search the codebase for the PDF structure type mentioned

- When Katharine says "this PDF doesn't pass PAC", the first step is always: run it through the pipeline with `--verbose` and check which specific clause/check failed. Don't guess.

### If Mohak is unavailable

1. **For bug reports:** This document's [Troubleshooting Guide](#8-troubleshooting-guide) covers the most common failures with exact file locations and fixes.

2. **For new features:** Follow the [How to Extend](#9-how-to-extend) section. The pipeline is modular - each stage can be modified independently.

3. **For deployment issues:** Check [Environment Setup](#11-environment-setup). Font embedding failures on servers are the most common deployment issue (install `fonts-liberation`).

4. **For understanding the code:** Start with `main.py` (the orchestrator), then read the stage you need to modify. The entry points are: `extract_document()`, `tag_pdf()`, `postprocess_pdf()`, `validate_pdf()`.

---

## 13. Glossary

| Term | Definition |
|---|---|
| **PDF/UA-1** | ISO 14289-1 - Universal Accessibility standard for PDFs. Defines what makes a PDF accessible to assistive technology |
| **PAC** | PDF Accessibility Checker - validator by the Swiss PDF Association. The industry-standard tool for checking PDF/UA compliance |
| **veraPDF** | Open-source PDF validator maintained by the PDF Association and Digital Preservation Coalition. More strict than PAC on some checks |
| **Matterhorn Protocol** | Set of 136 failure conditions for PDF/UA-1 testing, maintained by the PDF Association. Each has a clause number (e.g., 01-006) |
| **BDC** | Begin Marked Content with Dictionary - PDF operator that starts a tagged content region. Takes a tag name and property dictionary with MCID |
| **BMC** | Begin Marked Content - PDF operator that starts a marked content region without properties. Used for artifacts |
| **EMC** | End Marked Content - PDF operator that closes a BDC or BMC region |
| **MCID** | Marked Content Identifier - integer that links a content stream region (BDC/EMC) to a structure tree element |
| **MCR** | Marked Content Reference - structure tree entry pointing to tagged content (MCID + page reference) |
| **OBJR** | Object Reference - structure tree entry pointing to a PDF object (e.g., link annotation) |
| **StructTreeRoot** | Root of the PDF's logical structure tree. Contains the /Document element, ParentTree, and RoleMap |
| **ParentTree** | Number tree (reverse index) mapping MCIDs and StructParent keys to their parent structure elements |
| **RoleMap** | Dictionary mapping custom tag names to standard PDF structure types. Self-mappings of standard types are a violation |
| **CTM** | Current Transformation Matrix - 6-element array `[a,b,c,d,e,f]` tracking position, scale, and rotation state in a content stream |
| **Text Matrix (Tm)** | Transformation matrix for text positioning within a BT/ET text object. Combined with CTM to get absolute position |
| **CIDFont** | Character-ID font - used in Type0 composite font structures. Common for CJK (Chinese/Japanese/Korean) and modern fonts |
| **CIDFontType2** | A CIDFont backed by a TrueType font program. Requires `/CIDToGIDMap` per ISO 32000-1 |
| **CIDToGIDMap** | Maps Character IDs to Glyph IDs in a TrueType-backed CIDFont. `/Identity` means 1:1 mapping |
| **CIDSet** | Optional stream in CIDFont descriptor listing which CIDs are present. If incomplete, causes veraPDF failure - safer to remove |
| **ToUnicode** | CMap (Character Map) that maps character codes to Unicode code points. Essential for text extraction and accessibility |
| **WinAnsiEncoding** | Windows-1252 character encoding. Used by most Western fonts in PDF. Bytes 0x80-0x9F map differently than Latin-1 |
| **XMP** | Extensible Metadata Platform - XML-based metadata format embedded in PDFs. Carries dc:title, pdfuaid:part, etc. |
| **pikepdf** | Python library for low-level PDF read/write, based on QPDF. Used for content stream manipulation, structure tree building, and font embedding |
| **pdfminer.six** | Python library for PDF text extraction with layout analysis. Provides font metrics, bounding boxes, and color information |
| **fonttools** | Python library for TTF/OTF font file manipulation. Used for loading, subsetting, and reading font metrics |
| **TTF** | TrueType Font - font format with `.ttf` extension. The format we embed into PDFs |
| **TTC** | TrueType Collection - multiple fonts bundled in a single `.ttc` file. Common on macOS (e.g., Helvetica.ttc) |
| **Liberation fonts** | Open-source fonts metrically compatible with Arial (Liberation Sans), Times New Roman (Liberation Serif), and Courier New (Liberation Mono). Used as fallbacks on Linux |
| **OCG** | Optional Content Group - PDF layers. Used by some tools for watermarks. Controlled via OCProperties in the catalog |
| **OCProperties** | PDF catalog entry managing Optional Content (layers). Must have `/Name` on configs, must not have `/AS` |
| **/Artifact** | PDF tag for decorative/non-content elements that screen readers should ignore (watermarks, page numbers, headers/footers) |
| **Form XObject** | Reusable mini-PDF embedded as a resource. Some watermarks are implemented as Form XObjects |
| **Subsetting** | Reducing a font to only the glyphs actually used in the document. Reduces file size and avoids licensing issues |
| **`_set_if_missing()`** | Critical guard function (postprocess.py:886) that only sets font descriptor metrics if absent. Prevents visual corruption |

---

*This document covers the complete system as of April 2026. If something here contradicts what you see in the code, the code is the source of truth - update this document.*
