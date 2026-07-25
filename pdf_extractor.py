"""
pdf_extractor.py - Extract structured content from a PDF.

Uses pdfminer.six for text extraction with full font/position/color metadata,
and pikepdf for reliable image extraction.
"""
import logging
import math
from pathlib import Path
from collections import Counter
from io import BytesIO

from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LAParams, LTTextBox, LTTextLine, LTChar, LTAnno,
    LTFigure, LTImage, LTPage, LTRect, LTCurve,
)
import pikepdf
from langdetect import detect as detect_language

from models import (
    DocumentContent, PageContent, TextBlock, ImageBlock,
    TableBlock, TableCell, FontInfo, BBox, ElementType,
)
import config

logger = logging.getLogger(__name__)


def extract_document(pdf_path: str) -> DocumentContent:
    """Main entry point: extract all content from a PDF file.

    Raises pikepdf.PasswordError for encrypted PDFs.
    """
    # Pre-check: verify PDF is readable and not encrypted
    try:
        _pdf = pikepdf.Pdf.open(pdf_path)
        _pdf.close()
    except pikepdf.PasswordError:
        logger.error("PDF is encrypted/password-protected: %s", pdf_path)
        raise
    except Exception as e:
        logger.warning("PDF pre-check warning: %s", e)

    # Phase 1: Extract text blocks with full metadata via pdfminer
    raw_pages = _extract_text_blocks(pdf_path)

    # Phase 2: Merge split text blocks (drop-cap / first-char splits)
    for page in raw_pages:
        page.text_blocks = _merge_split_blocks(page.text_blocks)

    # Phase 3: Extract images via pikepdf
    _extract_images(pdf_path, raw_pages)

    # Phase 4: Detect the body font size (most common size)
    body_font_size = _detect_body_font_size(raw_pages)

    # Phase 5: Detect repeated header/footer text across pages
    hf_signatures = _detect_header_footer_signatures(raw_pages)

    # Phase 6: Classify elements
    for page in raw_pages:
        _classify_elements(page, body_font_size, hf_signatures)

    # Phase 7: Normalize heading hierarchy (start at H1, no skipped levels)
    _normalize_heading_hierarchy(raw_pages)

    # Phase 8: Detect tables
    for page in raw_pages:
        _detect_tables(page, body_font_size)

    # Phase 9: Detect document language (use all text, not just paragraphs)
    all_text = " ".join(
        tb.text for p in raw_pages for tb in p.text_blocks
        if tb.element_type not in (ElementType.WATERMARK, ElementType.HEADER_FOOTER)
        and tb.text.strip()
    )
    try:
        language = detect_language(all_text[:5000]) if all_text.strip() else "en"
    except Exception as e:
        logger.warning("Language detection failed, defaulting to 'en': %s", e)
        language = "en"

    # Phase 10: Extract title (try PDF metadata first, then largest font)
    title = _detect_title(raw_pages, pdf_path)

    return DocumentContent(
        title=title,
        language=language,
        pages=raw_pages,
        source_path=pdf_path,
    )


def _extract_text_blocks(pdf_path: str) -> list:
    """Use pdfminer.six extract_pages to get text with full metadata."""
    laparams = LAParams(
        line_margin=0.5,
        word_margin=0.1,
        char_margin=2.0,
        boxes_flow=0.5,
        detect_vertical=True,
    )

    pages = []
    for page_num, page_layout in enumerate(extract_pages(pdf_path, laparams=laparams)):
        page_content = PageContent(
            page_number=page_num,
            width=page_layout.width,
            height=page_layout.height,
        )
        try:
            for element in page_layout:
                _process_layout_element(element, page_content, page_num)
        except Exception as e:
            logger.warning("Partial extraction on page %d: %s", page_num, e)
        pages.append(page_content)

    return pages


def _process_layout_element(element, page_content: PageContent, page_num: int):
    """Recursively process layout elements.

    When a text box contains lines with very different font sizes,
    splits it into separate blocks per line group so headings aren't
    averaged with body text.
    """
    if isinstance(element, LTTextBox):
        # Collect per-line info to detect mixed-size boxes
        line_infos = []  # (text, chars, line_obj)
        for line in element:
            if isinstance(line, LTTextLine):
                line_chars = [item for item in line if isinstance(item, LTChar)]
                line_text = line.get_text().strip()
                if line_chars and line_text:
                    line_infos.append((line_text, line_chars, line))

        if not line_infos:
            return

        # Check if lines have significantly different font sizes
        line_sizes = []
        for text, chars, line_obj in line_infos:
            avg_size = sum(c.size for c in chars) / len(chars)
            line_sizes.append(avg_size)

        # If max/min ratio > 1.3, split by line groups of similar size
        min_size = min(line_sizes) if line_sizes else 1.0
        if min_size < 1.0:
            min_size = 1.0
        if line_sizes and max(line_sizes) / min_size > 1.3:
            _split_text_box_by_lines(line_infos, line_sizes, page_content, page_num)
        else:
            # All lines similar size — create one block
            all_chars = [c for _, chars, _ in line_infos for c in chars]
            full_text = "\n".join(t for t, _, _ in line_infos)
            font_info = _dominant_font(all_chars)
            rotation = _calc_rotation(all_chars[0].matrix) if all_chars else 0.0

            text_block = TextBlock(
                text=full_text,
                bbox=BBox(element.x0, element.y0, element.x1, element.y1),
                font=font_info,
                rotation_degrees=rotation,
                page_number=page_num,
            )
            page_content.text_blocks.append(text_block)

    elif isinstance(element, LTFigure):
        for child in element:
            _process_layout_element(child, page_content, page_num)

    elif isinstance(element, LTImage):
        page_content.images.append(ImageBlock(
            image_bytes=b"",
            format="unknown",
            bbox=BBox(element.x0, element.y0, element.x1, element.y1),
            page_number=page_num,
        ))

    elif isinstance(element, (LTRect, LTCurve)):
        # Filled rectangles form table cell backgrounds; capture them so tables
        # can be reconstructed from their true cell grid rather than guessed
        # from text positions.
        if getattr(element, "fill", False):
            w = element.x1 - element.x0
            h = element.y1 - element.y0
            if w >= 20 and h >= 8:
                page_content.fill_rects.append(
                    BBox(element.x0, element.y0, element.x1, element.y1)
                )


def _split_text_box_by_lines(line_infos, line_sizes, page_content, page_num):
    """Split a text box into separate blocks when lines have different font sizes.

    Groups consecutive lines with similar sizes together.
    """
    SIZE_TOLERANCE = 1.5  # lines within this pt difference are grouped

    groups = []  # list of (texts, chars, line_objs)
    current_group_texts = []
    current_group_chars = []
    current_group_lines = []
    current_group_size = line_sizes[0]

    for i, (text, chars, line_obj) in enumerate(line_infos):
        size = line_sizes[i]
        if abs(size - current_group_size) <= SIZE_TOLERANCE:
            current_group_texts.append(text)
            current_group_chars.extend(chars)
            current_group_lines.append(line_obj)
        else:
            if current_group_texts:
                groups.append((current_group_texts, current_group_chars, current_group_lines))
            current_group_texts = [text]
            current_group_chars = list(chars)
            current_group_lines = [line_obj]
            current_group_size = size

    if current_group_texts:
        groups.append((current_group_texts, current_group_chars, current_group_lines))

    for texts, chars, lines in groups:
        full_text = "\n".join(texts)
        font_info = _dominant_font(chars)
        rotation = _calc_rotation(chars[0].matrix) if chars else 0.0

        # Calculate bbox from the lines
        x0 = min(l.x0 for l in lines)
        y0 = min(l.y0 for l in lines)
        x1 = max(l.x1 for l in lines)
        y1 = max(l.y1 for l in lines)

        text_block = TextBlock(
            text=full_text,
            bbox=BBox(x0, y0, x1, y1),
            font=font_info,
            rotation_degrees=rotation,
            page_number=page_num,
        )
        page_content.text_blocks.append(text_block)


def _merge_split_blocks(text_blocks: list) -> list:
    """Merge text blocks split by pdfminer (e.g. 'P' + 'ublicly Available').

    pdfminer sometimes splits the first character into a separate text box
    when it has a slightly different position or formatting (drop caps, etc.).
    This merges them back together.
    """
    if len(text_blocks) < 2:
        return text_blocks

    X_GAP_MAX = 30.0   # max gap between blocks to consider them adjacent
    SHORT_THRESHOLD = 3  # blocks with <= this many chars are merge candidates

    merged = []
    used = set()

    indexed = list(enumerate(text_blocks))

    for i, block_a in indexed:
        if i in used:
            continue

        # Look for a short fragment to merge with this block, or vice versa
        best_merge = None
        for j, block_b in indexed:
            if j in used or j == i:
                continue

            # Check if blocks are on the same line using Y overlap or proximity
            y_overlap = (min(block_a.bbox.y1, block_b.bbox.y1) -
                         max(block_a.bbox.y0, block_b.bbox.y0))
            y_close = abs(block_a.bbox.y0 - block_b.bbox.y0) < 5.0
            if y_overlap <= 0 and not y_close:
                continue

            # One must be short (1-3 chars)
            a_short = len(block_a.text.strip()) <= SHORT_THRESHOLD
            b_short = len(block_b.text.strip()) <= SHORT_THRESHOLD
            if not a_short and not b_short:
                continue

            # Determine which is the short (fragment) and long block
            if a_short:
                short_b, long_b = block_a, block_b
                short_idx, long_idx = i, j
            else:
                short_b, long_b = block_b, block_a
                short_idx, long_idx = j, i

            # Check X adjacency using multiple strategies:
            # 1. Short block is left-adjacent to long block (normal split)
            x_gap = long_b.bbox.x0 - short_b.bbox.x1
            left_adjacent = -10.0 <= x_gap <= X_GAP_MAX

            # 2. Blocks share same x0 (drop-cap style: tall first letter)
            same_column = abs(short_b.bbox.x0 - long_b.bbox.x0) < 5.0

            # 3. Short block's x range is within long block's x range
            contained = (short_b.bbox.x0 >= long_b.bbox.x0 - 5.0 and
                         short_b.bbox.x1 <= long_b.bbox.x1 + 5.0)

            if left_adjacent or same_column or contained:
                best_merge = (short_b, long_b, short_idx, long_idx)
                break

        if best_merge:
            short_b, long_b, short_idx, long_idx = best_merge
            # Prepend short fragment to long block text
            merged_text = short_b.text.strip() + long_b.text.strip()
            merged_bbox = BBox(
                x0=min(short_b.bbox.x0, long_b.bbox.x0),
                y0=min(short_b.bbox.y0, long_b.bbox.y0),
                x1=max(short_b.bbox.x1, long_b.bbox.x1),
                y1=max(short_b.bbox.y1, long_b.bbox.y1),
            )
            # Use the longer block's font info (more representative)
            merged_block = TextBlock(
                text=merged_text,
                bbox=merged_bbox,
                font=long_b.font,
                rotation_degrees=long_b.rotation_degrees,
                page_number=long_b.page_number,
            )
            merged.append(merged_block)
            used.add(short_idx)
            used.add(long_idx)
        else:
            merged.append(block_a)
            used.add(i)

    return merged


def _dominant_font(chars: list) -> FontInfo:
    """Determine the dominant font in a list of LTChar objects."""
    if not chars:
        return FontInfo(name="unknown", size=12.0, is_bold=False,
                        is_italic=False, color=(0.0, 0.0, 0.0))

    font_counter = Counter()
    sizes = []

    for ch in chars:
        try:
            font_counter[ch.fontname] += 1
            sizes.append(ch.size)
        except (AttributeError, TypeError):
            continue

    if not font_counter:
        return FontInfo(name="unknown", size=12.0, is_bold=False,
                        is_italic=False, color=(0.0, 0.0, 0.0))

    dominant_name = font_counter.most_common(1)[0][0]
    avg_size = sum(sizes) / len(sizes) if sizes else 12.0

    name_lower = dominant_name.lower() if dominant_name else ""
    is_bold = any(kw in name_lower for kw in ["bold", "black", "heavy", "demi"])
    is_italic = any(kw in name_lower for kw in ["italic", "oblique", "slant"])

    # Extract color safely from graphicstate
    color = (0.0, 0.0, 0.0)
    try:
        gs = getattr(chars[0], 'graphicstate', None)
        ncolor = getattr(gs, 'ncolor', None) if gs else None
        if ncolor is not None:
            if isinstance(ncolor, (list, tuple)):
                if len(ncolor) == 3:
                    color = tuple(float(c) for c in ncolor)
                elif len(ncolor) == 4:
                    c, m, y, k = [float(v) for v in ncolor]
                    color = ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
                elif len(ncolor) >= 1:
                    g = float(ncolor[0])
                    color = (g, g, g)
            elif isinstance(ncolor, (int, float)):
                g = float(ncolor)
                color = (g, g, g)
    except (ValueError, TypeError, IndexError, AttributeError):
        color = (0.0, 0.0, 0.0)

    return FontInfo(
        name=dominant_name or "unknown",
        size=avg_size,
        is_bold=is_bold,
        is_italic=is_italic,
        color=color,
    )


def _calc_rotation(matrix: tuple) -> float:
    """Extract rotation angle in degrees from a PDF transformation matrix."""
    if not matrix or len(matrix) < 2:
        return 0.0
    a, b = matrix[0], matrix[1]
    angle_rad = math.atan2(b, a)
    return math.degrees(angle_rad)


_LIST_ITEM_RE = None


def _is_list_item(text: str) -> bool:
    """Check if text starts with a bullet or numbered list marker.

    Handles: bullets (-, *, bullet chars), numbers (1., 2), letters (a., b)),
    and roman numerals (i., ii.) as list prefixes.
    """
    global _LIST_ITEM_RE
    import re
    if _LIST_ITEM_RE is None:
        _LIST_ITEM_RE = re.compile(
            r'^\s*(?:'
            r'[\u2022\u2023\u2043\u25aa\u25ab\u25b8\u25b9\u25cb\u25cf\u25e6\u2013\u2014\u2219\u00b7]'  # bullet chars
            r'|[-*\u2010\u2011]'       # dash/asterisk bullets
            r'|\d{1,3}[.)]\s'          # numbered: 1. 2) 10.
            r'|[a-zA-Z][.)]\s'         # lettered: a. b) A.
            r'|[ivxIVX]{1,4}[.)]\s'    # roman: i. ii) IV.
            r'|\(\d{1,3}\)\s'          # parenthesized: (1) (2)
            r'|\([a-zA-Z]\)\s'         # parenthesized: (a) (b)
            r')',
            re.UNICODE,
        )
    text = text.strip()
    if not text or len(text) < 2:
        return False
    return bool(_LIST_ITEM_RE.match(text))


_HEADING_PROSE_RE = None


def _looks_like_heading_text(text: str) -> bool:
    """Heuristic gate: is this text plausibly a heading rather than body prose?

    Font size / bold alone over-fires — bold body sentences, lead-in
    sentences, and citations get mis-tagged as headings, which breaks
    heading-based navigation (WCAG 1.3.1). A real heading is a short label,
    not a sentence. This rejects:
      - empty / whitespace-only text
      - long text (headings are short)
      - multi-sentence prose (sentence punctuation followed by more text)
      - full sentences that end in terminal/lead-in punctuation

    It is intentionally conservative: it only vetoes text that clearly reads
    as prose, so genuine headings (even long ones) still pass.
    """
    global _HEADING_PROSE_RE
    import re
    if _HEADING_PROSE_RE is None:
        # sentence break: .?! or ) followed by whitespace then an opening
        # character (capital letter, digit, quote, or paren) => >1 sentence
        _HEADING_PROSE_RE = re.compile(r'[.?!)]\s+[\"\'(A-Z0-9]')

    t = " ".join(text.split())  # normalize newlines / runs of whitespace
    if not t:
        return False
    if len(t) > 120:
        return False
    words = t.split()
    # Headings are short labels. A dozen words is already generous; anything
    # longer reads as a sentence/clause.
    if len(words) > 12:
        return False
    # Multi-sentence prose is never a heading.
    core = t[:-1] if t[-1] in ".?!" else t
    if _HEADING_PROSE_RE.search(core):
        return False
    # A clause of several words that ends in terminal/lead-in punctuation
    # (". " "? " "! " ": " "; ") reads as a sentence or question, not a label.
    # Short labels that end in a colon ("Note:", "Table 1:") are kept.
    if t[-1] in ".:;?!" and len(words) >= 6:
        return False
    # Headings are capitalized. Text whose first letter is lowercase is almost
    # always a prose fragment split off mid-sentence ("concede?", "based
    # outcomes?"), not a heading.
    first_alpha = next((c for c in t if c.isalpha()), "")
    if first_alpha and first_alpha.islower():
        return False
    return True


_ROMAN_NUMERAL_RE = None


def _is_page_number(text: str) -> bool:
    """Check if text looks like a page number in any common format.

    Handles: "1", "- 1 -", "Page 1", "page 1 of 10", "i", "ii", "iv", etc.
    """
    global _ROMAN_NUMERAL_RE
    import re
    if _ROMAN_NUMERAL_RE is None:
        _ROMAN_NUMERAL_RE = re.compile(
            r'^[- ]*(?:page\s+)?\d+(?:\s+of\s+\d+)?[- ]*$|'
            r'^[- ]*(?:page\s+)?[ivxlcdm]+[- ]*$',
            re.IGNORECASE,
        )
    text = text.strip()
    if not text:
        return False
    return bool(_ROMAN_NUMERAL_RE.match(text))


def _detect_body_font_size(pages: list) -> float:
    """Find the most common font size across all text blocks."""
    size_counter = Counter()
    for page in pages:
        for tb in page.text_blocks:
            rounded = round(tb.font.size * 2) / 2
            size_counter[rounded] += len(tb.text)

    if not size_counter:
        return 12.0
    return size_counter.most_common(1)[0][0]


def _detect_single_page_hf(pages: list) -> set:
    """Detect header/footer elements on single-page documents.

    For single-page docs, we cannot use cross-page repetition. Instead,
    detect page numbers and very small footer-zone text as header/footer.
    """
    if not pages:
        return set()

    page = pages[0]
    page_height = page.height
    header_threshold = page_height * (1 - config.HEADER_ZONE_FRACTION)
    footer_threshold = page_height * config.FOOTER_ZONE_FRACTION

    signatures = set()
    for tb in page.text_blocks:
        in_header = tb.bbox.y1 > header_threshold
        in_footer = tb.bbox.y0 < footer_threshold
        if not (in_header or in_footer):
            continue

        zone = "header" if in_header else "footer"
        normalized = tb.text.strip().lower().replace("\n", " ").strip()

        # Page numbers in header/footer zones
        if _is_page_number(normalized):
            signatures.add(("__page_number__", zone))
            continue

        # Very small text in footer zone (copyright lines, disclaimers)
        if in_footer and len(normalized) < 120 and tb.font.size < 9.0:
            signatures.add((normalized, zone))

    return signatures


def _detect_header_footer_signatures(pages: list) -> set:
    """Detect text that repeats across pages in header/footer zones.

    Only text that appears at similar positions on 2+ pages is considered
    a running header/footer. This prevents falsely excluding title text
    or body content that happens to be near the top/bottom of a page.

    Returns a set of (normalized_text, zone) signatures.
    """
    if len(pages) < 2:
        return _detect_single_page_hf(pages)

    # Collect candidate texts from header/footer zones on each page
    zone_texts = {}  # (normalized_text, zone) -> set of page numbers

    for page in pages:
        page_height = page.height
        header_threshold = page_height * (1 - config.HEADER_ZONE_FRACTION)
        footer_threshold = page_height * config.FOOTER_ZONE_FRACTION

        for tb in page.text_blocks:
            in_header = tb.bbox.y1 > header_threshold
            in_footer = tb.bbox.y0 < footer_threshold

            if not (in_header or in_footer):
                continue

            zone = "header" if in_header else "footer"
            # Normalize: strip, lowercase, remove page numbers
            normalized = tb.text.strip().lower()
            # Remove standalone page numbers in various formats:
            # "1", "- 1 -", "Page 1", "page 1 of 10", "i", "ii", "iii", etc.
            if _is_page_number(normalized):
                normalized = "__page_number__"
            # Remove common variations
            normalized = normalized.replace("\n", " ").strip()

            key = (normalized, zone)
            zone_texts.setdefault(key, set()).add(page.page_number)

    # Only keep signatures that appear on 2+ pages
    return {key for key, page_nums in zone_texts.items() if len(page_nums) >= 2}


def _classify_elements(page: PageContent, body_font_size: float, hf_signatures: set):
    """Classify text blocks as headings, watermarks, headers/footers, etc."""
    page_height = page.height
    header_threshold = page_height * (1 - config.HEADER_ZONE_FRACTION)
    footer_threshold = page_height * config.FOOTER_ZONE_FRACTION

    for tb in page.text_blocks:
        # Watermark detection
        abs_rotation = abs(tb.rotation_degrees)
        is_rotated = (config.WATERMARK_MIN_ROTATION <= abs_rotation
                      <= config.WATERMARK_MAX_ROTATION)
        is_large = tb.font.size >= config.WATERMARK_MIN_FONT_SIZE
        is_light = all(
            c > config.WATERMARK_LIGHT_COLOR_THRESHOLD
            for c in tb.font.color
        )
        watermark_signals = sum([is_rotated, is_large, is_light])
        if watermark_signals >= 2:
            tb.element_type = ElementType.WATERMARK
            continue

        # Header/footer detection — only if text repeats across pages
        in_header = tb.bbox.y1 > header_threshold
        in_footer = tb.bbox.y0 < footer_threshold
        if in_header or in_footer:
            zone = "header" if in_header else "footer"
            normalized = tb.text.strip().lower()
            if _is_page_number(normalized):
                normalized = "__page_number__"
            normalized = normalized.replace("\n", " ").strip()

            if (normalized, zone) in hf_signatures:
                tb.element_type = ElementType.HEADER_FOOTER
                continue
            # Also catch small-font text in footer zone (copyright notices etc.)
            if in_footer and tb.font.size < body_font_size * 0.85:
                tb.element_type = ElementType.HEADER_FOOTER
                continue

        # List item detection (before heading detection)
        if _is_list_item(tb.text):
            tb.element_type = ElementType.LIST_ITEM
            continue

        # Heading detection by font size ratio.
        # Gate on heading-like text: font size / bold alone over-fires and
        # tags body prose as headings, which wrecks heading navigation
        # (WCAG 1.3.1). A block must both look like a heading (short label,
        # not a sentence) AND carry a heading style signal.
        ratio = tb.font.size / body_font_size if body_font_size > 0 else 1.0
        head_ok = _looks_like_heading_text(tb.text)

        if head_ok and ratio >= config.HEADING_SIZE_RATIO_H1:
            tb.element_type = ElementType.HEADING
            tb.heading_level = 1
        elif head_ok and ratio >= config.HEADING_SIZE_RATIO_H2:
            tb.element_type = ElementType.HEADING
            tb.heading_level = 2
        elif head_ok and ratio >= config.HEADING_SIZE_RATIO_H3:
            tb.element_type = ElementType.HEADING
            tb.heading_level = 3
        elif head_ok and ratio >= config.HEADING_SIZE_RATIO_H4 and tb.font.is_bold:
            tb.element_type = ElementType.HEADING
            tb.heading_level = 4
        elif head_ok and tb.font.is_bold and ratio >= 1.0:
            tb.element_type = ElementType.HEADING
            tb.heading_level = 5
        elif (head_ok and ratio >= 0.95
              and tb.text.strip() == tb.text.strip().upper()
              and any(c.isalpha() for c in tb.text)):
            tb.element_type = ElementType.HEADING
            tb.heading_level = 6
        else:
            tb.element_type = ElementType.PARAGRAPH

        # Text in the header/footer band is page furniture (document codes,
        # running heads), not document structure — never a heading.
        if tb.element_type == ElementType.HEADING and (in_header or in_footer):
            tb.element_type = ElementType.PARAGRAPH
            tb.heading_level = None


def _normalize_heading_hierarchy(pages: list):
    """Normalize heading levels so they start at H1 and don't skip levels.

    E.g. if all headings are H3-H5, remap to H1-H3.
    If headings jump from H1 to H4, compress to H1-H2.
    """
    # Collect all heading levels used across all pages
    used_levels = sorted(set(
        tb.heading_level
        for page in pages for tb in page.text_blocks
        if tb.element_type == ElementType.HEADING and tb.heading_level
    ))

    if not used_levels:
        return

    # Create mapping: old_level -> new_level (sequential starting at 1)
    level_map = {}
    for new_level, old_level in enumerate(used_levels, start=1):
        level_map[old_level] = min(new_level, 6)

    # Apply mapping
    for page in pages:
        for tb in page.text_blocks:
            if tb.element_type == ElementType.HEADING and tb.heading_level:
                tb.heading_level = level_map.get(tb.heading_level, tb.heading_level)


def _detect_tables(page: PageContent, body_font_size: float = 12.0):
    """Reconstruct tables from filled cell-background rectangles.

    DRRC/Word tables draw a filled rectangle behind every cell, so the rects
    define the exact column/row grid — far more reliable than guessing a grid
    from text positions (which fails on wrapped multi-line cells). Text blocks
    are assigned to the cell rect that contains them; row 0 is the header row.
    Regions whose rects don't form a clean grid are skipped, and any leftover
    table-like region's bold cells are demoted from heading to paragraph so
    table content never leaks out as document headings.
    """
    _detect_tables_from_rects(page, body_font_size)
    _demote_stray_table_headings(page, body_font_size)


def _detect_tables_from_rects(page: PageContent, body_font_size: float):
    rects = list(page.fill_rects)
    if len(rects) < 4:
        return

    # Keep maximal rects (drop those contained within another): a cell often
    # has both an outer background rect and an inset text-clip rect.
    tuples = sorted(
        {(r.x0, r.y0, r.x1, r.y1) for r in rects},
        key=lambda b: -((b[2]-b[0]) * (b[3]-b[1])),
    )

    def _contains(a, b):
        return (a[0] <= b[0] + 1 and a[1] <= b[1] + 1
                and a[2] >= b[2] - 1 and a[3] >= b[3] - 1 and a != b)

    maximal = [b for b in tuples if not any(_contains(o, b) for o in tuples)]
    if len(maximal) < 4:
        return

    # Group rects into rows by vertical overlap.
    maximal.sort(key=lambda b: -b[3])  # top→bottom
    rows_r: list = []
    for b in maximal:
        placed = False
        for row in rows_r:
            ry0 = min(c[1] for c in row); ry1 = max(c[3] for c in row)
            overlap = min(b[3], ry1) - max(b[1], ry0)
            if overlap > 0.4 * min(b[3]-b[1], ry1-ry0):
                row.append(b); placed = True; break
        if not placed:
            rows_r.append([b])
    rows_r.sort(key=lambda row: -max(c[3] for c in row))

    COL_TOL = max(6.0, body_font_size * 0.7)

    def _col_set(row):
        xs = sorted(b[0] for b in row)
        cols = []
        for x in xs:
            if not cols or x - cols[-1] > COL_TOL:
                cols.append(x)
        return cols

    # Segment consecutive rows into separate tables: a new table starts on a
    # large vertical gap or when the column layout changes (a page can hold
    # several tables — e.g. a continued table plus a new one).
    segments: list = []
    cur: list = []
    for row in rows_r:
        if not cur:
            cur = [row]; continue
        prev = cur[-1]
        gap = min(c[1] for c in prev) - max(c[3] for c in row)
        prev_cols = _col_set(prev); cur_cols = _col_set(row)
        same_cols = (len(prev_cols) == len(cur_cols) and all(
            any(abs(a - b) <= COL_TOL for b in cur_cols) for a in prev_cols))
        if gap > body_font_size * 1.6 or not same_cols:
            segments.append(cur); cur = [row]
        else:
            cur.append(row)
    if cur:
        segments.append(cur)

    for seg in segments:
        _build_table_from_rows(page, seg, body_font_size, COL_TOL)


def _build_table_from_rows(page, seg_rows, body_font_size, COL_TOL):
    if len(seg_rows) < 2:
        return
    lefts = sorted(b[0] for row in seg_rows for b in row)
    col_lefts: list = []
    for lx in lefts:
        if not col_lefts or lx - col_lefts[-1] > COL_TOL:
            col_lefts.append(lx)
    col_count = len(col_lefts)
    if col_count < 2:
        return

    def _col_of_x(x0):
        for i, cl in enumerate(col_lefts):
            if abs(x0 - cl) <= COL_TOL:
                return i
        return -1

    grid_rows = []
    for row in seg_rows:
        cols: dict = {}
        for b in row:
            c = _col_of_x(b[0])
            if c >= 0 and c not in cols:
                cols[c] = b
        if cols:
            grid_rows.append((min(c[1] for c in row),
                              max(c[3] for c in row), cols))
    grid_rows.sort(key=lambda r: -r[1])

    full = [r for r in grid_rows if len(r[2]) >= max(2, col_count - 1)]
    if len(full) < 2 or len(full) < len(grid_rows) * 0.75:
        return

    text_candidates = [
        tb for tb in page.text_blocks
        if tb.element_type in (
            ElementType.PARAGRAPH, ElementType.HEADING, ElementType.LIST_ITEM
        ) and tb.text.strip()
    ]

    cells = []; used = []
    n_rows = len(grid_rows)
    for r_idx, (ry0, ry1, cols) in enumerate(grid_rows):
        for c_idx in range(col_count):
            rect = cols.get(c_idx)
            if rect is None:
                continue
            inside = [
                tb for tb in text_candidates
                if (rect[0] - COL_TOL <= (tb.bbox.x0 + tb.bbox.x1) / 2
                    <= rect[2] + COL_TOL)
                and (rect[1] - 2 <= (tb.bbox.y0 + tb.bbox.y1) / 2 <= rect[3] + 2)
            ]
            if not inside:
                continue
            inside.sort(key=lambda b: (-b.bbox.y1, b.bbox.x0))
            text = " ".join(" ".join(b.text.split()) for b in inside)
            cb = BBox(
                x0=min(b.bbox.x0 for b in inside),
                y0=min(b.bbox.y0 for b in inside),
                x1=max(b.bbox.x1 for b in inside),
                y1=max(b.bbox.y1 for b in inside),
            )
            cells.append(TableCell(text=text, bbox=cb, row=r_idx, col=c_idx,
                                   is_header=(r_idx == 0)))
            used.extend(inside)

    if len({c.row for c in cells}) < 2 or not used:
        return

    table_bbox = BBox(
        x0=min(t.bbox.x0 for t in used), y0=min(t.bbox.y0 for t in used),
        x1=max(t.bbox.x1 for t in used), y1=max(t.bbox.y1 for t in used),
    )
    page.tables.append(TableBlock(
        rows=[[c.text for c in cells if c.row == ri] for ri in range(n_rows)],
        cells=cells, n_cols=col_count, header_rows=1,
        bbox=table_bbox, page_number=page.page_number,
    ))
    used_ids = {id(b) for b in used}
    page.text_blocks = [tb for tb in page.text_blocks if id(tb) not in used_ids]


def _demote_stray_table_headings(page: PageContent, body_font_size: float):
    """Demote bold cells of tables we could not reconstruct from heading to
    paragraph, so table content never leaks out as document headings.

    A heading is demoted only when >=2 cell-background rects sit side by side
    at its vertical level — i.e. it is inside a multi-column table row. Single
    full-width banner rects (section banners like "Teaching Note") have only
    one rect at that level and are left as genuine headings.
    """
    rects = page.fill_rects
    if not rects:
        return
    COL_TOL = max(6.0, body_font_size * 0.7)
    for tb in page.text_blocks:
        if tb.element_type != ElementType.HEADING:
            continue
        cy = (tb.bbox.y0 + tb.bbox.y1) / 2
        band = [r for r in rects if r.y0 - 2 <= cy <= r.y1 + 2]
        xs = sorted(r.x0 for r in band)
        cols: list = []
        for x in xs:
            if not cols or x - cols[-1] > COL_TOL:
                cols.append(x)
        if len(cols) >= 2:
            tb.element_type = ElementType.PARAGRAPH
            tb.heading_level = None


def _is_decorative_image(bbox, page_height: float) -> bool:
    """Letterhead/footer-band images are branding/decoration.

    An image whose vertical center sits in the top ~12% (letterhead) or bottom
    ~8% (footer) of the page is treated as decorative — tagged as an Artifact
    with no alt text, which is WCAG 1.1.1-correct when the source/organization
    is also identified in text (as in DRRC letterheads). Body images are left
    as /Figure so a real description can be supplied. Rule-based code cannot
    invent a meaningful description for an arbitrary content image.
    """
    if page_height <= 0:
        return False
    cy = (bbox.y0 + bbox.y1) / 2
    return cy > page_height * 0.88 or cy < page_height * 0.08


def _extract_images(pdf_path: str, pages: list):
    """Extract images from PDF using pikepdf."""
    try:
        pdf = pikepdf.Pdf.open(pdf_path)
    except Exception as e:
        logger.warning("Could not open PDF for image extraction: %s", e)
        return

    for page_idx, pdf_page in enumerate(pdf.pages):
        if page_idx >= len(pages):
            break

        page_content = pages[page_idx]
        img_index = 0

        try:
            # Check page Resources and inherited Resources
            resources = pdf_page.get("/Resources")
            if resources is None:
                parent = pdf_page.get("/Parent")
                while parent and resources is None:
                    resources = parent.get("/Resources")
                    parent = parent.get("/Parent")
            if resources is None:
                continue
            xobjects = resources.get("/XObject")
            if xobjects is None:
                continue
        except Exception as e:
            logger.warning("Could not read resources on page %d: %s", page_idx, e)
            continue

        for name, obj in xobjects.items():
            try:
                if not hasattr(obj, "keys"):
                    continue
                if obj.get("/Subtype") != pikepdf.Name.Image:
                    continue

                pdfimage = pikepdf.PdfImage(obj)
                pil_image = pdfimage.as_pil_image()
                buf = BytesIO()
                img_format = "PNG"
                pil_image.save(buf, format=img_format)

                image_block = ImageBlock(
                    image_bytes=buf.getvalue(),
                    format=img_format.lower(),
                    bbox=BBox(0, 0, pdfimage.width, pdfimage.height),
                    page_number=page_idx,
                    alt_text=f"Figure {img_index + 1} on page {page_idx + 1}",
                    is_decorative=False,
                )

                if img_index < len(page_content.images):
                    image_block.bbox = page_content.images[img_index].bbox
                    image_block.is_decorative = _is_decorative_image(
                        image_block.bbox, page_content.height
                    )
                    page_content.images[img_index] = image_block
                else:
                    page_content.images.append(image_block)

                img_index += 1
            except Exception as e:
                logger.debug("Could not extract image '%s' on page %d: %s",
                             name, page_idx, e)
                continue

    pdf.close()


def _detect_title(pages: list, pdf_path: str) -> str:
    """Detect document title using multiple strategies.

    1. Try original PDF metadata (skip generic titles like 'Title', 'Untitled')
    2. Find the largest-font text on page 1
    3. Find the first heading on page 1
    4. Fallback to filename
    """
    # Strategy 1: PDF metadata
    try:
        pdf = pikepdf.Pdf.open(pdf_path)
        # Try XMP metadata first
        with pdf.open_metadata() as meta:
            xmp_title = meta.get("dc:title", "")
        # Try DocInfo
        doc_title = str(pdf.docinfo.get("/Title", ""))
        pdf.close()

        for candidate in [xmp_title, doc_title]:
            if candidate and _is_meaningful_title(candidate):
                return candidate[:200]
    except Exception as e:
        logger.debug("Could not read PDF metadata for title: %s", e)

    # Strategy 2: Largest font text on page 1
    if pages:
        largest_size = 0
        largest_text = ""
        for tb in pages[0].text_blocks:
            if tb.font.size > largest_size and len(tb.text.strip()) > 3:
                largest_size = tb.font.size
                largest_text = tb.text.strip()

        # Only use if it's notably larger than body text
        body_size = _detect_body_font_size(pages)
        if largest_text and largest_size > body_size * 1.2:
            # Take just the first line if multi-line
            first_line = largest_text.split("\n")[0].strip()
            if first_line:
                return first_line[:200]

    # Strategy 3: First heading on page 1
    if pages:
        for tb in pages[0].text_blocks:
            if tb.element_type == ElementType.HEADING:
                return tb.text.split("\n")[0].strip()[:200]

    # Strategy 4: First paragraph on page 1
    if pages:
        for tb in pages[0].text_blocks:
            if tb.element_type == ElementType.PARAGRAPH:
                return tb.text[:100]

    # Strategy 5: Filename
    return Path(pdf_path).stem.replace("_", " ").replace("-", " ").title()


def _is_meaningful_title(title: str) -> bool:
    """Check if a title from PDF metadata is meaningful (not a generic placeholder)."""
    if not title or not title.strip():
        return False
    normalized = title.strip().lower()
    generic_titles = {
        "title", "untitled", "document", "doc", "pdf", "file",
        "microsoft word", "powerpoint presentation", "slide 1",
        "new document", "unnamed",
    }
    if normalized in generic_titles:
        return False
    if len(normalized) < 3:
        return False
    return True
