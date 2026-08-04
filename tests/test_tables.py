"""
Tests for rect-based table cell assignment in pdf_extractor.

The grid comes from filled cell rectangles, but the *text* for each cell is
matched separately, and pdfminer does not respect cell boundaries when it groups
characters into blocks: two adjacent column headers on a shared baseline are
frequently returned as ONE text block spanning both columns.

Two failure modes follow from that, both of which produce structurally valid but
semantically wrong tables (PDF/UA passes, WCAG 1.3.1 does not):

  A. duplication — each column accepts blocks whose center falls within
     +/- COL_TOL of its rect, so adjacent columns' acceptance windows overlap by
     2*COL_TOL. A block centered in that overlap is claimed by BOTH columns.
  B. conflation — the straddling block's full text ("Left Right") is used
     verbatim, so no column gets its own label.

Correct behavior is to split a straddling block at the column boundary using
per-word x positions, and to assign every word to exactly one column.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pdf_extractor as E
from models import BBox, FontInfo, PageContent, TextBlock


BODY = 12.0


def _font():
    return FontInfo(name="Test", size=BODY, is_bold=False, is_italic=False)


def _tb(text, x0, y0, x1, y1, words=None):
    """A text block; `words` carries per-word x extents as real extraction does."""
    b = TextBlock(text=text, bbox=BBox(x0, y0, x1, y1), font=_font(), page_number=0)
    if words is not None:
        b.words = words
    return b


def _two_col_page(header_blocks):
    """A 2x2 shaded grid: cols at x=100-200 and x=200-300."""
    page = PageContent(page_number=0, width=612.0, height=792.0)
    page.fill_rects = [
        BBox(100, 180, 200, 200), BBox(200, 180, 300, 200),   # header row
        BBox(100, 150, 200, 170), BBox(200, 150, 300, 170),   # data row
    ]
    page.text_blocks = list(header_blocks) + [
        _tb("A", 105, 155, 150, 165, words=[(105, 150, "A")]),
        _tb("B", 205, 155, 250, 165, words=[(205, 250, "B")]),
    ]
    return page


def _cell(table, row, col):
    for c in table.cells:
        if c.row == row and c.col == col:
            return c
    return None


# --------------------------------------------------------------------------- #
# The straddling-header case. This is the exact shape seen on the DRRC
# "Joey Bagdona" note page 5, where "Reservation Point" and "Aspiration Point"
# came back as one block and both columns reported the conflated text.
# --------------------------------------------------------------------------- #
def test_header_block_spanning_two_columns_is_split_per_column():
    merged = _tb(
        "Left Header Right Header", 105, 185, 295, 195,
        words=[(105, 160, "Left"), (162, 195, "Header"),
               (205, 245, "Right"), (247, 295, "Header")],
    )
    page = _two_col_page([merged])

    E._detect_tables_from_rects(page, BODY)

    assert len(page.tables) == 1, "the shaded 2x2 grid should yield one table"
    t = page.tables[0]
    assert t.n_cols == 2

    left, right = _cell(t, 0, 0), _cell(t, 0, 1)
    assert left is not None and right is not None, "both header cells must exist"
    assert left.text.strip() == "Left Header"
    assert right.text.strip() == "Right Header"


def test_straddling_block_is_not_duplicated_into_both_columns():
    """Regression guard for failure mode A, independent of how the split lands."""
    merged = _tb(
        "Left Header Right Header", 105, 185, 295, 195,
        words=[(105, 160, "Left"), (162, 195, "Header"),
               (205, 245, "Right"), (247, 295, "Header")],
    )
    page = _two_col_page([merged])

    E._detect_tables_from_rects(page, BODY)

    t = page.tables[0]
    header_texts = [c.text.strip() for c in t.cells if c.row == 0 and c.text.strip()]
    assert len(header_texts) == len(set(header_texts)), (
        f"header cells must not repeat the same text: {header_texts}"
    )


def test_no_row_repeats_a_cell_value_across_columns():
    """
    General invariant: within one row, a single source block must never appear
    in two columns. Holds for any document, not just the straddling case.
    """
    merged = _tb(
        "Left Header Right Header", 105, 185, 295, 195,
        words=[(105, 160, "Left"), (162, 195, "Header"),
               (205, 245, "Right"), (247, 295, "Header")],
    )
    page = _two_col_page([merged])

    E._detect_tables_from_rects(page, BODY)

    for t in page.tables:
        for row in {c.row for c in t.cells}:
            vals = [c.text.strip() for c in t.cells
                    if c.row == row and c.text.strip()]
            assert len(vals) == len(set(vals)), f"row {row} repeats a value: {vals}"


# --------------------------------------------------------------------------- #
# Negative controls — the ordinary case must keep working untouched.
# --------------------------------------------------------------------------- #
def test_separate_header_blocks_are_unaffected():
    page = _two_col_page([
        _tb("Left", 105, 185, 150, 195, words=[(105, 150, "Left")]),
        _tb("Right", 205, 185, 250, 195, words=[(205, 250, "Right")]),
    ])

    E._detect_tables_from_rects(page, BODY)

    t = page.tables[0]
    assert _cell(t, 0, 0).text.strip() == "Left"
    assert _cell(t, 0, 1).text.strip() == "Right"
    assert _cell(t, 1, 0).text.strip() == "A"
    assert _cell(t, 1, 1).text.strip() == "B"


def test_split_never_drops_words_that_match_no_column():
    """
    A block is removed from the page once it is consumed into a table, so every
    word of it must survive into some cell. If any word falls outside all
    columns the block must NOT be split word-wise, or that word's text is lost
    from the document entirely (and its glyphs become untagged content).
    """
    page = _two_col_page([
        _tb(
            "Left Right Stray", 105, 185, 450, 195,
            words=[(105, 150, "Left"), (205, 250, "Right"),
                   (400, 450, "Stray")],   # far outside both column rects
        ),
    ])

    E._detect_tables_from_rects(page, BODY)

    t = page.tables[0]
    emitted = " ".join(c.text for c in t.cells if c.row == 0)
    for word in ("Left", "Right", "Stray"):
        assert word in emitted, (
            f"{word!r} was dropped; row 0 emitted {emitted!r}"
        )


def test_block_without_word_positions_still_assigned_to_one_column():
    """
    Robustness: blocks lacking per-word data (any path that does not populate
    them) must still land in exactly one column rather than being duplicated.
    """
    page = _two_col_page([
        _tb("Left Header Right Header", 105, 185, 295, 195),   # no words
    ])

    E._detect_tables_from_rects(page, BODY)

    t = page.tables[0]
    holders = [c.col for c in t.cells if c.row == 0 and c.text.strip()]
    assert len(holders) == 1, (
        f"un-splittable block must occupy one column only, got cols {holders}"
    )
