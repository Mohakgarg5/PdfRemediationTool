"""
Tests for list and run-in label detection in pdf_extractor.

pdfminer packs adjacent lines into one text box without regard to structure,
so a bold label plus the bullets under it, or a run of bullets, arrive as one
paragraph; and with a hanging indent a bullet's wrapped lines are often boxed
separately. Either way the list is tagged as /P (PDF/UA passes, WCAG 1.3.1
does not).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pdf_extractor as E
from models import BBox, ElementType, FontInfo, TextBlock


class _Char:
    def __init__(self, ch, fontname):
        self._ch = ch
        self.fontname = fontname

    def get_text(self):
        return self._ch


class _Line:
    def __init__(self, x0, x1):
        self.x0, self.x1 = x0, x1


def _line(text, x0=80, x1=535, bold=False):
    font = "TimesNewRomanPS-BoldMT" if bold else "TimesNewRomanPSMT"
    return (text, [_Char(c, font) for c in text], _Line(x0, x1))


def _groups(lines):
    return E._line_groups(lines, [11.0] * len(lines), split_sizes=False)


def test_symbol_pua_bullet_is_list_item():
    assert E._is_list_item(" Deal: fill in who made the first offer")


def test_bold_label_splits_from_bullets_and_each_bullet_is_own_group():
    lines = [
        _line("Keep totals accurate", x1=176, bold=True),
        _line("• Numbers only, in millions, with no dollar sign,"),
        _line("not 1.2 and not $1.2B.", x1=364),
        _line("• On no-deal rows, leave the final price blank,"),
    ]
    assert _groups(lines) == [[0], [1, 2], [3]]


def test_plain_paragraph_is_not_split():
    lines = [
        _line("This paragraph wraps onto a line that starts with a dash"),
        _line("- which is prose, not a list marker", x1=300),
    ]
    assert _groups(lines) == [[0, 1]]


def test_full_width_bold_first_line_is_not_a_label():
    lines = [
        _line("A bold sentence that runs the whole measure of the", bold=True),
        _line("text and continues in regular weight.", x1=300),
    ]
    assert _groups(lines) == [[0, 1]]


def _block(text, x0, y0, y1, kind):
    return TextBlock(
        text=text,
        bbox=BBox(x0, y0, 543, y1),
        font=FontInfo(name="T", size=11.0, is_bold=False, is_italic=False,
                      color=(0, 0, 0)),
        element_type=kind,
    )


def test_hanging_indent_continuation_merges_into_list_item():
    blocks = [
        _block(" No deal: fill in the name. Column I will", 89, 357, 369,
               ElementType.LIST_ITEM),
        _block("read No deal on its own.", 104, 345, 356, ElementType.PARAGRAPH),
        _block("Next paragraph at the margin", 72, 320, 331,
               ElementType.PARAGRAPH),
    ]
    out = E._merge_list_continuations(blocks)
    assert [b.element_type for b in out] == [
        ElementType.LIST_ITEM, ElementType.PARAGRAPH]
    assert out[0].text.endswith("read No deal on its own.")
    assert out[0].bbox.y0 == 345


def test_negative_number_is_not_a_dash_bullet():
    assert not E._is_list_item("-70")
    assert E._is_list_item("- a dash bullet")
