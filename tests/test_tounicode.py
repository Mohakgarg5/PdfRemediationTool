"""
Tests for the ToUnicode recovery logic in pdf_postprocess.

Covers the three-layer recovery used to satisfy PDF/UA clause 7.21.7 ("all
used character codes shall map to Unicode") — see _ensure_used_codes_mapped:

  A. glyph name -> Unicode via the Adobe Glyph List
  B. deterministic system-font outline match  (not exercised here — needs the
     real font installed; covered by the no-op safety path)
  C. context inference for ligatures Word omits from its ToUnicode CMap

The focus is the parts that make wrong output possible — the CMap parser and
the inference confidence gate — plus an end-to-end gap-fill on a synthetic PDF
and a negative control proving already-complete fonts are left untouched.
"""
import os
import sys

import pikepdf
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pdf_postprocess as pp


# --------------------------------------------------------------------------- #
# _parse_tounicode_full — must increment bfrange destinations and ignore the
# codespacerange (getting this wrong corrupts the inference context).
# --------------------------------------------------------------------------- #
def test_parse_bfrange_increments_destination():
    cmap = (
        "begincodespacerange\n<00> <FF>\nendcodespacerange\n"
        "1 beginbfrange\n<32><33><006d>\nendbfrange\n"
    )
    m = pp._parse_tounicode_full(cmap)
    assert m[0x32] == "m"
    assert m[0x33] == "n"          # incremented, not repeated
    assert 0x00 not in m           # codespacerange is not a mapping


def test_parse_bfchar_single_and_multiunit():
    cmap = (
        "beginbfchar\n<28> <0074>\n<3b> <00740069>\nendbfchar\n"
    )
    m = pp._parse_tounicode_full(cmap)
    assert m[0x28] == "t"
    assert m[0x3b] == "ti"         # multi-code-unit destination


# --------------------------------------------------------------------------- #
# _glyphname_to_unicode — AGL resolution, ligature names, and the all-important
# "return empty for stripped subset names" behaviour.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,expected", [
    ("bullet", "•"),
    ("f_i", "fi"),
    ("A", "A"),
    ("glyph00041", ""),   # stripped subset name -> not resolvable
    (".notdef", ""),
    ("", ""),
])
def test_glyphname_to_unicode(name, expected):
    assert pp._glyphname_to_unicode(name) == expected


# --------------------------------------------------------------------------- #
# _is_english_word — morphology tolerance so a base wordlist still matches
# inflected forms.
# --------------------------------------------------------------------------- #
def test_is_english_word_morphology():
    words = frozenset({"motive", "emotion", "negotiation", "attach"})
    assert pp._is_english_word("motives", words)        # plural
    assert pp._is_english_word("emotions,", words)      # plural + punctuation
    assert pp._is_english_word("negotiations", words)   # plural
    assert pp._is_english_word("attached", words)       # -ed
    assert not pp._is_english_word("xqz", words)
    assert not pp._is_english_word("z", words)          # too short


def test_bundled_wordlist_loads():
    words = pp._load_english_words()
    assert len(words) > 50000
    assert "negotiation" in words or "negotiations" in words


# --------------------------------------------------------------------------- #
# _recover_via_context — the inference layer. Must resolve clear cases and
# REFUSE to guess on weak/ambiguous evidence.
# --------------------------------------------------------------------------- #
def _ascii_map(extra_unmapped=()):
    m = {ord(c): c for c in
         "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,."}
    for c in extra_unmapped:
        m.pop(c, None)
    return m


def test_context_infers_clear_ligature(monkeypatch):
    monkeypatch.setattr(pp, "_ENGLISH_WORDS",
                        frozenset({"attached", "butter", "letter"}))
    UNK = 0x01
    code_to_unicode = _ascii_map(extra_unmapped=(UNK,))
    runs = [b"A\x01ached", b"bu\x01er", b"le\x01er"]   # tt ligature
    out = pp._recover_via_context(runs, code_to_unicode, [UNK])
    assert out == {UNK: "tt"}


def test_context_refuses_when_no_candidate_forms_words(monkeypatch):
    monkeypatch.setattr(pp, "_ENGLISH_WORDS",
                        frozenset({"hello", "world"}))
    UNK = 0x01
    code_to_unicode = _ascii_map(extra_unmapped=(UNK,))
    runs = [b"zx\x01qv", b"\x01k"]      # no ligature yields a real word
    out = pp._recover_via_context(runs, code_to_unicode, [UNK])
    assert out == {}                    # declines rather than guessing


def test_context_disabled_without_wordlist(monkeypatch):
    monkeypatch.setattr(pp, "_ENGLISH_WORDS", frozenset())
    UNK = 0x01
    out = pp._recover_via_context([b"A\x01ached"], _ascii_map((UNK,)), [UNK])
    assert out == {}


# --------------------------------------------------------------------------- #
# _generate_tounicode_from_map round-trips through the parser.
# --------------------------------------------------------------------------- #
def test_generate_tounicode_roundtrip():
    src = {0x41: "A", 0x28: "tt", 0xA5: "•"}
    cmap = pp._generate_tounicode_from_map(src)
    parsed = pp._parse_tounicode_full(cmap)
    for code, val in src.items():
        assert parsed[code] == val


# --------------------------------------------------------------------------- #
# End-to-end: a synthetic PDF whose font has a ToUnicode gap at a ligature code
# gets that code filled by inference; a font already fully mapped is untouched.
# --------------------------------------------------------------------------- #
def _make_pdf_with_font(tounicode_cmap, content_bytes, base_font="/ZZTestFont"):
    pdf = pikepdf.new()
    tu_stream = pikepdf.Stream(pdf, tounicode_cmap.encode("latin-1"))
    font = pdf.make_indirect(pikepdf.Dictionary(
        Type=pikepdf.Name.Font,
        Subtype=pikepdf.Name("/TrueType"),
        BaseFont=pikepdf.Name(base_font),
        FirstChar=0,
        LastChar=255,
        ToUnicode=tu_stream,
    ))
    resources = pikepdf.Dictionary(
        Font=pikepdf.Dictionary(F1=font))
    content = pikepdf.Stream(pdf, content_bytes)
    page = pdf.make_indirect(pikepdf.Dictionary(
        Type=pikepdf.Name.Page,
        MediaBox=[0, 0, 612, 792],
        Resources=resources,
        Contents=content,
    ))
    pdf.pages.append(pikepdf.Page(page))
    return pdf, font


def _cmap_for(mapping):
    return pp._generate_tounicode_from_map(mapping)


def test_end_to_end_gap_filled_by_inference(monkeypatch):
    monkeypatch.setattr(pp, "_ENGLISH_WORDS",
                        frozenset({"attached", "butter", "letter"}))
    UNK = 0x01
    # ToUnicode maps the ASCII letters used, but NOT the ligature code 0x01.
    mapping = {ord(c): c for c in "Aachedbutler "}
    content = b"BT /F1 12 Tf (A\x01ached bu\x01er le\x01er) Tj ET"
    pdf, font = _make_pdf_with_font(_cmap_for(mapping), content)

    report = {}
    pp._ensure_used_codes_mapped(pdf, report=report)

    result = pp._parse_tounicode_full(font.ToUnicode.read_bytes().decode("latin-1"))
    assert result[UNK] == "tt"
    assert report["tounicode"]["inferred"][0]["text"] == "tt"
    assert "unresolved" not in report.get("tounicode", {})


def test_end_to_end_complete_font_untouched():
    # Every used code is already mapped -> pass must make no change at all.
    mapping = {ord(c): c for c in "Hello "}
    cmap = _cmap_for(mapping)
    content = b"BT /F1 12 Tf (Hello) Tj ET"
    pdf, font = _make_pdf_with_font(cmap, content)
    before = font.ToUnicode.read_bytes()

    report = {}
    pp._ensure_used_codes_mapped(pdf, report=report)

    assert font.ToUnicode.read_bytes() == before     # byte-for-byte unchanged
    assert report == {}                                # nothing reported
