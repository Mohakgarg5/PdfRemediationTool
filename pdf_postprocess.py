"""
pdf_postprocess.py - Post-process PDF to fix remaining PDF/UA issues.

Uses pikepdf to ensure all catalog-level requirements are met:
- /MarkInfo /Marked true
- /Lang on catalog
- /ViewerPreferences /DisplayDocTitle true
- /Tabs /S on every page (tab order = structure)
- XMP metadata: dc:title, dc:language, pdfuaid:part
- Font fixes: ToUnicode CMap + embedding for non-embedded fonts
- RoleMap for structure types
"""
import logging
import os
import sys
from io import BytesIO

import pikepdf

logger = logging.getLogger(__name__)


def postprocess_pdf(pdf_path: str, title: str, language: str,
                    source_path: str = None, report: dict = None) -> str:
    """Fix catalog-level metadata in the PDF for PDF/UA-1 compliance.

    If ``report`` is a dict, it is populated with details a caller may want to
    surface to a user — currently ``report['tounicode']`` with the list of
    context-inferred and still-unresolved character mappings (see
    _ensure_used_codes_mapped). Backward-compatible: pass nothing to ignore it.
    """
    try:
        pdf = pikepdf.Pdf.open(pdf_path, allow_overwriting_input=True)
    except pikepdf.PasswordError:
        logger.error("PDF is encrypted/password-protected: %s", pdf_path)
        raise
    except Exception as e:
        logger.error("Could not open PDF for postprocessing: %s", e)
        raise

    _ensure_mark_info(pdf)
    _ensure_language(pdf, language)
    _ensure_viewer_preferences(pdf)
    _ensure_tab_order(pdf)
    _ensure_xmp_metadata(pdf, title, language)
    _ensure_role_map(pdf)
    _fix_optional_content(pdf)
    _fix_fonts(pdf)
    _ensure_used_codes_mapped(pdf, report=report)
    _fix_cid_to_gid_map(pdf)
    _fix_cidset_streams(pdf)
    _fix_annotations(pdf)
    _cleanup_empty_markers(pdf)

    # PDF/UA-1 requires PDF 1.7+
    pdf.save(pdf_path, min_version="1.7")
    pdf.close()

    return pdf_path


# ---------------------------------------------------------------------------
# Catalog-level fixes
# ---------------------------------------------------------------------------

def _ensure_mark_info(pdf: pikepdf.Pdf):
    if "/MarkInfo" not in pdf.Root:
        pdf.Root.MarkInfo = pikepdf.Dictionary()
    # Preserve existing keys, only set /Marked
    pdf.Root.MarkInfo[pikepdf.Name.Marked] = True


def _ensure_language(pdf: pikepdf.Pdf, language: str):
    pdf.Root.Lang = pikepdf.String(language)


def _ensure_viewer_preferences(pdf: pikepdf.Pdf):
    if "/ViewerPreferences" not in pdf.Root:
        pdf.Root.ViewerPreferences = pikepdf.Dictionary()
    pdf.Root.ViewerPreferences[pikepdf.Name.DisplayDocTitle] = True


def _ensure_tab_order(pdf: pikepdf.Pdf):
    for page in pdf.pages:
        page.obj[pikepdf.Name.Tabs] = pikepdf.Name.S


_PLACEHOLDER_TITLES = frozenset({
    "title", "untitled", "document", "untitled document",
    "microsoft word", "microsoft word document", "word document",
    "powerpoint presentation", "microsoft powerpoint",
})


def _ensure_xmp_metadata(pdf: pikepdf.Pdf, title: str, language: str):
    with pdf.open_metadata() as meta:
        # Overwrite blank/whitespace titles AND known Word/template placeholder
        # titles (e.g. 'Title', 'Untitled') — these display as useless in PAC
        # and cause a hard PDF/UA metadata failure (Matterhorn 06-003).
        existing = meta.get("dc:title") or ""
        if not existing.strip() or existing.strip().lower() in _PLACEHOLDER_TITLES:
            meta["dc:title"] = title
        if not meta.get("dc:language"):
            meta["dc:language"] = language
        if not meta.get("pdfuaid:part"):
            meta["pdfuaid:part"] = "1"
        if not meta.get("pdf:Producer"):
            meta["pdf:Producer"] = "VAPT Accessibility Pipeline (pikepdf)"
        meta["xmp:CreatorTool"] = "VAPT PDF Accessibility Remediation Pipeline"


_STANDARD_STRUCT_TYPES = frozenset([
    "/Document", "/Part", "/Art", "/Sect", "/Div", "/BlockQuote",
    "/Caption", "/TOC", "/TOCI", "/Index", "/NonStruct", "/Private",
    "/P", "/H", "/H1", "/H2", "/H3", "/H4", "/H5", "/H6",
    "/L", "/LI", "/Lbl", "/LBody",
    "/Table", "/TR", "/TH", "/TD", "/THead", "/TBody", "/TFoot",
    "/Span", "/Quote", "/Note", "/Reference", "/BibEntry", "/Code",
    "/Link", "/Annot", "/Ruby", "/Warichu", "/RB", "/RT", "/RP", "/WT", "/WP",
    "/Figure", "/Formula", "/Form",
    "/Artifact",
])


def _ensure_role_map(pdf: pikepdf.Pdf):
    """Fix RoleMap: remove self-mappings of standard types, keep custom mappings.

    PDF/UA requires a RoleMap for non-standard structure types so they can
    be resolved to standard types. Standard types must NOT be remapped.
    """
    stroot = pdf.Root.get("/StructTreeRoot")
    if not stroot:
        return
    role_map = stroot.get("/RoleMap")
    if not role_map:
        return

    keys_to_remove = []
    for key in role_map.keys():
        key_name = str(key) if not str(key).startswith("/") else str(key)
        val_name = str(role_map[key])
        # Remove self-mappings and standard-to-standard mappings
        if key_name == val_name:
            keys_to_remove.append(key)
        elif key_name in _STANDARD_STRUCT_TYPES and val_name in _STANDARD_STRUCT_TYPES:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del role_map[key]

    # Remove empty RoleMap
    if len(role_map.keys()) == 0:
        del stroot[pikepdf.Name("/RoleMap")]


def _fix_optional_content(pdf: pikepdf.Pdf):
    """Fix Optional Content (OCProperties) for PDF/UA-1 compliance.

    Clause 7.10 requires:
    - Each OC config dict (D key and Configs array) must have a /Name key
    - The /AS key must not appear in any OC config dict
    """
    oc_props = pdf.Root.get("/OCProperties")
    if not oc_props:
        return

    def _fix_config(config_dict):
        if not hasattr(config_dict, 'get'):
            return
        # Ensure /Name key exists
        if "/Name" not in config_dict or not str(config_dict.get("/Name", "")):
            config_dict[pikepdf.Name("/Name")] = pikepdf.String("Default")
        # Remove forbidden /AS key
        if "/AS" in config_dict:
            del config_dict[pikepdf.Name("/AS")]

    # Fix the default configuration (D key)
    d_config = oc_props.get("/D")
    if d_config:
        _fix_config(d_config)

    # Fix alternate configurations (Configs array)
    configs = oc_props.get("/Configs")
    if configs and isinstance(configs, pikepdf.Array):
        for cfg in configs:
            _fix_config(cfg)


# ---------------------------------------------------------------------------
# Content stream cleanup
# ---------------------------------------------------------------------------

def _cleanup_empty_markers(pdf: pikepdf.Pdf):
    """Remove empty BMC/BDC...EMC pairs that contain no content operators.

    These are created by the q/Q boundary fix and confuse some validators.
    """
    _CONTENT_OPS = frozenset([
        "Tj", "TJ", "'", '"',                          # text drawing
        "m", "l", "c", "v", "y", "h", "re",            # path construction
        "S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n",  # path painting
        "W", "W*",                                       # clipping
        "Do",                                            # XObject
        "sh",                                            # shading
        "BI",                                            # inline image
        "BT",                                            # text object (has content inside)
    ])

    for page in pdf.pages:
        try:
            ops = list(pikepdf.parse_content_stream(page))
        except Exception as e:
            logger.debug("Could not parse content stream for cleanup: %s", e)
            continue

        # Find empty marker spans (open_idx, close_idx) to remove
        changed = True
        while changed:
            changed = False
            marker_starts = []  # stack of (index, has_content)
            remove_indices = set()

            for i, (operands, operator) in enumerate(ops):
                op = bytes(operator).decode()
                if op in ("BDC", "BMC"):
                    marker_starts.append((i, False))
                elif op == "EMC":
                    if marker_starts:
                        start_idx, has_content = marker_starts.pop()
                        if not has_content:
                            remove_indices.add(start_idx)
                            remove_indices.add(i)
                            changed = True
                elif op in _CONTENT_OPS:
                    if marker_starts:
                        # Mark the current (innermost) marker as having content
                        idx, _ = marker_starts[-1]
                        marker_starts[-1] = (idx, True)

            if remove_indices:
                ops = [op for j, op in enumerate(ops) if j not in remove_indices]

        new_data = pikepdf.unparse_content_stream(ops)
        page.Contents = pikepdf.Stream(pdf, new_data)


# ---------------------------------------------------------------------------
# Link annotation fixes (clauses 7.18.1, 7.18.5)
# ---------------------------------------------------------------------------

def _fix_annotations(pdf: pikepdf.Pdf):
    """Tag annotations (Link, Widget) as structure elements with Contents.

    PDF/UA-1 requires:
    - Every link annotation is tagged as /Link in structure tree (7.18.5)
    - Every widget annotation is tagged as /Form in structure tree (7.18.1)
    - Annotations have /Contents or /Alt text

    For veraPDF, the annotation's /StructParent must map in the ParentTree
    to the correct structure element. We assign each annotation a new unique
    StructParent key beyond ParentTreeNextKey and add the mapping.
    """
    stroot = pdf.Root.get("/StructTreeRoot")
    if not stroot:
        return

    doc_elem = stroot.get("/K")
    if not doc_elem:
        return

    kids = doc_elem.get("/K")
    if not isinstance(kids, pikepdf.Array):
        return

    # Get or create the ParentTree number tree
    parent_tree = stroot.get("/ParentTree")
    if not parent_tree:
        parent_tree = pdf.make_indirect(pikepdf.Dictionary({
            "/Nums": pikepdf.Array(),
        }))
        stroot[pikepdf.Name("/ParentTree")] = parent_tree

    nums = parent_tree.get("/Nums")
    if nums is None:
        nums = pikepdf.Array()
        parent_tree[pikepdf.Name("/Nums")] = nums

    # Determine the next available key
    next_key = int(stroot.get("/ParentTreeNextKey", 0))
    for i in range(0, len(nums) - 1, 2):
        try:
            k = int(nums[i])
            if k >= next_key:
                next_key = k + 1
        except Exception:
            pass

    # Annotation subtype → structure type mapping
    _ANNOT_STRUCT_MAP = {
        "/Link": "/Link",
        "/Widget": "/Form",
    }

    for page_idx, page in enumerate(pdf.pages):
        annots = page.get("/Annots")
        if not annots:
            continue

        page_ref = page.obj

        for annot in annots:
            try:
                subtype = str(annot.get("/Subtype", ""))
                struct_type = _ANNOT_STRUCT_MAP.get(subtype)
                if not struct_type:
                    continue

                # 1. Ensure /Contents key exists with descriptive text
                if "/Contents" not in annot or not str(annot.get("/Contents", "")):
                    contents_text = _derive_annot_contents(annot, subtype)
                    annot[pikepdf.Name("/Contents")] = pikepdf.String(contents_text)

                # 2. Check if already properly tagged in ParentTree
                existing_sp = annot.get("/StructParent")
                if _is_already_tagged(stroot, existing_sp, struct_type):
                    continue

                # 3. Create structure element with OBJR.
                # /Link and /Form are inline elements — they must not be direct
                # children of /Document. Wrap them in a /P block so PAC does
                # not warn about improper nesting (Matterhorn structure tree).
                objr = pikepdf.Dictionary({
                    "/Type": pikepdf.Name("/OBJR"),
                    "/Pg": page_ref,
                    "/Obj": annot,
                })

                _INLINE_TYPES = {"/Link", "/Form"}
                if struct_type in _INLINE_TYPES:
                    elem = pdf.make_indirect(pikepdf.Dictionary({
                        "/Type": pikepdf.Name("/StructElem"),
                        "/S": pikepdf.Name(struct_type),
                        "/K": objr,
                    }))
                    p_elem = pdf.make_indirect(pikepdf.Dictionary({
                        "/Type": pikepdf.Name("/StructElem"),
                        "/S": pikepdf.Name("/P"),
                        "/P": doc_elem,
                        "/K": pikepdf.Array([elem]),
                    }))
                    elem[pikepdf.Name("/P")] = p_elem
                    kids.append(p_elem)
                else:
                    elem = pdf.make_indirect(pikepdf.Dictionary({
                        "/Type": pikepdf.Name("/StructElem"),
                        "/S": pikepdf.Name(struct_type),
                        "/P": doc_elem,
                        "/K": objr,
                    }))
                    kids.append(elem)

                # 4. Assign a new unique StructParent and add to ParentTree
                annot[pikepdf.Name("/StructParent")] = next_key
                nums.append(next_key)
                nums.append(elem)
                next_key += 1

            except Exception as e:
                logger.debug("Could not fix %s annotation on page %d: %s",
                             subtype, page_idx, e)
                continue

    # Update ParentTreeNextKey
    stroot[pikepdf.Name("/ParentTreeNextKey")] = next_key


def _derive_annot_contents(annot, subtype: str) -> str:
    """Derive descriptive Contents text for an annotation."""
    if subtype == "/Link":
        action = annot.get("/A")
        if action:
            uri = action.get("/URI")
            if uri:
                return str(uri)
            # GoTo action
            s_type = str(action.get("/S", ""))
            if s_type == "/GoTo":
                return "Internal link"
            if s_type == "/GoToR":
                f = action.get("/F")
                return f"Link to {f}" if f else "External document link"
            if s_type == "/Named":
                n = action.get("/N")
                return str(n) if n else "Named action"
            dest = action.get("/D")
            if dest:
                return "Internal link"
        dest = annot.get("/Dest")
        if dest:
            return "Internal link"
        return "Link"

    if subtype == "/Widget":
        # Try field name (T), tooltip (TU), or alternate description
        tu = annot.get("/TU")
        if tu:
            return str(tu)
        t = annot.get("/T")
        if t:
            return f"Form field: {str(t)}"
        return "Form field"

    return "Annotation"


def _is_already_tagged(stroot, struct_parent, expected_type: str) -> bool:
    """Check if an annotation with given StructParent is already tagged correctly."""
    if struct_parent is None:
        return False

    parent_tree = stroot.get("/ParentTree")
    if not parent_tree:
        return False

    nums = parent_tree.get("/Nums")
    if not nums:
        return False

    sp_val = int(struct_parent)
    for i in range(0, len(nums) - 1, 2):
        try:
            if int(nums[i]) == sp_val:
                elem = nums[i + 1]
                if isinstance(elem, pikepdf.Array):
                    for e in elem:
                        if hasattr(e, 'get') and str(e.get("/S", "")) == expected_type:
                            return True
                elif hasattr(elem, 'get'):
                    if str(elem.get("/S", "")) == expected_type:
                        return True
                return False
        except Exception:
            continue

    return False



# ---------------------------------------------------------------------------
# CIDSet stream fix (clause 7.21.4.2)
# ---------------------------------------------------------------------------

def _fix_cidset_streams(pdf: pikepdf.Pdf):
    """Remove CIDSet streams from CID font descriptors.

    PDF/UA-1 clause 7.21.4.2 requires that if a CIDSet is present, it must
    identify ALL CIDs in the font. Since CIDSet is optional, the safest
    compliant fix is to simply remove it.
    """
    seen_objgen = set()

    for page in pdf.pages:
        res = _resolve_page_resources(page)
        if not res:
            continue
        font_dict = res.get("/Font")
        if not font_dict:
            continue

        for name, font_obj in font_dict.items():
            try:
                objgen = font_obj.objgen
                if objgen in seen_objgen:
                    continue
                seen_objgen.add(objgen)

                font_type = str(font_obj.get("/Subtype", ""))

                # Collect all font descriptors to check
                descriptors = []
                if font_type == "/Type0":
                    descendants = font_obj.get("/DescendantFonts")
                    if descendants:
                        for desc_font in descendants:
                            d = desc_font.get("/FontDescriptor")
                            if d:
                                descriptors.append(d)
                desc = font_obj.get("/FontDescriptor")
                if desc:
                    descriptors.append(desc)

                for d in descriptors:
                    if "/CIDSet" in d:
                        del d[pikepdf.Name("/CIDSet")]
                        logger.debug("Removed CIDSet from font '%s'", name)

            except Exception as e:
                logger.debug("CIDSet fix failed for font '%s': %s", name, e)
                continue


# ---------------------------------------------------------------------------
# CIDToGIDMap fix (clause 7.21.3.2)
# ---------------------------------------------------------------------------

def _fix_cid_to_gid_map(pdf: pikepdf.Pdf):
    """Add CIDToGIDMap /Identity to embedded Type 2 CIDFont dicts missing it.

    ISO 32000-1 Table 117 requires embedded CIDFontType2 fonts to have a
    CIDToGIDMap entry (either /Identity or a stream). Without it veraPDF
    fails clause 7.21.3.2. Only adds the entry when it is absent.
    """
    seen_objgen = set()

    for page in pdf.pages:
        res = _resolve_page_resources(page)
        if not res:
            continue
        font_dict = res.get("/Font")
        if not font_dict:
            continue

        for name, font_obj in font_dict.items():
            try:
                if str(font_obj.get("/Subtype", "")) != "/Type0":
                    continue

                objgen = font_obj.objgen
                if objgen in seen_objgen:
                    continue
                seen_objgen.add(objgen)

                descendants = font_obj.get("/DescendantFonts")
                if not descendants:
                    continue

                for desc_font in descendants:
                    try:
                        if str(desc_font.get("/Subtype", "")) != "/CIDFontType2":
                            continue
                        if "/CIDToGIDMap" not in desc_font:
                            desc_font[pikepdf.Name("/CIDToGIDMap")] = \
                                pikepdf.Name("/Identity")
                            logger.debug(
                                "Added CIDToGIDMap /Identity to '%s'", name)
                    except Exception as e:
                        logger.debug(
                            "CIDToGIDMap fix failed for descendant of '%s': %s",
                            name, e)
            except Exception as e:
                logger.debug("CIDToGIDMap fix failed for font '%s': %s", name, e)


# ---------------------------------------------------------------------------
# Font fixes — ToUnicode CMap + embedding
# ---------------------------------------------------------------------------

# Windows-1252 byte values 0x80-0x9F that map to non-obvious Unicode points
_WIN1252_SPECIAL = {
    0x80: 0x20AC, 0x82: 0x201A, 0x83: 0x0192, 0x84: 0x201E,
    0x85: 0x2026, 0x86: 0x2020, 0x87: 0x2021, 0x88: 0x02C6,
    0x89: 0x2030, 0x8A: 0x0160, 0x8B: 0x2039, 0x8C: 0x0152,
    0x8E: 0x017D, 0x91: 0x2018, 0x92: 0x2019, 0x93: 0x201C,
    0x94: 0x201D, 0x95: 0x2022, 0x96: 0x2013, 0x97: 0x2014,
    0x98: 0x02DC, 0x99: 0x2122, 0x9A: 0x0161, 0x9B: 0x203A,
    0x9C: 0x0153, 0x9E: 0x017E, 0x9F: 0x0178,
}

# Map PostScript font names to possible TTF file names.
# Each list is tried in order — first match wins.
# Liberation fonts (apt-get install fonts-liberation) are metrically compatible
# substitutes for Arial/Times/Courier on Linux/Docker where MS fonts are absent.
_FONT_FILE_NAMES = {
    "TimesNewRomanPSMT": ["Times New Roman.ttf", "times.ttf", "LiberationSerif-Regular.ttf"],
    "TimesNewRomanPS-BoldMT": ["Times New Roman Bold.ttf", "timesbd.ttf", "LiberationSerif-Bold.ttf"],
    "TimesNewRomanPS-ItalicMT": ["Times New Roman Italic.ttf", "timesi.ttf", "LiberationSerif-Italic.ttf"],
    "TimesNewRomanPS-BoldItalicMT": ["Times New Roman Bold Italic.ttf", "timesbi.ttf", "LiberationSerif-BoldItalic.ttf"],
    "ArialMT": ["Arial.ttf", "arial.ttf", "LiberationSans-Regular.ttf"],
    "Arial-BoldMT": ["Arial Bold.ttf", "arialbd.ttf", "LiberationSans-Bold.ttf"],
    "Arial-ItalicMT": ["Arial Italic.ttf", "ariali.ttf", "LiberationSans-Italic.ttf"],
    "Arial-BoldItalicMT": ["Arial Bold Italic.ttf", "arialbi.ttf", "LiberationSans-BoldItalic.ttf"],
    "CourierNewPSMT": ["Courier New.ttf", "cour.ttf", "LiberationMono-Regular.ttf"],
    "CourierNewPS-BoldMT": ["Courier New Bold.ttf", "courbd.ttf", "LiberationMono-Bold.ttf"],
    "Verdana": ["Verdana.ttf", "verdana.ttf"],
    "Verdana-Bold": ["Verdana Bold.ttf", "verdanab.ttf"],
    "Georgia": ["Georgia.ttf", "georgia.ttf"],
    "Georgia-Bold": ["Georgia Bold.ttf", "georgiab.ttf"],
    "Tahoma": ["Tahoma.ttf", "tahoma.ttf"],
    "Tahoma-Bold": ["Tahoma Bold.ttf", "tahomabd.ttf"],
    "Calibri": ["Calibri.ttf", "calibri.ttf"],
    "Calibri-Bold": ["Calibri Bold.ttf", "calibrib.ttf"],
    "Cambria": ["Cambria.ttf", "cambria.ttf"],
    # Helvetica → Arial / Liberation Sans fallback
    "Helvetica": ["Arial.ttf", "arial.ttf", "LiberationSans-Regular.ttf"],
    "Helvetica,Bold": ["Arial Bold.ttf", "arialbd.ttf", "LiberationSans-Bold.ttf"],
    "Helvetica-Bold": ["Arial Bold.ttf", "arialbd.ttf", "LiberationSans-Bold.ttf"],
    "Helvetica,Italic": ["Arial Italic.ttf", "ariali.ttf", "LiberationSans-Italic.ttf"],
    "Helvetica-Oblique": ["Arial Italic.ttf", "ariali.ttf", "LiberationSans-Italic.ttf"],
    "Helvetica,BoldItalic": ["Arial Bold Italic.ttf", "arialbi.ttf", "LiberationSans-BoldItalic.ttf"],
    "Helvetica-BoldOblique": ["Arial Bold Italic.ttf", "arialbi.ttf", "LiberationSans-BoldItalic.ttf"],
}

# Map PostScript font names to TTC (TrueType Collection) files + font index
# Used when a standalone TTF is not available
_FONT_TTC_MAP = {}
if sys.platform == "darwin":
    _FONT_TTC_MAP = {
        "Helvetica": ("/System/Library/Fonts/Helvetica.ttc", 0),
        "Helvetica,Bold": ("/System/Library/Fonts/Helvetica.ttc", 1),
        "Helvetica-Bold": ("/System/Library/Fonts/Helvetica.ttc", 1),
        "Helvetica,Italic": ("/System/Library/Fonts/Helvetica.ttc", 2),
        "Helvetica-Oblique": ("/System/Library/Fonts/Helvetica.ttc", 2),
        "Helvetica,BoldItalic": ("/System/Library/Fonts/Helvetica.ttc", 3),
        "Helvetica-BoldOblique": ("/System/Library/Fonts/Helvetica.ttc", 3),
    }

# Platform-specific font directories
_FONT_DIRS = []
if sys.platform == "darwin":
    _FONT_DIRS = [
        "/System/Library/Fonts/Supplemental",
        "/System/Library/Fonts",
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
    ]
elif sys.platform.startswith("linux"):
    _FONT_DIRS = [
        "/usr/share/fonts/truetype",
        "/usr/share/fonts/truetype/msttcorefonts",
        "/usr/share/fonts/truetype/liberation",
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
    ]
elif sys.platform == "win32":
    _FONT_DIRS = [
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
    ]


def _resolve_page_resources(page):
    """Get Resources for a page, checking inheritance from page tree."""
    res = page.get("/Resources")
    if res:
        return res
    parent = page.get("/Parent")
    seen = set()
    while parent:
        try:
            obj_id = parent.objgen
            if obj_id in seen:
                break  # Circular reference protection
            seen.add(obj_id)
            res = parent.get("/Resources")
            if res:
                return res
            parent = parent.get("/Parent")
        except Exception:
            break
    return None


def _fix_fonts(pdf: pikepdf.Pdf):
    """Fix all non-embedded fonts: add ToUnicode CMap and embed font data."""
    seen_objgen = set()

    for page in pdf.pages:
        res = _resolve_page_resources(page)
        if not res:
            continue
        font_dict = res.get("/Font")
        if not font_dict:
            continue

        for name, font_obj in font_dict.items():
            try:
                objgen = font_obj.objgen
                if objgen in seen_objgen:
                    continue
                seen_objgen.add(objgen)

                has_tounicode = "/ToUnicode" in font_obj
                desc = font_obj.get("/FontDescriptor")
                embedded = False
                if desc:
                    embedded = any(k in desc for k in
                                   ["/FontFile", "/FontFile2", "/FontFile3"])

                # For Type0 (CID) fonts, check DescendantFonts for descriptor
                # and embedding status — Type0 wrappers don't have their own
                # FontDescriptor; it lives on the CIDFont descendant.
                font_type = str(font_obj.get("/Subtype", ""))
                if font_type == "/Type0" and not embedded:
                    descendants = font_obj.get("/DescendantFonts")
                    if descendants:
                        for desc_font in descendants:
                            d = desc_font.get("/FontDescriptor")
                            if d and any(k in d for k in
                                         ["/FontFile", "/FontFile2",
                                          "/FontFile3"]):
                                embedded = True
                                break

                if has_tounicode and embedded:
                    continue

                encoding_obj = font_obj.get("/Encoding")
                encoding = str(encoding_obj) if encoding_obj else ""
                base_font_raw = str(font_obj.get("/BaseFont", "")).lstrip("/")
                # Strip subset prefix like ABCDEF+
                is_subset = "+" in base_font_raw
                base_font = base_font_raw.split("+", 1)[1] if is_subset else base_font_raw

                # Add ToUnicode CMap for simple WinAnsi fonts.
                # Skip if encoding has /Differences (custom remapping) or if
                # font is Type0/CID (uses its own CMap), or if already present.
                if not has_tounicode and _is_simple_winansi(encoding_obj):
                    _add_tounicode_cmap(pdf, font_obj)

                if not embedded:
                    _try_embed_font(pdf, font_obj, base_font)

                # Fallback: any remaining simple font that still has no
                # ToUnicode (e.g. SymbolMT / MacRomanEncoding fonts used for
                # bullet glyphs in Word text boxes) gets a ToUnicode CMap
                # derived from its embedded font program's cmap. Without this,
                # PAC reports those glyphs as "unreadable text" (Content fail).
                # Purely additive: only runs when ToUnicode is still absent and
                # never touches Type0 fonts (they carry their own CMap).
                if (font_type != "/Type0"
                        and "/ToUnicode" not in font_obj):
                    _add_tounicode_from_font_program(pdf, font_obj)

            except Exception as e:
                logger.warning("Font fix failed for '%s': %s", name, e)
                continue


def _is_simple_winansi(encoding_obj) -> bool:
    """Check if encoding is plain /WinAnsiEncoding without /Differences.

    Returns False for:
    - None / missing encoding
    - Encoding dictionaries with /Differences array (custom glyph remapping)
    - Non-WinAnsi encodings (/MacRomanEncoding, /Identity-H, etc.)
    """
    if encoding_obj is None:
        return False
    # Simple Name: /WinAnsiEncoding
    if isinstance(encoding_obj, pikepdf.Name):
        return str(encoding_obj) == "/WinAnsiEncoding"
    # Dictionary: check /BaseEncoding and /Differences
    if isinstance(encoding_obj, pikepdf.Dictionary):
        base = str(encoding_obj.get("/BaseEncoding", ""))
        if "WinAnsi" not in base:
            return False
        # If /Differences is present, encoding is customized — skip
        if "/Differences" in encoding_obj:
            return False
        return True
    # String or other: check for WinAnsi substring
    return "WinAnsi" in str(encoding_obj)


def _add_tounicode_cmap(pdf: pikepdf.Pdf, font_obj):
    """Generate and attach a ToUnicode CMap for WinAnsiEncoding."""
    cmap_str = _generate_winansi_tounicode()
    cmap_stream = pikepdf.Stream(pdf, cmap_str.encode("latin-1"))
    font_obj[pikepdf.Name("/ToUnicode")] = cmap_stream


def _generate_winansi_tounicode() -> str:
    """Generate a standard ToUnicode CMap for WinAnsiEncoding (Windows-1252)."""
    entries = []
    for code in range(0x20, 0x100):
        if code == 0x7F:
            continue
        if 0x80 <= code <= 0x9F:
            if code in _WIN1252_SPECIAL:
                entries.append((code, _WIN1252_SPECIAL[code]))
            # Skip undefined codes (0x81, 0x8D, 0x8F, 0x90, 0x9D)
        else:
            entries.append((code, code))

    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        "/CMapName /Adobe-Identity-UCS def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        "<00> <FF>",
        "endcodespacerange",
    ]

    # Split into chunks of 100 (PDF CMap limit per block)
    for i in range(0, len(entries), 100):
        chunk = entries[i:i + 100]
        lines.append(f"{len(chunk)} beginbfchar")
        for byte_code, unicode_val in chunk:
            lines.append(f"<{byte_code:02X}> <{unicode_val:04X}>")
        lines.append("endbfchar")

    lines.extend([
        "endcmap",
        "CMapName currentdict /CMap defineresource pop",
        "end",
        "end",
    ])
    return "\n".join(lines)


def _add_tounicode_from_font_program(pdf: pikepdf.Pdf, font_obj):
    """Derive and attach a ToUnicode CMap from a simple font's embedded program.

    Handles fonts the WinAnsi path can't (Symbol, MacRoman, custom encodings)
    by mapping each character code to a glyph name via the embedded font's
    cmap (and any /Encoding /Differences), then to Unicode via the Adobe Glyph
    List. Used codes that resolve to a real Unicode value get a bfchar entry.

    No-op (leaves the font untouched) if there is no embedded TrueType program,
    fontTools is unavailable, or no glyph resolves — never overwrites anything.
    """
    desc = font_obj.get("/FontDescriptor")
    if not desc or "/FontFile2" not in desc:
        # Only TrueType (FontFile2) programs are parsed here. Type1/CFF
        # (FontFile/FontFile3) are left alone to avoid risky guesses.
        return

    code_to_glyph = _build_code_to_glyphname(font_obj)
    if not code_to_glyph:
        return

    # Resolve glyph names to Unicode via the Adobe Glyph List.
    code_to_unicode = {}
    for code, gname in code_to_glyph.items():
        uni = _glyphname_to_unicode(gname)
        if uni:
            code_to_unicode[code] = uni

    if not code_to_unicode:
        return

    cmap_str = _generate_tounicode_from_map(code_to_unicode)
    cmap_stream = pikepdf.Stream(pdf, cmap_str.encode("latin-1"))
    font_obj[pikepdf.Name("/ToUnicode")] = cmap_stream
    logger.info("Derived ToUnicode CMap for '%s' (%d glyph(s))",
                font_obj.get("/BaseFont"), len(code_to_unicode))


def _glyphname_to_unicode(gname: str) -> str:
    """Map a glyph name to a Unicode string via the Adobe Glyph List.

    Handles AGL names, uniXXXX/uXXXXXX forms, and ligature names written with
    underscores (e.g. 'f_i' -> 'fi'). Returns '' for .notdef or unresolvable
    (e.g. subset names like 'glyph00041' that carry no semantic information).
    """
    if not gname or gname == ".notdef":
        return ""
    try:
        from fontTools import agl
    except ImportError:
        return ""
    try:
        return agl.toUnicode(gname)
    except Exception:
        return ""


def _build_code_to_glyphname(font_obj) -> dict:
    """Map single-byte char codes to glyph names for a simple TrueType font.

    Combines the embedded font program's cmap (preferring single-byte Mac Roman
    and Symbol subtables) with any explicit /Encoding /Differences overrides.
    Returns {} if there is no embedded TrueType program or it cannot be parsed.
    """
    desc = font_obj.get("/FontDescriptor")
    if not desc or "/FontFile2" not in desc:
        return {}

    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        logger.warning("fontTools unavailable — cannot read glyph names for "
                       "'%s'", font_obj.get("/BaseFont"))
        return {}

    try:
        ttf = TTFont(BytesIO(desc["/FontFile2"].read_bytes()))
    except Exception as e:
        logger.warning("Could not parse embedded font program for '%s': %s",
                       font_obj.get("/BaseFont"), e)
        return {}

    cmap_tables = ttf["cmap"].tables if ttf.get("cmap") else []
    code_to_glyph = {}
    if cmap_tables:
        # Prefer single-byte subtables (Mac Roman 1,0 and Symbol 3,0) since PDF
        # simple-font codes are single bytes; fall back to whatever exists.
        chosen = None
        for pid, eid in ((1, 0), (3, 0), (3, 1), (0, 3)):
            for st in cmap_tables:
                if st.platformID == pid and st.platEncID == eid:
                    chosen = st
                    break
            if chosen:
                break
        if chosen is None:
            chosen = cmap_tables[0]
        for code, gname in chosen.cmap.items():
            c = code
            # Symbol (3,0) cmaps store glyphs in the 0xF000-0xF0FF range.
            if chosen.platformID == 3 and chosen.platEncID == 0 \
                    and 0xF000 <= code <= 0xF0FF:
                c = code - 0xF000
            if 0 <= c <= 0xFF:
                code_to_glyph[c] = gname

    # Apply explicit /Encoding /Differences (overrides the font's own cmap).
    encoding_obj = font_obj.get("/Encoding")
    if isinstance(encoding_obj, pikepdf.Dictionary) \
            and "/Differences" in encoding_obj:
        cur = 0
        for item in encoding_obj["/Differences"]:
            if isinstance(item, int):
                cur = item
            else:
                code_to_glyph[cur] = str(item).lstrip("/")
                cur += 1

    return code_to_glyph


def _generate_tounicode_from_map(code_to_unicode: dict) -> str:
    """Build a single-byte ToUnicode CMap from a {code: unicode_str} mapping."""
    entries = sorted(code_to_unicode.items())

    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        "/CMapName /Adobe-Identity-UCS def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        "<00> <FF>",
        "endcodespacerange",
    ]

    # Split into chunks of 100 (PDF CMap limit per bfchar block).
    for i in range(0, len(entries), 100):
        chunk = entries[i:i + 100]
        lines.append(f"{len(chunk)} beginbfchar")
        for code, uni in chunk:
            dst = "".join(f"{ord(ch):04X}" for ch in uni)
            lines.append(f"<{code:02X}> <{dst}>")
        lines.append("endbfchar")

    lines.extend([
        "endcmap",
        "CMapName currentdict /CMap defineresource pop",
        "end",
        "end",
    ])
    return "\n".join(lines)


# Common Latin ligatures used as the candidate set for context inference.
# A given subset code is one of these everywhere it appears; the surrounding
# words disambiguate which. Ordered most-common first.
_LIGATURE_CANDIDATES = (
    "ti", "fi", "fl", "ff", "tt", "ffi", "ffl", "ft", "st", "ct",
    "tf", "fj", "fb", "fh", "fk", "ll", "si", "ssi",
)

_ENGLISH_WORDS = None  # lazily-loaded frozenset, shared across calls


def _ensure_used_codes_mapped(pdf: pikepdf.Pdf, report: dict = None):
    """Ensure every character code actually drawn has a ToUnicode entry.

    Catches the common Word bug where a generated ToUnicode CMap omits some
    glyph codes (notably ligatures like 'ti'/'tt'/'fi'), which veraPDF flags
    under PDF/UA clause 7.21.7 ("all used character codes shall map to
    Unicode"). For each simple (non-Type0) font, collect the codes used in
    page content streams together with the text runs they appear in, then for
    any used code missing from the font's ToUnicode recover Unicode via a
    layered strategy (see _recover_missing_tounicode) and append it.

    Purely additive: existing ToUnicode entries are never modified, and the
    pass is a no-op for any font whose used codes are already fully mapped.
    """
    # objgen -> (font_obj, list of byte-string runs drawn with that font)
    runs_by_font = {}

    for page in pdf.pages:
        res = _resolve_page_resources(page)
        if not res:
            continue
        font_dict = res.get("/Font")
        if not font_dict:
            continue
        name_to_font = {}
        for name, fobj in font_dict.items():
            try:
                if str(fobj.get("/Subtype", "")) != "/Type0":
                    name_to_font[str(name)] = fobj
            except Exception:
                continue
        if not name_to_font:
            continue

        try:
            ops = pikepdf.parse_content_stream(page)
        except Exception:
            continue

        cur = None
        for instr in ops:
            op = str(instr.operator)
            if op == "Tf":
                cur = str(instr.operands[0])
            elif cur in name_to_font and op in ("Tj", "'", "\"", "TJ"):
                fobj = name_to_font[cur]
                try:
                    objgen = fobj.objgen
                except Exception:
                    continue
                runs = runs_by_font.setdefault(objgen, (fobj, []))[1]
                if op == "TJ":
                    # Concatenate the array's strings into one run so word
                    # boundaries (space glyphs / kerning gaps) are preserved.
                    parts = [bytes(e) for e in instr.operands[0]
                             if isinstance(e, (pikepdf.String, bytes))]
                    if parts:
                        runs.append(b"".join(parts))
                else:
                    runs.append(bytes(instr.operands[-1]))

    for objgen, (font_obj, byte_runs) in runs_by_font.items():
        try:
            _recover_missing_tounicode(pdf, font_obj, byte_runs, report=report)
        except Exception as e:
            logger.warning("ToUnicode gap-fill failed for '%s': %s",
                           font_obj.get("/BaseFont"), e)


def _recover_missing_tounicode(pdf: pikepdf.Pdf, font_obj, byte_runs,
                               report: dict = None):
    """Recover ToUnicode entries for used codes missing from the existing CMap.

    Layered, most-reliable-first; each layer only handles codes the previous
    ones could not. Purely additive — existing entries are never touched:

      A. Glyph name -> Unicode via the Adobe Glyph List (handles real or
         'f_i'-style ligature names retained in the cmap/Differences).
      B. Deterministic recovery from the installed system font: match the
         subset glyph's outline to the full font and read its Unicode (and
         decompose ligatures via the full font's GSUB). Exact when available.
      C. Context inference: a missing code is almost always a ligature; the
         surrounding words disambiguate which (e.g. 'A·ached' -> 'tt'). Only
         accepted when one candidate is a clear, confident winner.

    Codes still unresolved after all layers are logged for manual review.
    """
    tu = font_obj.get("/ToUnicode")
    if tu is None:
        return  # fonts with no ToUnicode are handled earlier in _fix_fonts
    try:
        cmap_text = tu.read_bytes().decode("latin-1")
    except Exception:
        return

    code_to_unicode = _parse_tounicode_full(cmap_text)
    used = {b for run in byte_runs for b in run}
    missing = sorted(c for c in used if c not in code_to_unicode)
    if not missing:
        return

    derived = {}
    code_to_glyph = _build_code_to_glyphname(font_obj)

    # Layer A — glyph name via AGL.
    for code in list(missing):
        uni = _glyphname_to_unicode(code_to_glyph.get(code, ""))
        if uni:
            derived[code] = uni
    remaining = [c for c in missing if c not in derived]

    # Layer B — deterministic system-font outline match.
    if remaining:
        try:
            sys_map = _recover_via_system_font(font_obj, code_to_glyph,
                                               remaining)
        except Exception as e:
            logger.debug("System-font recovery failed for '%s': %s",
                         font_obj.get("/BaseFont"), e)
            sys_map = {}
        derived.update(sys_map)
        remaining = [c for c in remaining if c not in derived]

    # Layer C — context inference.
    if remaining:
        try:
            ctx_map = _recover_via_context(byte_runs, code_to_unicode,
                                           remaining)
        except Exception as e:
            logger.debug("Context inference failed for '%s': %s",
                         font_obj.get("/BaseFont"), e)
            ctx_map = {}
        base_font = str(font_obj.get("/BaseFont", "")).lstrip("/")
        for code, uni in ctx_map.items():
            derived[code] = uni
            logger.info("Inferred code 0x%02X -> %r for '%s' from context",
                        code, uni, font_obj.get("/BaseFont"))
            if report is not None:
                report.setdefault("tounicode", {}).setdefault(
                    "inferred", []).append(
                        {"font": base_font, "code": code, "text": uni})
        remaining = [c for c in remaining if c not in derived]

    if remaining:
        logger.warning(
            "Font '%s' uses code(s) %s with no recoverable Unicode mapping; "
            "ToUnicode left incomplete — manual mapping may be required for "
            "full PDF/UA compliance.",
            font_obj.get("/BaseFont"),
            ", ".join(f"0x{c:02X}" for c in remaining))
        if report is not None:
            base_font = str(font_obj.get("/BaseFont", "")).lstrip("/")
            for c in remaining:
                report.setdefault("tounicode", {}).setdefault(
                    "unresolved", []).append({"font": base_font, "code": c})

    if not derived:
        return

    # Insert a new bfchar block immediately before endcmap — additive, leaves
    # all existing entries untouched.
    block_lines = [f"{len(derived)} beginbfchar"]
    for code, uni in sorted(derived.items()):
        dst = "".join(f"{ord(ch):04X}" for ch in uni)
        block_lines.append(f"<{code:02X}> <{dst}>")
    block_lines.append("endbfchar")
    block = "\n".join(block_lines) + "\n"

    new_text = cmap_text.replace("endcmap", block + "endcmap", 1)
    font_obj[pikepdf.Name("/ToUnicode")] = pikepdf.Stream(
        pdf, new_text.encode("latin-1"))
    logger.info("Filled %d missing ToUnicode entry(ies) for '%s'",
                len(derived), font_obj.get("/BaseFont"))


def _parse_tounicode_full(cmap_text: str) -> dict:
    """Parse a ToUnicode CMap into {code: unicode_str} for single-byte codes.

    Reads only inside begin/endbfchar and begin/endbfrange blocks (so the
    codespacerange is ignored), and correctly increments the destination
    across bfrange runs (`<32><33><006d>` -> 0x32:'m', 0x33:'n').
    """
    import re
    out = {}

    def _hex_to_str(h):
        if len(h) % 4:
            h = h.zfill(4)
        return bytes.fromhex(h).decode("utf-16-be", "replace")

    for blk in re.findall(r"beginbfchar(.*?)endbfchar", cmap_text, re.S):
        for code, dst in re.findall(
                r"<([0-9A-Fa-f]{1,2})>\s*<([0-9A-Fa-f]+)>", blk):
            out[int(code, 16)] = _hex_to_str(dst)

    for blk in re.findall(r"beginbfrange(.*?)endbfrange", cmap_text, re.S):
        for lo, hi, dst in re.findall(
                r"<([0-9A-Fa-f]{1,2})>\s*<([0-9A-Fa-f]{1,2})>\s*<([0-9A-Fa-f]+)>",
                blk):
            lo_i, hi_i, base = int(lo, 16), int(hi, 16), int(dst, 16)
            for i, code in enumerate(range(lo_i, hi_i + 1)):
                try:
                    out[code] = chr(base + i)
                except ValueError:
                    pass
    return out


def _recover_via_system_font(font_obj, code_to_glyph, codes):
    """Deterministically recover Unicode by matching glyph outlines to the
    installed full font (which still has names, cmap and ligature GSUB).

    Returns {code: unicode_str} for codes whose subset glyph outline matches a
    glyph in the system font. Returns {} when the system font is not installed
    or anything cannot be parsed — never guesses.
    """
    desc = font_obj.get("/FontDescriptor")
    if not desc or "/FontFile2" not in desc:
        return {}

    base_raw = str(font_obj.get("/BaseFont", "")).lstrip("/")
    base = base_raw.split("+", 1)[1] if "+" in base_raw else base_raw
    location = _find_system_font(base)
    if not location:
        return {}

    try:
        from fontTools.ttLib import TTFont
        from fontTools.pens.recordingPen import RecordingPen
    except ImportError:
        return {}

    def _sig(glyphset, gname):
        if gname not in glyphset:
            return None
        pen = RecordingPen()
        glyphset[gname].draw(pen)
        norm = []
        for op, args in pen.value:
            pts = tuple(round(c, 1) for pt in args
                        for c in (pt if isinstance(pt, (tuple, list)) else (pt,)))
            norm.append((op, pts))
        return tuple(norm)

    try:
        if isinstance(location, tuple):
            full = TTFont(location[0], fontNumber=location[1])
        else:
            full = TTFont(location)
        sub = TTFont(BytesIO(desc["/FontFile2"].read_bytes()))
    except Exception:
        return {}

    # full font: glyph name -> Unicode string (direct cmap, plus GSUB ligatures)
    glyph_to_uni = {}
    try:
        for cp, gname in full.getBestCmap().items():
            glyph_to_uni.setdefault(gname, chr(cp))
    except Exception:
        return {}
    rev_cmap = {}
    try:
        for cp, gname in full.getBestCmap().items():
            rev_cmap.setdefault(gname, cp)
    except Exception:
        rev_cmap = {}
    if "GSUB" in full:
        try:
            for lk in full["GSUB"].table.LookupList.Lookup:
                if lk.LookupType != 4:
                    continue
                for st in lk.SubTable:
                    for first, liglist in getattr(st, "ligatures", {}).items():
                        for lg in liglist:
                            comps = [first] + list(lg.Component)
                            cps = [rev_cmap.get(c) for c in comps]
                            if all(cp is not None for cp in cps):
                                glyph_to_uni.setdefault(
                                    lg.LigGlyph,
                                    "".join(chr(cp) for cp in cps))
        except Exception:
            pass

    # Index the full font's outlines so we can look a subset glyph up by shape.
    full_gs = full.getGlyphSet()
    sig_to_uni = {}
    for gname, uni in glyph_to_uni.items():
        sig = _sig(full_gs, gname)
        if sig:
            sig_to_uni.setdefault(sig, uni)

    sub_gs = sub.getGlyphSet()
    out = {}
    for code in codes:
        gname = code_to_glyph.get(code)
        if not gname:
            continue
        sig = _sig(sub_gs, gname)
        if sig and sig in sig_to_uni:
            out[code] = sig_to_uni[sig]
    return out


def _recover_via_context(byte_runs, code_to_unicode, codes):
    """Infer the Unicode for missing codes (almost always ligatures) from the
    words they appear in.

    For each missing code, tokenise every run (decoding known glyphs, marking
    the unknown), then test each ligature candidate by substitution: the
    candidate that turns the most tokens into real English words wins, but only
    if it is a *clear* winner (strictly beats the runner-up, hits a high
    fraction, and has enough evidence). Returns {code: unicode_str} for codes
    resolved confidently; never guesses otherwise.
    """
    words = _load_english_words()
    if not words:
        return {}

    # Build, per run, a list of glyphs where each is either a decoded char or
    # ('UNK', code). Then split into whitespace-delimited tokens.
    all_tokens = []
    for run in byte_runs:
        seq = []
        for b in run:
            if b in code_to_unicode:
                seq.append(code_to_unicode[b])
            else:
                seq.append(("UNK", b))
        token = []
        for g in seq:
            if g == " ":
                if token:
                    all_tokens.append(token)
                token = []
            else:
                token.append(g)
        if token:
            all_tokens.append(token)

    out = {}
    for code in codes:
        toks = [t for t in all_tokens
                if any(isinstance(g, tuple) and g[1] == code for g in t)]
        if not toks:
            continue

        scored = []
        for cand in _LIGATURE_CANDIDATES:
            hits = total = 0
            for t in toks:
                s = "".join(
                    (cand if g[1] == code else "") if isinstance(g, tuple)
                    else g
                    for g in t)
                if any(ch.isalpha() for ch in s):
                    total += 1
                    if _is_english_word(s, words):
                        hits += 1
            if total:
                scored.append((hits, total, cand))
        if not scored:
            continue
        scored.sort(reverse=True)
        best = scored[0]
        runner = scored[1] if len(scored) > 1 else (0, 0, "")

        # Confidence gate: clear winner, strong hit fraction, real evidence.
        best_hits, best_total, best_cand = best
        if (best_hits > runner[0]
                and best_hits >= 1
                and best_hits / best_total >= 0.6):
            out[code] = best_cand
    return out


def _load_english_words():
    """Load the bundled English wordlist (lowercased) as a frozenset, cached.

    Falls back to /usr/share/dict/words, then to an empty set (which disables
    context inference) if no wordlist is available.
    """
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is not None:
        return _ENGLISH_WORDS

    words = set()
    bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "words_en.txt.gz")
    try:
        import gzip
        with gzip.open(bundled, "rt", encoding="ascii") as fh:
            for line in fh:
                w = line.strip()
                if w:
                    words.add(w)
    except Exception:
        try:
            with open("/usr/share/dict/words", encoding="utf-8",
                      errors="ignore") as fh:
                for line in fh:
                    w = line.strip().lower()
                    if len(w) >= 2 and w.isalpha():
                        words.add(w)
        except Exception:
            logger.info("No English wordlist available — context-based "
                        "ToUnicode inference disabled.")

    _ENGLISH_WORDS = frozenset(words)
    return _ENGLISH_WORDS


def _is_english_word(token: str, words) -> bool:
    """True if a token (after stripping non-letters) is a real English word,
    tolerating common inflections so a limited base wordlist still matches
    plurals/past tenses (e.g. 'motives' -> 'motive')."""
    import re
    t = re.sub(r"[^A-Za-z]", "", token).lower()
    if len(t) < 2:
        return False
    if t in words:
        return True
    for suf in ("s", "es", "ed", "ing", "d", "ly"):
        if t.endswith(suf) and t[:-len(suf)] in words:
            return True
    if t.endswith("ies") and (t[:-3] + "y") in words:
        return True
    return False


def _try_embed_font(pdf: pikepdf.Pdf, font_obj, base_font: str):
    """Try to find and embed a system font file."""
    font_location = _find_system_font(base_font)
    if not font_location:
        logger.warning(
            "Could not find system font file for '%s' — font will NOT be embedded. "
            "On Linux, install: apt-get install fonts-liberation ttf-mscorefonts-installer",
            base_font,
        )
        return

    try:
        from fontTools.ttLib import TTFont
        from fontTools.subset import Subsetter
    except ImportError:
        msg = (
            "fontTools is not installed — font embedding skipped for '%s'. "
            "Run: pip install fonttools" % base_font
        )
        logger.error(msg)
        print(f"ERROR: {msg}")
        return

    try:
        if isinstance(font_location, tuple):
            # TTC file: (path, index)
            ttc_path, ttc_index = font_location
            from fontTools.ttLib import TTCollection
            ttc = TTCollection(ttc_path)
            tt = ttc.fonts[ttc_index]
        else:
            tt = TTFont(font_location)
    except Exception as e:
        logger.debug("Could not open font file '%s': %s", font_location, e)
        return

    try:
        head = tt.get("head")
        os2 = tt.get("OS/2")
        post = tt.get("post")
        if not head or not os2:
            tt.close()
            return

        units_per_em = head.unitsPerEm
        scale = 1000.0 / units_per_em

        # Subset to characters used (from FirstChar/LastChar)
        first_char = int(font_obj.get("/FirstChar", 0))
        last_char = int(font_obj.get("/LastChar", 255))
        unicodes = set()
        for code in range(first_char, last_char + 1):
            if code in _WIN1252_SPECIAL:
                unicodes.add(_WIN1252_SPECIAL[code])
            elif 0x20 <= code <= 0x7E or 0xA0 <= code <= 0xFF:
                unicodes.add(code)

        try:
            subsetter = Subsetter()
            subsetter.populate(unicodes=unicodes)
            subsetter.subset(tt)
        except Exception:
            # Reload full font if subsetting fails
            tt.close()
            if isinstance(font_location, tuple):
                from fontTools.ttLib import TTCollection as _TTC
                _ttc = _TTC(font_location[0])
                tt = _ttc.fonts[font_location[1]]
            else:
                tt = TTFont(font_location)
            head = tt["head"]
            os2 = tt["OS/2"]
            post = tt.get("post")

        buf = BytesIO()
        tt.save(buf)
        font_data = buf.getvalue()

        # Ensure FontDescriptor exists
        desc = font_obj.get("/FontDescriptor")
        if desc is None:
            desc = pdf.make_indirect(pikepdf.Dictionary({
                "/Type": pikepdf.Name("/FontDescriptor"),
            }))
            font_obj[pikepdf.Name("/FontDescriptor")] = desc

        # Only set metrics that are MISSING from the existing descriptor.
        # Overwriting existing metrics causes text layout corruption because
        # the original metrics match the document's text positioning.
        def _set_if_missing(key, value):
            if key not in desc:
                desc[pikepdf.Name(key)] = value

        _set_if_missing("/FontName", font_obj.get(
            "/BaseFont", pikepdf.Name("/Unknown")))
        if "/Flags" not in desc:
            flags = 32  # Nonsymbolic
            if post and post.italicAngle != 0:
                flags |= 64  # Italic
            desc[pikepdf.Name("/Flags")] = flags
        _set_if_missing("/FontBBox", pikepdf.Array([
            int(head.xMin * scale),
            int(head.yMin * scale),
            int(head.xMax * scale),
            int(head.yMax * scale),
        ]))
        _set_if_missing("/ItalicAngle",
                         int(post.italicAngle) if post else 0)
        _set_if_missing("/Ascent", int(os2.sTypoAscender * scale))
        _set_if_missing("/Descent", int(os2.sTypoDescender * scale))
        _set_if_missing("/CapHeight", int(
            getattr(os2, "sCapHeight", 700) * scale))
        _set_if_missing("/StemV", 80)

        # Embed font data
        font_stream = pikepdf.Stream(pdf, font_data)
        font_stream[pikepdf.Name("/Length1")] = len(font_data)
        desc[pikepdf.Name("/FontFile2")] = font_stream

    except Exception as e:
        logger.warning("Font embedding failed for %s: %s", base_font, e)
    finally:
        tt.close()


def _find_system_font(base_font: str):
    """Find a system font file matching the PDF BaseFont name.

    Uses multiple strategies:
    1. Exact match from known font name → filename map
    2. Direct filename match (BaseFont.ttf)
    3. Fuzzy match: strip style suffixes, try common variants
    4. TTC (TrueType Collection) map for macOS system fonts
    5. fc-match on Linux

    Returns:
        str path for TTF files, or (str path, int index) tuple for TTC files,
        or None if not found.
    """
    candidates = list(_FONT_FILE_NAMES.get(base_font, []))
    # Also try the base font name directly
    candidates.append(base_font + ".ttf")
    candidates.append(base_font + ".TTF")

    # Try fuzzy variants: strip PS suffixes like MT, PS, PSMT
    clean = base_font
    for suffix in ("PSMT", "PSMt", "PS-BoldMT", "PS-ItalicMT", "PS-BoldItalicMT",
                   "-Roman", "MT", ",Regular"):
        clean = clean.replace(suffix, "")
    # Add space-separated variants (e.g. "TimesNewRoman" -> "Times New Roman")
    import re as _re
    spaced = _re.sub(r'([a-z])([A-Z])', r'\1 \2', clean)
    if spaced != clean:
        candidates.append(spaced + ".ttf")
        candidates.append(spaced + ".TTF")
    # Try with hyphen variants
    for sep in ("-", ","):
        if sep in base_font:
            family = base_font.split(sep)[0]
            candidates.append(family + ".ttf")
            candidates.append(family + ".TTF")

    for font_dir in _FONT_DIRS:
        if not os.path.isdir(font_dir):
            continue
        for candidate in candidates:
            path = os.path.join(font_dir, candidate)
            if os.path.isfile(path):
                return path

    # Check TTC (TrueType Collection) map
    if base_font in _FONT_TTC_MAP:
        ttc_path, ttc_index = _FONT_TTC_MAP[base_font]
        if os.path.isfile(ttc_path):
            return (ttc_path, ttc_index)

    # Scan font dirs for case-insensitive partial match as last resort
    clean_lower = clean.lower().replace(" ", "")
    for font_dir in _FONT_DIRS:
        if not os.path.isdir(font_dir):
            continue
        try:
            for fname in os.listdir(font_dir):
                if not fname.lower().endswith((".ttf", ".otf")):
                    continue
                if clean_lower in fname.lower().replace(" ", ""):
                    return os.path.join(font_dir, fname)
        except OSError:
            continue

    # Fallback: try fc-match on Linux
    if sys.platform.startswith("linux"):
        try:
            import subprocess
            result = subprocess.run(
                ["fc-match", "-f", "%{file}", base_font],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                path = result.stdout.strip()
                if os.path.isfile(path):
                    return path
        except Exception as e:
            logger.debug("fc-match failed for '%s': %s", base_font, e)

    return None
