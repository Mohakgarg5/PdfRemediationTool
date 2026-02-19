"""
pdf_tagger.py - Add PDF/UA structure tags directly to existing PDFs.

Instead of reconstructing PDFs from scratch (which alters layout),
this module adds accessibility structure tags directly to the original
PDF's content streams, preserving the exact visual appearance.

Hardened for diverse PDF types:
- Handles inherited Resources from page tree
- Handles inline images (BI/ID/EI)
- Supports multiple images per page
- Per-page error handling (one bad page doesn't kill the PDF)
- Safe float conversion for operands
- Adaptive position matching tolerance
- Full table structure tagging (/Table, /TR, /TH, /TD)
- Position-based image matching using CTM coordinates
"""
import logging
import re
import sys
from typing import Optional

import pikepdf

from models import DocumentContent, PageContent, TextBlock, ImageBlock, TableBlock, ElementType

logger = logging.getLogger(__name__)


def tag_pdf(input_path: str, output_path: str, doc_content: DocumentContent) -> str:
    """Add PDF/UA structure tags to the original PDF."""
    try:
        pdf = pikepdf.Pdf.open(input_path)
    except pikepdf.PasswordError:
        logger.error("PDF is encrypted/password-protected: %s", input_path)
        raise
    except Exception as e:
        logger.error("Could not open PDF for tagging: %s", e)
        raise

    _remove_existing_structure(pdf)

    all_page_elems = []

    for page_idx, page in enumerate(pdf.pages):
        page_content = (doc_content.pages[page_idx]
                        if page_idx < len(doc_content.pages) else None)
        mcid_counter = [0]  # Reset per page — ParentTree is indexed by MCID per page
        try:
            page_elems = _tag_page(pdf, page, page_content, page_idx, mcid_counter)
        except Exception as e:
            logger.warning("Page %d tagging failed (%s), wrapping as artifact", page_idx, e)
            mcid_counter = [0]
            page_elems = _tag_page_fallback(pdf, page, page_idx, mcid_counter)
        all_page_elems.append((page_idx, page_elems))

    _build_structure_tree(pdf, all_page_elems, doc_content)

    pdf.save(output_path)
    pdf.close()
    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val, default=0.0) -> float:
    """Convert a pikepdf operand to float safely."""
    try:
        return float(val)
    except (ValueError, TypeError, OverflowError):
        return default


def _resolve_resources(page) -> Optional[pikepdf.Dictionary]:
    """Get Resources for a page, checking inheritance from the page tree."""
    res = page.get("/Resources")
    if res:
        return res
    # Walk up the page tree with circular reference protection
    parent = page.get("/Parent")
    seen = set()
    while parent:
        try:
            obj_id = parent.objgen
            if obj_id in seen:
                break
            seen.add(obj_id)
            res = parent.get("/Resources")
            if res:
                return res
            parent = parent.get("/Parent")
        except Exception:
            break
    return None


def _get_xobjects(page) -> Optional[pikepdf.Dictionary]:
    """Get XObject dictionary, handling inherited Resources."""
    res = _resolve_resources(page)
    if not res:
        return None
    return res.get("/XObject")


# ---------------------------------------------------------------------------
# Structure tree removal
# ---------------------------------------------------------------------------

def _remove_existing_structure(pdf: pikepdf.Pdf):
    """Remove existing structure tree and related entries."""
    if "/StructTreeRoot" in pdf.Root:
        del pdf.Root[pikepdf.Name.StructTreeRoot]
    for page in pdf.pages:
        if "/StructParents" in page:
            del page[pikepdf.Name.StructParents]


# ---------------------------------------------------------------------------
# Page-level tagging
# ---------------------------------------------------------------------------

def _tag_page(pdf, page, page_content: Optional[PageContent],
              page_idx: int, mcid_counter: list) -> list:
    """Tag a single page's content stream with structure markers."""
    blocks = _build_block_index(page_content)

    try:
        ops = list(pikepdf.parse_content_stream(page))
    except Exception:
        # Can't parse → wrap entire page as artifact
        return _tag_page_fallback(pdf, page, page_idx, mcid_counter)

    ops = _strip_markers(ops)

    watermark_forms = _detect_watermark_forms(page)
    link_annots = _collect_link_annots(page)

    new_ops, struct_elems = _insert_markers(
        ops, blocks, page, watermark_forms, mcid_counter, link_annots
    )

    new_stream_data = pikepdf.unparse_content_stream(new_ops)
    page.Contents = pikepdf.Stream(pdf, new_stream_data)
    page[pikepdf.Name.StructParents] = page_idx

    return struct_elems


def _tag_page_fallback(pdf, page, page_idx: int, mcid_counter: list) -> list:
    """Fallback: wrap entire page content in a single /P tag.

    Used when content stream parsing fails.
    """
    mcid = mcid_counter[0]
    mcid_counter[0] += 1

    try:
        ops = list(pikepdf.parse_content_stream(page))
    except Exception:
        page[pikepdf.Name.StructParents] = page_idx
        return [(mcid, "/P", "")]

    new_ops = [
        ([pikepdf.Name("/P"), pikepdf.Dictionary({"/MCID": mcid})],
         pikepdf.Operator("BDC")),
    ]
    new_ops.extend(ops)
    new_ops.append(([], pikepdf.Operator("EMC")))

    new_stream_data = pikepdf.unparse_content_stream(new_ops)
    page.Contents = pikepdf.Stream(pdf, new_stream_data)
    page[pikepdf.Name.StructParents] = page_idx

    return [(mcid, "/P", "")]


def _build_block_index(page_content: Optional[PageContent]) -> list:
    """Build a list of classified blocks with bboxes for position matching."""
    blocks = []
    if not page_content:
        return blocks

    for tb in page_content.text_blocks:
        struct_type = _element_to_struct_type(tb)
        blocks.append({
            "bbox": tb.bbox,
            "struct_type": struct_type,
            "text": tb.text,
            "is_artifact": tb.element_type in (
                ElementType.WATERMARK, ElementType.HEADER_FOOTER
            ),
        })

    for img in page_content.images:
        blocks.append({
            "bbox": img.bbox,
            "struct_type": "/Figure",
            "alt_text": img.alt_text or "Image",
            "is_artifact": img.is_decorative,
            "used": False,
        })

    return blocks


# ---------------------------------------------------------------------------
# Content stream manipulation
# ---------------------------------------------------------------------------

def _strip_markers(ops: list) -> list:
    """Remove all existing BDC/BMC/EMC markers from the content stream."""
    return [
        (operands, op)
        for operands, op in ops
        if str(op) not in ("BDC", "BMC", "EMC")
    ]


def _collect_link_annots(page) -> list:
    """Collect link annotations with their rects for position matching during tagging.

    Skips annotations with abnormally large rects (> 300pt wide) to avoid
    greedily capturing unrelated text. Such annotations typically cover
    entire paragraphs and are better handled by the postprocessor.
    """
    annots = page.get("/Annots")
    if not annots:
        return []

    # Get page width for size guard
    try:
        mediabox = page.get("/MediaBox")
        page_width = float(mediabox[2]) - float(mediabox[0]) if mediabox else 612
    except Exception:
        page_width = 612

    max_annot_width = min(500.0, page_width * 0.8)

    result = []
    for annot in annots:
        try:
            if str(annot.get("/Subtype", "")) != "/Link":
                continue
            rect = annot.get("/Rect")
            if not rect or len(rect) < 4:
                continue
            r = [float(rect[i]) for i in range(4)]
            x0, y0 = min(r[0], r[2]), min(r[1], r[3])
            x1, y1 = max(r[0], r[2]), max(r[1], r[3])
            width = x1 - x0
            height = y1 - y0
            # Skip degenerate or abnormally large rects
            if width <= 0 or height <= 0:
                continue
            if width > max_annot_width:
                logger.debug("Skipping large link annotation (%.0fx%.0f)", width, height)
                continue
            result.append({
                "rect": (x0, y0, x1, y1),
                "annot": annot,
            })
        except Exception:
            continue
    return result


def _find_link_annot(x: float, y: float, link_annots: list) -> Optional[int]:
    """Find the link annotation whose rect contains position (x, y).

    Uses asymmetric tolerances:
    - Left/bottom: generous (text may start slightly before rect due to rounding)
    - Right: tight (text starting past the rect end is not part of the link;
      the text position is the START of the glyph, so text past x1 belongs
      to the next content, not the link)
    - Vertical: generous (text baseline varies relative to annotation rect)

    Does NOT skip already-matched annotations — consecutive text operators
    within the same annotation rect must all be tagged as /Link.
    """
    best_idx = None
    best_dist = float("inf")

    for i, la in enumerate(link_annots):
        x0, y0, x1, y1 = la["rect"]
        height = y1 - y0
        width = x1 - x0
        tol_y = max(3.0, height * 0.5)
        tol_x_left = max(2.0, width * 0.1)
        # Tight right tolerance: only for floating-point rounding (< 0.5pt)
        tol_x_right = 0.5
        if (x0 - tol_x_left <= x <= x1 + tol_x_right and
                y0 - tol_y <= y <= y1 + tol_y):
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            dist = abs(x - cx) + abs(y - cy)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

    return best_idx


def _insert_markers(ops, blocks, page, watermark_forms, mcid_counter,
                    link_annots=None):
    """Walk through content stream ops, inserting BDC/EMC structure markers.

    Uses an "artifact-as-default" strategy so nothing is ever untagged.
    When link_annots is provided, text falling within a link annotation's
    rect is tagged as /Link with both MCR and annotation reference.
    """
    if link_annots is None:
        link_annots = []
    new_ops = []
    struct_elems = []

    # -- state tracking --
    # Account for page /Rotate: adjust initial CTM so position calculations
    # remain correct for rotated pages
    ctm = [1, 0, 0, 1, 0, 0]
    try:
        rotate = int(page.get("/Rotate", 0)) % 360
    except (ValueError, TypeError):
        rotate = 0
    if rotate == 90:
        ctm = [0, 1, -1, 0, 0, 0]
    elif rotate == 180:
        ctm = [-1, 0, 0, -1, 0, 0]
    elif rotate == 270:
        ctm = [0, -1, 1, 0, 0, 0]
    ctm_stack = []
    in_text = False
    tm = [1, 0, 0, 1, 0, 0]
    tlm = [1, 0, 0, 1, 0, 0]
    leading = 0.0

    artifact_open = False
    struct_open = False
    current_block_idx = -1

    def _open_artifact():
        nonlocal artifact_open
        if not artifact_open:
            new_ops.append((
                [pikepdf.Name("/Artifact")],
                pikepdf.Operator("BMC"),
            ))
            artifact_open = True

    def _close_artifact():
        nonlocal artifact_open
        if artifact_open:
            new_ops.append(([], pikepdf.Operator("EMC")))
            artifact_open = False

    current_link_idx = -1
    # Track which annotation indices already have a struct_elem with OBJR,
    # so q/Q splits don't create duplicate OBJR entries for the same annotation.
    linked_annot_indices = set()

    def _close_struct():
        nonlocal struct_open, current_block_idx, current_link_idx
        if struct_open:
            new_ops.append(([], pikepdf.Operator("EMC")))
            struct_open = False
            current_block_idx = -1
            current_link_idx = -1

    def _open_struct_for_block(bidx):
        nonlocal struct_open, current_block_idx

        if struct_open and current_block_idx == bidx:
            return

        _close_struct()
        _close_artifact()

        block = blocks[bidx]
        if block.get("is_artifact"):
            new_ops.append((
                [pikepdf.Name("/Artifact"),
                 pikepdf.Dictionary({"/Type": pikepdf.Name("/Pagination")})],
                pikepdf.Operator("BDC"),
            ))
            struct_open = True
            current_block_idx = bidx
        else:
            mcid = mcid_counter[0]
            mcid_counter[0] += 1
            new_ops.append((
                [pikepdf.Name(block["struct_type"]),
                 pikepdf.Dictionary({"/MCID": mcid})],
                pikepdf.Operator("BDC"),
            ))
            struct_open = True
            current_block_idx = bidx
            struct_elems.append((
                mcid, block["struct_type"], block.get("alt_text", ""),
            ))

    def _open_struct_for_link(lidx):
        nonlocal struct_open, current_block_idx, current_link_idx

        # Same link already open — keep it (handles consecutive text ops)
        if struct_open and current_link_idx == lidx:
            return

        _close_struct()
        _close_artifact()

        mcid = mcid_counter[0]
        mcid_counter[0] += 1
        new_ops.append((
            [pikepdf.Name("/Link"),
             pikepdf.Dictionary({"/MCID": mcid})],
            pikepdf.Operator("BDC"),
        ))
        struct_open = True
        current_block_idx = -3
        current_link_idx = lidx

        if lidx not in linked_annot_indices:
            # First time seeing this annotation — include OBJR reference
            linked_annot_indices.add(lidx)
            struct_elems.append((
                mcid, "/Link", "", None, link_annots[lidx]["annot"],
            ))
        else:
            # Re-opening after q/Q split — MCR-only /Span (no duplicate OBJR).
            # The primary /Link element already has the OBJR; this just
            # ensures the continuation text is still properly tagged.
            struct_elems.append((mcid, "/Span", ""))
            logger.debug("Link annot %d re-tagged as /Span (q/Q split)", lidx)

    def _open_struct_unmatched():
        nonlocal struct_open, current_block_idx

        if struct_open and current_block_idx == -2:
            return

        _close_struct()
        _close_artifact()

        mcid = mcid_counter[0]
        mcid_counter[0] += 1
        new_ops.append((
            [pikepdf.Name("/P"),
             pikepdf.Dictionary({"/MCID": mcid})],
            pikepdf.Operator("BDC"),
        ))
        struct_open = True
        current_block_idx = -2
        struct_elems.append((mcid, "/P", ""))

    # -- start with artifact wrapper --
    _open_artifact()

    for operands, operator in ops:
        op = str(operator)

        # ---- Inline image (BI/ID/EI) → stays in artifact ----
        if op in ("BI", "ID", "EI"):
            if not artifact_open and not struct_open:
                _open_artifact()
            new_ops.append((operands, operator))
            continue

        # ---- Graphics state save/restore ----
        # CRITICAL: close markers before q/Q to prevent crossing boundaries.
        # PDF/UA requires BDC/EMC and q/Q to be properly nested (no crossing).
        if op == "q":
            _close_struct()
            _close_artifact()
            ctm_stack.append(ctm[:])
            new_ops.append((operands, operator))
            _open_artifact()
            continue
        if op == "Q":
            _close_struct()
            _close_artifact()
            if ctm_stack:
                ctm = ctm_stack.pop()
            new_ops.append((operands, operator))
            _open_artifact()
            continue
        if op == "cm" and len(operands) >= 6:
            m = [_safe_float(operands[j]) for j in range(6)]
            ctm = _mat_mul(m, ctm)
            new_ops.append((operands, operator))
            continue

        # ---- Text object begin/end ----
        if op == "BT":
            in_text = True
            tm = [1, 0, 0, 1, 0, 0]
            tlm = [1, 0, 0, 1, 0, 0]
            new_ops.append((operands, operator))
            continue

        if op == "ET":
            _close_struct()
            if not artifact_open:
                _open_artifact()
            in_text = False
            new_ops.append((operands, operator))
            continue

        # ---- Inside text object ----
        if in_text:
            if op == "Tm" and len(operands) >= 6:
                tm = [_safe_float(operands[j]) for j in range(6)]
                tlm = tm[:]
            elif op in ("Td", "TD") and len(operands) >= 2:
                tx = _safe_float(operands[0])
                ty = _safe_float(operands[1])
                tlm = _mat_mul([1, 0, 0, 1, tx, ty], tlm)
                tm = tlm[:]
                if op == "TD":
                    leading = -ty
            elif op == "T*":
                tlm = _mat_mul([1, 0, 0, 1, 0, -leading], tlm)
                tm = tlm[:]
            elif op == "TL" and operands:
                leading = _safe_float(operands[0])

            if op in ("Tj", "TJ", "'", '"'):
                if op == "'":
                    tlm = _mat_mul([1, 0, 0, 1, 0, -leading], tlm)
                    tm = tlm[:]
                elif op == '"':
                    tlm = _mat_mul([1, 0, 0, 1, 0, -leading], tlm)
                    tm = tlm[:]

                ux = ctm[0] * tm[4] + ctm[2] * tm[5] + ctm[4]
                uy = ctm[1] * tm[4] + ctm[3] * tm[5] + ctm[5]

                # Check link annotations first (higher priority)
                lidx = _find_link_annot(ux, uy, link_annots)
                if lidx is not None:
                    _open_struct_for_link(lidx)
                else:
                    bidx = _find_block(ux, uy, blocks)
                    if bidx >= 0:
                        _open_struct_for_block(bidx)
                    else:
                        _open_struct_unmatched()

            new_ops.append((operands, operator))
            continue

        # ---- XObject Do (outside text) ----
        if op == "Do" and operands:
            xobj_name = str(operands[0])

            if xobj_name in watermark_forms:
                _close_struct()
                _close_artifact()
                new_ops.append((
                    [pikepdf.Name("/Artifact"),
                     pikepdf.Dictionary({
                         "/Subtype": pikepdf.Name("/Watermark"),
                         "/Type": pikepdf.Name("/Pagination"),
                     })],
                    pikepdf.Operator("BDC"),
                ))
                new_ops.append((operands, operator))
                new_ops.append(([], pikepdf.Operator("EMC")))
                _open_artifact()
                continue

            xobj_type = _get_xobject_subtype(page, xobj_name)

            if xobj_type == "Image":
                # Position-based image matching using full CTM transform
                # The image is placed at (0,0)-(1,1) in image space; CTM maps
                # this to page space. Use CTM translation + half the scale
                # as the center position for matching.
                img_x = ctm[4] + ctm[0] * 0.5 + ctm[2] * 0.5
                img_y = ctm[5] + ctm[1] * 0.5 + ctm[3] * 0.5
                img_idx = _find_image_block_by_position(blocks, img_x, img_y)

                if img_idx is not None and not blocks[img_idx].get("is_artifact"):
                    _close_struct()
                    _close_artifact()
                    mcid = mcid_counter[0]
                    mcid_counter[0] += 1
                    blocks[img_idx]["used"] = True
                    new_ops.append((
                        [pikepdf.Name("/Figure"),
                         pikepdf.Dictionary({"/MCID": mcid})],
                        pikepdf.Operator("BDC"),
                    ))
                    new_ops.append((operands, operator))
                    new_ops.append(([], pikepdf.Operator("EMC")))
                    alt = blocks[img_idx].get("alt_text", "Image")
                    # Compute bbox from CTM (image space is unit square)
                    x0 = ctm[4]
                    y0 = ctm[5]
                    x1 = ctm[4] + ctm[0]
                    y1 = ctm[5] + ctm[3]
                    fig_bbox = [min(x0, x1), min(y0, y1),
                                max(x0, x1), max(y0, y1)]
                    struct_elems.append((mcid, "/Figure", alt, fig_bbox))
                    _open_artifact()
                    continue

            # Any other Do → stays in artifact wrapper
            if not artifact_open:
                _close_struct()
                _open_artifact()
            new_ops.append((operands, operator))
            continue

        # ---- Everything else stays in current wrapper ----
        if not artifact_open and not struct_open:
            _open_artifact()
        new_ops.append((operands, operator))

    _close_struct()
    _close_artifact()

    return new_ops, struct_elems


# ---------------------------------------------------------------------------
# Position matching
# ---------------------------------------------------------------------------

def _find_block(x: float, y: float, blocks: list) -> int:
    """Find the text block whose bbox best matches position (x, y).

    Uses adaptive tolerance based on the block's own size.
    """
    best_idx = -1
    best_dist = float("inf")

    for idx, block in enumerate(blocks):
        if block["struct_type"] == "/Figure":
            continue
        bbox = block["bbox"]
        # Adaptive tolerance: 20pt or 30% of block height, whichever is larger
        bh = max(bbox.y1 - bbox.y0, 1.0)
        bw = max(bbox.x1 - bbox.x0, 1.0)
        tol_y = max(20.0, bh * 0.3)
        tol_x = max(20.0, bw * 0.15)

        if (bbox.x0 - tol_x <= x <= bbox.x1 + tol_x and
                bbox.y0 - tol_y <= y <= bbox.y1 + tol_y):
            cx = (bbox.x0 + bbox.x1) / 2
            cy = (bbox.y0 + bbox.y1) / 2
            dist = abs(x - cx) + abs(y - cy)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

    return best_idx


def _find_image_block_by_position(blocks: list, x: float, y: float) -> Optional[int]:
    """Find the closest unused image block by CTM position.

    Falls back to the next sequential unused image if no position match.
    """
    best_idx = None
    best_dist = float("inf")

    for idx, block in enumerate(blocks):
        if block["struct_type"] != "/Figure" or block.get("used"):
            continue
        bbox = block["bbox"]
        # Use generous tolerance since CTM position may not perfectly align
        tol = max(50.0, max(bbox.width, bbox.height) * 0.5)
        cx = (bbox.x0 + bbox.x1) / 2
        cy = (bbox.y0 + bbox.y1) / 2
        dist = abs(x - cx) + abs(y - cy)
        if dist < tol and dist < best_dist:
            best_dist = dist
            best_idx = idx

    # Fallback: next unused image block if position matching fails
    if best_idx is None:
        for idx, block in enumerate(blocks):
            if block["struct_type"] == "/Figure" and not block.get("used"):
                best_idx = idx
                break

    return best_idx


# ---------------------------------------------------------------------------
# Watermark detection
# ---------------------------------------------------------------------------

_WATERMARK_KEYWORDS = [
    # English
    "draft", "confidential", "copy", "do not",
    "sample", "watermark", "instructor", "reproduce",
    "preliminary", "internal", "restricted", "void",
    "duplicate", "unofficial", "not for distribution",
    "review", "not for publication", "internal use",
    "proprietary", "for review", "proof",
    # French
    "brouillon", "confidentiel", "copie", "ne pas",
    "projet", "filigrane",
    # German
    "entwurf", "vertraulich", "kopie", "muster",
    "wasserzeichen",
    # Spanish
    "borrador", "confidencial", "copia", "muestra",
]


def _detect_watermark_forms(page) -> set:
    """Return set of XObject names that are watermark Form XObjects."""
    wm_names = set()
    xobjects = _get_xobjects(page)
    if not xobjects:
        return wm_names

    for name, obj in xobjects.items():
        try:
            if not hasattr(obj, "keys"):
                continue
            if obj.get("/Subtype") != pikepdf.Name.Form:
                continue

            # Check for Adobe watermark marker
            piece_info = obj.get("/PieceInfo")
            if piece_info:
                compound = piece_info.get("/ADBE_CompoundType")
                if compound:
                    private = compound.get("/Private")
                    if private and str(private) == "/Watermark":
                        wm_names.add(str(name))
                        continue

            # Check for Optional Content group named "Watermark"
            oc = obj.get("/OC")
            if oc:
                oc_name = ""
                try:
                    if "/Name" in oc:
                        oc_name = str(oc["/Name"])
                    elif "/OCGs" in oc:
                        for ocg in oc["/OCGs"]:
                            if "/Name" in ocg:
                                oc_name = str(ocg["/Name"])
                                break
                except Exception:
                    pass
                if "watermark" in oc_name.lower():
                    wm_names.add(str(name))
                    continue

            # Check content for watermark keywords
            try:
                data = obj.read_bytes().decode("latin-1", errors="replace")
            except Exception:
                continue
            # Only check if stream is small (avoid processing huge Form XObjects)
            if len(data) > 10000:
                continue
            tj_texts = re.findall(r"\((.*?)\)", data)
            full_text = " ".join(tj_texts).strip().lower()

            if any(kw in full_text for kw in _WATERMARK_KEYWORDS):
                wm_names.add(str(name))
        except Exception as e:
            logger.debug("Watermark detection failed for XObject '%s': %s", name, e)
            continue

    return wm_names


# ---------------------------------------------------------------------------
# XObject helpers
# ---------------------------------------------------------------------------

def _get_xobject_subtype(page, xobj_name: str) -> str:
    """Return 'Image', 'Form', or '' for the named XObject.

    Handles inherited Resources from the page tree.
    """
    xobjects = _get_xobjects(page)
    if not xobjects:
        return ""
    obj = xobjects.get(xobj_name)
    if obj is None:
        # Try without leading /
        obj = xobjects.get(xobj_name.lstrip("/"))
    if obj is None:
        return ""
    try:
        subtype = obj.get("/Subtype")
        if subtype == pikepdf.Name.Image:
            return "Image"
        elif subtype == pikepdf.Name.Form:
            return "Form"
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Structure tree construction — with table support
# ---------------------------------------------------------------------------

def _build_structure_tree(pdf: pikepdf.Pdf, all_page_elems: list,
                          doc_content: DocumentContent):
    """Build StructTreeRoot -> Document -> elements hierarchy.

    Groups elements that require parent containers:
    - Consecutive /LI elements are wrapped in /L (List)
    - Consecutive /TD|/TH elements are wrapped in /Table -> /TR
    - All other types (/P, /H1-/H6, /Figure, etc.) go directly under /Document
    """
    # Pass 1: sequence-based heading normalization (PDF/UA Clause 7.4.2).
    # The rule: when heading level increases, it must go up by exactly 1.
    # Walk headings in document order; if a heading jumps more than +1 from
    # the previous heading, clamp it to prev_level + 1.
    # e.g. sequence H3,H1,H4,H2,H3 → H1,H1,H2,H2,H3 (no forward jumps > 1).
    _heading_re = re.compile(r'^/H(\d+)$')
    _elem_overrides: dict = {}   # (page_idx, elem_idx) → new struct_type
    _prev_level = 0
    for _pi, _se in all_page_elems:
        for _ei, _ed in enumerate(_se):
            _m = _heading_re.match(_ed[1])
            if _m:
                _lvl = int(_m.group(1))
                if _lvl > _prev_level + 1:
                    _lvl = _prev_level + 1
                    _elem_overrides[(_pi, _ei)] = f"/H{_lvl}"
                _prev_level = _lvl

    def _remap_struct_type(page_idx: int, elem_idx: int, st: str) -> str:
        return _elem_overrides.get((page_idx, elem_idx), st)

    doc_kids = pikepdf.Array()
    doc_elem = pdf.make_indirect(pikepdf.Dictionary({
        "/Type": pikepdf.Name("/StructElem"),
        "/S": pikepdf.Name("/Document"),
        "/K": doc_kids,
    }))

    parent_tree_nums = pikepdf.Array()
    all_leaf_elems = []  # (elem_ref, struct_type) for grouping
    annot_parent_entries = []  # (annot_obj, elem_ref) for annotation ParentTree

    for page_idx, struct_elems in all_page_elems:
        if not struct_elems:
            continue

        if page_idx >= len(pdf.pages):
            continue

        page_ref = pdf.pages[page_idx].obj
        page_elem_refs = []

        for elem_idx, elem_data in enumerate(struct_elems):
            # Tuples: (mcid, type, alt) or (mcid, type, alt, bbox)
            #     or: (mcid, "/Link", alt, None, annot_obj)
            mcid = elem_data[0]
            struct_type = _remap_struct_type(page_idx, elem_idx, elem_data[1])
            alt_text = elem_data[2]
            fig_bbox = elem_data[3] if len(elem_data) > 3 else None
            annot_obj = elem_data[4] if len(elem_data) > 4 else None

            mcr = pikepdf.Dictionary({
                "/Type": pikepdf.Name("/MCR"),
                "/Pg": page_ref,
                "/MCID": mcid,
            })

            elem_dict = {
                "/Type": pikepdf.Name("/StructElem"),
                "/S": pikepdf.Name(struct_type),
            }

            if struct_type == "/Link" and annot_obj is not None:
                # /Link with annotation: K = [MCR, OBJR]
                objr = pikepdf.Dictionary({
                    "/Type": pikepdf.Name("/OBJR"),
                    "/Pg": page_ref,
                    "/Obj": annot_obj,
                })
                elem_dict["/K"] = pikepdf.Array([mcr, objr])
            else:
                elem_dict["/K"] = mcr

            if struct_type == "/Figure" and alt_text:
                elem_dict["/Alt"] = pikepdf.String(alt_text)

            if struct_type == "/Figure" and fig_bbox:
                elem_dict["/A"] = pikepdf.Dictionary({
                    "/O": pikepdf.Name("/Layout"),
                    "/BBox": pikepdf.Array([
                        fig_bbox[0], fig_bbox[1],
                        fig_bbox[2], fig_bbox[3],
                    ]),
                    "/Placement": pikepdf.Name("/Block"),
                })

            elem = pdf.make_indirect(pikepdf.Dictionary(elem_dict))
            page_elem_refs.append(elem)
            all_leaf_elems.append((elem, struct_type))

            if struct_type == "/Link" and annot_obj is not None:
                annot_parent_entries.append((annot_obj, elem))

        parent_tree_nums.append(page_idx)
        parent_tree_nums.append(pikepdf.Array(page_elem_refs))

    # Group elements under proper parent containers
    _group_and_add_children(pdf, doc_elem, doc_kids, all_leaf_elems)

    # Determine next key for annotation StructParent entries
    max_key = -1
    for i in range(0, len(parent_tree_nums) - 1, 2):
        try:
            k = int(parent_tree_nums[i])
            if k > max_key:
                max_key = k
        except Exception:
            pass
    next_key = max_key + 1

    # Add annotation StructParent entries to ParentTree
    for annot_obj, elem in annot_parent_entries:
        annot_obj[pikepdf.Name("/StructParent")] = next_key
        parent_tree_nums.append(next_key)
        parent_tree_nums.append(elem)
        next_key += 1

    parent_tree = pdf.make_indirect(pikepdf.Dictionary({
        "/Nums": parent_tree_nums,
    }))

    struct_tree_root = pdf.make_indirect(pikepdf.Dictionary({
        "/Type": pikepdf.Name("/StructTreeRoot"),
        "/K": doc_elem,
        "/ParentTree": parent_tree,
        "/ParentTreeNextKey": next_key,
    }))

    doc_elem["/P"] = struct_tree_root
    pdf.Root[pikepdf.Name.StructTreeRoot] = struct_tree_root



def _group_and_add_children(pdf: pikepdf.Pdf, doc_elem, doc_kids,
                             all_leaf_elems: list):
    """Group leaf structure elements under proper parent containers.

    - Consecutive /LI elements → wrapped in /L (List)
    - Consecutive /TD or /TH elements → wrapped in /Table -> /TR
    - Inline elements (/Link, /Span) → wrapped in /P (cannot be direct /Document children)
    - Everything else → direct child of /Document
    """
    _NEEDS_LIST = frozenset(["/LI"])
    _NEEDS_TABLE = frozenset(["/TD", "/TH"])
    # Inline-level elements must not be direct children of /Document.
    # PAC warns (Matterhorn structure tree check) about each one.
    # Wrap them in a /P block so they're properly nested.
    _NEEDS_P_WRAP = frozenset(["/Link", "/Span"])

    def _flush_list(items):
        """Wrap accumulated /LI elements in an /L container."""
        l_kids = pikepdf.Array()
        l_elem = pdf.make_indirect(pikepdf.Dictionary({
            "/Type": pikepdf.Name("/StructElem"),
            "/S": pikepdf.Name("/L"),
            "/P": doc_elem,
            "/K": l_kids,
        }))
        for item_elem in items:
            item_elem[pikepdf.Name("/P")] = l_elem
            l_kids.append(item_elem)
        doc_kids.append(l_elem)

    def _flush_table(cells):
        """Wrap accumulated /TD|/TH elements in /Table -> /TR."""
        table_kids = pikepdf.Array()
        table_elem = pdf.make_indirect(pikepdf.Dictionary({
            "/Type": pikepdf.Name("/StructElem"),
            "/S": pikepdf.Name("/Table"),
            "/P": doc_elem,
            "/K": table_kids,
        }))

        # Group cells into rows. Since we process sequentially, we create
        # one /TR per consecutive group. Each cell becomes a child of /TR.
        tr_kids = pikepdf.Array()
        tr_elem = pdf.make_indirect(pikepdf.Dictionary({
            "/Type": pikepdf.Name("/StructElem"),
            "/S": pikepdf.Name("/TR"),
            "/P": table_elem,
            "/K": tr_kids,
        }))

        for cell_elem in cells:
            cell_elem[pikepdf.Name("/P")] = tr_elem
            tr_kids.append(cell_elem)

        table_kids.append(tr_elem)
        doc_kids.append(table_elem)

    def _wrap_in_p(child_elem):
        """Wrap a single inline element in a /P block under /Document."""
        p_kids = pikepdf.Array([child_elem])
        p_elem = pdf.make_indirect(pikepdf.Dictionary({
            "/Type": pikepdf.Name("/StructElem"),
            "/S": pikepdf.Name("/P"),
            "/P": doc_elem,
            "/K": p_kids,
        }))
        child_elem[pikepdf.Name("/P")] = p_elem
        doc_kids.append(p_elem)

    # Accumulate consecutive same-type elements
    pending_list = []
    pending_table = []

    def _flush_pending():
        if pending_list:
            _flush_list(pending_list)
            pending_list.clear()
        if pending_table:
            _flush_table(pending_table)
            pending_table.clear()

    for elem, struct_type in all_leaf_elems:
        if struct_type in _NEEDS_LIST:
            if pending_table:
                _flush_table(pending_table)
                pending_table.clear()
            pending_list.append(elem)
        elif struct_type in _NEEDS_TABLE:
            if pending_list:
                _flush_list(pending_list)
                pending_list.clear()
            pending_table.append(elem)
        elif struct_type in _NEEDS_P_WRAP:
            # Inline element — flush pending groups, then wrap in /P
            _flush_pending()
            _wrap_in_p(elem)
        else:
            # Block-level or neutral element — flush pending, add directly
            _flush_pending()
            elem[pikepdf.Name("/P")] = doc_elem
            doc_kids.append(elem)

    # Flush remaining
    _flush_pending()



# ---------------------------------------------------------------------------
# Element type mapping
# ---------------------------------------------------------------------------

def _element_to_struct_type(tb: TextBlock) -> str:
    """Map a TextBlock's ElementType to a PDF structure tag name."""
    if tb.element_type == ElementType.HEADING:
        level = max(1, min(tb.heading_level or 1, 6))
        return f"/H{level}"
    elif tb.element_type == ElementType.LIST_ITEM:
        return "/LI"
    elif tb.element_type == ElementType.TABLE_CELL:
        return "/TD"
    elif tb.element_type == ElementType.TABLE_HEADER:
        return "/TH"
    elif tb.element_type == ElementType.WATERMARK:
        return "/Artifact"
    elif tb.element_type == ElementType.HEADER_FOOTER:
        return "/Artifact"
    else:
        return "/P"


# ---------------------------------------------------------------------------
# Matrix math
# ---------------------------------------------------------------------------

def _mat_mul(m1: list, m2: list) -> list:
    """Multiply two 2D affine matrices [a, b, c, d, e, f]."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return [
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    ]
