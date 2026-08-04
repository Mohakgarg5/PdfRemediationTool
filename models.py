"""
models.py - Shared data structures for the PDF accessibility pipeline.

Defines the data classes that flow between pipeline stages.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ElementType(Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_CELL = "table_cell"
    TABLE_HEADER = "table_header"
    IMAGE = "image"
    WATERMARK = "watermark"
    HEADER_FOOTER = "header_footer"


@dataclass
class FontInfo:
    name: str
    size: float
    is_bold: bool
    is_italic: bool
    color: tuple = (0.0, 0.0, 0.0)


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class TextBlock:
    """A contiguous block of text with uniform formatting."""
    text: str
    bbox: BBox
    font: FontInfo
    element_type: ElementType = ElementType.PARAGRAPH
    heading_level: Optional[int] = None
    rotation_degrees: float = 0.0
    page_number: int = 0
    # Per-word horizontal extents as (x0, x1, text). pdfminer groups characters
    # into blocks without regard to table cell boundaries, so two adjacent
    # column headers sharing a baseline can arrive as one block; word extents
    # let such a block be split back onto the correct columns.
    words: list = field(default_factory=list)


@dataclass
class ImageBlock:
    """An extracted image with metadata."""
    image_bytes: bytes
    format: str
    bbox: BBox
    page_number: int
    alt_text: str = ""
    is_decorative: bool = False


@dataclass
class TableCell:
    """One cell of a detected table, with position and role."""
    text: str
    bbox: BBox
    row: int = 0
    col: int = 0
    is_header: bool = False


@dataclass
class TableBlock:
    """A detected table structure."""
    rows: list = field(default_factory=list)      # list[list[str]] (legacy view)
    cells: list = field(default_factory=list)     # list[TableCell] (structured)
    n_cols: int = 0
    header_rows: int = 1
    bbox: Optional[BBox] = None
    page_number: int = 0


@dataclass
class PageContent:
    """All extracted content for a single page."""
    page_number: int
    width: float
    height: float
    text_blocks: list = field(default_factory=list)
    images: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    fill_rects: list = field(default_factory=list)  # BBox of filled rectangles


@dataclass
class DocumentContent:
    """Complete extracted content for one PDF."""
    title: str
    language: str
    pages: list = field(default_factory=list)
    source_path: str = ""
