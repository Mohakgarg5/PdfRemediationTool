"""
config.py - Configuration constants for the PDF accessibility pipeline.
"""
from pathlib import Path

# Directories
DEFAULT_INPUT_DIR = Path("input")
DEFAULT_OUTPUT_DIR = Path("output")

# Heading detection thresholds (ratio of font size to body text size)
HEADING_SIZE_RATIO_H1 = 1.8
HEADING_SIZE_RATIO_H2 = 1.5
HEADING_SIZE_RATIO_H3 = 1.25
HEADING_SIZE_RATIO_H4 = 1.1

# Watermark detection
WATERMARK_MIN_ROTATION = 15.0
WATERMARK_MAX_ROTATION = 75.0
WATERMARK_MIN_FONT_SIZE = 36.0
WATERMARK_LIGHT_COLOR_THRESHOLD = 0.7

# Header/footer detection (fraction of page height)
HEADER_ZONE_FRACTION = 0.08
FOOTER_ZONE_FRACTION = 0.08

# veraPDF
VERAPDF_PROFILE = "ua1"

# Image alt text placeholder
DEFAULT_IMAGE_ALT = "Figure"
