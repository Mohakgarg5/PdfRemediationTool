#!/bin/bash
# Convenience script to run the PDF Accessibility Pipeline
# Usage:
#   ./run.sh                           # Process all PDFs in input/
#   ./run.sh --input file.pdf          # Process a single file
#   ./run.sh --skip-validation         # Skip veraPDF validation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3.12"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found."
    echo "Run the following to set up:"
    echo "  python3.12 -m venv venv"
    echo "  ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

cd "$SCRIPT_DIR"
exec "$VENV_PYTHON" main.py "$@"
