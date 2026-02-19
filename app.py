"""
app.py - Streamlit UI for the PDF Accessibility Remediation Pipeline.

Run with:
    streamlit run app.py
"""
import logging
import tempfile
import time
from pathlib import Path

import streamlit as st

from pdf_extractor import extract_document
from pdf_tagger import tag_pdf
from pdf_postprocess import postprocess_pdf

# Configure logging for the web UI
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(name)s] %(message)s",
)

# Maximum upload size per file (100 MB)
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

st.set_page_config(
    page_title="PDF Accessibility Pipeline",
    page_icon="♿",
    layout="centered",
)

st.title("PDF Accessibility Pipeline")
st.markdown("Upload PDFs to make them **PDF/UA-1 compliant** and pass the PAC report.")
st.caption(f"Maximum file size: {MAX_FILE_SIZE_MB} MB per file")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    # Validate file sizes before processing
    oversized = [f for f in uploaded_files if f.size > MAX_FILE_SIZE_BYTES]
    if oversized:
        for f in oversized:
            st.error(f"'{f.name}' exceeds the {MAX_FILE_SIZE_MB} MB limit "
                     f"({f.size / 1024 / 1024:.1f} MB)")
        uploaded_files = [f for f in uploaded_files if f.size <= MAX_FILE_SIZE_BYTES]

    if not uploaded_files:
        st.warning("No valid files to process.")
    else:
        st.markdown(f"**{len(uploaded_files)}** file(s) selected")

        if st.button("Make Accessible", type="primary", use_container_width=True):
            results = []

            for uploaded in uploaded_files:
                st.divider()
                st.subheader(uploaded.name)
                progress = st.progress(0, text="Starting...")

                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Write uploaded file to disk
                    input_path = Path(tmp_dir) / uploaded.name
                    input_path.write_bytes(uploaded.getvalue())
                    output_path = Path(tmp_dir) / f"{input_path.stem}_accessible.pdf"

                    try:
                        start = time.time()

                        # Stage 1: Extract & classify content
                        progress.progress(15, text="Extracting content...")
                        doc_content = extract_document(str(input_path))
                        text_count = sum(len(p.text_blocks) for p in doc_content.pages)
                        img_count = sum(len(p.images) for p in doc_content.pages)

                        # Stage 2: Tag original PDF with structure markers
                        progress.progress(40, text="Adding structure tags...")
                        tag_pdf(str(input_path), str(output_path), doc_content)

                        # Stage 3: Post-process metadata
                        progress.progress(75, text="Fixing metadata...")
                        postprocess_pdf(
                            str(output_path),
                            doc_content.title,
                            doc_content.language,
                        )

                        duration = time.time() - start
                        progress.progress(100, text="Done!")

                        # Show stats
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Pages", len(doc_content.pages))
                        col2.metric("Text Blocks", text_count)
                        col3.metric("Images", img_count)
                        col4.metric("Time", f"{duration:.1f}s")

                        st.success(f"Title: {doc_content.title} | Language: {doc_content.language}")

                        # Download button
                        pdf_bytes = output_path.read_bytes()
                        st.download_button(
                            label=f"Download {output_path.name}",
                            data=pdf_bytes,
                            file_name=output_path.name,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                        results.append((uploaded.name, True, None))

                    except Exception as e:
                        progress.progress(100, text="Failed")
                        st.error(f"Error: {type(e).__name__}: {e}")
                        results.append((uploaded.name, False, str(e)))

            # Summary
            if len(results) > 1:
                st.divider()
                success = sum(1 for _, ok, _ in results if ok)
                st.markdown(f"### Summary: {success}/{len(results)} files processed successfully")
