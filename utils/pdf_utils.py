"""
Utilities for extracting raw text from uploaded PDF files
(resumes and job description documents).
"""
from io import BytesIO
from pypdf import PdfReader


def extract_text_from_pdf(file_obj) -> str:
    """
    Extract text from a PDF file-like object (e.g. a Streamlit
    UploadedFile) or a raw bytes object.
    """
    if isinstance(file_obj, (bytes, bytearray)):
        file_obj = BytesIO(file_obj)

    reader = PdfReader(file_obj)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)

    return "\n".join(text_parts).strip()


def extract_text_from_multiple_pdfs(file_objs) -> list[dict]:
    """
    Extract text from multiple uploaded PDFs.
    Returns a list of {"filename": str, "text": str} dicts,
    skipping files that fail to parse or contain no text.
    """
    documents = []
    for f in file_objs:
        try:
            text = extract_text_from_pdf(f)
            if text:
                documents.append({"filename": f.name, "text": text})
        except Exception as e:
            print(f"[warn] failed to parse {getattr(f, 'name', 'file')}: {e}")
    return documents
