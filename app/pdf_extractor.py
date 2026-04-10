# pdf_extractor.py
import fitz
import datetime

def extract_text_from_pdf(uploaded_file) -> list[dict]:
    raw_bytes = uploaded_file.read()
    doc       = fitz.open(stream=raw_bytes, filetype="pdf")

    pages = []
    for i in range(len(doc)):
        page     = doc.load_page(i)
        raw_text = page.get_text("text")

        # get_text() returns str in most versions but list in some — normalise it
        if isinstance(raw_text, list):
            text = " ".join(str(t) for t in raw_text).strip()
        else:
            text = str(raw_text).strip()

        if not text:
            continue

        pages.append({
            "headline": f"Page {i + 1} — {uploaded_file.name}",
            "summary":  "",
            "text":     text,
            "datetime": datetime.datetime.now(),
            "url":      "",
            "source":   uploaded_file.name,
        })

    doc.close()
    return pages


def pdf_meta(uploaded_file) -> dict:
    raw_bytes = uploaded_file.read()
    doc       = fitz.open(stream=raw_bytes, filetype="pdf")

    metadata = doc.metadata or {}
    meta = {
        "pages":  len(doc),
        "title":  metadata.get("title")  or uploaded_file.name,
        "author": metadata.get("author") or "Unknown",
    }
    doc.close()
    return meta