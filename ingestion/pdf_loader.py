from pypdf import PdfReader


def extract_pdf_text(pdf_path):
    """
    Extract text from a PDF file.
    """
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text