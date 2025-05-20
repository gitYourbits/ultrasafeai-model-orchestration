import pdfplumber
from typing import List
import os
import logging

class DocumentParserAgent:
    """
    Agent to extract text from financial PDF documents.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def parse_pdf(self, pdf_path: str) -> str:
        self.logger.info(f"Starting PDF parsing: {pdf_path}")
        if not os.path.exists(pdf_path):
            self.logger.error(f"File not found: {pdf_path}")
            raise FileNotFoundError(f"File not found: {pdf_path}")
        text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            self.logger.info(f"Successfully parsed PDF: {pdf_path}")
        except Exception as e:
            self.logger.error(f"Error parsing PDF: {e}")
            raise
        return "\n".join(text)

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python document_parser.py <pdf_path>")
        exit(1)
    parser = DocumentParserAgent()
    extracted_text = parser.parse_pdf(sys.argv[1])
    print(extracted_text) 