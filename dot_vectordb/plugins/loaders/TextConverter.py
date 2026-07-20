from pathlib import Path
from typing import Dict, List
import pdfplumber
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation


class DocumentTextExtractor:
    """
    Extracts raw text from TXT, PDF, DOCX, Excel, and PowerPoint files.
    """

    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".pdf",
        ".docx",
        ".xlsx",
        ".xlsm",
        ".pptx",
    }

    def __init__(self, directories: List[str]):
        self.directories = [Path(directory) for directory in directories]

    def extract_txt(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8")

    def extract_pdf(self, file_path: Path) -> str:
        text = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)

        return "\n".join(text)

    def extract_docx(self, file_path: Path) -> str:
        document = Document(file_path)
        text = []

        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text)

        for table in document.tables:
            for row in table.rows:
                row_text = [
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text.strip()
                ]
                if row_text:
                    text.append(" | ".join(row_text))

        return "\n".join(text)

    def extract_excel(self, file_path: Path) -> str:
        workbook = load_workbook(file_path, data_only=True)
        text = []

        for sheet in workbook.worksheets:
            text.append(f"Sheet: {sheet.title}")

            for row in sheet.iter_rows(values_only=True):
                row_values = [
                    str(cell)
                    for cell in row
                    if cell is not None
                ]

                if row_values:
                    text.append(" | ".join(row_values))

        return "\n".join(text)

    def extract_slide(self, file_path: Path) -> str:
        presentation = Presentation(file_path)
        text = []

        for slide_number, slide in enumerate(presentation.slides, start=1):
            text.append(f"Slide {slide_number}:")

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text.append(shape.text.strip())

        return "\n".join(text)

    def extract_file(self, file_path: Path) -> str:
        extension = file_path.suffix.lower()

        if extension == ".txt":
            return self.extract_txt(file_path)

        if extension == ".pdf":
            return self.extract_pdf(file_path)

        if extension == ".docx":
            return self.extract_docx(file_path)

        if extension in {".xlsx", ".xlsm"}:
            return self.extract_excel(file_path)

        if extension == ".pptx":
            return self.extract_slide(file_path)

        raise ValueError(f"Unsupported file type: {extension}")

    def controller(self, recursive: bool = True) -> Dict[str, str]:
        extracted_text = {}

        for directory in self.directories:
            if not directory.exists():
                continue

            files = directory.rglob("*") if recursive else directory.glob("*")

            for file_path in files:
                if not file_path.is_file():
                    continue

                if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    continue

                try:
                    extracted_text[str(file_path)] = self.extract_file(file_path)
                except Exception as error:
                    extracted_text[str(file_path)] = f"ERROR: {error}"

        return extracted_text