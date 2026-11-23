import os
import tempfile
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def extract_text_from_pdf_stream(uploaded_file) -> List[Document]:
    """
    Extracts text page by page from an uploaded Streamlit file object (or BytesIO).
    Preserves document metadata such as source filename, page number, and total pages.
    """
    filename = getattr(uploaded_file, "name", "document.pdf")
    pdf_reader = PdfReader(uploaded_file)
    total_pages = len(pdf_reader.pages)
    documents = []

    for page_idx, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if page_text:
            metadata = {
                "source": filename,
                "page": page_idx + 1,
                "total_pages": total_pages,
            }
            documents.append(Document(page_content=page_text, metadata=metadata))

    return documents


def extract_text_from_pdf_path(file_path: str) -> List[Document]:
    """
    Extracts text page by page from a local PDF file path.
    """
    with open(file_path, "rb") as f:
        return extract_text_from_pdf_stream(f)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Splits documents into smaller chunks using RecursiveCharacterTextSplitter
    while maintaining metadata and chunk tracking IDs.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    # Enrich metadata with chunk index per document
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        # Ensure clean preview
        chunk.metadata["snippet"] = chunk.page_content[:150].replace("\n", " ") + "..."

    return chunks
