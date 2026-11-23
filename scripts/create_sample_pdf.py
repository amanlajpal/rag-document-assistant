# Minimal valid single-page PDF generator using pure Python
from pathlib import Path

def create_sample_pdf(filepath: str):
    """
    Creates a minimal valid PDF document with test GenAI content for RAG testing.
    """
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<\n"
        b"/Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"5 0 obj<</Length 420>>stream\n"
        b"BT\n"
        b"/F1 16 Tf\n"
        b"50 720 Td\n"
        b"(GenAI Document Assistant - Project Overview) Tj\n"
        b"/F1 11 Tf\n"
        b"0 -30 Td\n"
        b"(This document serves as a test file for the RAG Document Assistant.) Tj\n"
        b"0 -20 Td\n"
        b"(Architecture: The application supports two distinct RAG pipelines:) Tj\n"
        b"0 -20 Td\n"
        b"(1. Agentic RAG using LangGraph ReAct agent with tool calling and in-memory checkpoints.) Tj\n"
        b"0 -20 Td\n"
        b"(2. Workflow RAG with separate retrieval, context preparation, and generation stages.) Tj\n"
        b"0 -20 Td\n"
        b"(Persistent vector storage is managed by Chroma DB with OpenAI embeddings.) Tj\n"
        b"0 -20 Td\n"
        b"(Fast inference is powered by Groq Llama-3.3-70B model.) Tj\n"
        b"ET\n"
        b"endstream\n"
        b"endobj\n"
        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f \n"
        b"0000000010 00000 n \n"
        b"0000000059 00000 n \n"
        b"0000000116 00000 n \n"
        b"0000000223 00000 n \n"
        b"0000000289 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n"
        b"760\n"
        b"%%EOF\n"
    )
    with open(filepath, "wb") as f:
        f.write(pdf_content)

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "sample_data"
    out_dir.mkdir(exist_ok=True)
    pdf_path = out_dir / "sample_policy.pdf"
    create_sample_pdf(str(pdf_path))
    print(f"Sample PDF created at: {pdf_path}")
