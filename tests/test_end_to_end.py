import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.document_processor import extract_text_from_pdf_path, chunk_documents
from src.vector_store import VectorStoreManager, get_embedding_function

def test_pdf_extraction_and_chroma():
    pdf_path = root_dir / "sample_data" / "sample_policy.pdf"
    print(f"Reading sample PDF from {pdf_path}...")
    docs = extract_text_from_pdf_path(str(pdf_path))
    assert len(docs) > 0, "No documents extracted from PDF!"
    print(f"Extracted {len(docs)} page(s). First page length: {len(docs[0].page_content)} characters.")

    print("Splitting into chunks...")
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 0, "No chunks generated!"
    print(f"Created {len(chunks)} chunks.")

    print("Testing VectorStore with HuggingFace embedding (free/offline test)...")
    test_chroma_dir = str(root_dir / "sample_data" / "test_chroma_db")
    emb_fn = get_embedding_function(provider="huggingface")
    manager = VectorStoreManager(persist_directory=test_chroma_dir, collection_name="test_col")
    manager.initialize_store(embedding_function=emb_fn)
    manager.add_documents(chunks)

    count = manager.count_documents()
    print(f"Chroma DB successfully indexed {count} items.")
    assert count == len(chunks), f"Expected {len(chunks)} items, got {count}"

    print("Testing similarity search retriever...")
    retriever = manager.get_retriever(k=2)
    results = retriever.invoke("What model powers fast inference?")
    assert len(results) > 0, "Retriever returned 0 results!"
    print(f"Retriever matched {len(results)} chunks. Top snippet:\n{results[0].page_content[:100]}...")

    # Cleanup test chroma db
    manager.reset_store()
    print("Test Chroma DB cleaned up. End-to-end extraction, chunking, and Chroma persistence test PASSED!")

if __name__ == "__main__":
    test_pdf_extraction_and_chroma()
