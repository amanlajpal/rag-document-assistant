import sys
import os
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from langchain_core.documents import Document
from src.document_processor import chunk_documents
from src.vector_store import VectorStoreManager
from src.agentic_rag import AgenticRAGAssistant
from src.workflow_rag import WorkflowRAGAssistant


def test_chunking():
    print("Testing document chunking...")
    sample_docs = [
        Document(
            page_content="Artificial Intelligence (AI) is transforming software development.\n\n"
                         "LangChain provides tools to compose LLM chains and agents.\n"
                         "LangGraph introduces cyclic state graphs for complex reasoning.\n"
                         "RAG connects LLMs with external proprietary documents to reduce hallucinations.",
            metadata={"source": "test_document.pdf", "page": 1, "total_pages": 1},
        )
    ]
    chunks = chunk_documents(sample_docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 0, "No chunks generated!"
    print(f"Chunking test passed! Generated {len(chunks)} chunks.")
    return chunks


def test_imports():
    print("Testing imports for AgenticRAG and WorkflowRAG...")
    assert AgenticRAGAssistant is not None
    assert WorkflowRAGAssistant is not None
    assert VectorStoreManager is not None
    print("Imports test passed!")


if __name__ == "__main__":
    test_imports()
    test_chunking()
    print("All basic unit tests passed successfully!")
