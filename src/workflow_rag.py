import os
from typing import Annotated, Dict, Any, List, Optional, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from src.config import DEFAULT_GROQ_MODEL
from src.llm_factory import get_llm


class WorkflowState(TypedDict):
    """
    Shared state for the LangGraph 3-stage RAG workflow.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    retrieved_docs: List[Document]
    context: str
    answer: str


WORKFLOW_SYSTEM_PROMPT = """You are an accurate, reliable PDF Document Assistant.
Answer the user's question using ONLY the provided document context below.
If the information is not present in the context, truthfully state that the uploaded documents do not contain this information.
Do not hallucinate or speculate beyond the provided text.
Always include citations referencing the source file and page numbers (e.g. "[Source: filename.pdf, Page 2]").

Document Context:
{context}
"""


class WorkflowRAGAssistant:
    """
    RAG Assistant using an explicit LangGraph multi-stage workflow.
    Matches Resume Bullet:
    'Developed additional RAG implementations using persistent Chroma storage and a
    LangGraph workflow with separate retrieval, context preparation, and answer generation stages.'
    """

    def __init__(
        self,
        retriever,
        groq_api_key: Optional[str] = None,
        model_name: str = DEFAULT_GROQ_MODEL,
        temperature: float = 0.1,
    ):
        self.retriever = retriever
        if not groq_api_key:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                    groq_api_key = str(st.secrets["GROQ_API_KEY"]).strip()
            except Exception:
                pass
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Groq API key is required. Please set GROQ_API_KEY.")

        self.model_name = model_name
        self.temperature = temperature
        self.llm = get_llm(
            api_key=self.groq_api_key,
            model_name=self.model_name,
            temperature=self.temperature,
        )

        self.checkpointer = MemorySaver()
        self.workflow = self._build_graph()

    def _retrieve_stage(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Stage 1: Retrieval stage.
        Queries the persistent Chroma vector store for top semantic matches.
        """
        query = state.get("question")
        if not query and state.get("messages"):
            # Extract from latest human message
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break

        retrieved_docs = self.retriever.invoke(query or "")
        return {"retrieved_docs": retrieved_docs}

    def _prepare_context_stage(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Stage 2: Context Preparation stage.
        Deduplicates, extracts source metadata, and formats chunks with citation headers.
        """
        docs = state.get("retrieved_docs", [])
        if not docs:
            return {"context": "No relevant documents found."}

        formatted_snippets = []
        seen_texts = set()

        for doc in docs:
            cleaned = doc.page_content.strip()
            if cleaned in seen_texts:
                continue
            seen_texts.add(cleaned)

            source = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page", "Unknown Page")
            snippet = f"--- [Document: {source} | Page: {page}] ---\n{cleaned}"
            formatted_snippets.append(snippet)

        formatted_context = "\n\n".join(formatted_snippets)
        return {"context": formatted_context}

    def _generate_answer_stage(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Stage 3: Answer Generation stage.
        Synthesizes final grounded answer using Groq LLM with context and chat history.
        """
        context = state.get("context", "")
        question = state.get("question", "")

        # Format system prompt with prepared context
        system_content = WORKFLOW_SYSTEM_PROMPT.format(context=context)
        prompt_messages = [SystemMessage(content=system_content)]

        # Include prior conversation history for conversational context
        chat_history = state.get("messages", [])
        for msg in chat_history:
            if isinstance(msg, (HumanMessage, AIMessage)):
                prompt_messages.append(msg)

        # Include latest question if not already in messages
        if not chat_history or chat_history[-1].content != question:
            prompt_messages.append(HumanMessage(content=question))

        response = self.llm.invoke(prompt_messages)
        return {
            "answer": response.content,
            "messages": [AIMessage(content=response.content)],
        }

    def _build_graph(self):
        """
        Constructs and compiles the 3-stage LangGraph workflow.
        """
        builder = StateGraph(WorkflowState)

        # Register stages as nodes
        builder.add_node("retrieval_stage", self._retrieve_stage)
        builder.add_node("context_preparation_stage", self._prepare_context_stage)
        builder.add_node("answer_generation_stage", self._generate_answer_stage)

        # Connect edges sequentially: retrieval -> context prep -> generation
        builder.add_edge(START, "retrieval_stage")
        builder.add_edge("retrieval_stage", "context_preparation_stage")
        builder.add_edge("context_preparation_stage", "answer_generation_stage")
        builder.add_edge("answer_generation_stage", END)

        # Compile with in-memory checkpointer for thread persistence
        return builder.compile(checkpointer=self.checkpointer)

    def query(self, user_message: str, thread_id: str = "default_thread") -> Dict[str, Any]:
        """
        Executes the LangGraph 3-stage pipeline for a query.
        """
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {
            "question": user_message,
            "messages": [HumanMessage(content=user_message)],
        }

        final_state = self.workflow.invoke(inputs, config=config)

        # Extract deduplicated source citations for UI display
        sources = []
        seen_snippets = set()
        for doc in final_state.get("retrieved_docs", []):
            snippet = doc.page_content.strip()
            if snippet in seen_snippets:
                continue
            seen_snippets.add(snippet)
            sources.append({
                "source": doc.metadata.get("source", "Document"),
                "page": doc.metadata.get("page", "1"),
                "snippet": snippet[:250] + ("..." if len(snippet) > 250 else ""),
            })

        return {
            "answer": final_state.get("answer", "No response generated."),
            "retrieved_docs": final_state.get("retrieved_docs", []),
            "context": final_state.get("context", ""),
            "sources": sources,
        }
