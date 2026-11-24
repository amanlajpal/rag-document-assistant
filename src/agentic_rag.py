import os
from typing import Optional, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool, create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from src.config import DEFAULT_GROQ_MODEL
from src.llm_factory import get_llm


AGENT_SYSTEM_PROMPT = """You are an intelligent, precise conversational PDF assistant.
You have access to a tool called 'search_pdf_documents' which retrieves relevant passages from uploaded PDF documents.

Guidelines:
1. When asked about documents, policies, notes, summaries, or facts from uploaded files, ALWAYS use the 'search_pdf_documents' tool to fetch accurate context before answering.
2. If the user greets you or asks general conversational questions not related to the documents, answer naturally and politely without calling the tool.
3. Base your answers strictly on the retrieved passages. If the passages do not contain enough information to answer the question, state clearly that the uploaded documents do not contain that information.
4. Always cite the document source filename and page number whenever quoting or referencing facts (e.g. "[Source: report.pdf, Page 3]").
5. Be concise, well-structured, and helpful.
"""


class AgenticRAGAssistant:
    """
    RAG Assistant using a Tool-Calling Agent with LangGraph and MemorySaver.
    Matches Resume Bullet:
    'Integrated document retrieval as an agent tool to fetch relevant passages for
    answering questions, with in-memory checkpoints to maintain conversational context.'
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

        # In-memory checkpointer for conversational context preservation
        self.checkpointer = MemorySaver()
        self.agent_executor = self._build_agent()

    def _build_agent(self):
        # 1. Initialize Chat Model with tool calling capability
        llm = get_llm(
            api_key=self.groq_api_key,
            model_name=self.model_name,
            temperature=self.temperature,
        )

        # 2. Package document retrieval as a LangChain Agent Tool
        retriever_tool = create_retriever_tool(
            self.retriever,
            name="search_pdf_documents",
            description=(
                "Search and retrieve relevant text snippets and passages from the "
                "uploaded PDF documents. Use this whenever the user asks about facts, "
                "summaries, or specific topics in their files."
            ),
        )

        # 3. Create LangGraph ReAct agent with system prompt and MemorySaver
        # Supports 'prompt' (modern LangGraph) with fallback to 'state_modifier' (legacy)
        try:
            agent = create_react_agent(
                model=llm,
                tools=[retriever_tool],
                prompt=AGENT_SYSTEM_PROMPT,
                checkpointer=self.checkpointer,
            )
        except TypeError:
            agent = create_react_agent(
                model=llm,
                tools=[retriever_tool],
                state_modifier=AGENT_SYSTEM_PROMPT,
                checkpointer=self.checkpointer,
            )
        return agent

    def query(self, user_message: str, thread_id: str = "default_thread") -> Dict[str, Any]:
        """
        Processes a user query through the agent with checkpointed conversation memory.
        Returns the final answer and any retrieved document references.
        """
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=user_message)]}

        # Invoke the agent graph
        result = self.agent_executor.invoke(inputs, config=config)

        # Extract messages from result
        messages = result.get("messages", [])
        final_answer = ""
        tool_sources = []

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if isinstance(msg.content, str):
                    final_answer = msg.content
                elif isinstance(msg.content, list):
                    text_parts = [
                        p["text"] if isinstance(p, dict) and "text" in p else str(p)
                        for p in msg.content
                    ]
                    final_answer = "\n".join(text_parts)
                if final_answer.strip():
                    break

        # Check for tool call messages to extract retrieved context snippets
        raw_tool_chunks = []
        for msg in messages:
            if getattr(msg, "name", None) == "search_pdf_documents" or type(msg).__name__ == "ToolMessage":
                if msg.content:
                    # In create_retriever_tool, chunks are separated by double newlines
                    snippets = [s.strip() for s in str(msg.content).split("\n\n") if s.strip()]
                    raw_tool_chunks.extend(snippets if snippets else [msg.content.strip()])

        # Deduplicate snippets while preserving order
        seen_snippets = set()
        for snippet in raw_tool_chunks:
            if snippet not in seen_snippets:
                seen_snippets.add(snippet)
                tool_sources.append(snippet)

        return {
            "answer": final_answer,
            "sources": tool_sources,
            "raw_messages": messages,
        }
