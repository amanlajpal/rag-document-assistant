import os
from typing import Optional, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from src.config import DEFAULT_GROQ_MODEL, AVAILABLE_GROQ_MODELS


def fetch_available_groq_models(groq_api_key: str) -> List[str]:
    """
    Dynamically fetches the exact list of available models for a given Groq API key.
    Filters for chat models suitable for tool calling and RAG.
    """
    if not groq_api_key or not groq_api_key.startswith("gsk_"):
        return AVAILABLE_GROQ_MODELS

    try:
        import groq

        client = groq.Groq(api_key=groq_api_key)
        models_data = client.models.list()
        # Extract model IDs
        model_ids = [m.id for m in models_data.data if m.active]
        # Prefer llama/mixtral chat models
        chat_models = [
            m for m in model_ids if any(k in m.lower() for k in ["llama", "mixtral", "gemma"])
        ]
        return chat_models if chat_models else model_ids
    except Exception:
        return AVAILABLE_GROQ_MODELS


def get_llm(
    api_key: str,
    model_name: Optional[str] = None,
    temperature: float = 0.1,
) -> BaseChatModel:
    """
    Factory function returning the appropriate Chat Model.
    - If key starts with 'xai-': Configures xAI Grok (console.x.ai).
    - If key starts with 'gsk_': Configures Groq (console.groq.com) as per resume.
    - Otherwise defaults to Groq or OpenAI.
    """
    key_clean = (api_key or "").strip()

    if key_clean.startswith("xai-"):
        # User provided an xAI (Grok) key
        actual_model = model_name if model_name and "grok" in model_name else "grok-beta"
        return ChatOpenAI(
            model=actual_model,
            api_key=key_clean,
            base_url="https://api.x.ai/v1",
            temperature=temperature,
        )
    else:
        # Default to Groq (console.groq.com)
        actual_model = model_name or DEFAULT_GROQ_MODEL
        return ChatGroq(
            model=actual_model,
            groq_api_key=key_clean,
            temperature=temperature,
        )
