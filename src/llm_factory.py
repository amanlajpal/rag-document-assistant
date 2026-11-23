import os
from typing import Optional, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Supported default models
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
AVAILABLE_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]


DISALLOWED_MODEL_KEYWORDS = [
    "guard",
    "prompt-guard",
    "safeguard",
    "whisper",
    "embed",
    "tts",
    "stt",
    "classification",
    "moderation",
]

PRIORITY_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]


def fetch_available_groq_models(groq_api_key: str) -> List[str]:
    """
    Dynamically fetches the active models for a given Groq API key,
    strictly filtering out safety guardrails, classifiers, and speech models.
    Prioritizes top tool-calling conversational LLMs.
    """
    if not groq_api_key or not groq_api_key.startswith("gsk_"):
        return AVAILABLE_GROQ_MODELS

    try:
        import groq

        client = groq.Groq(api_key=groq_api_key)
        models_data = client.models.list()

        # Extract active model IDs
        all_active = [m.id for m in models_data.data if getattr(m, "active", True)]

        # Filter out safety guards, audio models, and classifiers
        valid_chat_models = []
        for model_id in all_active:
            m_lower = model_id.lower()
            if any(bad in m_lower for bad in DISALLOWED_MODEL_KEYWORDS):
                continue
            valid_chat_models.append(model_id)

        # Sort so that proven general chat models appear first
        sorted_models = []
        for p in PRIORITY_MODELS:
            if p in valid_chat_models:
                sorted_models.append(p)

        # Append remaining valid chat models
        for m in valid_chat_models:
            if m not in sorted_models:
                sorted_models.append(m)

        return sorted_models if sorted_models else AVAILABLE_GROQ_MODELS
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
