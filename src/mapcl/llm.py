"""LangChain chains for the two optional LLM steps: label expansion and candidate assessment."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

SYSTEM = "You are an expert in cell biology and the Cell Ontology (CL)."


class Expansion(BaseModel):
    """Standard cell type name for a short or ambiguous label."""

    cell_type: str = Field(description="Most likely standard Cell Ontology cell type name")
    synonyms: list[str] = Field(description="Up to 3 common synonyms or abbreviations; may be empty")


class Assessment(BaseModel):
    """Choice of the best candidate term for a user label."""

    best_index: int = Field(description="1-based index of the best candidate")
    score: int = Field(description="Similarity of the best candidate on the 1-10 scale")
    reason: str = Field(description="One short sentence justifying the choice")


EXPAND_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human",
     "Interpret this cell type label from a single-cell study and give the standard "
     "Cell Ontology cell type name it most likely refers to. Expand abbreviations.\n\n"
     "Label: {label}\n{context}"),
])

ASSESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human",
     "Score how well the best candidate term matches the user's cell type label.\n"
     "Scale:\n"
     "10: exact match\n"
     "8-9: synonym or minor naming variant of the same cell type\n"
     "6-7: closely related (direct parent or child term with the same function)\n"
     "4-5: loosely related (same broad category but different function or tissue)\n"
     "1-3: unrelated\n"
     "Judge by cell identity and function, not wording. Use the context when given. "
     "If two candidates fit equally, prefer the one whose function matches the label best.\n"
     "A candidate that is not a cell type (organelle, tissue, drug, gene, disease) scores 1.\n"
     "{hint}\n\n"
     "User label: {label}\n{context}\n\nCandidates:\n{candidates}"),
])


def format_context(**fields: str | None) -> str:
    lines = [f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in fields.items() if v]
    return "Context:\n" + "\n".join(lines) if lines else "Context: none given"


def format_candidates(candidates: Sequence[Any]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        syn = "; ".join(c.synonyms[:6]) or "none"
        lines.append(f"{i}. {c.label} [{c.id}] synonyms: {syn}. definition: {c.definition or 'none'}")
    return "\n".join(lines)


# DeepSeek's API name "deepseek-flash" is served by DeepSeek V4.1 Flash.
DEFAULT_MODEL = "deepseek:deepseek-flash"
DEFAULT_REASONING_EFFORT: str | None = None


def make_chat_model(
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a chat model with ``init_chat_model``.

    ``model`` is "provider:model", e.g. the default "deepseek:deepseek-flash"
    (DeepSeek V4.1 Flash), "openai:gpt-6-luna" or "anthropic:claude-sonnet-5"
    (install the matching langchain-<provider> package). Without ``api_key`` and
    ``base_url`` the provider's environment variables are used (DEEPSEEK_API_KEY
    and DEEPSEEK_API_BASE for DeepSeek). ``reasoning_effort`` only applies to
    OpenAI reasoning models.
    """
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if reasoning_effort and model.startswith("openai:"):
        kwargs.setdefault("reasoning_effort", reasoning_effort)
        if base_url or os.getenv("OPENAI_BASE_URL"):
            # OpenAI-compatible proxies usually serve Chat Completions only.
            kwargs.setdefault("use_responses_api", False)
    else:
        kwargs.setdefault("temperature", 0)
    kwargs.setdefault("max_retries", 1)
    kwargs.setdefault("timeout", 90)
    return init_chat_model(model, **kwargs)


def _structured(llm: BaseChatModel, schema: type[BaseModel]) -> Runnable:
    # Tool calling works on OpenAI, Anthropic and OpenAI-compatible proxies;
    # langchain-openai's default (json_schema) is often ignored by proxies.
    try:
        return llm.with_structured_output(schema, method="function_calling")
    except TypeError:  # models without a ``method`` argument
        return llm.with_structured_output(schema)


def expansion_chain(llm: BaseChatModel) -> Runnable:
    """Inputs: label, context. Output: :class:`Expansion`."""
    return EXPAND_PROMPT | _structured(llm, Expansion)


def assessment_chain(llm: BaseChatModel) -> Runnable:
    """Inputs: label, context, hint, candidates. Output: :class:`Assessment`."""
    return ASSESS_PROMPT | _structured(llm, Assessment)
