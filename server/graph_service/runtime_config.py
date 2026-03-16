from dataclasses import dataclass
from typing import Protocol


class GraphServiceSettings(Protocol):
    openai_api_key: str
    openai_base_url: str | None
    model_name: str | None
    embedding_api_key: str | None
    embedding_base_url: str | None
    embedding_model_name: str | None


@dataclass(frozen=True)
class LLMRuntimeConfig:
    api_key: str
    base_url: str | None
    model_name: str | None


@dataclass(frozen=True)
class EmbeddingRuntimeConfig:
    api_key: str
    base_url: str | None
    model_name: str | None


@dataclass(frozen=True)
class GraphitiClientRuntimeConfig:
    llm: LLMRuntimeConfig
    embedding: EmbeddingRuntimeConfig


def resolve_graphiti_client_settings(
    settings: GraphServiceSettings,
) -> GraphitiClientRuntimeConfig:
    embedding_api_key = settings.embedding_api_key or settings.openai_api_key

    return GraphitiClientRuntimeConfig(
        llm=LLMRuntimeConfig(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model_name=settings.model_name,
        ),
        embedding=EmbeddingRuntimeConfig(
            api_key=embedding_api_key,
            base_url=settings.embedding_base_url,
            model_name=settings.embedding_model_name,
        ),
    )
