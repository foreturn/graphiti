from dataclasses import dataclass
from typing import Protocol

DEFAULT_LOCAL_EMBEDDING_MODEL = 'BAAI/bge-m3'
DEFAULT_LOCAL_RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'


class GraphServiceSettings(Protocol):
    openai_api_key: str
    openai_base_url: str | None
    model_name: str | None
    embedding_provider: str
    embedding_api_key: str | None
    embedding_base_url: str | None
    embedding_model_name: str | None
    reranker_provider: str
    reranker_api_key: str | None
    reranker_base_url: str | None
    reranker_model_name: str | None


@dataclass(frozen=True)
class LLMRuntimeConfig:
    api_key: str | None
    base_url: str | None
    model_name: str | None


@dataclass(frozen=True)
class EmbeddingRuntimeConfig:
    provider: str
    api_key: str | None
    base_url: str | None
    model_name: str | None


@dataclass(frozen=True)
class RerankerRuntimeConfig:
    provider: str
    api_key: str | None
    base_url: str | None
    model_name: str | None


@dataclass(frozen=True)
class GraphitiClientRuntimeConfig:
    llm: LLMRuntimeConfig
    embedding: EmbeddingRuntimeConfig
    reranker: RerankerRuntimeConfig


def resolve_graphiti_client_settings(
    settings: GraphServiceSettings,
) -> GraphitiClientRuntimeConfig:
    embedding_provider = settings.embedding_provider
    reranker_provider = settings.reranker_provider

    return GraphitiClientRuntimeConfig(
        llm=LLMRuntimeConfig(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model_name=settings.model_name,
        ),
        embedding=EmbeddingRuntimeConfig(
            provider=embedding_provider,
            api_key=(
                settings.embedding_api_key or settings.openai_api_key
                if embedding_provider == 'openai_compatible'
                else None
            ),
            base_url=settings.embedding_base_url if embedding_provider == 'openai_compatible' else None,
            model_name=settings.embedding_model_name or DEFAULT_LOCAL_EMBEDDING_MODEL,
        ),
        reranker=RerankerRuntimeConfig(
            provider=reranker_provider,
            api_key=settings.reranker_api_key if reranker_provider == 'openai_compatible' else None,
            base_url=(
                settings.reranker_base_url if reranker_provider == 'openai_compatible' else None
            ),
            model_name=settings.reranker_model_name or DEFAULT_LOCAL_RERANKER_MODEL,
        ),
    )
