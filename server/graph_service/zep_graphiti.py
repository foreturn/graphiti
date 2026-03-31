import logging
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.cross_encoder import CrossEncoderClient  # type: ignore
from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder import EmbedderClient, OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.embedder.sentence_transformer import (
    SentenceTransformerEmbedder,
    SentenceTransformerEmbedderConfig,
)
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient, LLMConfig  # type: ignore
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore

from graph_service.config import ZepEnvDep
from graph_service.dto import FactResult
from graph_service.runtime_config import (
    DEFAULT_LOCAL_RERANKER_MODEL,
    resolve_graphiti_client_settings,
)

logger = logging.getLogger(__name__)


class ZepGraphiti(Graphiti):
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        llm_client: LLMClient | None = None,
        embedder: EmbedderClient | None = None,
        cross_encoder: CrossEncoderClient | None = None,
    ):
        super().__init__(
            uri,
            user,
            password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(self.driver)
        return new_node

    async def get_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            return edge
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_group(self, group_id: str):
        try:
            edges = await EntityEdge.get_by_group_ids(self.driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(self.driver, [group_id])

        episodes = await EpisodicNode.get_by_group_ids(self.driver, [group_id])

        for edge in edges:
            await edge.delete(self.driver)

        for node in nodes:
            await node.delete(self.driver)

        for episode in episodes:
            await episode.delete(self.driver)

    async def delete_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            await edge.delete(self.driver)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e


def _build_graphiti_client(settings: ZepEnvDep) -> ZepGraphiti:
    runtime_config = resolve_graphiti_client_settings(settings)
    llm_config = LLMConfig(
        api_key=runtime_config.llm.api_key,
        base_url=runtime_config.llm.base_url,
        model=runtime_config.llm.model_name,
    )

    if runtime_config.embedding.provider == 'local':
        embedder: EmbedderClient = SentenceTransformerEmbedder(
            config=SentenceTransformerEmbedderConfig(
                embedding_model=runtime_config.embedding.model_name
                or SentenceTransformerEmbedderConfig().embedding_model
            )
        )
    else:
        embedder_config_kwargs: dict[str, str] = {
            'api_key': runtime_config.embedding.api_key or '',
        }
        if runtime_config.embedding.base_url is not None:
            embedder_config_kwargs['base_url'] = runtime_config.embedding.base_url
        if runtime_config.embedding.model_name is not None:
            embedder_config_kwargs['embedding_model'] = runtime_config.embedding.model_name
        embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(**embedder_config_kwargs))

    if runtime_config.reranker.provider == 'local':
        cross_encoder: CrossEncoderClient = BGERerankerClient(
            model_name=runtime_config.reranker.model_name or DEFAULT_LOCAL_RERANKER_MODEL
        )
    else:
        reranker_config = LLMConfig(
            api_key=runtime_config.reranker.api_key,
            base_url=runtime_config.reranker.base_url,
            model=runtime_config.reranker.model_name,
        )
        cross_encoder = OpenAIRerankerClient(config=reranker_config)

    return ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        llm_client=OpenAIGenericClient(config=llm_config),
        embedder=embedder,
        cross_encoder=cross_encoder,
    )


async def get_graphiti(settings: ZepEnvDep):
    client = _build_graphiti_client(settings)

    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    client = _build_graphiti_client(settings)
    await client.build_indices_and_constraints()


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
