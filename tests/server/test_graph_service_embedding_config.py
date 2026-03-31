import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import types

SERVER_ROOT = Path(__file__).resolve().parents[2] / 'server'
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from graph_service.runtime_config import resolve_graphiti_client_settings


class ResolveGraphitiClientSettingsTests(unittest.TestCase):
    def test_resolves_separate_embedding_configuration(self) -> None:
        settings = SimpleNamespace(
            openai_api_key='llm-key',
            openai_base_url='https://llm.example/v1',
            model_name='gpt-4.1-mini',
            embedding_provider='openai_compatible',
            embedding_api_key='embed-key',
            embedding_base_url='https://embed.example/v1',
            embedding_model_name='text-embedding-3-large',
            reranker_provider='openai_compatible',
            reranker_api_key='rerank-key',
            reranker_base_url='https://rerank.example/v1',
            reranker_model_name='rerank-model',
        )

        resolved = resolve_graphiti_client_settings(settings)

        self.assertEqual(resolved.llm.api_key, 'llm-key')
        self.assertEqual(resolved.llm.base_url, 'https://llm.example/v1')
        self.assertEqual(resolved.llm.model_name, 'gpt-4.1-mini')
        self.assertEqual(resolved.embedding.provider, 'openai_compatible')
        self.assertEqual(resolved.embedding.api_key, 'embed-key')
        self.assertEqual(resolved.embedding.base_url, 'https://embed.example/v1')
        self.assertEqual(resolved.embedding.model_name, 'text-embedding-3-large')
        self.assertEqual(resolved.reranker.provider, 'openai_compatible')
        self.assertEqual(resolved.reranker.api_key, 'rerank-key')
        self.assertEqual(resolved.reranker.base_url, 'https://rerank.example/v1')
        self.assertEqual(resolved.reranker.model_name, 'rerank-model')

    def test_local_provider_defaults_do_not_require_remote_credentials(self) -> None:
        settings = SimpleNamespace(
            openai_api_key='shared-key',
            openai_base_url='https://llm.example/v1',
            model_name=None,
            embedding_provider='local',
            embedding_api_key=None,
            embedding_base_url=None,
            embedding_model_name=None,
            reranker_provider='local',
            reranker_api_key=None,
            reranker_base_url=None,
            reranker_model_name=None,
        )

        resolved = resolve_graphiti_client_settings(settings)

        self.assertEqual(resolved.embedding.provider, 'local')
        self.assertIsNone(resolved.embedding.api_key)
        self.assertIsNone(resolved.embedding.base_url)
        self.assertEqual(resolved.embedding.model_name, 'BAAI/bge-m3')
        self.assertEqual(resolved.reranker.provider, 'local')
        self.assertIsNone(resolved.reranker.api_key)
        self.assertIsNone(resolved.reranker.base_url)
        self.assertEqual(resolved.reranker.model_name, 'BAAI/bge-reranker-v2-m3')


class BuildGraphitiClientTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.modules.pop('graph_service.zep_graphiti', None)

    def tearDown(self) -> None:
        sys.modules.pop('graph_service.zep_graphiti', None)

    def test_build_graphiti_client_uses_local_embedding_and_reranker_by_default(self) -> None:
        module_map = self._build_stub_modules()

        with patch.dict(sys.modules, module_map):
            from graph_service.zep_graphiti import _build_graphiti_client

            settings = SimpleNamespace(
                openai_api_key='llm-key',
                openai_base_url='https://llm.example/v1',
                model_name='gpt-4.1-mini',
                embedding_provider='local',
                embedding_api_key=None,
                embedding_base_url=None,
                embedding_model_name='BAAI/bge-m3',
                reranker_provider='local',
                reranker_api_key=None,
                reranker_base_url=None,
                reranker_model_name='BAAI/bge-reranker-v2-m3',
                neo4j_uri='bolt://neo4j:7687',
                neo4j_user='neo4j',
                neo4j_password='secret',
            )

            client = _build_graphiti_client(settings)

        self.assertEqual(client.uri, 'bolt://neo4j:7687')
        self.assertEqual(client.user, 'neo4j')
        self.assertEqual(client.password, 'secret')
        self.assertEqual(client.llm_client.__class__.__name__, 'OpenAIGenericClient')
        self.assertEqual(client.llm_client.config.api_key, 'llm-key')
        self.assertEqual(client.llm_client.config.base_url, 'https://llm.example/v1')
        self.assertEqual(client.llm_client.config.model, 'gpt-4.1-mini')
        self.assertEqual(client.embedder.__class__.__name__, 'SentenceTransformerEmbedder')
        self.assertEqual(client.embedder.config.embedding_model, 'BAAI/bge-m3')
        self.assertEqual(client.cross_encoder.__class__.__name__, 'BGERerankerClient')
        self.assertEqual(client.cross_encoder.model_name, 'BAAI/bge-reranker-v2-m3')

    def test_build_graphiti_client_uses_openai_compatible_clients_when_requested(self) -> None:
        module_map = self._build_stub_modules()

        with patch.dict(sys.modules, module_map):
            from graph_service.zep_graphiti import _build_graphiti_client

            settings = SimpleNamespace(
                openai_api_key='llm-key',
                openai_base_url='https://llm.example/v1',
                model_name='gpt-4.1-mini',
                embedding_provider='openai_compatible',
                embedding_api_key='embed-key',
                embedding_base_url='https://embed.example/v1',
                embedding_model_name='text-embedding-3-large',
                reranker_provider='openai_compatible',
                reranker_api_key='rerank-key',
                reranker_base_url='https://rerank.example/v1',
                reranker_model_name='rerank-model',
                neo4j_uri='bolt://neo4j:7687',
                neo4j_user='neo4j',
                neo4j_password='secret',
            )

            client = _build_graphiti_client(settings)

        self.assertEqual(client.embedder.__class__.__name__, 'OpenAIEmbedder')
        self.assertEqual(client.embedder.config.api_key, 'embed-key')
        self.assertEqual(client.embedder.config.base_url, 'https://embed.example/v1')
        self.assertEqual(client.embedder.config.embedding_model, 'text-embedding-3-large')
        self.assertEqual(client.cross_encoder.__class__.__name__, 'OpenAIRerankerClient')
        self.assertEqual(client.cross_encoder.config.api_key, 'rerank-key')
        self.assertEqual(client.cross_encoder.config.base_url, 'https://rerank.example/v1')
        self.assertEqual(client.cross_encoder.config.model, 'rerank-model')

    def _build_stub_modules(self) -> dict[str, types.ModuleType]:
        fastapi_module = types.ModuleType('fastapi')

        class HTTPException(Exception):
            def __init__(self, status_code: int, detail: str):
                self.status_code = status_code
                self.detail = detail

        fastapi_module.Depends = lambda dependency: dependency
        fastapi_module.HTTPException = HTTPException

        graphiti_core_module = types.ModuleType('graphiti_core')
        graphiti_core_module.__path__ = []  # type: ignore[attr-defined]
        cross_encoder_package = types.ModuleType('graphiti_core.cross_encoder')
        bge_reranker_module = types.ModuleType('graphiti_core.cross_encoder.bge_reranker_client')
        embedder_module = types.ModuleType('graphiti_core.embedder')
        sentence_embedder_module = types.ModuleType('graphiti_core.embedder.sentence_transformer')
        edges_module = types.ModuleType('graphiti_core.edges')
        errors_module = types.ModuleType('graphiti_core.errors')
        llm_client_module = types.ModuleType('graphiti_core.llm_client')
        llm_generic_module = types.ModuleType('graphiti_core.llm_client.openai_generic_client')
        cross_encoder_module = types.ModuleType('graphiti_core.cross_encoder.openai_reranker_client')
        nodes_module = types.ModuleType('graphiti_core.nodes')
        config_module = types.ModuleType('graph_service.config')
        dto_module = types.ModuleType('graph_service.dto')

        class Graphiti:
            def __init__(
                self,
                uri,
                user,
                password,
                llm_client=None,
                embedder=None,
                cross_encoder=None,
            ):
                self.uri = uri
                self.user = user
                self.password = password
                self.llm_client = llm_client
                self.embedder = embedder
                self.cross_encoder = cross_encoder

        class EmbedderClient:
            pass

        class OpenAIEmbedderConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class OpenAIEmbedder(EmbedderClient):
            def __init__(self, config):
                self.config = config

        class SentenceTransformerEmbedderConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class SentenceTransformerEmbedder(EmbedderClient):
            def __init__(self, config):
                self.config = config

        class LLMClient:
            pass

        class LLMConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class OpenAIClient(LLMClient):
            def __init__(self, config):
                self.config = config

        class OpenAIGenericClient(LLMClient):
            def __init__(self, config):
                self.config = config

        class OpenAIRerankerClient:
            def __init__(self, config):
                self.config = config

        class BGERerankerClient:
            def __init__(self, model_name='BAAI/bge-reranker-v2-m3'):
                self.model_name = model_name

        class EntityEdge:
            pass

        class EntityNode:
            pass

        class EpisodicNode:
            pass

        class EdgeNotFoundError(Exception):
            pass

        class GroupsEdgesNotFoundError(Exception):
            pass

        class NodeNotFoundError(Exception):
            pass

        class FactResult:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        graphiti_core_module.Graphiti = Graphiti
        cross_encoder_package.CrossEncoderClient = OpenAIRerankerClient
        cross_encoder_package.OpenAIRerankerClient = OpenAIRerankerClient
        cross_encoder_package.BGERerankerClient = BGERerankerClient
        embedder_module.EmbedderClient = EmbedderClient
        embedder_module.OpenAIEmbedder = OpenAIEmbedder
        embedder_module.OpenAIEmbedderConfig = OpenAIEmbedderConfig
        embedder_module.SentenceTransformerEmbedder = SentenceTransformerEmbedder
        embedder_module.SentenceTransformerEmbedderConfig = SentenceTransformerEmbedderConfig
        sentence_embedder_module.SentenceTransformerEmbedder = SentenceTransformerEmbedder
        sentence_embedder_module.SentenceTransformerEmbedderConfig = SentenceTransformerEmbedderConfig
        edges_module.EntityEdge = EntityEdge
        errors_module.EdgeNotFoundError = EdgeNotFoundError
        errors_module.GroupsEdgesNotFoundError = GroupsEdgesNotFoundError
        errors_module.NodeNotFoundError = NodeNotFoundError
        llm_client_module.LLMClient = LLMClient
        llm_client_module.LLMConfig = LLMConfig
        llm_client_module.OpenAIClient = OpenAIClient
        llm_generic_module.OpenAIGenericClient = OpenAIGenericClient
        cross_encoder_module.OpenAIRerankerClient = OpenAIRerankerClient
        bge_reranker_module.BGERerankerClient = BGERerankerClient
        nodes_module.EntityNode = EntityNode
        nodes_module.EpisodicNode = EpisodicNode
        config_module.ZepEnvDep = object
        dto_module.FactResult = FactResult

        return {
            'fastapi': fastapi_module,
            'graphiti_core': graphiti_core_module,
            'graphiti_core.cross_encoder': cross_encoder_package,
            'graphiti_core.cross_encoder.bge_reranker_client': bge_reranker_module,
            'graphiti_core.embedder': embedder_module,
            'graphiti_core.embedder.sentence_transformer': sentence_embedder_module,
            'graphiti_core.edges': edges_module,
            'graphiti_core.errors': errors_module,
            'graphiti_core.llm_client': llm_client_module,
            'graphiti_core.llm_client.openai_generic_client': llm_generic_module,
            'graphiti_core.cross_encoder.openai_reranker_client': cross_encoder_module,
            'graphiti_core.nodes': nodes_module,
            'graph_service.config': config_module,
            'graph_service.dto': dto_module,
        }


if __name__ == '__main__':
    unittest.main()
