"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import MagicMock

MODULE_PATH = Path(__file__).resolve().parents[2] / 'graphiti_core' / 'embedder' / 'sentence_transformer.py'
FULL_MODULE_NAME = 'graphiti_core.embedder.sentence_transformer'

graphiti_core_module = types.ModuleType('graphiti_core')
graphiti_core_module.__path__ = []  # type: ignore[attr-defined]
embedder_package = types.ModuleType('graphiti_core.embedder')
embedder_package.__path__ = []  # type: ignore[attr-defined]
client_module = types.ModuleType('graphiti_core.embedder.client')
pydantic_module = types.ModuleType('pydantic')
sentence_transformers_module = types.ModuleType('sentence_transformers')


class EmbedderConfig:
    def __init__(self, embedding_dim: int = 1024, **kwargs):
        self.embedding_dim = embedding_dim
        for key, value in kwargs.items():
            setattr(self, key, value)


class EmbedderClient:
    pass


class SentenceTransformer:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, _input):
        return []


client_module.EmbedderClient = EmbedderClient
client_module.EmbedderConfig = EmbedderConfig
pydantic_module.Field = lambda default=None, **kwargs: default
sentence_transformers_module.SentenceTransformer = SentenceTransformer

sys.modules['graphiti_core'] = graphiti_core_module
sys.modules['graphiti_core.embedder'] = embedder_package
sys.modules['graphiti_core.embedder.client'] = client_module
sys.modules['pydantic'] = pydantic_module
sys.modules['sentence_transformers'] = sentence_transformers_module

SPEC = importlib.util.spec_from_file_location(FULL_MODULE_NAME, MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f'Unable to load module from {MODULE_PATH}')
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[FULL_MODULE_NAME] = MODULE
SPEC.loader.exec_module(MODULE)

SentenceTransformerEmbedder = MODULE.SentenceTransformerEmbedder
SentenceTransformerEmbedderConfig = MODULE.SentenceTransformerEmbedderConfig


class SentenceTransformerEmbedderTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_returns_trimmed_embedding(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3, 0.4]
        embedder = SentenceTransformerEmbedder(
            config=SentenceTransformerEmbedderConfig(
                embedding_model='BAAI/bge-m3',
                embedding_dim=3,
            ),
            model=mock_model,
        )

        result = await embedder.create('hello world')

        mock_model.encode.assert_called_once_with('hello world')
        self.assertEqual(result, [0.1, 0.2, 0.3])

    async def test_create_batch_returns_trimmed_embeddings(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ]
        embedder = SentenceTransformerEmbedder(
            config=SentenceTransformerEmbedderConfig(
                embedding_model='BAAI/bge-m3',
                embedding_dim=3,
            ),
            model=mock_model,
        )

        result = await embedder.create_batch(['a', 'b'])

        mock_model.encode.assert_called_once_with(['a', 'b'])
        self.assertEqual(
            result,
            [
                [0.5, 0.6, 0.7],
                [0.9, 1.0, 1.1],
            ],
        )


if __name__ == '__main__':
    unittest.main()
