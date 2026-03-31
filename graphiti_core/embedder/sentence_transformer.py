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

import asyncio
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from pydantic import Field

from .client import EmbedderClient, EmbedderConfig

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            'sentence-transformers is required for SentenceTransformerEmbedder. '
            'Install it with: pip install graphiti-core[sentence-transformers]'
        ) from None

DEFAULT_EMBEDDING_MODEL = 'BAAI/bge-m3'


class SentenceTransformerEmbedderConfig(EmbedderConfig):
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)


class SentenceTransformerEmbedder(EmbedderClient):
    def __init__(
        self,
        config: SentenceTransformerEmbedderConfig | None = None,
        model: 'SentenceTransformer | None' = None,
    ):
        if config is None:
            config = SentenceTransformerEmbedderConfig()

        self.config = config
        self.model = model or SentenceTransformer(config.embedding_model)

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        text = input_data if isinstance(input_data, str) else str(input_data)
        embedding = await self._encode(text)
        return list(embedding)[: self.config.embedding_dim]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        if not input_data_list:
            return []

        embeddings = await self._encode(input_data_list)
        return [list(embedding)[: self.config.embedding_dim] for embedding in embeddings]

    async def _encode(self, input_data: str | list[str]) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.model.encode, input_data)
