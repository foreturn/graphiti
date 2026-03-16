import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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
            embedding_api_key='embed-key',
            embedding_base_url='https://embed.example/v1',
            embedding_model_name='text-embedding-3-large',
        )

        resolved = resolve_graphiti_client_settings(settings)

        self.assertEqual(resolved.llm.api_key, 'llm-key')
        self.assertEqual(resolved.llm.base_url, 'https://llm.example/v1')
        self.assertEqual(resolved.llm.model_name, 'gpt-4.1-mini')
        self.assertEqual(resolved.embedding.api_key, 'embed-key')
        self.assertEqual(resolved.embedding.base_url, 'https://embed.example/v1')
        self.assertEqual(resolved.embedding.model_name, 'text-embedding-3-large')

    def test_embedding_api_key_falls_back_to_openai_api_key(self) -> None:
        settings = SimpleNamespace(
            openai_api_key='shared-key',
            openai_base_url='https://llm.example/v1',
            model_name=None,
            embedding_api_key=None,
            embedding_base_url='https://embed.example/v1',
            embedding_model_name=None,
        )

        resolved = resolve_graphiti_client_settings(settings)

        self.assertEqual(resolved.embedding.api_key, 'shared-key')
        self.assertEqual(resolved.embedding.base_url, 'https://embed.example/v1')
        self.assertIsNone(resolved.embedding.model_name)


if __name__ == '__main__':
    unittest.main()
