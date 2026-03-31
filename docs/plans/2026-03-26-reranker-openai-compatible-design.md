# Reranker OpenAI-Compatible Config Design

**Goal:** Let `server/graph_service` configure the reranker with its own API key, base URL, and model name, without falling back to the primary LLM settings.

**Context:** The graph service already separates LLM and embedding configuration. The reranker currently reuses the LLM config, which makes it impossible to target a distinct OpenAI-compatible endpoint for reranking.

**Design:**
- Add `reranker_api_key`, `reranker_base_url`, and `reranker_model_name` to the server settings model and runtime settings protocol.
- Extend `GraphitiClientRuntimeConfig` with a dedicated `reranker` field shaped like `LLMRuntimeConfig`.
- Build `reranker_config = LLMConfig(...)` from only the reranker settings in `server/graph_service/zep_graphiti.py`.
- Keep the reranker config independent: if reranker values are unset, they remain `None` and are not copied from the main LLM config.

**Testing:**
- Extend `tests/server/test_graph_service_embedding_config.py` to verify reranker runtime config stays independent from the LLM values.
- Extend the `_build_graphiti_client()` assembly test to assert `OpenAIRerankerClient` receives the reranker-specific `api_key`, `base_url`, and `model`.
