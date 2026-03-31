# Server Local Embedding And Reranker Design

**Goal:** Make the Dockerized graph service start with local embedding and reranking models by default, while keeping the existing OpenAI-compatible LLM path intact.

**Scope:** This design applies to `server/graph_service` only. The MCP server keeps its current provider-based configuration for now.

## Architecture

The graph service will continue to use the current OpenAI-compatible LLM wiring for extraction and graph-building prompts. Embedding and reranking will gain a provider switch so Docker deployments can default to local model execution inside the same container.

Two local defaults will be used:
- Embedding: `BAAI/bge-m3`
- Reranker: `BAAI/bge-reranker-v2-m3`

When `EMBEDDING_PROVIDER=local`, the service will instantiate a new local sentence-transformers embedder instead of `OpenAIEmbedder`. When `RERANKER_PROVIDER=local`, the service will instantiate a BGE reranker client configured with `BAAI/bge-reranker-v2-m3`. The existing OpenAI-compatible paths remain available behind explicit provider selection.

## Components

### Local Embedding Client

Add a new embedder implementation under `graphiti_core/embedder/` backed by `sentence-transformers`. It will:
- load a configured local embedding model
- expose the existing `EmbedderClient` interface
- support single-text and batch embedding creation
- trim vectors to `embedding_dim` when needed

The default local model is `BAAI/bge-m3`.

### Local Reranker Client

Extend the existing BGE reranker client so the model name can be configured, while defaulting to `BAAI/bge-reranker-v2-m3`. This preserves the current `CrossEncoderClient` contract and lets the server select either local or OpenAI-compatible reranking.

### Server Provider Selection

`server/graph_service/config.py` and `runtime_config.py` will gain provider fields for embedding and reranking:
- `embedding_provider`
- `reranker_provider`

The graph-service builder in `server/graph_service/zep_graphiti.py` will route construction based on those fields:
- LLM: unchanged OpenAI-compatible path
- embedding: local sentence-transformers client or `OpenAIEmbedder`
- reranker: local BGE reranker or `OpenAIRerankerClient`

## Docker Behavior

Docker deployments should default to local embedding and reranking. The image will install the required local-model dependencies and define a model cache directory so first-run downloads are persisted across restarts.

Suggested defaults:
- `EMBEDDING_PROVIDER=local`
- `EMBEDDING_MODEL_NAME=BAAI/bge-m3`
- `RERANKER_PROVIDER=local`
- `RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3`

The cache directory should be mounted as a volume to avoid repeated model downloads.

## Error Handling

There should be no silent fallback between local and remote providers.

Expected behavior:
- if `embedding_provider=local` and local dependencies are missing, service startup fails with a clear import/config error
- if `reranker_provider=local` and model loading fails, service startup fails clearly
- if `embedding_provider=openai_compatible`, the existing API-key/base-url path is used
- if `reranker_provider=openai_compatible`, the existing reranker API-key/base-url path is used

This keeps deployment intent explicit and avoids hidden cross-provider behavior.

## Testing

The change should be covered in three layers:
- unit tests for local embedder behavior and configurable BGE reranker construction
- server assembly tests for provider-based client selection
- Docker/config tests for local-default environment wiring and cache-related configuration

## Risks

- First container startup will be slower because model weights must download
- Image size will increase because of local inference dependencies
- CPU-only deployments may have slower reranking and embedding throughput than external APIs

These tradeoffs are acceptable because the primary goal is zero-config local deployment.
