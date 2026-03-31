# Server Local Embedding And Reranker Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add local Docker-default embedding and reranking to `server/graph_service` using `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3`, while preserving the current OpenAI-compatible LLM path.

**Architecture:** Add a new local sentence-transformers embedder, make the existing BGE reranker configurable, then route graph-service client construction through explicit embedding and reranker providers. Update Docker and env examples so local models are the default deployment behavior and model downloads persist through a cache volume.

**Tech Stack:** Python, sentence-transformers, unittest, Docker Compose, Graphiti server configuration

---

### Task 1: Add failing tests for local provider selection

**Files:**
- Modify: `tests/server/test_graph_service_embedding_config.py`
- Create: `tests/embedder/test_sentence_transformer.py`
- Test: `tests/server/test_graph_service_embedding_config.py`
- Test: `tests/embedder/test_sentence_transformer.py`

**Step 1: Write the failing tests**

Add tests that assert:
- `embedding_provider=local` selects the new local embedder
- `reranker_provider=local` selects the configurable BGE reranker
- model names default to `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3`

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.server.test_graph_service_embedding_config`
Expected: FAIL because provider-based local selection does not exist yet.

**Step 3: Add embedder unit tests**

Write tests for local embedding creation and batching using mocked sentence-transformers model calls.

**Step 4: Run the embedder test to verify it fails**

Run: `python -m pytest tests/embedder/test_sentence_transformer.py`
Expected: FAIL because the local embedder class does not exist yet.

### Task 2: Implement the local embedding client

**Files:**
- Create: `graphiti_core/embedder/sentence_transformer.py`
- Modify: `graphiti_core/embedder/__init__.py`

**Step 1: Add config and client classes**

Implement:
- `SentenceTransformerEmbedderConfig`
- `SentenceTransformerEmbedder`

The client should:
- implement `EmbedderClient`
- load the sentence-transformers model lazily in `__init__`
- support `create()` and `create_batch()`
- honor `embedding_dim`

**Step 2: Export the client**

Add the new classes to `graphiti_core.embedder.__all__`.

**Step 3: Run embedder tests**

Run: `python -m pytest tests/embedder/test_sentence_transformer.py`
Expected: PASS

### Task 3: Make the BGE reranker configurable

**Files:**
- Modify: `graphiti_core/cross_encoder/bge_reranker_client.py`

**Step 1: Add model parameter**

Change the constructor to accept a model name, defaulting to `BAAI/bge-reranker-v2-m3`.

**Step 2: Keep behavior unchanged otherwise**

Preserve the same `rank()` contract and execution model.

**Step 3: Add or extend tests if needed**

If no reranker test exists, add a focused constructor-level test.

### Task 4: Add provider-based server routing

**Files:**
- Modify: `server/graph_service/config.py`
- Modify: `server/graph_service/runtime_config.py`
- Modify: `server/graph_service/zep_graphiti.py`
- Modify: `tests/server/test_graph_service_embedding_config.py`

**Step 1: Add provider fields**

Add:
- `embedding_provider`
- `reranker_provider`

**Step 2: Add local-model defaults**

Set defaults:
- `EMBEDDING_MODEL_NAME=BAAI/bge-m3`
- `RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3`

**Step 3: Route by provider**

In `_build_graphiti_client()`:
- if `embedding_provider == 'local'`, build the local sentence-transformers embedder
- else build the existing `OpenAIEmbedder`
- if `reranker_provider == 'local'`, build `BGERerankerClient`
- else build `OpenAIRerankerClient`

**Step 4: Run the server tests**

Run: `python -m unittest tests.server.test_graph_service_embedding_config`
Expected: PASS

### Task 5: Update Docker defaults

**Files:**
- Modify: `server/.env.example`
- Locate and modify the graph-service Dockerfile or compose assets used to build the server image
- Modify any graph-service Docker documentation if present

**Step 1: Add local provider defaults**

Document:
- `EMBEDDING_PROVIDER=local`
- `EMBEDDING_MODEL_NAME=BAAI/bge-m3`
- `RERANKER_PROVIDER=local`
- `RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3`

**Step 2: Install runtime dependencies**

Ensure the Docker image includes `sentence-transformers` and related local-model dependencies.

**Step 3: Add cache volume guidance**

Document or wire a cache path such as `HF_HOME` so model weights persist across restarts.

### Task 6: Verify the full change

**Files:**
- Test: `tests/server/test_graph_service_embedding_config.py`
- Test: `tests/embedder/test_sentence_transformer.py`

**Step 1: Run targeted verification**

Run: `python -m unittest tests.server.test_graph_service_embedding_config`
Expected: PASS

Run: `python -m pytest tests/embedder/test_sentence_transformer.py`
Expected: PASS

**Step 2: Run syntax verification**

Run: `python -m py_compile server/graph_service/config.py server/graph_service/runtime_config.py server/graph_service/zep_graphiti.py graphiti_core/embedder/sentence_transformer.py`
Expected: PASS

**Step 3: Note Docker follow-up**

If the exact graph-service Docker build files are not present in this workspace, report that gap and specify which files still need to be updated.
