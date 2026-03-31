# Server OpenAI-Compatible Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `server/graph_service` explicitly support OpenAI-compatible LLM, embedding, and reranker configuration instead of relying on Graphiti defaults.

**Architecture:** Extend the server runtime settings so the graph service can build a complete OpenAI-compatible client set from environment variables. Use TDD to add assembly-level tests first, then wire `OpenAIGenericClient`, `OpenAIEmbedder`, and `OpenAIRerankerClient` with shared or provider-specific config.

**Tech Stack:** Python, unittest, FastAPI settings, Graphiti core OpenAI-compatible clients

---

### Task 1: Add failing server assembly tests

**Files:**
- Modify: `tests/server/test_graph_service_embedding_config.py`
- Test: `tests/server/test_graph_service_embedding_config.py`

**Step 1: Write the failing test**

Add tests that import `graph_service.zep_graphiti` with patched dependencies and assert `_build_graphiti_client()` constructs:
- `OpenAIGenericClient` for the main LLM
- `OpenAIEmbedder` with embedding-specific `base_url`, `api_key`, and `embedding_model`
- `OpenAIRerankerClient` with the OpenAI-compatible LLM config

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.server.test_graph_service_embedding_config`
Expected: FAIL because the server does not yet fully wire OpenAI-compatible imports/configuration.

**Step 3: Write minimal implementation**

Update the server runtime config and builder so the tested client construction matches the expected OpenAI-compatible setup.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.server.test_graph_service_embedding_config`
Expected: PASS

### Task 2: Extend server runtime configuration

**Files:**
- Modify: `server/graph_service/config.py`
- Modify: `server/graph_service/runtime_config.py`

**Step 1: Add settings fields**

Add `small_model_name: str | None` to the server settings protocol and runtime config models.

**Step 2: Preserve fallback behavior**

Keep `embedding_api_key` falling back to `openai_api_key` when unset.

### Task 3: Wire explicit OpenAI-compatible clients

**Files:**
- Modify: `server/graph_service/zep_graphiti.py`

**Step 1: Fix imports**

Import `OpenAIGenericClient` from `graphiti_core.llm_client.openai_generic_client` and `OpenAIRerankerClient` from `graphiti_core.cross_encoder.openai_reranker_client`.

**Step 2: Build shared LLM config**

Construct `LLMConfig` with `api_key`, `base_url`, `model`, and `small_model`.

**Step 3: Pass explicit clients to `ZepGraphiti`**

Instantiate:
- `OpenAIGenericClient(config=llm_config)`
- `OpenAIEmbedder(config=OpenAIEmbedderConfig(...))`
- `OpenAIRerankerClient(config=llm_config)`

### Task 4: Verify the change

**Files:**
- Test: `tests/server/test_graph_service_embedding_config.py`

**Step 1: Run targeted verification**

Run: `python -m unittest tests.server.test_graph_service_embedding_config`
Expected: PASS

**Step 2: Note remaining gaps**

Document any limitations found during verification, especially if optional server dependencies are unavailable in the environment.
