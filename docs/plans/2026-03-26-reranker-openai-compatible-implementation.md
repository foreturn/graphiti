# Reranker OpenAI-Compatible Config Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add independent reranker configuration for the graph service so OpenAI-compatible reranking can target its own endpoint, key, and model without inheriting the main LLM config.

**Architecture:** Extend the server settings/runtime config with reranker-specific fields, then build a dedicated `LLMConfig` for `OpenAIRerankerClient`. Verify the new separation with runtime-config tests and graph-service assembly tests.

**Tech Stack:** Python, unittest, FastAPI settings, Graphiti OpenAI-compatible clients

---

### Task 1: Add failing reranker configuration tests

**Files:**
- Modify: `tests/server/test_graph_service_embedding_config.py`
- Test: `tests/server/test_graph_service_embedding_config.py`

**Step 1: Write the failing test**

Add tests that assert:
- `resolve_graphiti_client_settings()` returns reranker values from `reranker_*` settings only
- missing reranker values remain `None`
- `_build_graphiti_client()` injects reranker-specific config into `OpenAIRerankerClient`

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.server.test_graph_service_embedding_config`
Expected: FAIL because the runtime config and builder do not yet expose standalone reranker settings.

**Step 3: Write minimal implementation**

Update server settings, runtime config, and `zep_graphiti.py` to build a dedicated reranker config.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.server.test_graph_service_embedding_config`
Expected: PASS

### Task 2: Extend server settings and runtime config

**Files:**
- Modify: `server/graph_service/config.py`
- Modify: `server/graph_service/runtime_config.py`

**Step 1: Add settings fields**

Add:
- `reranker_api_key: str | None`
- `reranker_base_url: str | None`
- `reranker_model_name: str | None`

**Step 2: Extend resolved runtime config**

Add a `reranker` field to `GraphitiClientRuntimeConfig` and populate it directly from the reranker settings without any fallback.

### Task 3: Wire dedicated reranker config

**Files:**
- Modify: `server/graph_service/zep_graphiti.py`

**Step 1: Build reranker config**

Create:

```python
reranker_config = LLMConfig(
    api_key=runtime_config.reranker.api_key,
    base_url=runtime_config.reranker.base_url,
    model=runtime_config.reranker.model_name,
)
```

**Step 2: Inject reranker config**

Pass `OpenAIRerankerClient(config=reranker_config)` to `ZepGraphiti`.

### Task 4: Verify the change

**Files:**
- Test: `tests/server/test_graph_service_embedding_config.py`

**Step 1: Run targeted verification**

Run: `python -m unittest tests.server.test_graph_service_embedding_config`
Expected: PASS

**Step 2: Run syntax verification**

Run: `python -m py_compile server/graph_service/config.py server/graph_service/runtime_config.py server/graph_service/zep_graphiti.py tests/server/test_graph_service_embedding_config.py`
Expected: PASS
