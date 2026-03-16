# Separate Embedding Config Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add independent embedding API configuration for the Dockerized graph service so embedding requests can use a different base URL, model name, and API key than the LLM client.

**Architecture:** Extend `graph_service` settings with embedding-specific environment variables, then construct the LLM client and embedder with explicit configs before creating `Graphiti`. This ensures both clients receive the intended runtime settings instead of mutating config objects after client construction.

**Tech Stack:** Python 3.10+, FastAPI settings, Graphiti core OpenAI clients, Docker Compose, unittest

---

### Task 1: Add a failing regression test

**Files:**
- Create: `tests/server/test_graph_service_embedding_config.py`
- Test: `tests/server/test_graph_service_embedding_config.py`

**Step 1: Write the failing test**

```python
class GetGraphitiConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_graphiti_uses_independent_embedding_settings(self):
        ...
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.server.test_graph_service_embedding_config`
Expected: FAIL because `get_graphiti` does not yet construct clients from separate embedding settings.

**Step 3: Write minimal implementation**

Add helpers that build an `LLMConfig` and `OpenAIEmbedderConfig` from settings, then inject them into `ZepGraphiti`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.server.test_graph_service_embedding_config`
Expected: PASS

### Task 2: Add embedding-specific settings

**Files:**
- Modify: `server/graph_service/config.py`

**Step 1: Add settings fields**

Add `embedding_base_url`, `embedding_model_name`, and `embedding_api_key`.

**Step 2: Keep fallback behavior simple**

Use `OPENAI_API_KEY` as the embedding API key fallback when `EMBEDDING_API_KEY` is unset.

### Task 3: Wire explicit client construction

**Files:**
- Modify: `server/graph_service/zep_graphiti.py`

**Step 1: Build LLM client config before `ZepGraphiti` creation**

Construct `OpenAIClient(LLMConfig(...))` with `OPENAI_BASE_URL` and `MODEL_NAME`.

**Step 2: Build embedder config before `ZepGraphiti` creation**

Construct `OpenAIEmbedder(OpenAIEmbedderConfig(...))` with `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL_NAME`, and `EMBEDDING_API_KEY`.

**Step 3: Reuse the same helper for initialization**

Ensure `initialize_graphiti` builds the same configured client pair.

### Task 4: Update runtime examples

**Files:**
- Modify: `docker-compose.yml`
- Modify: `.env.example`
- Modify: `server/.env.example`

**Step 1: Expose new variables**

Add `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL_NAME`, and `EMBEDDING_API_KEY` to examples.

**Step 2: Keep names aligned**

Use the same environment variable names in Docker and server examples.

### Task 5: Verify the change

**Files:**
- Test: `tests/server/test_graph_service_embedding_config.py`

**Step 1: Run targeted verification**

Run: `python -m unittest tests.server.test_graph_service_embedding_config`
Expected: PASS

**Step 2: Note any gaps**

If project-wide pytest is unavailable in the environment, report that limitation with the exact command failure.
