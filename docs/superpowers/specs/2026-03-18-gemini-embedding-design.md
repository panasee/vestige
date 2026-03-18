# Gemini Embedding Integration — Design Spec

**Date:** 2026-03-18
**Status:** Approved (v2, post-review)
**Scope:** Replace local nomic-embed-text-v1.5 embedding model with Google Gemini API

---

## Goal

Replace the local ONNX-based embedding model (`nomic-embed-text-v1.5`, 256D) with `gemini-embedding-2-preview` via Google's REST API. Motivation: higher semantic precision and larger input context (8192 tokens). No fallback — Gemini is the sole embedding backend.

---

## Architecture

### What Changes

`EmbeddingService` (currently in `src/embeddings/local.rs`) is replaced by `GeminiEmbeddingService` (`src/embeddings/gemini.rs`). The `fastembed` dependency is retained only for the Reranker (Jina Reranker v1 Turbo). No `EmbeddingProvider` trait abstraction is needed — single backend, direct replacement.

### New Feature Flag

```toml
# Cargo.toml
gemini-embeddings = ["dep:reqwest", "dep:toml"]
```

The `embeddings` feature continues to gate `fastembed` for the reranker. `gemini-embeddings` is a separate, additive flag. The default feature set will be updated to include `gemini-embeddings`.

`reqwest` is added with `blocking` + `json` + `rustls-tls` features. **Async `reqwest` is NOT used** (see Sync/Async section below).

---

## Configuration

File: `~/.vestige/config.toml`

```toml
[embeddings]
provider = "gemini"
api_key = "AIza..."
model = "gemini-embedding-2-preview"
output_dimensions = 1536   # optional, default 1536, max 3072
```

Config is loaded once at startup. If the file is missing or `api_key` is absent, the server starts with semantic search **disabled** (graceful degradation — FTS5/BM25 keyword search still works), logging a clear warning. This is consistent with how the current `embedding_service.is_ready()` pattern handles an unavailable ONNX model.

`output_dimensions` defaults to **1536**. Changing this value after embeddings have been generated requires re-running the full migration; this is unsupported without a manual reset. Valid range: 1–3072. Validation at config load time.

---

## Sync/Async Design

**`Storage` is entirely synchronous** and holds `Mutex<Connection>` guards. Calling async `reqwest` from sync code via `block_on` panics inside a Tokio runtime.

Resolution: use **`reqwest::blocking`** for all inline embedding calls within the synchronous storage layer (`generate_embedding_for_node`, `generate_missing_embeddings`). The background migration task, which runs inside a Tokio-spawned task, also uses `reqwest::blocking` via `spawn_blocking` to avoid blocking the async executor.

```rust
// Blocking client — used from sync Storage methods
pub struct GeminiEmbeddingService {
    api_key: String,
    model: String,
    output_dimensions: usize,
    client: reqwest::blocking::Client,
}
```

The Gemini REST endpoint:

```
POST https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents
x-goog-api-key: {api_key}
```

Batch chunk size: **32** (well within the API's 100-request-per-batch limit; intentionally conservative).

---

## Configuration File

File Changes:

| File | Change |
|------|--------|
| `crates/vestige-core/Cargo.toml` | Add `gemini-embeddings` feature; add `reqwest` (blocking, json, rustls-tls), `toml` as optional deps |
| `crates/vestige-core/src/embeddings/local.rs` | Delete (replaced) |
| `crates/vestige-core/src/embeddings/gemini.rs` | New — `GeminiEmbeddingService` with blocking reqwest |
| `crates/vestige-core/src/embeddings/config.rs` | New — `VestigeConfig` / `EmbeddingConfig`, loaded from `~/.vestige/config.toml` |
| `crates/vestige-core/src/embeddings/mod.rs` | Update exports; remove local, add gemini + config |
| `crates/vestige-core/src/search/vector.rs` | Make `VectorIndex` dimension configurable (remove hardcoded 256) |
| `crates/vestige-core/src/storage/sqlite.rs` | Plumb configured dimensions through `Storage::new()` and `VectorIndex::new()`; update `generate_embedding_for_node`, `load_embeddings_into_index` |
| `crates/vestige-core/src/storage/migrations.rs` | New migration V10 (see below) |
| `crates/vestige-core/src/lib.rs` | Update `DEFAULT_EMBEDDING_MODEL` constant |

---

## GeminiEmbeddingService Interface

```rust
pub struct GeminiEmbeddingService {
    api_key: String,
    model: String,
    output_dimensions: usize,  // default: 1536
    client: reqwest::blocking::Client,
}

impl GeminiEmbeddingService {
    pub fn from_config(config: &EmbeddingConfig) -> Result<Self, EmbeddingError>;
    pub fn embed(&self, text: &str) -> Result<Embedding, EmbeddingError>;
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbeddingError>;
    pub fn is_available(&self) -> bool;  // true if config loaded successfully
    pub fn dimensions(&self) -> usize;
    pub fn model_name(&self) -> &str;
}
```

---

## USearch VectorIndex — Dimension Change

`VectorIndex` currently hardcodes `DEFAULT_DIMENSIONS: usize = 256` (vector.rs line 22) and `VectorIndexConfig::default()` uses it (line 91). This must be made configurable:

```rust
impl VectorIndexConfig {
    pub fn with_dimensions(dimensions: usize) -> Self { ... }
}
```

### Threading Config into `Storage::new()`

`Storage::new()` current signature: `pub fn new(db_path: Option<PathBuf>) -> Result<Self>`.

**Chosen approach:** Add an `embedding_config: Option<EmbeddingConfig>` parameter:

```rust
pub fn new(db_path: Option<PathBuf>, embedding_config: Option<EmbeddingConfig>) -> Result<Self>
```

The MCP server loads `EmbeddingConfig` from `~/.vestige/config.toml` at startup (before calling `Storage::new()`) and passes it in. If config is absent, `None` is passed and semantic search is disabled.

**Affected callsites** that must be updated to pass the new parameter:
- `crates/vestige-mcp/src/main.rs` — production server init (where `McpServer` / `Storage::new` is called)
- `crates/vestige-mcp/src/server.rs` — `test_storage()` helper (line 1041)
- `crates/vestige-core/src/lib.rs` — any `Storage::new()` used in integration helpers
- `tests/e2e/src/mocks/fixtures.rs` (line 514) — pass `None`
- `tests/e2e/src/harness/db_manager.rs` (lines 71, 85, 276, 325) — pass `None`

All test callsites pass `None`, disabling Gemini in tests (uses FTS5-only search).

`VectorIndex` is then constructed with the configured dimension:
```rust
let dim = embedding_config.as_ref().map(|c| c.output_dimensions).unwrap_or(256);
let vector_index = VectorIndex::with_config(VectorIndexConfig::with_dimensions(dim))
    .map_err(|e| StorageError::Init(...))?;
```

### `load_embeddings_into_index` — Table Change

Currently queries `node_embeddings.embedding` (sqlite.rs line 215). After V10 migration, Gemini vectors live in `knowledge_nodes.embedding_v2`. The query must change:

```sql
-- Old (nomic, from node_embeddings table)
SELECT node_id, embedding FROM node_embeddings

-- New (gemini, from knowledge_nodes table)
SELECT id, embedding_v2 FROM knowledge_nodes
WHERE embedding_v2 IS NOT NULL
  AND embedding_model = 'gemini-embedding-2-preview'
```

`get_all_embeddings` (sqlite.rs line 541) which also reads `node_embeddings.embedding` must be updated to the same new query. The `node_embeddings` table is **no longer written to** once `gemini-embeddings` feature is active. It is preserved only for the V11 cleanup migration.

---

## Database Migration (V10)

The existing V1 migration already created `embedding_model TEXT` on `knowledge_nodes` (migrations.rs line 101), written as `'nomic-embed-text-v1.5'` by `generate_embedding_for_node`. **No new `embedding_model` column is added** — the existing one is reused.

Migration V10 adds two columns:

```sql
-- New column for 1536D Gemini vectors (6144 bytes per vector)
ALTER TABLE knowledge_nodes ADD COLUMN embedding_v2 BLOB DEFAULT NULL;

-- Retry counter to skip permanently failing entries (avoids quota burn)
ALTER TABLE knowledge_nodes ADD COLUMN gemini_retry_count INTEGER DEFAULT 0;
```

After this migration:
- Old `embedding` column on `node_embeddings` (256D nomic vectors) remains; it is **not dropped** in V10.
  - SQLite `DROP COLUMN` requires ≥ 3.35.0 (bundled SQLite version must be verified). Dropping is deferred to a future migration (V11) after full migration is confirmed.
- `embedding_model` is used as the migration status indicator:
  - `'nomic-embed-text-v1.5'` or `NULL` → not yet migrated
  - `'gemini-embedding-2-preview'` → migrated

---

## Automatic Migration Flow

The background migration task and all Gemini-specific code paths are gated `#[cfg(feature = "gemini-embeddings")]`.

On startup, after migrations run, check count of memories where `embedding_model != 'gemini-embedding-2-preview'`:

```sql
SELECT COUNT(*) FROM knowledge_nodes
WHERE embedding_model IS NULL OR embedding_model != 'gemini-embedding-2-preview';
```

If count > 0, spawn a Tokio background task:

```
tokio::task::spawn_blocking(|| {
    loop {
        fetch batch of 32 unmigrated memories (by content)
        call embed_batch() via reqwest::blocking
        write embedding_v2, set embedding_model = 'gemini-embedding-2-preview'
        if no more → break
    }
})
```

**Live write path cutover**: As soon as `GeminiEmbeddingService::is_available()` returns true (config loaded, API key present), `generate_embedding_for_node` writes to `embedding_v2` and sets `embedding_model = 'gemini-embedding-2-preview'`. New memories are never written with nomic vectors.

**Search during migration**: `load_embeddings_into_index` and `get_all_embeddings` both load only rows where `embedding_v2 IS NOT NULL AND embedding_model = 'gemini-embedding-2-preview'`. Unmigrated memories are accessible via FTS5 keyword search only until migrated. This is intentional — returning mixed-model vectors to the HNSW index would produce meaningless similarity scores. `auto_dedup_consolidation`, which relies on `get_all_embeddings`, will only deduplicate Gemini-migrated memories during the migration window.

**Failure handling**: API errors are logged per-memory. The `gemini_retry_count` column (added in V10 migration) tracks failed attempts; memories with `gemini_retry_count >= 3` are skipped to avoid burning quota on permanently broken entries. On each failure the count is incremented immediately in the same DB write.

---

## Error Handling

- Missing config / missing `api_key`: server starts with semantic search disabled, logs warning
- Missing config / missing `api_key`: `EmbeddingError::ConfigError(String)` (new variant — `ModelInit` is not reused as it implies a local model)
- API call failure (network, quota, 4xx/5xx): `EmbeddingError::ApiError(String)` propagated to caller
- Empty text input: `EmbeddingError::InvalidInput`
- `output_dimensions` out of range (> 3072 or < 1): startup error at config load
- `output_dimensions` changed after embeddings generated: logged warning; semantic search results will be wrong until re-migration (unsupported path, user must reset manually)

---

## Testing

- Unit: config parsing (valid, missing key, malformed TOML, out-of-range dimensions)
- Unit: `Embedding` bytes round-trip and cosine similarity (unchanged)
- Unit: `VectorIndex` with configurable dimensions
- Integration: mock Gemini API via `wiremock` — verify `batchEmbedContents` request format, chunk size, `output_dimensionality` field
- Integration: migration test — populate DB with nomic-style rows, run V10 migration, run background migration, verify `embedding_model = 'gemini-embedding-2-preview'` and `embedding_v2` non-null
- Integration: graceful degradation — start server with no config file, verify FTS5 search still works, semantic search returns empty with warning
- Existing reranker tests: unchanged (fastembed/Jina path not affected)
- `wiremock` added to `[dev-dependencies]` in `crates/vestige-core/Cargo.toml`

---

## Out of Scope

- Replacing the Jina Reranker with a Gemini reranker
- Supporting multiple embedding providers simultaneously
- Encrypting the API key in config
- V11 migration (drop old `embedding` column) — separate PR after full migration confirmed
