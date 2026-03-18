# Gemini Embedding Integration — Design Spec

**Date:** 2026-03-18
**Status:** Approved
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

The `embeddings` feature continues to gate `fastembed` for the reranker. `gemini-embeddings` is a separate, additive flag. The default feature set will be updated to include `gemini-embeddings` and remove the nomic model dependency.

---

## Configuration

File: `~/.vestige/config.toml`

```toml
[embeddings]
provider = "gemini"
api_key = "AIza..."
model = "gemini-embedding-2-preview"
output_dimensions = 1536   # optional, default 1536
```

Config is loaded once at startup via a `VestigeConfig` struct. Missing or malformed config produces a clear error at startup rather than a runtime panic.

`output_dimensions` defaults to **1536**. This is half of the model's maximum (3072D) and offers strong quality at moderate storage cost — significantly better than the previous 256D nomic vectors.

---

## File Changes

| File | Change |
|------|--------|
| `crates/vestige-core/Cargo.toml` | Add `gemini-embeddings` feature; add `reqwest` (async, json, rustls-tls), `toml` as optional deps |
| `crates/vestige-core/src/embeddings/local.rs` | Delete (replaced) |
| `crates/vestige-core/src/embeddings/gemini.rs` | New — `GeminiEmbeddingService` implementation |
| `crates/vestige-core/src/embeddings/config.rs` | New — config loading from `~/.vestige/config.toml` |
| `crates/vestige-core/src/embeddings/mod.rs` | Update exports; remove local, add gemini + config |
| `crates/vestige-core/src/storage/migrations.rs` | New migration: drop old embedding blob, add new column sized for 1536D floats, add `embedding_model` TEXT column |
| `crates/vestige-core/src/storage/sqlite.rs` | Update embedding read/write to use new dimensions; trigger migration on open |

---

## GeminiEmbeddingService Interface

```rust
pub struct GeminiEmbeddingService {
    api_key: String,
    model: String,
    output_dimensions: usize,  // default: 1536
    client: reqwest::Client,
}

impl GeminiEmbeddingService {
    pub fn from_config(config: &EmbeddingConfig) -> Result<Self, EmbeddingError>;
    pub async fn embed(&self, text: &str) -> Result<Embedding, EmbeddingError>;
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbeddingError>;
    pub fn dimensions(&self) -> usize;
    pub fn model_name(&self) -> &str;
}
```

Batch calls use a chunk size of 32 (same as current local implementation). The Gemini REST endpoint used:

```
POST https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent
Authorization: Bearer {api_key}
```

Batch via `batchEmbedContents` endpoint for efficiency.

---

## Database Migration

On startup, `Storage::open()` runs migrations in order. The new migration (version N+1):

1. Add column `embedding_model TEXT DEFAULT NULL` to the memories table
2. Add column `embedding_v2 BLOB DEFAULT NULL` for the new 1536D vectors (6144 bytes)
3. Keep old `embedding` column temporarily (dropped after full migration)

The `embedding_model` field is `NULL` for unmigrated memories and `'gemini-embedding-2-preview'` after migration.

---

## Automatic Migration Flow

On every startup, after DB migrations run:

1. Query count of memories where `embedding_model IS NULL`
2. If count > 0, start background migration task (Tokio spawned task)
3. Fetch memories in batches of 32
4. Call `embed_batch()` for each batch
5. Write new vectors to `embedding_v2`, set `embedding_model = 'gemini-embedding-2-preview'`
6. Search logic uses `embedding_v2` when available, skips unmigrated memories for semantic search (they remain accessible via FTS5/BM25 keyword search)
7. Once all memories migrated, drop old `embedding` column (in a follow-up migration)

Migration failures (API errors, network issues) are logged and retried on next startup. They do not block server startup.

---

## Error Handling

- Missing config file or missing `api_key`: startup error with clear message pointing to `~/.vestige/config.toml`
- API call failure (network, quota, auth): `EmbeddingError::ApiError(String)` — propagated to caller
- Empty text input: `EmbeddingError::InvalidInput` (same as current)
- Dimension mismatch after schema change: caught in migration, logged, memory skipped

---

## Testing

- Unit tests for config parsing (valid, missing key, malformed TOML)
- Unit tests for `Embedding` struct (bytes round-trip, cosine similarity) — unchanged from current
- Integration test: mock Gemini API server (using `wiremock` or `httpmock`) to verify request format and batch logic
- Migration test: populate DB with nomic-style embeddings, run migration, verify `embedding_model` field updated and new vectors present
- Existing reranker tests unchanged (fastembed/Jina path not affected)

---

## Out of Scope

- Replacing the Jina Reranker with a Gemini reranker — separate decision
- Supporting multiple embedding providers simultaneously
- Encrypting the API key in config
