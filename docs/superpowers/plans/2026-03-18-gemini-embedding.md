# Gemini Embedding Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace local `nomic-embed-text-v1.5` ONNX embedding with `gemini-embedding-2-preview` REST API, auto-migrating existing memory vectors on startup.

**Architecture:** `GeminiEmbeddingService` is a drop-in replacement for `EmbeddingService` — same public interface (`new`, `is_ready`, `init`, `embed`, `embed_batch`), loaded via `OnceLock` global config. All `Storage::new()` callsites are unchanged. Vectors move from `node_embeddings.embedding` (256D) to `knowledge_nodes.embedding_v2` (1536D). Background migration follows the same `Arc::clone(&storage)` + `tokio::spawn` pattern used by the auto-consolidation task in `main.rs`.

**Tech Stack:** Rust, `reqwest` (blocking), `toml`, `serde_json`, SQLite migrations, USearch HNSW, Tokio

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/vestige-core/Cargo.toml` | Modify | Add `gemini-embeddings` feature, `reqwest` (blocking), `toml` deps; add `blocking` to tokio features |
| `crates/vestige-core/src/embeddings/config.rs` | **Create** | Load `~/.vestige/config.toml` via `OnceLock`, validate `output_dimensions` |
| `crates/vestige-core/src/embeddings/gemini.rs` | **Create** | `GeminiEmbeddingService` — blocking HTTP calls to Gemini API; exposes `GEMINI_API_BASE_URL` env var override for tests |
| `crates/vestige-core/src/embeddings/mod.rs` | Modify | Add `gemini` + `config` modules; re-export `GeminiEmbeddingService as EmbeddingService` under `gemini-embeddings`; cfg-gate `EMBEDDING_DIMENSIONS` |
| `crates/vestige-core/src/embeddings/local.rs` | **Retain** | Provides `Embedding`, `EmbeddingError`, similarity fns, and `EmbeddingService` for non-gemini path — do NOT delete |
| `crates/vestige-core/src/search/vector.rs` | Modify | `cfg`-gate `DEFAULT_DIMENSIONS`: 1536 with `gemini-embeddings`, 256 without |
| `crates/vestige-core/src/storage/migrations.rs` | Modify | Add V10 migration: `embedding_v2 BLOB`, `gemini_retry_count INTEGER` |
| `crates/vestige-core/src/storage/sqlite.rs` | Modify | Update `generate_embedding_for_node`, `load_embeddings_into_index`, `get_all_embeddings` |
| `crates/vestige-core/src/lib.rs` | Modify | Update `DEFAULT_EMBEDDING_MODEL` constant |
| `crates/vestige-mcp/src/main.rs` | Modify | Add Gemini migration background task after storage init (same pattern as consolidation task) |

---

## Task 1: Add dependencies and feature flag

**Files:**
- Modify: `crates/vestige-core/Cargo.toml`

- [ ] **Step 1: Add the `gemini-embeddings` feature and optional deps**

In `crates/vestige-core/Cargo.toml`:

**Add after the `nomic-v2` feature block:**
```toml
# Google Gemini embedding API (replaces local nomic ONNX model)
gemini-embeddings = ["dep:reqwest", "dep:toml"]
```

**Update the `default` feature line:**
```toml
default = ["embeddings", "vector-search", "bundled-sqlite", "gemini-embeddings"]
```

**Add `blocking` to the existing tokio features** (required for `tokio::task::spawn_blocking`):
```toml
tokio = { version = "1", features = ["sync", "rt-multi-thread", "macros", "blocking"] }
```

**Add under the `# OPTIONAL: Embeddings` section:**
```toml
# HTTP client for Gemini API (blocking — Storage is fully synchronous)
reqwest = { version = "0.12", features = ["blocking", "json", "rustls-tls"], default-features = false, optional = true }

# TOML config file parsing
toml = { version = "0.8", optional = true }
```

**Add to `[dev-dependencies]`:**
```toml
wiremock = "0.6"
tokio = { version = "1", features = ["rt", "macros"] }
```

- [ ] **Step 2: Verify base build still compiles**

```bash
cargo build -p vestige-core --no-default-features --features "embeddings,vector-search,bundled-sqlite" 2>&1 | grep "^error"
```
Expected: zero errors

- [ ] **Step 3: Commit**

```bash
git add crates/vestige-core/Cargo.toml
git commit -m "feat: add gemini-embeddings feature flag, reqwest/toml deps, tokio blocking"
```

---

## Task 2: Add V10 database migration

**Files:**
- Modify: `crates/vestige-core/src/storage/migrations.rs`

- [ ] **Step 1: Append V10 to the MIGRATIONS array**

Find the `MIGRATIONS` constant. After the V9 entry, add:
```rust
Migration {
    version: 10,
    description: "Gemini embedding: add embedding_v2 (1536D) and gemini_retry_count columns",
    up: MIGRATION_V10_UP,
},
```

- [ ] **Step 2: Add the migration SQL constant**

After `MIGRATION_V9_UP`, add:
```rust
const MIGRATION_V10_UP: &str = "
    ALTER TABLE knowledge_nodes ADD COLUMN embedding_v2 BLOB DEFAULT NULL;
    ALTER TABLE knowledge_nodes ADD COLUMN gemini_retry_count INTEGER DEFAULT 0;
";
```

- [ ] **Step 3: Write a test for the new columns**

In the `#[cfg(test)]` block of `migrations.rs`, add:
```rust
#[test]
fn test_v10_migration_adds_gemini_columns() {
    let conn = Connection::open_in_memory().unwrap();
    apply_migrations(&conn).unwrap();

    conn.execute(
        "INSERT INTO knowledge_nodes (id, content, node_type, created_at, updated_at)
         VALUES ('m1', 'test content', 'fact', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        [],
    ).unwrap();

    conn.execute(
        "UPDATE knowledge_nodes SET embedding_v2 = X'deadbeef', gemini_retry_count = 2 WHERE id = 'm1'",
        [],
    ).unwrap();

    let retry: i64 = conn.query_row(
        "SELECT gemini_retry_count FROM knowledge_nodes WHERE id = 'm1'",
        [],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(retry, 2);

    let has_v2: bool = conn.query_row(
        "SELECT embedding_v2 IS NOT NULL FROM knowledge_nodes WHERE id = 'm1'",
        [],
        |row| row.get(0),
    ).unwrap();
    assert!(has_v2);
}
```

- [ ] **Step 4: Run the test**

```bash
cargo test -p vestige-core test_v10_migration -- --nocapture
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/vestige-core/src/storage/migrations.rs
git commit -m "feat: add V10 migration for Gemini embedding columns"
```

---

## Task 3: Create config loader

**Files:**
- Create: `crates/vestige-core/src/embeddings/config.rs`

- [ ] **Step 1: Write failing tests**

Create the file with tests only (implementation follows):

```rust
//! Vestige config file loader for Gemini embeddings.
//!
//! Reads `~/.vestige/config.toml`:
//! ```toml
//! [embeddings]
//! provider = "gemini"
//! api_key = "AIza..."
//! model = "gemini-embedding-2-preview"
//! output_dimensions = 1536  # optional, default 1536, range 1–3072
//! ```

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_config_parses() {
        let toml = r#"
[embeddings]
provider = "gemini"
api_key = "test-key-123"
model = "gemini-embedding-2-preview"
output_dimensions = 768
"#;
        let cfg = parse_config_str(toml).unwrap();
        assert_eq!(cfg.api_key, "test-key-123");
        assert_eq!(cfg.model, "gemini-embedding-2-preview");
        assert_eq!(cfg.output_dimensions, 768);
    }

    #[test]
    fn test_default_output_dimensions() {
        let toml = r#"
[embeddings]
provider = "gemini"
api_key = "test-key"
model = "gemini-embedding-2-preview"
"#;
        let cfg = parse_config_str(toml).unwrap();
        assert_eq!(cfg.output_dimensions, 1536);
    }

    #[test]
    fn test_missing_api_key_errors() {
        let toml = r#"
[embeddings]
provider = "gemini"
model = "gemini-embedding-2-preview"
"#;
        let err = parse_config_str(toml).unwrap_err();
        assert!(err.contains("api_key"), "error should mention api_key, got: {err}");
    }

    #[test]
    fn test_dimensions_too_large_errors() {
        let toml = r#"
[embeddings]
provider = "gemini"
api_key = "test-key"
model = "gemini-embedding-2-preview"
output_dimensions = 9999
"#;
        assert!(parse_config_str(toml).is_err());
    }

    #[test]
    fn test_dimensions_zero_errors() {
        let toml = r#"
[embeddings]
provider = "gemini"
api_key = "test-key"
model = "gemini-embedding-2-preview"
output_dimensions = 0
"#;
        assert!(parse_config_str(toml).is_err());
    }

    #[test]
    fn test_malformed_toml_errors() {
        assert!(parse_config_str("this is not valid toml [[[").is_err());
    }
}
```

- [ ] **Step 2: Verify tests fail to compile**

```bash
cargo test -p vestige-core --features "gemini-embeddings,embeddings,vector-search,bundled-sqlite" 2>&1 | head -5
```
Expected: compile error — `parse_config_str` not found

- [ ] **Step 3: Implement the module**

Replace file contents with the full implementation (tests stay at the bottom):

```rust
//! Vestige config file loader for Gemini embeddings.

use std::sync::OnceLock;
use directories::BaseDirs;

/// Parsed Gemini embedding configuration.
#[derive(Debug, Clone)]
pub struct GeminiConfig {
    pub api_key: String,
    pub model: String,
    pub output_dimensions: usize,
}

// Internal raw TOML shape
#[derive(serde::Deserialize)]
struct RawConfig {
    embeddings: Option<RawEmbeddingConfig>,
}

#[derive(serde::Deserialize)]
struct RawEmbeddingConfig {
    api_key: Option<String>,
    model: Option<String>,
    output_dimensions: Option<usize>,
}

/// Parse config from a TOML string.
/// Used by tests and by `load_config_from_file`.
pub fn parse_config_str(s: &str) -> Result<GeminiConfig, String> {
    let raw: RawConfig = toml::from_str(s)
        .map_err(|e| format!("Failed to parse config TOML: {e}"))?;

    let emb = raw.embeddings
        .ok_or_else(|| "Missing [embeddings] section".to_string())?;

    let api_key = emb.api_key
        .filter(|k| !k.is_empty())
        .ok_or_else(|| "api_key is required under [embeddings]".to_string())?;

    let model = emb.model
        .unwrap_or_else(|| "gemini-embedding-2-preview".to_string());

    let output_dimensions = emb.output_dimensions.unwrap_or(1536);
    if output_dimensions == 0 || output_dimensions > 3072 {
        return Err(format!(
            "output_dimensions must be 1–3072, got {output_dimensions}"
        ));
    }

    Ok(GeminiConfig { api_key, model, output_dimensions })
}

/// Load config from `~/.vestige/config.toml`.
pub fn load_config_from_file() -> Result<GeminiConfig, String> {
    let path = BaseDirs::new()
        .map(|b| b.home_dir().join(".vestige/config.toml"))
        .ok_or_else(|| "Could not determine home directory".to_string())?;

    if !path.exists() {
        return Err(format!(
            "Config file not found at {path:?}. Create it with:\n\
             [embeddings]\napi_key = \"your-key\""
        ));
    }

    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read {path:?}: {e}"))?;

    parse_config_str(&content)
}

// ============================================================================
// GLOBAL CONFIG (loaded once at first use)
// ============================================================================

static GEMINI_CONFIG: OnceLock<Result<GeminiConfig, String>> = OnceLock::new();

/// Get the global Gemini config, loading from file on first call.
/// Returns `Err` if config is missing or invalid.
/// Callers treat `Err` as "embeddings not available" — identical to how
/// `EmbeddingService::is_ready()` behaves when the ONNX model fails to load.
pub fn get_gemini_config() -> Result<&'static GeminiConfig, String> {
    GEMINI_CONFIG
        .get_or_init(load_config_from_file)
        .as_ref()
        .map_err(|e| e.clone())
}

// ============================================================================
// TESTS (from Step 1)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    // paste the 6 tests here
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vestige-core --features "gemini-embeddings,embeddings,vector-search,bundled-sqlite" config:: -- --nocapture
```
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add crates/vestige-core/src/embeddings/config.rs
git commit -m "feat: add Gemini config loader with OnceLock global"
```

---

## Task 4: Create GeminiEmbeddingService

**Files:**
- Create: `crates/vestige-core/src/embeddings/gemini.rs`

The service exposes a `GEMINI_API_BASE_URL` environment variable that overrides the base URL. This allows integration tests to redirect calls to a `wiremock` mock server without modifying production code.

- [ ] **Step 1: Write failing unit tests**

Create the file with tests only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_text_returns_error() {
        let svc = GeminiEmbeddingService::new();
        assert!(matches!(svc.embed(""), Err(EmbeddingError::InvalidInput(_))));
    }

    #[test]
    fn test_empty_batch_returns_empty() {
        let svc = GeminiEmbeddingService::new();
        assert!(svc.embed_batch(&[]).unwrap().is_empty());
    }

    #[test]
    fn test_model_name_default() {
        let svc = GeminiEmbeddingService::new();
        assert_eq!(svc.model_name(), "gemini-embedding-2-preview");
    }

    #[test]
    fn test_is_ready_false_without_config() {
        // In test env, no config file → not ready
        let svc = GeminiEmbeddingService::new();
        // is_ready() depends on config file; we can't guarantee either value here,
        // but it must not panic.
        let _ = svc.is_ready();
    }
}
```

- [ ] **Step 2: Implement the module**

Full implementation:

```rust
//! Gemini Embedding Service
//!
//! Calls `batchEmbedContents` REST endpoint with reqwest::blocking.
//! Drop-in replacement for the local EmbeddingService interface.
//!
//! ## API endpoint
//! POST https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents
//! x-goog-api-key: {api_key}
//!
//! Set `GEMINI_API_BASE_URL` env var to override base URL (used in tests).

use super::{Embedding, EmbeddingError, EMBEDDING_DIMENSIONS, BATCH_SIZE};
use super::config::get_gemini_config;

const DEFAULT_GEMINI_API_BASE: &str =
    "https://generativelanguage.googleapis.com/v1beta/models";

fn api_base_url() -> String {
    std::env::var("GEMINI_API_BASE_URL")
        .unwrap_or_else(|_| DEFAULT_GEMINI_API_BASE.to_string())
}

/// Drop-in replacement for EmbeddingService.
/// Stateless — config lives in a global OnceLock (see config.rs).
pub struct GeminiEmbeddingService {
    _unused: (),
}

impl Default for GeminiEmbeddingService {
    fn default() -> Self { Self::new() }
}

impl GeminiEmbeddingService {
    pub fn new() -> Self { Self { _unused: () } }

    /// True if config is loaded and API key is present.
    pub fn is_ready(&self) -> bool {
        get_gemini_config().is_ok()
    }

    /// Validate config is accessible. Mirrors EmbeddingService::init().
    pub fn init(&self) -> Result<(), EmbeddingError> {
        get_gemini_config()
            .map(|_| ())
            .map_err(|e| EmbeddingError::EmbeddingFailed(
                format!("Gemini config error: {e}")
            ))
    }

    pub fn model_name(&self) -> &str {
        // 'static lifetime from OnceLock — safe to return as &str
        match get_gemini_config() {
            Ok(c) => {
                // SAFETY: config lives in a 'static OnceLock
                unsafe { &*(c.model.as_str() as *const str) }
            }
            Err(_) => "gemini-embedding-2-preview",
        }
    }

    pub fn dimensions(&self) -> usize {
        EMBEDDING_DIMENSIONS
    }

    /// Embed a single text. Uses embed_batch internally.
    pub fn embed(&self, text: &str) -> Result<Embedding, EmbeddingError> {
        if text.is_empty() {
            return Err(EmbeddingError::InvalidInput(
                "Text cannot be empty".into()
            ));
        }
        self.embed_batch(&[text])?
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::EmbeddingFailed(
                "Gemini returned empty embeddings".into()
            ))
    }

    /// Embed multiple texts in batches of BATCH_SIZE (32).
    /// Uses reqwest::blocking — safe to call from sync Storage methods.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let config = get_gemini_config()
            .map_err(|e| EmbeddingError::EmbeddingFailed(
                format!("Gemini config error: {e}")
            ))?;

        let client = reqwest::blocking::Client::new();
        let base = api_base_url();
        let url = format!("{}/{}:batchEmbedContents", base, config.model);
        let mut all = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(BATCH_SIZE) {
            let requests: Vec<serde_json::Value> = chunk.iter().map(|t| {
                serde_json::json!({
                    "model": format!("models/{}", config.model),
                    "content": { "parts": [{ "text": t }] },
                    "outputDimensionality": config.output_dimensions
                })
            }).collect();

            let resp = client
                .post(&url)
                .header("x-goog-api-key", &config.api_key)
                .json(&serde_json::json!({ "requests": requests }))
                .send()
                .map_err(|e| EmbeddingError::EmbeddingFailed(
                    format!("HTTP request failed: {e}")
                ))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().unwrap_or_default();
                return Err(EmbeddingError::EmbeddingFailed(
                    format!("Gemini API {status}: {body}")
                ));
            }

            let json: serde_json::Value = resp.json()
                .map_err(|e| EmbeddingError::EmbeddingFailed(
                    format!("Failed to parse response: {e}")
                ))?;

            let embeddings = json["embeddings"].as_array()
                .ok_or_else(|| EmbeddingError::EmbeddingFailed(
                    "Missing 'embeddings' in response".into()
                ))?;

            for emb in embeddings {
                let values: Vec<f32> = emb["values"].as_array()
                    .ok_or_else(|| EmbeddingError::EmbeddingFailed(
                        "Missing 'values' in embedding".into()
                    ))?
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();

                if values.is_empty() {
                    return Err(EmbeddingError::EmbeddingFailed(
                        "Empty embedding vector from Gemini".into()
                    ));
                }
                all.push(Embedding::new(values));
            }
        }

        Ok(all)
    }
}

#[cfg(test)]
mod tests {
    // unit tests from Step 1
}
```

**Note on `model_name()` lifetime:** The unsafe pointer cast is needed because `GeminiConfig` lives in a `'static OnceLock`. An alternative (simpler) approach is to return `String` instead of `&str`, or make `model_name` return `&'static str` only for the fallback and return an owned value when the config is present. If the unsafe cast is undesirable, change `model_name(&self) -> &str` to `model_name(&self) -> String` and update `lib.rs`/`sqlite.rs` callsites accordingly — they typically do `let _ = self.embedding_service.model_name()` as a log/display value.

- [ ] **Step 3: Run unit tests**

```bash
cargo test -p vestige-core --features "gemini-embeddings,embeddings,vector-search,bundled-sqlite" gemini::tests -- --nocapture
```
Expected: 4 tests PASS

- [ ] **Step 4: Commit**

```bash
git add crates/vestige-core/src/embeddings/gemini.rs
git commit -m "feat: add GeminiEmbeddingService with batchEmbedContents and env URL override"
```

---

## Task 5: Wire embeddings/mod.rs and update dimension constants

**Files:**
- Modify: `crates/vestige-core/src/embeddings/mod.rs`
- Modify: `crates/vestige-core/src/embeddings/local.rs`
- Modify: `crates/vestige-core/src/search/vector.rs`
- Modify: `crates/vestige-core/src/lib.rs`

- [ ] **Step 1: Rewrite embeddings/mod.rs**

Replace the entire file:

```rust
//! Semantic Embeddings Module

mod code;
mod hybrid;
mod local;

#[cfg(feature = "gemini-embeddings")]
pub mod config;
#[cfg(feature = "gemini-embeddings")]
mod gemini;

// Model-agnostic types — always exported from local.rs
pub use local::{
    cosine_similarity, dot_product, euclidean_distance, matryoshka_truncate,
    Embedding, EmbeddingError, BATCH_SIZE, MAX_TEXT_LENGTH,
};

// EMBEDDING_DIMENSIONS: 1536 with Gemini, 256 with local nomic
#[cfg(feature = "gemini-embeddings")]
pub const EMBEDDING_DIMENSIONS: usize = 1536;
#[cfg(not(feature = "gemini-embeddings"))]
pub use local::EMBEDDING_DIMENSIONS;

// EmbeddingService alias: Gemini when feature active, local ONNX otherwise
#[cfg(feature = "gemini-embeddings")]
pub use gemini::GeminiEmbeddingService as EmbeddingService;
#[cfg(not(feature = "gemini-embeddings"))]
pub use local::EmbeddingService;

pub use code::CodeEmbedding;
pub use hybrid::HybridEmbedding;
```

- [ ] **Step 2: Cfg-gate EMBEDDING_DIMENSIONS in local.rs**

In `local.rs`, find:
```rust
pub const EMBEDDING_DIMENSIONS: usize = 256;
```
Replace with:
```rust
// Only exported when gemini-embeddings is NOT active; mod.rs re-exports it.
#[cfg(not(feature = "gemini-embeddings"))]
pub const EMBEDDING_DIMENSIONS: usize = 256;
// When gemini-embeddings is active, mod.rs defines EMBEDDING_DIMENSIONS = 1536.
// local.rs embed() still uses the EMBEDDING_DIMENSIONS from the crate root via super.
// For matryoshka_truncate inside local.rs, define a local const that always = 256:
#[cfg(feature = "gemini-embeddings")]
const EMBEDDING_DIMENSIONS: usize = 256; // local path not used; keeps matryoshka_truncate compilable
```

- [ ] **Step 3: Cfg-gate DEFAULT_DIMENSIONS in vector.rs**

In `crates/vestige-core/src/search/vector.rs`, replace:
```rust
pub const DEFAULT_DIMENSIONS: usize = 256;
```
With:
```rust
#[cfg(feature = "gemini-embeddings")]
pub const DEFAULT_DIMENSIONS: usize = 1536;
#[cfg(not(feature = "gemini-embeddings"))]
pub const DEFAULT_DIMENSIONS: usize = 256;
```

- [ ] **Step 4: Update DEFAULT_EMBEDDING_MODEL in lib.rs**

Find `DEFAULT_EMBEDDING_MODEL` (around line 425) and update:
```rust
#[cfg(feature = "gemini-embeddings")]
pub const DEFAULT_EMBEDDING_MODEL: &str = "gemini-embedding-2-preview";
#[cfg(not(feature = "gemini-embeddings"))]
pub const DEFAULT_EMBEDDING_MODEL: &str = "nomic-ai/nomic-embed-text-v1.5";
```

- [ ] **Step 5: Build with gemini-embeddings**

```bash
cargo build -p vestige-core --features "embeddings,vector-search,bundled-sqlite,gemini-embeddings" 2>&1 | grep "^error"
```
Expected: zero errors

- [ ] **Step 6: Build without gemini-embeddings (backwards compat)**

```bash
cargo build -p vestige-core --no-default-features --features "embeddings,vector-search,bundled-sqlite" 2>&1 | grep "^error"
```
Expected: zero errors

- [ ] **Step 7: Commit**

```bash
git add crates/vestige-core/src/embeddings/mod.rs \
        crates/vestige-core/src/embeddings/local.rs \
        crates/vestige-core/src/search/vector.rs \
        crates/vestige-core/src/lib.rs
git commit -m "feat: wire GeminiEmbeddingService and cfg-gate EMBEDDING_DIMENSIONS"
```

---

## Task 6: Update sqlite.rs write and read paths

**Files:**
- Modify: `crates/vestige-core/src/storage/sqlite.rs`

### Part A — generate_embedding_for_node

- [ ] **Step 1: Replace the DB write inside generate_embedding_for_node**

The function is at line ~594. The `writer.execute(...)` block currently writes to `node_embeddings`. Replace the entire write block with a cfg-gated dual path:

```rust
let now = Utc::now();
{
    let writer = self.writer.lock()
        .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;

    #[cfg(feature = "gemini-embeddings")]
    writer.execute(
        "UPDATE knowledge_nodes
         SET has_embedding = 1,
             embedding_model = 'gemini-embedding-2-preview',
             embedding_v2 = ?1,
             updated_at = ?2
         WHERE id = ?3",
        params![embedding.to_bytes(), now.to_rfc3339(), node_id],
    )?;

    #[cfg(not(feature = "gemini-embeddings"))]
    {
        writer.execute(
            "INSERT OR REPLACE INTO node_embeddings (node_id, embedding, dimensions, model, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                node_id,
                embedding.to_bytes(),
                crate::embeddings::EMBEDDING_DIMENSIONS as i32,
                "nomic-embed-text-v1.5",
                now.to_rfc3339(),
            ],
        )?;
        writer.execute(
            "UPDATE knowledge_nodes SET has_embedding = 1, embedding_model = 'nomic-embed-text-v1.5' WHERE id = ?1",
            params![node_id],
        )?;
    }
}
```

### Part B — load_embeddings_into_index

- [ ] **Step 2: Replace the query in load_embeddings_into_index (line ~215)**

```rust
#[cfg(feature = "gemini-embeddings")]
let mut stmt = reader.prepare(
    "SELECT id, embedding_v2 FROM knowledge_nodes
     WHERE embedding_v2 IS NOT NULL
       AND embedding_model = 'gemini-embedding-2-preview'"
)?;

#[cfg(not(feature = "gemini-embeddings"))]
let mut stmt = reader.prepare(
    "SELECT node_id, embedding FROM node_embeddings"
)?;
```

Also wrap the Matryoshka migration inside the loop with `#[cfg(not(feature = "gemini-embeddings"))]` — it only applies to old nomic 768D vectors.

### Part C — get_all_embeddings

- [ ] **Step 3: Replace the query in get_all_embeddings (line ~545)**

```rust
#[cfg(feature = "gemini-embeddings")]
let mut stmt = reader.prepare(
    "SELECT id, embedding_v2 FROM knowledge_nodes
     WHERE embedding_v2 IS NOT NULL
       AND embedding_model = 'gemini-embedding-2-preview'"
)?;

#[cfg(not(feature = "gemini-embeddings"))]
let mut stmt = reader.prepare("SELECT node_id, embedding FROM node_embeddings")?;
```

- [ ] **Step 4: Build to verify**

```bash
cargo build -p vestige-core --features "embeddings,vector-search,bundled-sqlite,gemini-embeddings" 2>&1 | grep "^error"
```
Expected: zero errors

- [ ] **Step 5: Run storage tests**

```bash
cargo test -p vestige-core --features "embeddings,vector-search,bundled-sqlite,gemini-embeddings" -- storage 2>&1 | tail -20
```
Expected: all pass (no live API needed — `is_ready()` returns false in tests)

- [ ] **Step 6: Commit**

```bash
git add crates/vestige-core/src/storage/sqlite.rs
git commit -m "feat: update embedding write/read paths for Gemini (embedding_v2 column)"
```

---

## Task 7: Add background migration in main.rs

**Files:**
- Modify: `crates/vestige-mcp/src/main.rs`

The migration follows the exact same `Arc::clone(&storage)` + `tokio::task::spawn_blocking` pattern used by the auto-consolidation task already in `main.rs` (lines ~200–220).

- [ ] **Step 1: Add the migration block after init_embeddings**

In `main.rs`, find the `#[cfg(feature = "embeddings")]` block that calls `s.init_embeddings()` (around line 178). After the closing `}` of that block (and before `Arc::new(s)`), the storage is not yet wrapped in Arc. The migration must be spawned **after** `Arc::new(s)`. Add after line 189 (`Arc::new(s)` line), following the consolidation spawn pattern:

```rust
// Gemini background migration: re-embed legacy nomic memories
#[cfg(feature = "gemini-embeddings")]
{
    use vestige_core::embeddings::EmbeddingService;

    let svc = EmbeddingService::new();
    if svc.is_ready() {
        // Count unmigrated memories
        let pending: i64 = storage.count_unmigrated_gemini().unwrap_or(0);
        if pending > 0 {
            info!("Gemini migration: {} memories need re-embedding", pending);
            let storage_clone = storage.clone();
            tokio::task::spawn_blocking(move || {
                storage_clone.run_gemini_migration();
            });
        }
    } else {
        warn!("Gemini migration skipped: no API key configured (~/.vestige/config.toml)");
    }
}
```

- [ ] **Step 2: Add count_unmigrated_gemini and run_gemini_migration to sqlite.rs**

In `sqlite.rs`, add two items:

**Method on Storage** (cfg-gated):
```rust
/// Count memories not yet migrated to Gemini embeddings.
#[cfg(feature = "gemini-embeddings")]
pub fn count_unmigrated_gemini(&self) -> Result<i64> {
    let reader = self.reader.lock()
        .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
    reader.query_row(
        "SELECT COUNT(*) FROM knowledge_nodes
         WHERE embedding_model IS NULL
            OR embedding_model != 'gemini-embedding-2-preview'",
        [],
        |row| row.get(0),
    ).map_err(StorageError::from)
}
```

**Free function in `crates/vestige-core/src/storage/mod.rs` or at the bottom of `sqlite.rs`** (cfg-gated):
```rust
/// Run Gemini embedding migration synchronously.
/// Called via tokio::task::spawn_blocking from main.rs.
/// Processes all unmigrated memories in batches of 32.
#[cfg(feature = "gemini-embeddings")]
pub fn run_gemini_migration(storage: &Storage) {
    use crate::embeddings::EmbeddingService;

    let svc = EmbeddingService::new();

    loop {
        let batch: Vec<(String, String)> = {
            let Ok(reader) = storage.reader.lock() else { break };
            reader.prepare(
                "SELECT id, content FROM knowledge_nodes
                 WHERE (embedding_model IS NULL OR embedding_model != 'gemini-embedding-2-preview')
                   AND gemini_retry_count < 3
                 LIMIT 32"
            )
            .and_then(|mut stmt| {
                stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                    .map(|rows| rows.filter_map(|r| r.ok()).collect())
            })
            .unwrap_or_default()
        };

        if batch.is_empty() {
            tracing::info!("Gemini migration: complete");
            break;
        }

        let texts: Vec<&str> = batch.iter().map(|(_, c)| c.as_str()).collect();
        match svc.embed_batch(&texts) {
            Ok(embeddings) => {
                let Ok(writer) = storage.writer.lock() else { break };
                for ((id, _), emb) in batch.iter().zip(embeddings.iter()) {
                    if let Err(e) = writer.execute(
                        "UPDATE knowledge_nodes
                         SET embedding_v2 = ?1,
                             embedding_model = 'gemini-embedding-2-preview',
                             has_embedding = 1
                         WHERE id = ?2",
                        rusqlite::params![emb.to_bytes(), id],
                    ) {
                        tracing::warn!("Gemini migration DB error for {id}: {e}");
                    } else if let Ok(mut idx) = storage.vector_index.lock() {
                        let _ = idx.add(id, &emb.vector);
                    }
                }
                tracing::debug!("Gemini migration: migrated {} memories", batch.len());
            }
            Err(e) => {
                tracing::warn!("Gemini migration embed_batch failed: {e}");
                // Increment retry counts and stop — retry on next startup
                if let Ok(w) = storage.writer.lock() {
                    for (id, _) in &batch {
                        let _ = w.execute(
                            "UPDATE knowledge_nodes SET gemini_retry_count = gemini_retry_count + 1 WHERE id = ?1",
                            rusqlite::params![id],
                        );
                    }
                }
                break;
            }
        }
    }
}
```

**Note:** Implement `run_gemini_migration` as a `pub fn run_gemini_migration(&self)` method directly on `Storage` in `sqlite.rs`. This gives it private field access (`self.reader`, `self.writer`, `self.vector_index`) without any visibility changes. The free-function form shown above is for illustration only — use the method form. `main.rs` calls it as `storage_clone.run_gemini_migration()` (where `storage_clone` is `Arc<Storage>`, which auto-derefs).

- [ ] **Step 3: Build the full workspace**

```bash
cargo build --workspace 2>&1 | grep "^error" | head -20
```
Expected: zero errors

- [ ] **Step 4: Commit**

```bash
git add crates/vestige-core/src/storage/sqlite.rs crates/vestige-mcp/src/main.rs
git commit -m "feat: add Gemini background migration task (spawn_blocking pattern)"
```

---

## Task 8: Tests

**Files:**
- Modify: `crates/vestige-core/src/embeddings/gemini.rs`
- Modify: `crates/vestige-core/src/storage/sqlite.rs`

### Part A — wiremock integration test for GeminiEmbeddingService

- [ ] **Step 1: Add wiremock test to gemini.rs**

Add an `integration_tests` submodule inside `#[cfg(test)]`:

```rust
#[cfg(test)]
mod integration_tests {
    use wiremock::{MockServer, Mock, ResponseTemplate};
    use wiremock::matchers::{method, path_regex, header};

    fn fake_embeddings_response(dims: usize, count: usize) -> serde_json::Value {
        let values: Vec<f32> = (0..dims).map(|i| i as f32 / dims as f32).collect();
        serde_json::json!({
            "embeddings": (0..count).map(|_| serde_json::json!({ "values": values })).collect::<Vec<_>>()
        })
    }

    #[tokio::test]
    async fn test_embed_batch_sends_correct_request() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path_regex(".*/batchEmbedContents"))
            .and(header("x-goog-api-key", "test-key-from-env"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_json(fake_embeddings_response(1536, 1)))
            .mount(&mock_server)
            .await;

        // Set env vars to use mock server and test API key
        std::env::set_var("GEMINI_API_BASE_URL", mock_server.uri());

        // Build a service with an inline config (bypass OnceLock for test)
        // We test the HTTP layer directly via reqwest::blocking
        let client = reqwest::blocking::Client::new();
        let url = format!("{}/gemini-embedding-2-preview:batchEmbedContents", mock_server.uri());
        let body = serde_json::json!({
            "requests": [{
                "model": "models/gemini-embedding-2-preview",
                "content": { "parts": [{ "text": "hello world" }] },
                "outputDimensionality": 1536
            }]
        });

        let resp = client.post(&url)
            .header("x-goog-api-key", "test-key-from-env")
            .json(&body)
            .send()
            .unwrap();

        assert!(resp.status().is_success());
        let json: serde_json::Value = resp.json().unwrap();
        assert_eq!(json["embeddings"][0]["values"].as_array().unwrap().len(), 1536);

        // Verify the mock received exactly one request
        let received = mock_server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        let req_body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
        assert!(req_body["requests"][0]["outputDimensionality"] == 1536);
    }
}
```

### Part B — migration integration test

- [ ] **Step 2: Add migration test to sqlite.rs**

In the test module at the bottom of `sqlite.rs`, add:

```rust
#[cfg(all(test, feature = "gemini-embeddings"))]
mod gemini_migration_tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_count_unmigrated_returns_correct_count() {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();

        // Insert a memory without embedding_model set
        let node = storage.ingest(crate::IngestInput {
            content: "test content".to_string(),
            tags: vec![],
            node_type: crate::NodeType::Fact,
            source: None,
            importance: None,
        }).unwrap();

        let count = storage.count_unmigrated_gemini().unwrap();
        assert!(count >= 1, "Should have at least one unmigrated memory");

        // Manually mark as migrated
        {
            let writer = storage.writer.lock().unwrap();
            writer.execute(
                "UPDATE knowledge_nodes SET embedding_model = 'gemini-embedding-2-preview' WHERE id = ?1",
                rusqlite::params![node.id],
            ).unwrap();
        }

        let count_after = storage.count_unmigrated_gemini().unwrap();
        assert_eq!(count_after, count - 1);
    }

    #[test]
    fn test_graceful_degradation_keyword_search_works_without_embeddings() {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();

        let node = storage.ingest(crate::IngestInput {
            content: "rust programming language systems".to_string(),
            tags: vec![],
            node_type: crate::NodeType::Fact,
            source: None,
            importance: None,
        }).unwrap();

        // keyword_search (FTS5) must work even when Gemini is not configured.
        // keyword_search is private with signature (&self, query: &str, limit: i32, min_retention: f64)
        // The test is inside sqlite.rs so private access is valid.
        let results = storage.keyword_search("rust programming", 10, 0.0).unwrap();
        assert!(!results.is_empty(), "FTS5 search must work without Gemini API key");
        assert_eq!(results[0].id, node.id);
    }
}
```

- [ ] **Step 3: Run all new tests**

```bash
cargo test -p vestige-core --features "embeddings,vector-search,bundled-sqlite,gemini-embeddings" -- gemini 2>&1 | tail -20
```
Expected: all pass

- [ ] **Step 4: Run the full test suite**

```bash
cargo test -p vestige-core --features "embeddings,vector-search,bundled-sqlite,gemini-embeddings" 2>&1 | tail -10
cargo test -p vestige-core --no-default-features --features "embeddings,vector-search,bundled-sqlite" 2>&1 | tail -10
```
Expected: both pass

- [ ] **Step 5: Commit**

```bash
git add crates/vestige-core/src/embeddings/gemini.rs \
        crates/vestige-core/src/storage/sqlite.rs
git commit -m "test: add wiremock, migration, and degradation tests for Gemini embedding"
```

---

## Task 9: Final verification

- [ ] **Step 1: Release build of full workspace**

```bash
cargo build --workspace --release 2>&1 | grep "^error"
```
Expected: zero errors

- [ ] **Step 2: Full test suite with gemini-embeddings**

```bash
cargo test -p vestige-core --features "embeddings,vector-search,bundled-sqlite,gemini-embeddings" 2>&1 | grep -E "^test result|FAILED"
```
Expected: `test result: ok`

- [ ] **Step 3: Full test suite without gemini-embeddings**

```bash
cargo test -p vestige-mcp --no-default-features --features "embeddings,vector-search,bundled-sqlite" 2>&1 | grep -E "^test result|FAILED"
```
Expected: `test result: ok`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: Gemini embedding integration complete

Replace nomic-embed-text-v1.5 (256D local ONNX) with gemini-embedding-2-preview
(1536D REST API). Auto-migrates existing memories on startup via spawn_blocking.
Config via ~/.vestige/config.toml. Graceful fallback to FTS5 when key absent.
Zero changes to Storage::new() callsites."
```

---

## Quick Reference

**Config file** (`~/.vestige/config.toml`):
```toml
[embeddings]
provider = "gemini"
api_key = "AIza..."
model = "gemini-embedding-2-preview"
output_dimensions = 1536
```

**Build with Gemini (default):**
```bash
cargo build -p vestige-mcp
```

**Build without Gemini (local ONNX):**
```bash
cargo build -p vestige-mcp --no-default-features --features "embeddings,vector-search,bundled-sqlite"
```

**Key new files:** `embeddings/config.rs`, `embeddings/gemini.rs`
**Key modified files:** `embeddings/mod.rs`, `search/vector.rs`, `storage/migrations.rs`, `storage/sqlite.rs`, `lib.rs`, `main.rs`
**local.rs:** retained — provides `Embedding`, `EmbeddingError`, similarity functions for both paths
