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
/// Callers treat `Err` as "embeddings not available".
pub fn get_gemini_config() -> Result<&'static GeminiConfig, String> {
    GEMINI_CONFIG
        .get_or_init(load_config_from_file)
        .as_ref()
        .map_err(|e| e.clone())
}

// ============================================================================
// TESTS
// ============================================================================

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
