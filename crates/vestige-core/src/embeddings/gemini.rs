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

    pub fn model_name(&self) -> &'static str {
        "gemini-embedding-2-preview"
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
    fn test_is_ready_does_not_panic() {
        let svc = GeminiEmbeddingService::new();
        let _ = svc.is_ready();
    }
}
