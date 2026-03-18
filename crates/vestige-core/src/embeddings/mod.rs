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
