//! Error types for Thermogram

use thiserror::Error;

/// Result type alias for Thermogram operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during Thermogram operations
#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Hash chain verification failed: {0}")]
    HashChainVerification(String),

    #[error("Plasticity rule violation: {0}")]
    PlasticityViolation(String),

    #[error("Consolidation error: {0}")]
    Consolidation(String),

    #[error("Delta conflict: {0}")]
    DeltaConflict(String),

    #[error("Key not found: {0}")]
    KeyNotFound(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Compression error: {0}")]
    Compression(String),

    #[error("Decompression error: {0}")]
    Decompression(String),

    #[error("Signature verification failed")]
    SignatureVerification,

    #[error("Engram export failed: {0}")]
    EngramExport(String),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Serialization(e.to_string())
    }
}

impl From<bincode::Error> for Error {
    fn from(e: bincode::Error) -> Self {
        Error::Serialization(e.to_string())
    }
}
