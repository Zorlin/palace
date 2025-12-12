use thiserror::Error;

#[derive(Error, Debug)]
pub enum TranslatorError {
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Upstream error: {0}")]
    Upstream(String),

    #[error("Unknown model: {0}")]
    UnknownModel(String),
}
