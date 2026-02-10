use thiserror::Error;

#[derive(Debug, Error)]
pub enum MseedError {
    #[error("record too short: expected at least {expected} bytes, got {actual}")]
    RecordTooShort { expected: usize, actual: usize },

    #[error("invalid fixed header")]
    InvalidHeader,

    #[error("unsupported encoding format: {0}")]
    UnsupportedEncoding(u8),

    #[error("blockette 1000 not found")]
    MissingBlockette1000,

    #[error("steim decode error: {0}")]
    SteimDecode(String),

    #[error("sample count mismatch: header says {expected}, decoded {actual}")]
    SampleCountMismatch { expected: usize, actual: usize },
}

pub type Result<T> = std::result::Result<T, MseedError>;
