//! Error types for miniSEED decoding and encoding.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MseedError {
    #[error("record too short: expected at least {expected} bytes, got {actual}")]
    RecordTooShort { expected: usize, actual: usize },

    #[error("invalid fixed header")]
    InvalidHeader,

    #[error("invalid v3 header: {0}")]
    InvalidV3Header(String),

    #[error("CRC-32C mismatch: stored {stored:#010X}, computed {computed:#010X}")]
    CrcMismatch { stored: u32, computed: u32 },

    #[error("unsupported encoding format: {0}")]
    UnsupportedEncoding(u8),

    #[error("blockette 1000 not found")]
    MissingBlockette1000,

    #[error("unrecognized record format")]
    UnrecognizedFormat,

    #[error("steim decode error: {0}")]
    SteimDecode(String),

    #[error("sample count mismatch: header says {expected}, decoded {actual}")]
    SampleCountMismatch { expected: usize, actual: usize },

    #[error("encode error: {0}")]
    EncodeError(String),
}

pub type Result<T> = std::result::Result<T, MseedError>;
