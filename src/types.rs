//! Shared types: [`ByteOrder`], [`EncodingFormat`], and [`FormatVersion`].

use std::fmt;

use crate::{MseedError, Result};

/// miniSEED format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatVersion {
    /// miniSEED v2 (SEED Manual, 48-byte fixed header + blockettes).
    V2,
    /// miniSEED v3 (FDSN, 40-byte fixed header, little-endian).
    V3,
}

impl fmt::Display for FormatVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::V2 => write!(f, "miniSEED v2"),
            Self::V3 => write!(f, "miniSEED v3"),
        }
    }
}

/// Byte order for multi-byte fields in a miniSEED record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    Big,
    Little,
}

/// Encoding format for sample data in a miniSEED v2 record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingFormat {
    /// 16-bit signed integer (code 1).
    Int16,
    /// 32-bit signed integer (code 3).
    Int32,
    /// 32-bit IEEE float (code 4).
    Float32,
    /// 64-bit IEEE double (code 5).
    Float64,
    /// Steim-1 compressed integers (code 10).
    Steim1,
    /// Steim-2 compressed integers (code 11).
    Steim2,
}

impl EncodingFormat {
    /// Convert a raw encoding code (from Blockette 1000) to an `EncodingFormat`.
    pub fn from_code(code: u8) -> Result<Self> {
        match code {
            1 => Ok(Self::Int16),
            3 => Ok(Self::Int32),
            4 => Ok(Self::Float32),
            5 => Ok(Self::Float64),
            10 => Ok(Self::Steim1),
            11 => Ok(Self::Steim2),
            _ => Err(MseedError::UnsupportedEncoding(code)),
        }
    }

    /// Convert to the raw encoding code for Blockette 1000.
    pub fn to_code(self) -> u8 {
        match self {
            Self::Int16 => 1,
            Self::Int32 => 3,
            Self::Float32 => 4,
            Self::Float64 => 5,
            Self::Steim1 => 10,
            Self::Steim2 => 11,
        }
    }
}

impl fmt::Display for EncodingFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int16 => write!(f, "INT16"),
            Self::Int32 => write!(f, "INT32"),
            Self::Float32 => write!(f, "FLOAT32"),
            Self::Float64 => write!(f, "FLOAT64"),
            Self::Steim1 => write!(f, "Steim1"),
            Self::Steim2 => write!(f, "Steim2"),
        }
    }
}
