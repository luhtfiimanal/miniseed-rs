//! Pure Rust miniSEED v2 decoder and encoder.
//!
//! Decodes 512-byte miniSEED records (Steim1/2, INT16/32, FLOAT32/64)
//! into structured [`MseedRecord`] with header metadata and sample data.
//!
//! # Example
//!
//! ```ignore
//! use miniseed::decode;
//!
//! let record = decode(&raw_bytes)?;
//! println!("{}.{}.{}.{}", record.network, record.station, record.location, record.channel);
//! println!("samples: {}", record.samples.len());
//! ```

pub mod decode;
pub mod encode;
pub mod error;
pub mod steim;

pub use error::{MseedError, Result};
