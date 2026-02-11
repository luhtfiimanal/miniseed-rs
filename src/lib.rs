//! Pure Rust miniSEED v2 decoder and encoder.
//!
//! Zero `unsafe`, zero C dependencies. Supports Steim1/2 compression,
//! uncompressed INT16/32, FLOAT32/64, and any power-of-2 record length
//! (256, 512, 4096, etc.).
//!
//! # Decoding a record
//!
//! ```
//! use miniseed_rs::{decode, encode, MseedRecord, Samples, EncodingFormat, BTime};
//!
//! // Build a record, encode it, then decode the bytes
//! let record = MseedRecord::new()
//!     .with_nslc("IU", "ANMO", "00", "BHZ")
//!     .with_sample_rate(20.0)
//!     .with_samples(Samples::Int(vec![100, 200, 300]));
//!
//! let bytes = encode(&record).unwrap();
//! let decoded = decode(&bytes).unwrap();
//!
//! assert_eq!(decoded.network, "IU");
//! assert_eq!(decoded.station, "ANMO");
//! assert_eq!(decoded.samples.len(), 3);
//! ```
//!
//! # Iterating multi-record data
//!
//! ```
//! use miniseed_rs::{encode, MseedRecord, MseedReader, Samples};
//!
//! // Concatenate two encoded records
//! let r1 = MseedRecord::new()
//!     .with_nslc("IU", "ANMO", "00", "BHZ")
//!     .with_samples(Samples::Int(vec![1, 2, 3]));
//! let r2 = MseedRecord::new()
//!     .with_nslc("IU", "ANMO", "00", "BHN")
//!     .with_samples(Samples::Int(vec![4, 5, 6]));
//!
//! let mut data = encode(&r1).unwrap();
//! data.extend_from_slice(&encode(&r2).unwrap());
//!
//! let records: Vec<_> = MseedReader::new(&data)
//!     .collect::<Result<Vec<_>, _>>()
//!     .unwrap();
//!
//! assert_eq!(records.len(), 2);
//! assert_eq!(records[0].channel, "BHZ");
//! assert_eq!(records[1].channel, "BHN");
//! ```
//!
//! # Building a record from scratch
//!
//! ```
//! use miniseed_rs::{MseedRecord, Samples, EncodingFormat, BTime, encode};
//!
//! let record = MseedRecord::new()
//!     .with_nslc("XX", "TEST", "00", "BHZ")
//!     .with_start_time(BTime {
//!         year: 2025, day: 100, hour: 12,
//!         minute: 30, second: 45, fract: 0,
//!     })
//!     .with_sample_rate(20.0)
//!     .with_encoding(EncodingFormat::Int32)
//!     .with_samples(Samples::Int(vec![1, -2, 3, -4]));
//!
//! let bytes = encode(&record).unwrap();
//! assert_eq!(bytes.len(), 512);
//! ```

pub mod decode;
pub mod encode;
pub mod error;
pub mod reader;
pub mod steim;
pub mod types;

pub use decode::{BTime, MseedRecord, Samples};
pub use error::{MseedError, Result};
pub use reader::MseedReader;
pub use types::{ByteOrder, EncodingFormat};

pub use decode::decode;
pub use encode::encode;
