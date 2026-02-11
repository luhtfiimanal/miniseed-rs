//! Unified miniSEED record type for v2 and v3.
//!
//! [`MseedRecord`] represents a single decoded miniSEED record. It supports
//! both v2 and v3 formats through a unified struct, similar to libmseed's
//! `MS3Record`.

use std::fmt;

use crate::sid::SourceId;
use crate::time::{BTime, NanoTime};
use crate::types::{ByteOrder, EncodingFormat, FormatVersion};

/// A decoded miniSEED record (v2 or v3).
///
/// All fields are populated for both versions. Version-specific fields
/// use sensible defaults when not applicable:
/// - v2 records get `source_id` synthesized from NSLC codes
/// - v3 records get `sequence_number = "000000"`, `quality = 'D'`, etc.
#[derive(Debug, Clone, PartialEq)]
pub struct MseedRecord {
    // --- Format ---
    /// The miniSEED format version of this record.
    pub format_version: FormatVersion,

    // --- Identification (always populated) ---
    pub network: String,
    pub station: String,
    pub location: String,
    pub channel: String,
    /// FDSN Source Identifier. Synthesized from NSLC for v2 records.
    pub source_id: SourceId,

    // --- Time + Data ---
    /// Record start time with nanosecond precision.
    pub start_time: NanoTime,
    pub sample_rate: f64,
    pub encoding: EncodingFormat,
    pub samples: Samples,

    // --- v2-specific (defaults for v3) ---
    pub sequence_number: String,
    pub quality: char,
    pub byte_order: ByteOrder,
    /// Record length in bytes. Was `u16` in v0.1; now `u32` to support v3.
    pub record_length: u32,

    // --- v3-specific (defaults for v2) ---
    pub flags: u8,
    pub publication_version: u8,
    pub extra_headers: String,
    pub crc: u32,
}

impl MseedRecord {
    /// Create a new `MseedRecord` with v2 defaults.
    ///
    /// Defaults: sequence "000001", quality 'D', empty NSLC,
    /// big-endian, 512-byte records, INT32, no samples.
    pub fn new() -> Self {
        Self {
            format_version: FormatVersion::V2,
            network: String::new(),
            station: String::new(),
            location: String::new(),
            channel: String::new(),
            source_id: SourceId::from_nslc("", "", "", ""),
            start_time: NanoTime::epoch(),
            sample_rate: 1.0,
            encoding: EncodingFormat::Int32,
            samples: Samples::Int(vec![]),
            sequence_number: "000001".into(),
            quality: 'D',
            byte_order: ByteOrder::Big,
            record_length: 512,
            flags: 0,
            publication_version: 0,
            extra_headers: String::new(),
            crc: 0,
        }
    }

    /// Create a new `MseedRecord` with v3 defaults.
    pub fn new_v3() -> Self {
        Self {
            format_version: FormatVersion::V3,
            network: String::new(),
            station: String::new(),
            location: String::new(),
            channel: String::new(),
            source_id: SourceId::from_nslc("", "", "", ""),
            start_time: NanoTime::epoch(),
            sample_rate: 1.0,
            encoding: EncodingFormat::Int32,
            samples: Samples::Int(vec![]),
            sequence_number: "000000".into(),
            quality: 'D',
            byte_order: ByteOrder::Little,
            record_length: 0, // v3: variable length, set during encode
            flags: 0,
            publication_version: 0,
            extra_headers: String::new(),
            crc: 0,
        }
    }

    /// Set network, station, location, and channel codes.
    /// Also updates `source_id` to match.
    pub fn with_nslc(
        mut self,
        network: &str,
        station: &str,
        location: &str,
        channel: &str,
    ) -> Self {
        self.network = network.into();
        self.station = station.into();
        self.location = location.into();
        self.channel = channel.into();
        self.source_id = SourceId::from_nslc(network, station, location, channel);
        self
    }

    /// Set the start time (NanoTime).
    pub fn with_start_time(mut self, time: NanoTime) -> Self {
        self.start_time = time;
        self
    }

    /// Set the start time from a legacy BTime value.
    pub fn with_start_time_btime(mut self, bt: BTime) -> Self {
        self.start_time = NanoTime::from_btime(&bt);
        self
    }

    /// Set the sample rate in Hz.
    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Set the encoding format.
    pub fn with_encoding(mut self, enc: EncodingFormat) -> Self {
        self.encoding = enc;
        self
    }

    /// Set the sample data.
    pub fn with_samples(mut self, samples: Samples) -> Self {
        self.samples = samples;
        self
    }

    /// Set the record length (power of 2 for v2, any value for v3).
    pub fn with_record_length(mut self, len: u32) -> Self {
        self.record_length = len;
        self
    }

    /// Return the NSLC identifier: `"NET.STA.LOC.CHA"`.
    pub fn nslc(&self) -> String {
        format!(
            "{}.{}.{}.{}",
            self.network, self.station, self.location, self.channel
        )
    }
}

impl Default for MseedRecord {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MseedRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} | {} | {} Hz | {} samples ({}) [{}]",
            self.nslc(),
            self.start_time,
            self.sample_rate,
            self.samples.len(),
            self.encoding,
            self.format_version,
        )
    }
}

/// Decoded sample data.
#[derive(Debug, Clone, PartialEq)]
pub enum Samples {
    Int(Vec<i32>),
    Float(Vec<f32>),
    Double(Vec<f64>),
}

impl Samples {
    pub fn len(&self) -> usize {
        match self {
            Samples::Int(v) => v.len(),
            Samples::Float(v) => v.len(),
            Samples::Double(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
