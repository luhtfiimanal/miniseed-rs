//! Encode an [`MseedRecord`] into miniSEED v3 record bytes.
//!
//! The v3 format uses a 40-byte fixed header (little-endian), followed by
//! variable-length Source Identifier, optional extra headers (JSON), and
//! data payload. CRC-32C is computed over the entire record.

use crate::crc;
use crate::encode::encode_data;
use crate::record::MseedRecord;
use crate::types::{ByteOrder, EncodingFormat};
use crate::{MseedError, Result};

/// Encode a [`MseedRecord`] into miniSEED v3 record bytes.
pub fn encode_v3(record: &MseedRecord) -> Result<Vec<u8>> {
    let encoding = record.encoding;

    // v3: Steim1/2 are always BE, uncompressed are always LE
    let byte_order = match encoding {
        EncodingFormat::Steim1 | EncodingFormat::Steim2 => ByteOrder::Big,
        _ => ByteOrder::Little,
    };

    // Encode data payload
    let data_payload = encode_data(&record.samples, encoding, byte_order)?;

    // Build SID string
    let sid = record.source_id.as_str();
    let sid_bytes = sid.as_bytes();
    if sid_bytes.len() > 255 {
        return Err(MseedError::EncodeError(format!(
            "SID too long: {} bytes (max 255)",
            sid_bytes.len()
        )));
    }

    let extra_bytes = record.extra_headers.as_bytes();
    if extra_bytes.len() > u16::MAX as usize {
        return Err(MseedError::EncodeError(format!(
            "extra headers too long: {} bytes (max {})",
            extra_bytes.len(),
            u16::MAX
        )));
    }

    let total_length = 40 + sid_bytes.len() + extra_bytes.len() + data_payload.len();
    let mut buf = vec![0u8; total_length];

    // --- Fixed header (40 bytes, little-endian) ---

    // Magic + version
    buf[0] = b'M';
    buf[1] = b'S';
    buf[2] = 3;
    // Flags
    buf[3] = record.flags;
    // Nanosecond (4-7)
    buf[4..8].copy_from_slice(&record.start_time.nanosecond.to_le_bytes());
    // Year (8-9)
    buf[8..10].copy_from_slice(&record.start_time.year.to_le_bytes());
    // Day (10-11)
    buf[10..12].copy_from_slice(&record.start_time.day.to_le_bytes());
    // Hour, Minute, Second (12-14)
    buf[12] = record.start_time.hour;
    buf[13] = record.start_time.minute;
    buf[14] = record.start_time.second;
    // Encoding format (15)
    buf[15] = encoding.to_code();
    // Sample rate as f64 LE (16-23)
    buf[16..24].copy_from_slice(&record.sample_rate.to_le_bytes());
    // Number of samples (24-27)
    buf[24..28].copy_from_slice(&(record.samples.len() as u32).to_le_bytes());
    // CRC (28-31): zeroed, computed after
    // Publication version (32)
    buf[32] = record.publication_version;
    // SID length (33)
    buf[33] = sid_bytes.len() as u8;
    // Extra headers length (34-35)
    buf[34..36].copy_from_slice(&(extra_bytes.len() as u16).to_le_bytes());
    // Data payload length (36-39)
    buf[36..40].copy_from_slice(&(data_payload.len() as u32).to_le_bytes());

    // --- Variable sections ---

    let sid_start = 40;
    buf[sid_start..sid_start + sid_bytes.len()].copy_from_slice(sid_bytes);

    let extra_start = sid_start + sid_bytes.len();
    buf[extra_start..extra_start + extra_bytes.len()].copy_from_slice(extra_bytes);

    let data_start = extra_start + extra_bytes.len();
    buf[data_start..data_start + data_payload.len()].copy_from_slice(&data_payload);

    // Compute and write CRC-32C
    crc::compute_v3_crc(&mut buf);

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode;
    use crate::record::Samples;
    use crate::time::NanoTime;
    use crate::types::FormatVersion;

    #[test]
    fn test_v3_encode_decode_roundtrip_int32() {
        let record = MseedRecord::new_v3()
            .with_nslc("IU", "ANMO", "00", "BHZ")
            .with_start_time(NanoTime {
                year: 2025,
                day: 100,
                hour: 12,
                minute: 30,
                second: 45,
                nanosecond: 500_000_000,
            })
            .with_sample_rate(20.0)
            .with_encoding(EncodingFormat::Int32)
            .with_samples(Samples::Int(vec![1, -2, 3, -4, 100_000, -100_000]));

        let encoded = encode_v3(&record).unwrap();

        // Verify magic bytes
        assert_eq!(&encoded[0..3], b"MS\x03");

        // Decode via auto-detect
        let decoded = decode::decode(&encoded).unwrap();

        assert_eq!(decoded.format_version, FormatVersion::V3);
        assert_eq!(decoded.network, "IU");
        assert_eq!(decoded.station, "ANMO");
        assert_eq!(decoded.location, "00");
        assert_eq!(decoded.channel, "BHZ");
        assert_eq!(decoded.sample_rate, 20.0);
        assert_eq!(decoded.start_time.year, 2025);
        assert_eq!(decoded.start_time.day, 100);
        assert_eq!(decoded.start_time.hour, 12);
        assert_eq!(decoded.start_time.minute, 30);
        assert_eq!(decoded.start_time.second, 45);
        assert_eq!(decoded.start_time.nanosecond, 500_000_000);
        assert_eq!(
            decoded.samples,
            Samples::Int(vec![1, -2, 3, -4, 100_000, -100_000])
        );
    }

    #[test]
    fn test_v3_encode_decode_roundtrip_steim1() {
        let samples: Vec<i32> = (0..100).collect();
        let record = MseedRecord::new_v3()
            .with_nslc("XX", "TEST", "00", "BHZ")
            .with_sample_rate(20.0)
            .with_encoding(EncodingFormat::Steim1)
            .with_samples(Samples::Int(samples.clone()));

        let encoded = encode_v3(&record).unwrap();
        let decoded = decode::decode(&encoded).unwrap();

        assert_eq!(decoded.format_version, FormatVersion::V3);
        assert_eq!(decoded.samples, Samples::Int(samples));
    }

    #[test]
    fn test_v3_encode_decode_roundtrip_steim2() {
        let samples: Vec<i32> = (0..100).collect();
        let record = MseedRecord::new_v3()
            .with_nslc("XX", "TEST", "00", "BHZ")
            .with_sample_rate(20.0)
            .with_encoding(EncodingFormat::Steim2)
            .with_samples(Samples::Int(samples.clone()));

        let encoded = encode_v3(&record).unwrap();
        let decoded = decode::decode(&encoded).unwrap();

        assert_eq!(decoded.format_version, FormatVersion::V3);
        assert_eq!(decoded.samples, Samples::Int(samples));
    }

    #[test]
    fn test_v3_encode_decode_roundtrip_int16() {
        let record = MseedRecord::new_v3()
            .with_nslc("IU", "ANMO", "00", "BHZ")
            .with_sample_rate(20.0)
            .with_encoding(EncodingFormat::Int16)
            .with_samples(Samples::Int(vec![0, 100, -100, 32767, -32768]));

        let encoded = encode_v3(&record).unwrap();
        let decoded = decode::decode(&encoded).unwrap();

        assert_eq!(decoded.format_version, FormatVersion::V3);
        assert_eq!(
            decoded.samples,
            Samples::Int(vec![0, 100, -100, 32767, -32768])
        );
    }

    #[test]
    fn test_v3_encode_decode_roundtrip_float32() {
        let record = MseedRecord::new_v3()
            .with_nslc("IU", "ANMO", "00", "BHZ")
            .with_sample_rate(20.0)
            .with_encoding(EncodingFormat::Float32)
            .with_samples(Samples::Float(vec![0.0, 1.5, -1.5, 3.14]));

        let encoded = encode_v3(&record).unwrap();
        let decoded = decode::decode(&encoded).unwrap();

        assert_eq!(decoded.format_version, FormatVersion::V3);
        match &decoded.samples {
            Samples::Float(s) => {
                assert_eq!(s.len(), 4);
                assert!((s[0] - 0.0).abs() < 1e-6);
                assert!((s[1] - 1.5).abs() < 1e-6);
                assert!((s[2] - (-1.5)).abs() < 1e-6);
                assert!((s[3] - 3.14).abs() < 0.01);
            }
            other => panic!("expected Float, got {other:?}"),
        }
    }

    #[test]
    fn test_v3_encode_decode_roundtrip_float64() {
        let record = MseedRecord::new_v3()
            .with_nslc("IU", "ANMO", "00", "BHZ")
            .with_sample_rate(20.0)
            .with_encoding(EncodingFormat::Float64)
            .with_samples(Samples::Double(vec![0.0, 1.5, -1.5, 3.141592653589793]));

        let encoded = encode_v3(&record).unwrap();
        let decoded = decode::decode(&encoded).unwrap();

        assert_eq!(decoded.format_version, FormatVersion::V3);
        assert_eq!(
            decoded.samples,
            Samples::Double(vec![0.0, 1.5, -1.5, 3.141592653589793])
        );
    }

    #[test]
    fn test_v3_encode_crc_valid() {
        let record = MseedRecord::new_v3()
            .with_nslc("IU", "ANMO", "00", "BHZ")
            .with_sample_rate(20.0)
            .with_encoding(EncodingFormat::Int32)
            .with_samples(Samples::Int(vec![42]));

        let encoded = encode_v3(&record).unwrap();

        // CRC should be valid
        assert!(crc::verify_v3_crc(&encoded));
    }

    #[test]
    fn test_v3_encode_empty_location() {
        let record = MseedRecord::new_v3()
            .with_nslc("JP", "TSK", "", "LHN")
            .with_sample_rate(1.0)
            .with_encoding(EncodingFormat::Int32)
            .with_samples(Samples::Int(vec![1, 2, 3]));

        let encoded = encode_v3(&record).unwrap();
        let decoded = decode::decode(&encoded).unwrap();

        assert_eq!(decoded.network, "JP");
        assert_eq!(decoded.station, "TSK");
        assert_eq!(decoded.location, "");
        assert_eq!(decoded.channel, "LHN");
    }

    #[test]
    fn test_v3_encode_via_top_level_encode() {
        // Verify that top-level encode() dispatches to v3
        let record = MseedRecord::new_v3()
            .with_nslc("IU", "ANMO", "00", "BHZ")
            .with_sample_rate(20.0)
            .with_encoding(EncodingFormat::Int32)
            .with_samples(Samples::Int(vec![1, 2, 3]));

        let encoded = crate::encode::encode(&record).unwrap();

        // Should start with v3 magic
        assert_eq!(&encoded[0..3], b"MS\x03");

        let decoded = decode::decode(&encoded).unwrap();
        assert_eq!(decoded.format_version, FormatVersion::V3);
        assert_eq!(decoded.samples, Samples::Int(vec![1, 2, 3]));
    }
}
