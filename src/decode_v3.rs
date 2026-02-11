//! Decode miniSEED v3 records from raw bytes.
//!
//! The v3 format uses a 40-byte fixed header (little-endian), followed by
//! variable-length Source Identifier, optional extra headers (JSON), and
//! data payload. CRC-32C integrity checking is performed automatically.

use crate::crc;
use crate::decode::decode_data;
use crate::record::{MseedRecord, Samples};
use crate::sid::SourceId;
use crate::time::NanoTime;
use crate::types::{ByteOrder, EncodingFormat, FormatVersion};
use crate::{MseedError, Result};

/// Minimum size of a v3 fixed header.
const V3_HEADER_SIZE: usize = 40;

/// Decode a single miniSEED v3 record from raw bytes.
pub fn decode_v3(data: &[u8]) -> Result<MseedRecord> {
    if data.len() < V3_HEADER_SIZE {
        return Err(MseedError::RecordTooShort {
            expected: V3_HEADER_SIZE,
            actual: data.len(),
        });
    }

    // Verify magic bytes: 'M', 'S', 3
    if data[0] != b'M' || data[1] != b'S' || data[2] != 3 {
        return Err(MseedError::InvalidV3Header(
            "missing 'MS' magic or version != 3".into(),
        ));
    }

    let flags = data[3];
    let nanosecond = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let year = u16::from_le_bytes([data[8], data[9]]);
    let day = u16::from_le_bytes([data[10], data[11]]);
    let hour = data[12];
    let minute = data[13];
    let second = data[14];
    let encoding_code = data[15];
    let sample_rate = f64::from_le_bytes(
        data[16..24]
            .try_into()
            .map_err(|_| MseedError::InvalidV3Header("bad sample rate bytes".into()))?,
    );
    let num_samples = u32::from_le_bytes([data[24], data[25], data[26], data[27]]) as usize;
    let _crc_stored = u32::from_le_bytes([data[28], data[29], data[30], data[31]]);
    let pub_version = data[32];
    let sid_length = data[33] as usize;
    let extra_length = u16::from_le_bytes([data[34], data[35]]) as usize;
    let data_length = u32::from_le_bytes([data[36], data[37], data[38], data[39]]) as usize;

    let total_length = V3_HEADER_SIZE + sid_length + extra_length + data_length;

    if data.len() < total_length {
        return Err(MseedError::RecordTooShort {
            expected: total_length,
            actual: data.len(),
        });
    }

    let record_bytes = &data[..total_length];

    // Verify CRC-32C
    if !crc::verify_v3_crc(record_bytes) {
        let stored = u32::from_le_bytes([data[28], data[29], data[30], data[31]]);
        // Compute what the CRC should be
        let mut buf = record_bytes.to_vec();
        buf[28] = 0;
        buf[29] = 0;
        buf[30] = 0;
        buf[31] = 0;
        let computed = crc::crc32c(&buf);
        return Err(MseedError::CrcMismatch { stored, computed });
    }

    // Parse Source Identifier
    let sid_start = V3_HEADER_SIZE;
    let sid_end = sid_start + sid_length;
    let sid_str = std::str::from_utf8(&data[sid_start..sid_end])
        .map_err(|_| MseedError::InvalidV3Header("invalid UTF-8 in SID".into()))?;
    let source_id = SourceId::parse(sid_str);
    let (network, station, location, channel) = source_id.to_nslc();

    // Parse extra headers (JSON, if present)
    let extra_start = sid_end;
    let extra_end = extra_start + extra_length;
    let extra_headers = if extra_length > 0 {
        std::str::from_utf8(&data[extra_start..extra_end])
            .map_err(|_| MseedError::InvalidV3Header("invalid UTF-8 in extra headers".into()))?
            .to_string()
    } else {
        String::new()
    };

    // Decode data payload
    let data_start = extra_end;
    let data_section = &data[data_start..data_start + data_length];
    let encoding = EncodingFormat::from_code(encoding_code)?;

    // v3: Steim1/2 are always BE, uncompressed are always LE
    let byte_order = match encoding {
        EncodingFormat::Steim1 | EncodingFormat::Steim2 => ByteOrder::Big,
        _ => ByteOrder::Little,
    };

    let samples = if num_samples == 0 && data_length == 0 {
        Samples::Int(vec![])
    } else {
        decode_data(data_section, encoding, num_samples, byte_order)?
    };

    Ok(MseedRecord {
        format_version: FormatVersion::V3,
        network,
        station,
        location,
        channel,
        source_id,
        start_time: NanoTime {
            year,
            day,
            hour,
            minute,
            second,
            nanosecond,
        },
        sample_rate,
        encoding,
        samples,
        sequence_number: "000000".into(),
        quality: 'D',
        byte_order,
        record_length: total_length as u32,
        flags,
        publication_version: pub_version,
        extra_headers,
        crc: _crc_stored,
    })
}

/// Peek at a v3 record to determine its total length.
///
/// Requires at least 40 bytes (the fixed header).
pub(crate) fn peek_v3_record_length(data: &[u8]) -> Result<usize> {
    if data.len() < V3_HEADER_SIZE {
        return Err(MseedError::RecordTooShort {
            expected: V3_HEADER_SIZE,
            actual: data.len(),
        });
    }
    let sid_length = data[33] as usize;
    let extra_length = u16::from_le_bytes([data[34], data[35]]) as usize;
    let data_length = u32::from_le_bytes([data[36], data[37], data[38], data[39]]) as usize;
    Ok(V3_HEADER_SIZE + sid_length + extra_length + data_length)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn load_vectors(filename: &str) -> serde_json::Value {
        let vectors_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("pyscripts")
            .join("test_vectors");
        let path = vectors_dir.join(filename);
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
        serde_json::from_str(&content).unwrap()
    }

    fn decode_b64(s: &str) -> Vec<u8> {
        let mut result = Vec::new();
        let bytes: Vec<u8> = s
            .bytes()
            .filter(|b| !matches!(b, b'\n' | b'\r' | b' '))
            .collect();
        let mut i = 0;
        while i + 3 < bytes.len() {
            let a = b64val(bytes[i]);
            let b = b64val(bytes[i + 1]);
            let c = b64val(bytes[i + 2]);
            let d = b64val(bytes[i + 3]);
            let pad_c = bytes[i + 2] == b'=';
            let pad_d = bytes[i + 3] == b'=';
            let ca = if pad_c { 0u32 } else { c };
            let da = if pad_d { 0u32 } else { d };
            let triple = (a << 18) | (b << 12) | (ca << 6) | da;
            result.push((triple >> 16) as u8);
            if !pad_c {
                result.push((triple >> 8) as u8);
            }
            if !pad_d {
                result.push(triple as u8);
            }
            i += 4;
        }
        result
    }

    fn b64val(ch: u8) -> u32 {
        match ch {
            b'A'..=b'Z' => (ch - b'A') as u32,
            b'a'..=b'z' => (ch - b'a' + 26) as u32,
            b'0'..=b'9' => (ch - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => 0,
        }
    }

    #[test]
    fn test_v3_header_parsing() {
        let vectors = load_vectors("v3_header_vectors.json");
        let arr = vectors.as_array().unwrap();

        for v in arr {
            let name = v["name"].as_str().unwrap();
            let raw = decode_b64(v["record_b64"].as_str().unwrap());
            let expected = &v["expected"];

            let record = decode_v3(&raw).unwrap_or_else(|e| {
                panic!("decode_v3 failed for {name}: {e}");
            });

            assert_eq!(record.format_version, FormatVersion::V3, "{name}: version");
            assert_eq!(
                record.source_id.as_str(),
                expected["sid"].as_str().unwrap(),
                "{name}: sid"
            );
            assert_eq!(
                record.network,
                expected["network"].as_str().unwrap(),
                "{name}: network"
            );
            assert_eq!(
                record.station,
                expected["station"].as_str().unwrap(),
                "{name}: station"
            );
            assert_eq!(
                record.location,
                expected["location"].as_str().unwrap(),
                "{name}: location"
            );
            assert_eq!(
                record.channel,
                expected["channel"].as_str().unwrap(),
                "{name}: channel"
            );
            assert_eq!(
                record.sample_rate,
                expected["sample_rate"].as_f64().unwrap(),
                "{name}: sample_rate"
            );
            assert_eq!(
                record.start_time.year,
                expected["year"].as_u64().unwrap() as u16,
                "{name}: year"
            );
            assert_eq!(
                record.start_time.day,
                expected["day"].as_u64().unwrap() as u16,
                "{name}: day"
            );
            assert_eq!(
                record.start_time.hour,
                expected["hour"].as_u64().unwrap() as u8,
                "{name}: hour"
            );
            assert_eq!(
                record.start_time.minute,
                expected["minute"].as_u64().unwrap() as u8,
                "{name}: minute"
            );
            assert_eq!(
                record.start_time.second,
                expected["second"].as_u64().unwrap() as u8,
                "{name}: second"
            );
            assert_eq!(
                record.start_time.nanosecond,
                expected["nanosecond"].as_u64().unwrap() as u32,
                "{name}: nanosecond"
            );
            assert_eq!(
                record.encoding.to_code(),
                expected["encoding_format"].as_u64().unwrap() as u8,
                "{name}: encoding"
            );
            assert_eq!(
                record.samples.len(),
                expected["num_samples"].as_u64().unwrap() as usize,
                "{name}: num_samples"
            );
            assert_eq!(
                record.flags,
                expected["flags"].as_u64().unwrap() as u8,
                "{name}: flags"
            );
            assert_eq!(
                record.publication_version,
                expected["pub_version"].as_u64().unwrap() as u8,
                "{name}: pub_version"
            );
            assert_eq!(
                record.record_length,
                v["record_length"].as_u64().unwrap() as u32,
                "{name}: record_length"
            );
        }
    }

    #[test]
    fn test_v3_steim1_decode() {
        let vectors = load_vectors("v3_steim1_vectors.json");
        let arr = vectors.as_array().unwrap();

        for v in arr {
            let name = v["name"].as_str().unwrap();
            let raw = decode_b64(v["record_b64"].as_str().unwrap());

            let record = decode_v3(&raw).unwrap_or_else(|e| {
                panic!("decode_v3 failed for {name}: {e}");
            });

            let expected_samples: Vec<i32> = v["expected_samples"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect();

            assert_eq!(record.encoding, EncodingFormat::Steim1, "{name}: encoding");
            match &record.samples {
                Samples::Int(samples) => {
                    assert_eq!(
                        samples.len(),
                        expected_samples.len(),
                        "{name}: sample count"
                    );
                    assert_eq!(samples, &expected_samples, "{name}: samples mismatch");
                }
                other => panic!("{name}: expected Int samples, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_v3_steim2_decode() {
        let vectors = load_vectors("v3_steim2_vectors.json");
        let arr = vectors.as_array().unwrap();

        for v in arr {
            let name = v["name"].as_str().unwrap();
            let raw = decode_b64(v["record_b64"].as_str().unwrap());

            let record = decode_v3(&raw).unwrap_or_else(|e| {
                panic!("decode_v3 failed for {name}: {e}");
            });

            let expected_samples: Vec<i32> = v["expected_samples"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect();

            assert_eq!(record.encoding, EncodingFormat::Steim2, "{name}: encoding");
            match &record.samples {
                Samples::Int(samples) => {
                    assert_eq!(
                        samples.len(),
                        expected_samples.len(),
                        "{name}: sample count"
                    );
                    assert_eq!(samples, &expected_samples, "{name}: samples mismatch");
                }
                other => panic!("{name}: expected Int samples, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_v3_uncompressed_decode() {
        let vectors = load_vectors("v3_uncompressed_vectors.json");
        let arr = vectors.as_array().unwrap();

        for v in arr {
            let name = v["name"].as_str().unwrap();
            let raw = decode_b64(v["record_b64"].as_str().unwrap());

            let record = decode_v3(&raw).unwrap_or_else(|e| {
                panic!("decode_v3 failed for {name}: {e}");
            });

            let expected_samples = v["expected_samples"].as_array().unwrap();

            assert_eq!(
                record.record_length,
                v["record_length"].as_u64().unwrap() as u32,
                "{name}: record_length"
            );

            match &record.samples {
                Samples::Int(samples) => {
                    let expected: Vec<i32> = expected_samples
                        .iter()
                        .map(|x| x.as_i64().unwrap() as i32)
                        .collect();
                    assert_eq!(samples, &expected, "{name}: int samples mismatch");
                }
                Samples::Float(samples) => {
                    let expected: Vec<f32> = expected_samples
                        .iter()
                        .map(|x| x.as_f64().unwrap() as f32)
                        .collect();
                    assert_eq!(samples.len(), expected.len(), "{name}: float sample count");
                    for (i, (a, b)) in samples.iter().zip(expected.iter()).enumerate() {
                        assert!(
                            (a - b).abs() < 1e-6 * b.abs().max(1.0),
                            "{name}: float sample {i}: {a} != {b}"
                        );
                    }
                }
                Samples::Double(samples) => {
                    let expected: Vec<f64> = expected_samples
                        .iter()
                        .map(|x| x.as_f64().unwrap())
                        .collect();
                    assert_eq!(samples, &expected, "{name}: double samples mismatch");
                }
            }
        }
    }

    #[test]
    fn test_v3_crc_verification() {
        let vectors = load_vectors("v3_crc_vectors.json");
        let arr = vectors.as_array().unwrap();

        for v in arr {
            let name = v["name"].as_str().unwrap();
            let raw = decode_b64(v["record_b64"].as_str().unwrap());
            let expected_crc = v["crc"].as_u64().unwrap() as u32;

            // Verify CRC matches expected
            let stored_crc = u32::from_le_bytes([raw[28], raw[29], raw[30], raw[31]]);
            assert_eq!(stored_crc, expected_crc, "{name}: stored CRC mismatch");

            // Verify CRC passes validation
            assert!(crc::verify_v3_crc(&raw), "{name}: CRC verification failed");

            // Decode should succeed (CRC valid)
            let record = decode_v3(&raw).unwrap_or_else(|e| {
                panic!("decode_v3 failed for {name}: {e}");
            });
            assert_eq!(record.crc, expected_crc, "{name}: record.crc");
        }
    }

    #[test]
    fn test_v3_crc_corruption_detected() {
        let vectors = load_vectors("v3_crc_vectors.json");
        let v = &vectors.as_array().unwrap()[0];
        let mut raw = decode_b64(v["record_b64"].as_str().unwrap());

        // Corrupt a data byte
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;

        match decode_v3(&raw) {
            Err(MseedError::CrcMismatch { .. }) => {} // expected
            Ok(_) => panic!("expected CrcMismatch error for corrupted data"),
            Err(e) => panic!("expected CrcMismatch, got: {e}"),
        }
    }

    #[test]
    fn test_v3_roundtrip_decode() {
        let vectors = load_vectors("v3_roundtrip_vectors.json");
        let arr = vectors.as_array().unwrap();

        for v in arr {
            let name = v["name"].as_str().unwrap();
            let raw = decode_b64(v["record_b64"].as_str().unwrap());

            let record = decode_v3(&raw).unwrap_or_else(|e| {
                panic!("decode_v3 failed for {name}: {e}");
            });

            assert_eq!(record.format_version, FormatVersion::V3, "{name}: version");
            assert_eq!(
                record.network,
                v["network"].as_str().unwrap(),
                "{name}: network"
            );
            assert_eq!(
                record.station,
                v["station"].as_str().unwrap(),
                "{name}: station"
            );
            assert_eq!(
                record.location,
                v["location"].as_str().unwrap(),
                "{name}: location"
            );
            assert_eq!(
                record.channel,
                v["channel"].as_str().unwrap(),
                "{name}: channel"
            );
            assert_eq!(
                record.sample_rate,
                v["sample_rate"].as_f64().unwrap(),
                "{name}: sample_rate"
            );
            assert_eq!(
                record.samples.len(),
                v["num_samples"].as_u64().unwrap() as usize,
                "{name}: num_samples"
            );
            assert_eq!(
                record.record_length,
                v["record_length"].as_u64().unwrap() as u32,
                "{name}: record_length"
            );

            let expected_samples = v["expected_samples"].as_array().unwrap();
            match &record.samples {
                Samples::Int(samples) => {
                    let expected: Vec<i32> = expected_samples
                        .iter()
                        .map(|x| x.as_i64().unwrap() as i32)
                        .collect();
                    assert_eq!(samples, &expected, "{name}: samples mismatch");
                }
                Samples::Float(samples) => {
                    let expected: Vec<f32> = expected_samples
                        .iter()
                        .map(|x| x.as_f64().unwrap() as f32)
                        .collect();
                    for (i, (a, b)) in samples.iter().zip(expected.iter()).enumerate() {
                        assert!(
                            (a - b).abs() < 1e-6 * b.abs().max(1.0),
                            "{name}: float sample {i}: {a} != {b}"
                        );
                    }
                }
                Samples::Double(samples) => {
                    let expected: Vec<f64> = expected_samples
                        .iter()
                        .map(|x| x.as_f64().unwrap())
                        .collect();
                    assert_eq!(samples, &expected, "{name}: double samples mismatch");
                }
            }
        }
    }
}
