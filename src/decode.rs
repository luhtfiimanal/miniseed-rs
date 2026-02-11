//! Decode miniSEED records from raw bytes.
//!
//! The main entry point is [`decode()`], which auto-detects the format version
//! (v2 or v3) and parses a single miniSEED record into an [`MseedRecord`].
//! For multi-record data, see [`MseedReader`](crate::MseedReader).

use crate::record::{MseedRecord, Samples};
use crate::sid::SourceId;
use crate::steim;
use crate::time::{BTime, NanoTime};
use crate::types::{ByteOrder, EncodingFormat, FormatVersion};
use crate::{MseedError, Result};

/// Decode a single miniSEED record from raw bytes.
///
/// Auto-detects the format version:
/// - **v3**: starts with `"MS"` magic bytes and version byte `3`
/// - **v2**: starts with ASCII sequence number and quality indicator
pub fn decode(data: &[u8]) -> Result<MseedRecord> {
    if data.len() < 8 {
        return Err(MseedError::RecordTooShort {
            expected: 8,
            actual: data.len(),
        });
    }

    // Auto-detect: v3 starts with "MS" + version byte 3
    if data[0] == b'M' && data[1] == b'S' && data[2] == 3 {
        return crate::decode_v3::decode_v3(data);
    }

    // v2: bytes 0-5 should be ASCII digits/spaces, byte 6 is quality indicator
    let looks_like_v2 = data[0..6].iter().all(|&b| b.is_ascii_digit() || b == b' ')
        && matches!(data[6], b'D' | b'R' | b'Q' | b'M');

    if !looks_like_v2 {
        return Err(MseedError::UnrecognizedFormat);
    }

    decode_v2(data)
}

/// Decode a single miniSEED v2 record from raw bytes.
fn decode_v2(data: &[u8]) -> Result<MseedRecord> {
    if data.len() < 48 {
        return Err(MseedError::RecordTooShort {
            expected: 48,
            actual: data.len(),
        });
    }

    // Fixed header (48 bytes)
    let sequence_number = std::str::from_utf8(&data[0..6])
        .map_err(|_| MseedError::InvalidHeader)?
        .to_string();
    let quality = data[6] as char;
    let station = std::str::from_utf8(&data[8..13])
        .map_err(|_| MseedError::InvalidHeader)?
        .trim()
        .to_string();
    let location = std::str::from_utf8(&data[13..15])
        .map_err(|_| MseedError::InvalidHeader)?
        .trim()
        .to_string();
    let channel = std::str::from_utf8(&data[15..18])
        .map_err(|_| MseedError::InvalidHeader)?
        .trim()
        .to_string();
    let network = std::str::from_utf8(&data[18..20])
        .map_err(|_| MseedError::InvalidHeader)?
        .trim()
        .to_string();

    // BTIME (bytes 20-29)
    let btime = BTime {
        year: u16::from_be_bytes([data[20], data[21]]),
        day: u16::from_be_bytes([data[22], data[23]]),
        hour: data[24],
        minute: data[25],
        second: data[26],
        // byte 27 is unused
        fract: u16::from_be_bytes([data[28], data[29]]),
    };
    let start_time = NanoTime::from_btime(&btime);

    let num_samples = u16::from_be_bytes([data[30], data[31]]) as usize;
    let sample_rate_factor = i16::from_be_bytes([data[32], data[33]]);
    let sample_rate_multiplier = i16::from_be_bytes([data[34], data[35]]);
    let sample_rate = compute_sample_rate(sample_rate_factor, sample_rate_multiplier);

    let data_offset = u16::from_be_bytes([data[44], data[45]]) as usize;
    let first_blockette = u16::from_be_bytes([data[46], data[47]]) as usize;

    // Find Blockette 1000
    let (encoding, byte_order_val, record_length_power) =
        find_blockette_1000(data, first_blockette)?;

    let byte_order = if byte_order_val == 1 {
        ByteOrder::Big
    } else {
        ByteOrder::Little
    };
    let record_length = 1u32 << record_length_power;

    let encoding_format = EncodingFormat::from_code(encoding)?;

    // Decode data section
    let data_section = &data[data_offset..record_length as usize];
    let samples = decode_data(data_section, encoding_format, num_samples, byte_order)?;

    let source_id = SourceId::from_nslc(&network, &station, &location, &channel);

    Ok(MseedRecord {
        format_version: FormatVersion::V2,
        network,
        station,
        location,
        channel,
        source_id,
        start_time,
        sample_rate,
        encoding: encoding_format,
        samples,
        sequence_number,
        quality,
        byte_order,
        record_length,
        flags: 0,
        publication_version: 0,
        extra_headers: String::new(),
        crc: 0,
    })
}

fn compute_sample_rate(factor: i16, multiplier: i16) -> f64 {
    let f = factor as f64;
    let m = multiplier as f64;
    match (factor > 0, multiplier > 0) {
        (true, true) => f * m,
        (true, false) => -f / m,
        (false, true) => -m / f,
        (false, false) => 1.0 / (f * m),
    }
}

fn find_blockette_1000(data: &[u8], mut offset: usize) -> Result<(u8, u8, u8)> {
    loop {
        if offset + 4 > data.len() {
            return Err(MseedError::MissingBlockette1000);
        }
        let blockette_type = u16::from_be_bytes([data[offset], data[offset + 1]]);
        let next_offset = u16::from_be_bytes([data[offset + 2], data[offset + 3]]) as usize;

        if blockette_type == 1000 {
            if offset + 8 > data.len() {
                return Err(MseedError::MissingBlockette1000);
            }
            let encoding = data[offset + 4];
            let byte_order = data[offset + 5];
            let record_length_power = data[offset + 6];
            return Ok((encoding, byte_order, record_length_power));
        }

        if next_offset == 0 {
            return Err(MseedError::MissingBlockette1000);
        }
        offset = next_offset;
    }
}

pub(crate) fn decode_data(
    data: &[u8],
    encoding: EncodingFormat,
    num_samples: usize,
    byte_order: ByteOrder,
) -> Result<Samples> {
    match encoding {
        EncodingFormat::Int16 => decode_int16(data, num_samples, byte_order),
        EncodingFormat::Int32 => decode_int32(data, num_samples, byte_order),
        EncodingFormat::Float32 => decode_float32(data, num_samples, byte_order),
        EncodingFormat::Float64 => decode_float64(data, num_samples, byte_order),
        EncodingFormat::Steim1 => {
            let samples = steim::decode_steim1(data, num_samples, byte_order)?;
            Ok(Samples::Int(samples))
        }
        EncodingFormat::Steim2 => {
            let samples = steim::decode_steim2(data, num_samples, byte_order)?;
            Ok(Samples::Int(samples))
        }
    }
}

fn decode_int16(data: &[u8], num_samples: usize, byte_order: ByteOrder) -> Result<Samples> {
    let needed = num_samples * 2;
    if data.len() < needed {
        return Err(MseedError::RecordTooShort {
            expected: needed,
            actual: data.len(),
        });
    }
    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let offset = i * 2;
        let val = match byte_order {
            ByteOrder::Big => i16::from_be_bytes([data[offset], data[offset + 1]]),
            ByteOrder::Little => i16::from_le_bytes([data[offset], data[offset + 1]]),
        };
        samples.push(val as i32);
    }
    Ok(Samples::Int(samples))
}

fn decode_int32(data: &[u8], num_samples: usize, byte_order: ByteOrder) -> Result<Samples> {
    let needed = num_samples * 4;
    if data.len() < needed {
        return Err(MseedError::RecordTooShort {
            expected: needed,
            actual: data.len(),
        });
    }
    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let offset = i * 4;
        let bytes = [
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ];
        let val = match byte_order {
            ByteOrder::Big => i32::from_be_bytes(bytes),
            ByteOrder::Little => i32::from_le_bytes(bytes),
        };
        samples.push(val);
    }
    Ok(Samples::Int(samples))
}

fn decode_float32(data: &[u8], num_samples: usize, byte_order: ByteOrder) -> Result<Samples> {
    let needed = num_samples * 4;
    if data.len() < needed {
        return Err(MseedError::RecordTooShort {
            expected: needed,
            actual: data.len(),
        });
    }
    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let offset = i * 4;
        let bytes = [
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ];
        let val = match byte_order {
            ByteOrder::Big => f32::from_be_bytes(bytes),
            ByteOrder::Little => f32::from_le_bytes(bytes),
        };
        samples.push(val);
    }
    Ok(Samples::Float(samples))
}

fn decode_float64(data: &[u8], num_samples: usize, byte_order: ByteOrder) -> Result<Samples> {
    let needed = num_samples * 8;
    if data.len() < needed {
        return Err(MseedError::RecordTooShort {
            expected: needed,
            actual: data.len(),
        });
    }
    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let offset = i * 8;
        let bytes = [
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ];
        let val = match byte_order {
            ByteOrder::Big => f64::from_be_bytes(bytes),
            ByteOrder::Little => f64::from_le_bytes(bytes),
        };
        samples.push(val);
    }
    Ok(Samples::Double(samples))
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
        // Reuse from steim tests via a simple inline impl
        use std::io::Read;

        struct B64Reader<'a> {
            input: &'a [u8],
            pos: usize,
            buf: [u8; 3],
            buf_len: usize,
            buf_pos: usize,
        }

        impl std::io::Read for B64Reader<'_> {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                let mut written = 0;
                while written < out.len() {
                    if self.buf_pos < self.buf_len {
                        out[written] = self.buf[self.buf_pos];
                        self.buf_pos += 1;
                        written += 1;
                        continue;
                    }
                    let mut quad = [0u8; 4];
                    let mut qi = 0;
                    while qi < 4 {
                        if self.pos >= self.input.len() {
                            return Ok(written);
                        }
                        let ch = self.input[self.pos];
                        self.pos += 1;
                        if matches!(ch, b'\n' | b'\r' | b' ') {
                            continue;
                        }
                        quad[qi] = match ch {
                            b'A'..=b'Z' => ch - b'A',
                            b'a'..=b'z' => ch - b'a' + 26,
                            b'0'..=b'9' => ch - b'0' + 52,
                            b'+' => 62,
                            b'/' => 63,
                            _ => 0xFF,
                        };
                        qi += 1;
                    }
                    let pad = quad.iter().filter(|&&v| v == 0xFF).count();
                    let a = quad[0] as u32;
                    let b = quad[1] as u32;
                    let c = if quad[2] == 0xFF { 0 } else { quad[2] as u32 };
                    let d = if quad[3] == 0xFF { 0 } else { quad[3] as u32 };
                    let triple = (a << 18) | (b << 12) | (c << 6) | d;
                    self.buf[0] = (triple >> 16) as u8;
                    self.buf[1] = (triple >> 8) as u8;
                    self.buf[2] = triple as u8;
                    self.buf_len = 3 - pad;
                    self.buf_pos = 0;
                }
                Ok(written)
            }
        }

        let mut reader = B64Reader {
            input: s.as_bytes(),
            pos: 0,
            buf: [0; 3],
            buf_len: 0,
            buf_pos: 0,
        };
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).unwrap();
        buf
    }

    #[test]
    fn test_header_parsing() {
        let vectors = load_vectors("header_vectors.json");
        let arr = vectors.as_array().unwrap();

        for v in arr {
            let name = v["name"].as_str().unwrap();
            let raw = decode_b64(v["record_b64"].as_str().unwrap());
            let expected = &v["expected"];

            let record = decode(&raw).unwrap_or_else(|e| {
                panic!("decode failed for {name}: {e}");
            });

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
            // NanoTime fields: convert from BTime-style expected values
            let btime = record.start_time.to_btime();
            assert_eq!(
                btime.year,
                expected["year"].as_u64().unwrap() as u16,
                "{name}: year"
            );
            assert_eq!(
                btime.day,
                expected["day"].as_u64().unwrap() as u16,
                "{name}: day"
            );
            assert_eq!(
                btime.hour,
                expected["hour"].as_u64().unwrap() as u8,
                "{name}: hour"
            );
            assert_eq!(
                btime.minute,
                expected["minute"].as_u64().unwrap() as u8,
                "{name}: minute"
            );
            assert_eq!(
                btime.second,
                expected["second"].as_u64().unwrap() as u8,
                "{name}: second"
            );
            assert_eq!(
                btime.fract,
                expected["fract"].as_u64().unwrap() as u16,
                "{name}: fract"
            );
            assert_eq!(
                record.encoding.to_code(),
                expected["encoding_format"].as_u64().unwrap() as u8,
                "{name}: encoding"
            );
        }
    }

    #[test]
    fn test_uncompressed_decode() {
        let vectors = load_vectors("uncompressed_vectors.json");
        let arr = vectors.as_array().unwrap();

        for v in arr {
            let name = v["name"].as_str().unwrap();
            let raw = decode_b64(v["record_b64"].as_str().unwrap());

            let record = decode(&raw).unwrap_or_else(|e| {
                panic!("decode failed for {name}: {e}");
            });

            let expected_samples = v["expected_samples"].as_array().unwrap();

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
                            "{name}: float sample {i} mismatch: {a} != {b}"
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
    fn test_4096_record_decode() {
        let vectors = load_vectors("record_4096_vectors.json");
        let arr = vectors.as_array().unwrap();

        for v in arr {
            let name = v["name"].as_str().unwrap();
            let raw = decode_b64(v["record_b64"].as_str().unwrap());
            let expected_len = v["record_length"].as_u64().unwrap() as usize;

            assert_eq!(raw.len(), expected_len, "{name}: raw record length");

            let record = decode(&raw).unwrap_or_else(|e| {
                panic!("decode failed for {name}: {e}");
            });

            assert_eq!(
                record.record_length as usize, expected_len,
                "{name}: decoded record_length"
            );
            assert_eq!(
                record.station,
                v["station"].as_str().unwrap(),
                "{name}: station"
            );

            let expected_samples: Vec<i32> = v["expected_samples"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect();

            match &record.samples {
                Samples::Int(samples) => {
                    assert_eq!(samples, &expected_samples, "{name}: samples mismatch");
                }
                other => panic!("{name}: expected Int samples, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_roundtrip_decode() {
        let vectors = load_vectors("roundtrip_vectors.json");
        let arr = vectors.as_array().unwrap();

        for v in arr {
            let name = v["name"].as_str().unwrap();
            let raw = decode_b64(v["record_b64"].as_str().unwrap());

            let record = decode(&raw).unwrap_or_else(|e| {
                panic!("decode failed for {name}: {e}");
            });

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

            let num = v["num_samples"].as_u64().unwrap() as usize;
            assert_eq!(record.samples.len(), num, "{name}: sample count");
        }
    }

    #[test]
    fn test_autodetect_v3_via_decode() {
        // Verify top-level decode() auto-detects v3 records
        let vectors = load_vectors("v3_header_vectors.json");
        let arr = vectors.as_array().unwrap();

        for v in arr {
            let name = v["name"].as_str().unwrap();
            let raw = decode_b64(v["record_b64"].as_str().unwrap());
            let expected = &v["expected"];

            // Use top-level decode() — should auto-detect v3
            let record = decode(&raw).unwrap_or_else(|e| {
                panic!("auto-detect decode failed for {name}: {e}");
            });

            assert_eq!(
                record.format_version,
                crate::types::FormatVersion::V3,
                "{name}: should be v3"
            );
            assert_eq!(
                record.station,
                expected["station"].as_str().unwrap(),
                "{name}: station"
            );
            assert_eq!(
                record.samples.len(),
                expected["num_samples"].as_u64().unwrap() as usize,
                "{name}: num_samples"
            );
        }
    }

    #[test]
    fn test_autodetect_v2_via_decode() {
        // Verify top-level decode() still auto-detects v2 records
        let vectors = load_vectors("header_vectors.json");
        let v = &vectors.as_array().unwrap()[0];
        let raw = decode_b64(v["record_b64"].as_str().unwrap());

        let record = decode(&raw).unwrap();
        assert_eq!(record.format_version, crate::types::FormatVersion::V2);
    }

    #[test]
    fn test_unrecognized_format() {
        let garbage = vec![0xFF; 64];
        match decode(&garbage) {
            Err(MseedError::UnrecognizedFormat) => {}
            other => panic!("expected UnrecognizedFormat, got: {other:?}"),
        }
    }
}
