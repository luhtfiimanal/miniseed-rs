//! Encode an [`MseedRecord`] into miniSEED record bytes.
//!
//! The main entry point is [`encode()`], which serializes a record struct
//! into a `Vec<u8>`. For v2 records the output is fixed-length (default 512);
//! for v3 records it is variable-length.

use crate::record::{MseedRecord, Samples};
use crate::steim;
use crate::time::BTime;
use crate::types::{ByteOrder, EncodingFormat, FormatVersion};
use crate::{MseedError, Result};

/// Encode a [`MseedRecord`] into miniSEED record bytes.
///
/// Dispatches to v2 or v3 encoding based on [`MseedRecord::format_version`].
pub fn encode(record: &MseedRecord) -> Result<Vec<u8>> {
    match record.format_version {
        FormatVersion::V2 => encode_v2(record),
        FormatVersion::V3 => crate::encode_v3::encode_v3(record),
    }
}

/// Encode a [`MseedRecord`] into miniSEED v2 record bytes.
fn encode_v2(record: &MseedRecord) -> Result<Vec<u8>> {
    let rec_len = record.record_length as usize;
    let rec_len_power = record
        .record_length
        .checked_ilog2()
        .ok_or_else(|| MseedError::EncodeError("record_length must be a power of 2".into()))?
        as u8;

    let mut buf = vec![0u8; rec_len];

    // --- Fixed header (48 bytes) ---

    // Sequence number (bytes 0-5), right-padded with spaces
    let seq = record.sequence_number.as_bytes();
    for (i, &b) in seq.iter().enumerate().take(6) {
        buf[i] = b;
    }
    for slot in buf.iter_mut().take(6).skip(seq.len()) {
        *slot = b' ';
    }

    // Quality indicator (byte 6)
    buf[6] = record.quality as u8;
    // Reserved (byte 7)
    buf[7] = b' ';

    // Station (bytes 8-12), right-padded with spaces
    write_padded(&mut buf[8..13], &record.station);
    // Location (bytes 13-14)
    write_padded(&mut buf[13..15], &record.location);
    // Channel (bytes 15-17)
    write_padded(&mut buf[15..18], &record.channel);
    // Network (bytes 18-19)
    write_padded(&mut buf[18..20], &record.network);

    // BTIME (bytes 20-29) — convert NanoTime to BTime for v2
    let btime = record.start_time.to_btime();
    write_btime(&mut buf[20..30], &btime);

    // Number of samples (bytes 30-31)
    let num_samples = record.samples.len() as u16;
    buf[30..32].copy_from_slice(&num_samples.to_be_bytes());

    // Sample rate factor and multiplier (bytes 32-35)
    let (factor, multiplier) = decompose_sample_rate(record.sample_rate)?;
    buf[32..34].copy_from_slice(&factor.to_be_bytes());
    buf[34..36].copy_from_slice(&multiplier.to_be_bytes());

    // Activity flags (36), I/O flags (37), Data quality flags (38): all 0
    // Number of blockettes (byte 39)
    buf[39] = 1;

    // Time correction (bytes 40-43): 0
    // Beginning of data (bytes 44-45): set after encoding data
    // First blockette offset (bytes 46-47)
    buf[46..48].copy_from_slice(&48u16.to_be_bytes());

    // --- Blockette 1000 (8 bytes at offset 48) ---
    // Blockette type = 1000
    buf[48..50].copy_from_slice(&1000u16.to_be_bytes());
    // Next blockette offset = 0 (no more blockettes)
    buf[50..52].copy_from_slice(&0u16.to_be_bytes());
    // Encoding format
    buf[52] = record.encoding.to_code();
    // Byte order (0=little, 1=big)
    buf[53] = match record.byte_order {
        ByteOrder::Big => 1,
        ByteOrder::Little => 0,
    };
    // Record length power
    buf[54] = rec_len_power;
    // Reserved
    buf[55] = 0;

    // --- Data section ---
    // For Steim: data must start at 64-byte boundary
    // For uncompressed: right after blockette 1000 (offset 56)
    let data_offset: usize = match record.encoding {
        EncodingFormat::Steim1 | EncodingFormat::Steim2 => 64,
        _ => 56,
    };

    // Write data offset into header
    buf[44..46].copy_from_slice(&(data_offset as u16).to_be_bytes());

    let encoded_data = encode_data(&record.samples, record.encoding, record.byte_order)?;

    if data_offset + encoded_data.len() > rec_len {
        return Err(MseedError::EncodeError(format!(
            "encoded data ({} bytes) exceeds record capacity ({} bytes from offset {})",
            encoded_data.len(),
            rec_len - data_offset,
            data_offset,
        )));
    }

    buf[data_offset..data_offset + encoded_data.len()].copy_from_slice(&encoded_data);

    Ok(buf)
}

fn write_padded(dest: &mut [u8], src: &str) {
    let bytes = src.as_bytes();
    for (i, slot) in dest.iter_mut().enumerate() {
        *slot = if i < bytes.len() { bytes[i] } else { b' ' };
    }
}

fn write_btime(dest: &mut [u8], bt: &BTime) {
    dest[0..2].copy_from_slice(&bt.year.to_be_bytes());
    dest[2..4].copy_from_slice(&bt.day.to_be_bytes());
    dest[4] = bt.hour;
    dest[5] = bt.minute;
    dest[6] = bt.second;
    dest[7] = 0; // unused
    dest[8..10].copy_from_slice(&bt.fract.to_be_bytes());
}

/// Decompose a sample rate (Hz) into (factor, multiplier) pair.
fn decompose_sample_rate(rate: f64) -> Result<(i16, i16)> {
    if rate <= 0.0 {
        return Err(MseedError::EncodeError(
            "sample rate must be positive".into(),
        ));
    }

    if rate >= 1.0 {
        // Integer rate: factor = rate, multiplier = 1
        let f = rate.round() as i16;
        Ok((f, 1))
    } else {
        // Sub-hertz: period = 1/rate
        // Use factor < 0, multiplier > 0: rate = -multiplier / factor
        // So factor = -(1/rate), multiplier = 1
        let period = (1.0 / rate).round() as i16;
        Ok((-period, 1))
    }
}

pub(crate) fn encode_data(
    samples: &Samples,
    encoding: EncodingFormat,
    byte_order: ByteOrder,
) -> Result<Vec<u8>> {
    match encoding {
        EncodingFormat::Int16 => encode_int16(samples, byte_order),
        EncodingFormat::Int32 => encode_int32(samples, byte_order),
        EncodingFormat::Float32 => encode_float32(samples, byte_order),
        EncodingFormat::Float64 => encode_float64(samples, byte_order),
        EncodingFormat::Steim1 => {
            let ints = samples_as_int(samples)?;
            steim::encode_steim1(ints, byte_order)
        }
        EncodingFormat::Steim2 => {
            let ints = samples_as_int(samples)?;
            steim::encode_steim2(ints, byte_order)
        }
    }
}

fn samples_as_int(samples: &Samples) -> Result<&[i32]> {
    match samples {
        Samples::Int(v) => Ok(v),
        _ => Err(MseedError::EncodeError(
            "Steim encoding requires integer samples".into(),
        )),
    }
}

fn encode_int16(samples: &Samples, byte_order: ByteOrder) -> Result<Vec<u8>> {
    let ints = match samples {
        Samples::Int(v) => v,
        _ => {
            return Err(MseedError::EncodeError(
                "INT16 encoding requires integer samples".into(),
            ));
        }
    };
    let mut data = Vec::with_capacity(ints.len() * 2);
    for &val in ints {
        let s = val as i16;
        match byte_order {
            ByteOrder::Big => data.extend_from_slice(&s.to_be_bytes()),
            ByteOrder::Little => data.extend_from_slice(&s.to_le_bytes()),
        }
    }
    Ok(data)
}

fn encode_int32(samples: &Samples, byte_order: ByteOrder) -> Result<Vec<u8>> {
    let ints = match samples {
        Samples::Int(v) => v,
        _ => {
            return Err(MseedError::EncodeError(
                "INT32 encoding requires integer samples".into(),
            ));
        }
    };
    let mut data = Vec::with_capacity(ints.len() * 4);
    for &val in ints {
        match byte_order {
            ByteOrder::Big => data.extend_from_slice(&val.to_be_bytes()),
            ByteOrder::Little => data.extend_from_slice(&val.to_le_bytes()),
        }
    }
    Ok(data)
}

fn encode_float32(samples: &Samples, byte_order: ByteOrder) -> Result<Vec<u8>> {
    let floats = match samples {
        Samples::Float(v) => v,
        _ => {
            return Err(MseedError::EncodeError(
                "FLOAT32 encoding requires float samples".into(),
            ));
        }
    };
    let mut data = Vec::with_capacity(floats.len() * 4);
    for &val in floats {
        match byte_order {
            ByteOrder::Big => data.extend_from_slice(&val.to_be_bytes()),
            ByteOrder::Little => data.extend_from_slice(&val.to_le_bytes()),
        }
    }
    Ok(data)
}

fn encode_float64(samples: &Samples, byte_order: ByteOrder) -> Result<Vec<u8>> {
    let doubles = match samples {
        Samples::Double(v) => v,
        _ => {
            return Err(MseedError::EncodeError(
                "FLOAT64 encoding requires double samples".into(),
            ));
        }
    };
    let mut data = Vec::with_capacity(doubles.len() * 8);
    for &val in doubles {
        match byte_order {
            ByteOrder::Big => data.extend_from_slice(&val.to_be_bytes()),
            ByteOrder::Little => data.extend_from_slice(&val.to_le_bytes()),
        }
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode;
    use crate::time::BTime;
    use crate::types::EncodingFormat;
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
            let a = b64_val(bytes[i]);
            let b = b64_val(bytes[i + 1]);
            let c = b64_val(bytes[i + 2]);
            let d = b64_val(bytes[i + 3]);
            let triple = (a << 18) | (b << 12) | (c << 6) | d;
            result.push((triple >> 16) as u8);
            if bytes[i + 2] != b'=' {
                result.push((triple >> 8) as u8);
            }
            if bytes[i + 3] != b'=' {
                result.push(triple as u8);
            }
            i += 4;
        }
        result
    }

    fn b64_val(ch: u8) -> u32 {
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
    fn test_encode_roundtrip() {
        // Decode roundtrip vectors, re-encode, decode again, compare
        let vectors = load_vectors("roundtrip_vectors.json");
        let arr = vectors.as_array().unwrap();

        for v in arr {
            let name = v["name"].as_str().unwrap();
            let raw = decode_b64(v["record_b64"].as_str().unwrap());

            // Decode the original record
            let record = decode::decode(&raw).unwrap_or_else(|e| {
                panic!("initial decode failed for {name}: {e}");
            });

            // Re-encode
            let encoded = encode(&record).unwrap_or_else(|e| {
                panic!("encode failed for {name}: {e}");
            });

            // Decode again
            let record2 = decode::decode(&encoded).unwrap_or_else(|e| {
                panic!("re-decode failed for {name}: {e}");
            });

            // Compare fields
            assert_eq!(record.network, record2.network, "{name}: network");
            assert_eq!(record.station, record2.station, "{name}: station");
            assert_eq!(record.location, record2.location, "{name}: location");
            assert_eq!(record.channel, record2.channel, "{name}: channel");
            assert_eq!(record.quality, record2.quality, "{name}: quality");
            assert_eq!(record.start_time, record2.start_time, "{name}: start_time");
            assert_eq!(
                record.sample_rate, record2.sample_rate,
                "{name}: sample_rate"
            );
            assert_eq!(record.encoding, record2.encoding, "{name}: encoding");
            assert_eq!(record.byte_order, record2.byte_order, "{name}: byte_order");
            assert_eq!(record.samples, record2.samples, "{name}: samples");
        }
    }

    #[test]
    fn test_4096_roundtrip() {
        let vectors = load_vectors("record_4096_vectors.json");
        let arr = vectors.as_array().unwrap();

        for v in arr {
            let name = v["name"].as_str().unwrap();
            let raw = decode_b64(v["record_b64"].as_str().unwrap());

            // Decode the 4096-byte record
            let record = decode::decode(&raw).unwrap_or_else(|e| {
                panic!("initial decode failed for {name}: {e}");
            });
            assert_eq!(record.record_length, 4096, "{name}: record_length");

            // Re-encode
            let encoded = encode(&record).unwrap_or_else(|e| {
                panic!("encode failed for {name}: {e}");
            });
            assert_eq!(encoded.len(), 4096, "{name}: encoded length");

            // Decode again and compare
            let record2 = decode::decode(&encoded).unwrap_or_else(|e| {
                panic!("re-decode failed for {name}: {e}");
            });

            assert_eq!(record.station, record2.station, "{name}: station");
            assert_eq!(
                record.record_length, record2.record_length,
                "{name}: record_length"
            );
            assert_eq!(record.samples, record2.samples, "{name}: samples");
        }
    }

    #[test]
    fn test_encode_4096_from_scratch() {
        use crate::time::NanoTime;

        // Build a 4096-byte record from scratch
        let samples: Vec<i32> = (0..500).collect();
        let record = MseedRecord::new()
            .with_nslc("XX", "BIG", "00", "BHZ")
            .with_start_time(NanoTime {
                year: 2025,
                day: 1,
                hour: 0,
                minute: 0,
                second: 0,
                nanosecond: 0,
            })
            .with_sample_rate(100.0)
            .with_encoding(EncodingFormat::Int32)
            .with_record_length(4096)
            .with_samples(Samples::Int(samples.clone()));

        let encoded = encode(&record).unwrap();
        assert_eq!(encoded.len(), 4096);

        let decoded = decode::decode(&encoded).unwrap();
        assert_eq!(decoded.record_length, 4096);
        assert_eq!(decoded.station, "BIG");
        assert_eq!(decoded.samples, Samples::Int(samples));
    }

    #[test]
    fn test_encode_from_scratch() {
        use crate::time::NanoTime;

        // Build a record from scratch, encode, decode, verify
        let record = MseedRecord::new()
            .with_nslc("XX", "TEST", "00", "BHZ")
            .with_start_time(NanoTime::from_btime(&BTime {
                year: 2025,
                day: 100,
                hour: 12,
                minute: 30,
                second: 45,
                fract: 1234,
            }))
            .with_sample_rate(20.0)
            .with_encoding(EncodingFormat::Int32)
            .with_samples(Samples::Int(vec![1, -2, 3, -4, 100000, -100000]));

        let encoded = encode(&record).unwrap();
        assert_eq!(encoded.len(), 512);

        let decoded = decode::decode(&encoded).unwrap();
        assert_eq!(decoded.network, "XX");
        assert_eq!(decoded.station, "TEST");
        assert_eq!(decoded.location, "00");
        assert_eq!(decoded.channel, "BHZ");
        assert_eq!(decoded.quality, 'D');
        let btime = decoded.start_time.to_btime();
        assert_eq!(btime.year, 2025);
        assert_eq!(btime.day, 100);
        assert_eq!(btime.hour, 12);
        assert_eq!(btime.minute, 30);
        assert_eq!(btime.second, 45);
        assert_eq!(btime.fract, 1234);
        assert_eq!(decoded.sample_rate, 20.0);
        assert_eq!(decoded.encoding, EncodingFormat::Int32);
        assert_eq!(
            decoded.samples,
            Samples::Int(vec![1, -2, 3, -4, 100000, -100000])
        );
    }
}
