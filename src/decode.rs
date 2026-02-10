use crate::steim::{self, ByteOrder};
use crate::{MseedError, Result};

/// Decoded miniSEED v2 record.
#[derive(Debug, Clone)]
pub struct MseedRecord {
    pub sequence_number: String,
    pub quality: char,
    pub station: String,
    pub location: String,
    pub channel: String,
    pub network: String,
    pub start_time: BTime,
    pub sample_rate: f64,
    pub encoding: u8,
    pub byte_order: ByteOrder,
    pub record_length: u16,
    pub samples: Samples,
}

/// BTIME timestamp (10 bytes in the fixed header).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BTime {
    pub year: u16,
    pub day: u16,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub fract: u16, // 0.0001 second units
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

/// Decode a single miniSEED v2 record from raw bytes.
pub fn decode(data: &[u8]) -> Result<MseedRecord> {
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
    let start_time = BTime {
        year: u16::from_be_bytes([data[20], data[21]]),
        day: u16::from_be_bytes([data[22], data[23]]),
        hour: data[24],
        minute: data[25],
        second: data[26],
        // byte 27 is unused
        fract: u16::from_be_bytes([data[28], data[29]]),
    };

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
    let record_length = 1u16 << record_length_power;

    // Decode data section
    let data_section = &data[data_offset..record_length as usize];
    let samples = decode_data(data_section, encoding, num_samples, byte_order)?;

    Ok(MseedRecord {
        sequence_number,
        quality,
        station,
        location,
        channel,
        network,
        start_time,
        sample_rate,
        encoding,
        byte_order,
        record_length,
        samples,
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

fn decode_data(
    data: &[u8],
    encoding: u8,
    num_samples: usize,
    byte_order: ByteOrder,
) -> Result<Samples> {
    match encoding {
        1 => decode_int16(data, num_samples, byte_order),
        3 => decode_int32(data, num_samples, byte_order),
        4 => decode_float32(data, num_samples, byte_order),
        5 => decode_float64(data, num_samples, byte_order),
        10 => {
            let samples = steim::decode_steim1(data, num_samples, byte_order)?;
            Ok(Samples::Int(samples))
        }
        11 => {
            let samples = steim::decode_steim2(data, num_samples, byte_order)?;
            Ok(Samples::Int(samples))
        }
        _ => Err(MseedError::UnsupportedEncoding(encoding)),
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
                record.start_time.fract,
                expected["fract"].as_u64().unwrap() as u16,
                "{name}: fract"
            );
            assert_eq!(
                record.encoding,
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
}
