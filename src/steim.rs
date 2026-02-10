//! Steim1 and Steim2 compression and decompression.
//!
//! These are differential integer compression schemes used in seismological
//! data (SEED/miniSEED format). See Appendix B of the SEED Manual v2.4.

use crate::types::ByteOrder;
use crate::{MseedError, Result};

#[deprecated(note = "use `miniseed::ByteOrder` (re-exported from `types`) instead")]
pub type ByteOrderAlias = ByteOrder;

const FRAME_SIZE: usize = 64; // 16 x 32-bit words
const WORDS_PER_FRAME: usize = 16;

fn read_u32(data: &[u8], offset: usize, byte_order: ByteOrder) -> u32 {
    let bytes = [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ];
    match byte_order {
        ByteOrder::Big => u32::from_be_bytes(bytes),
        ByteOrder::Little => u32::from_le_bytes(bytes),
    }
}

fn extract_nibble(control_word: u32, word_index: usize) -> u8 {
    let shift = 30 - (word_index * 2);
    ((control_word >> shift) & 0x03) as u8
}

fn sign_extend(value: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    (value as i32).wrapping_shl(shift).wrapping_shr(shift)
}

/// Apply diffs using an accumulator, pushing each result into samples.
fn apply_diffs_1(word: u32, nibble: u8, acc: &mut i32, samples: &mut Vec<i32>, num_samples: usize) {
    match nibble {
        0b00 => {} // no data
        0b01 => {
            // four 8-bit signed diffs
            for i in 0..4u32 {
                if samples.len() >= num_samples {
                    break;
                }
                let diff = sign_extend((word >> (24 - i * 8)) & 0xFF, 8);
                *acc += diff;
                samples.push(*acc);
            }
        }
        0b10 => {
            // two 16-bit signed diffs
            for i in 0..2u32 {
                if samples.len() >= num_samples {
                    break;
                }
                let diff = sign_extend((word >> (16 - i * 16)) & 0xFFFF, 16);
                *acc += diff;
                samples.push(*acc);
            }
        }
        0b11 => {
            // one 32-bit signed diff
            if samples.len() < num_samples {
                let diff = word as i32;
                *acc += diff;
                samples.push(*acc);
            }
        }
        _ => unreachable!(),
    }
}

/// Decode Steim1 compressed data into i32 samples.
///
/// `data` must be frame-aligned (multiple of 64 bytes).
/// Frame 0, word 1 = X₀ (forward integration constant / accumulator seed).
/// Frame 0, word 2 = Xₙ (reverse integration constant, for validation).
pub fn decode_steim1(data: &[u8], num_samples: usize, byte_order: ByteOrder) -> Result<Vec<i32>> {
    if !data.len().is_multiple_of(FRAME_SIZE) {
        return Err(MseedError::SteimDecode(format!(
            "data length {} not a multiple of frame size {}",
            data.len(),
            FRAME_SIZE
        )));
    }

    let num_frames = data.len() / FRAME_SIZE;
    if num_frames == 0 {
        return Err(MseedError::SteimDecode("no frames in data".into()));
    }

    // X₀ = forward integration constant (accumulator seed)
    let x0 = read_u32(data, 4, byte_order) as i32;
    let mut acc = x0;
    let mut samples = Vec::with_capacity(num_samples);

    for frame_idx in 0..num_frames {
        let frame_offset = frame_idx * FRAME_SIZE;
        let control_word = read_u32(data, frame_offset, byte_order);

        for word_idx in 1..WORDS_PER_FRAME {
            if samples.len() >= num_samples {
                break;
            }

            // Skip X₀ and Xₙ words in frame 0
            if frame_idx == 0 && (word_idx == 1 || word_idx == 2) {
                continue;
            }

            let word_offset = frame_offset + word_idx * 4;
            let word = read_u32(data, word_offset, byte_order);
            let nibble = extract_nibble(control_word, word_idx);

            apply_diffs_1(word, nibble, &mut acc, &mut samples, num_samples);
        }
    }

    if samples.len() != num_samples {
        return Err(MseedError::SampleCountMismatch {
            expected: num_samples,
            actual: samples.len(),
        });
    }

    Ok(samples)
}

/// Decode Steim2 compressed data into i32 samples.
///
/// Extends Steim1 with additional packing formats using "dnib" (bits 31-30 of data word).
pub fn decode_steim2(data: &[u8], num_samples: usize, byte_order: ByteOrder) -> Result<Vec<i32>> {
    if !data.len().is_multiple_of(FRAME_SIZE) {
        return Err(MseedError::SteimDecode(format!(
            "data length {} not a multiple of frame size {}",
            data.len(),
            FRAME_SIZE
        )));
    }

    let num_frames = data.len() / FRAME_SIZE;
    if num_frames == 0 {
        return Err(MseedError::SteimDecode("no frames in data".into()));
    }

    let x0 = read_u32(data, 4, byte_order) as i32;
    let mut acc = x0;
    let mut samples = Vec::with_capacity(num_samples);

    for frame_idx in 0..num_frames {
        let frame_offset = frame_idx * FRAME_SIZE;
        let control_word = read_u32(data, frame_offset, byte_order);

        for word_idx in 1..WORDS_PER_FRAME {
            if samples.len() >= num_samples {
                break;
            }

            if frame_idx == 0 && (word_idx == 1 || word_idx == 2) {
                continue;
            }

            let word_offset = frame_offset + word_idx * 4;
            let word = read_u32(data, word_offset, byte_order);
            let nibble = extract_nibble(control_word, word_idx);

            steim2_apply_diffs(word, nibble, &mut acc, &mut samples, num_samples)?;
        }
    }

    if samples.len() != num_samples {
        return Err(MseedError::SampleCountMismatch {
            expected: num_samples,
            actual: samples.len(),
        });
    }

    Ok(samples)
}

fn steim2_apply_diffs(
    word: u32,
    nibble: u8,
    acc: &mut i32,
    samples: &mut Vec<i32>,
    num_samples: usize,
) -> Result<()> {
    let dnib = ((word >> 30) & 0x03) as u8;

    match nibble {
        0b00 => {} // no data
        0b01 => {
            // four 8-bit signed diffs (same as Steim1)
            for i in 0..4u32 {
                if samples.len() >= num_samples {
                    break;
                }
                let diff = sign_extend((word >> (24 - i * 8)) & 0xFF, 8);
                *acc += diff;
                samples.push(*acc);
            }
        }
        0b10 => match dnib {
            0b01 => {
                // one 30-bit diff
                if samples.len() < num_samples {
                    let diff = sign_extend(word & 0x3FFF_FFFF, 30);
                    *acc += diff;
                    samples.push(*acc);
                }
            }
            0b10 => {
                // two 15-bit diffs
                for i in 0..2u32 {
                    if samples.len() >= num_samples {
                        break;
                    }
                    let diff = sign_extend((word >> (15 - i * 15)) & 0x7FFF, 15);
                    *acc += diff;
                    samples.push(*acc);
                }
            }
            0b11 => {
                // three 10-bit diffs
                for i in 0..3u32 {
                    if samples.len() >= num_samples {
                        break;
                    }
                    let diff = sign_extend((word >> (20 - i * 10)) & 0x3FF, 10);
                    *acc += diff;
                    samples.push(*acc);
                }
            }
            _ => {
                return Err(MseedError::SteimDecode(format!(
                    "steim2 nibble=10 invalid dnib={dnib}"
                )));
            }
        },
        0b11 => match dnib {
            0b00 => {
                // five 6-bit diffs
                for i in 0..5u32 {
                    if samples.len() >= num_samples {
                        break;
                    }
                    let diff = sign_extend((word >> (24 - i * 6)) & 0x3F, 6);
                    *acc += diff;
                    samples.push(*acc);
                }
            }
            0b01 => {
                // six 5-bit diffs
                for i in 0..6u32 {
                    if samples.len() >= num_samples {
                        break;
                    }
                    let diff = sign_extend((word >> (25 - i * 5)) & 0x1F, 5);
                    *acc += diff;
                    samples.push(*acc);
                }
            }
            0b10 => {
                // seven 4-bit diffs
                for i in 0..7u32 {
                    if samples.len() >= num_samples {
                        break;
                    }
                    let diff = sign_extend((word >> (24 - i * 4)) & 0x0F, 4);
                    *acc += diff;
                    samples.push(*acc);
                }
            }
            _ => {
                return Err(MseedError::SteimDecode(format!(
                    "steim2 nibble=11 invalid dnib={dnib}"
                )));
            }
        },
        _ => unreachable!(),
    }

    Ok(())
}

/// Encode i32 samples using Steim1 compression.
pub fn encode_steim1(samples: &[i32], byte_order: ByteOrder) -> Result<Vec<u8>> {
    if samples.is_empty() {
        return Err(MseedError::SteimDecode("no samples to encode".into()));
    }

    // diffs[0] = samples[0] - x0 = 0 (identity diff)
    // diffs[i] = samples[i] - samples[i-1] for i > 0
    let x0 = samples[0];
    let xn = *samples.last().unwrap();

    let mut diffs = Vec::with_capacity(samples.len());
    diffs.push(0i32); // d₀ = x₀ - X₀ = 0
    for i in 1..samples.len() {
        diffs.push(samples[i].wrapping_sub(samples[i - 1]));
    }

    let mut frames: Vec<[u32; WORDS_PER_FRAME]> = Vec::new();
    let mut diff_idx = 0;

    loop {
        let is_first_frame = frames.is_empty();
        let mut frame = [0u32; WORDS_PER_FRAME];
        let mut control: u32 = 0;

        let start_word = if is_first_frame {
            frame[1] = x0 as u32;
            frame[2] = xn as u32;
            3
        } else {
            1
        };

        #[allow(clippy::needless_range_loop)]
        for word_idx in start_word..WORDS_PER_FRAME {
            if diff_idx >= diffs.len() {
                break;
            }

            let (packed_word, nibble, consumed) = steim1_pack_diffs(&diffs[diff_idx..]);
            frame[word_idx] = packed_word;
            control |= (nibble as u32) << (30 - word_idx * 2);
            diff_idx += consumed;
        }

        frame[0] = control;
        frames.push(frame);

        if diff_idx >= diffs.len() {
            break;
        }
    }

    let mut output = Vec::with_capacity(frames.len() * FRAME_SIZE);
    for frame in &frames {
        for &word in frame {
            match byte_order {
                ByteOrder::Big => output.extend_from_slice(&word.to_be_bytes()),
                ByteOrder::Little => output.extend_from_slice(&word.to_le_bytes()),
            }
        }
    }

    Ok(output)
}

/// Pack consecutive diffs into a single Steim1 word.
/// Returns (packed_word, nibble, num_consumed).
fn steim1_pack_diffs(diffs: &[i32]) -> (u32, u8, usize) {
    // Try four 8-bit diffs
    if diffs.len() >= 4 && diffs[..4].iter().all(|&d| (-128..=127).contains(&d)) {
        let word = ((diffs[0] as u8 as u32) << 24)
            | ((diffs[1] as u8 as u32) << 16)
            | ((diffs[2] as u8 as u32) << 8)
            | (diffs[3] as u8 as u32);
        return (word, 0b01, 4);
    }

    // Try two 16-bit diffs
    if diffs.len() >= 2 && diffs[..2].iter().all(|&d| (-32768..=32767).contains(&d)) {
        let word = ((diffs[0] as u16 as u32) << 16) | (diffs[1] as u16 as u32);
        return (word, 0b10, 2);
    }

    // Fallback: one 32-bit diff
    (diffs[0] as u32, 0b11, 1)
}

/// Encode i32 samples using Steim2 compression.
pub fn encode_steim2(samples: &[i32], byte_order: ByteOrder) -> Result<Vec<u8>> {
    if samples.is_empty() {
        return Err(MseedError::SteimDecode("no samples to encode".into()));
    }

    let x0 = samples[0];
    let xn = *samples.last().unwrap();

    let mut diffs = Vec::with_capacity(samples.len());
    diffs.push(0i32); // d₀ = x₀ - X₀ = 0
    for i in 1..samples.len() {
        diffs.push(samples[i].wrapping_sub(samples[i - 1]));
    }

    let mut frames: Vec<[u32; WORDS_PER_FRAME]> = Vec::new();
    let mut diff_idx = 0;

    loop {
        let is_first_frame = frames.is_empty();
        let mut frame = [0u32; WORDS_PER_FRAME];
        let mut control: u32 = 0;

        let start_word = if is_first_frame {
            frame[1] = x0 as u32;
            frame[2] = xn as u32;
            3
        } else {
            1
        };

        #[allow(clippy::needless_range_loop)]
        for word_idx in start_word..WORDS_PER_FRAME {
            if diff_idx >= diffs.len() {
                break;
            }

            let (packed_word, nibble, consumed) = steim2_pack_diffs(&diffs[diff_idx..]);
            frame[word_idx] = packed_word;
            control |= (nibble as u32) << (30 - word_idx * 2);
            diff_idx += consumed;
        }

        frame[0] = control;
        frames.push(frame);

        if diff_idx >= diffs.len() {
            break;
        }
    }

    let mut output = Vec::with_capacity(frames.len() * FRAME_SIZE);
    for frame in &frames {
        for &word in frame {
            match byte_order {
                ByteOrder::Big => output.extend_from_slice(&word.to_be_bytes()),
                ByteOrder::Little => output.extend_from_slice(&word.to_le_bytes()),
            }
        }
    }

    Ok(output)
}

/// Pack consecutive diffs into a single Steim2 word.
/// Returns (packed_word, nibble, num_consumed).
fn steim2_pack_diffs(diffs: &[i32]) -> (u32, u8, usize) {
    // Try 7 x 4-bit (fits -8..7), nibble=11, dnib=10
    if diffs.len() >= 7 && diffs[..7].iter().all(|&d| (-8..=7).contains(&d)) {
        let mut word: u32 = 0b10 << 30; // dnib=10
        for (i, &d) in diffs[..7].iter().enumerate() {
            word |= ((d as u32) & 0x0F) << (24 - i * 4);
        }
        return (word, 0b11, 7);
    }

    // Try 6 x 5-bit (fits -16..15), nibble=11, dnib=01
    if diffs.len() >= 6 && diffs[..6].iter().all(|&d| (-16..=15).contains(&d)) {
        let mut word: u32 = 0b01 << 30; // dnib=01
        for (i, &d) in diffs[..6].iter().enumerate() {
            word |= ((d as u32) & 0x1F) << (25 - i * 5);
        }
        return (word, 0b11, 6);
    }

    // Try 5 x 6-bit (fits -32..31), nibble=11, dnib=00
    if diffs.len() >= 5 && diffs[..5].iter().all(|&d| (-32..=31).contains(&d)) {
        let mut word: u32 = 0; // dnib=00
        for (i, &d) in diffs[..5].iter().enumerate() {
            word |= ((d as u32) & 0x3F) << (24 - i * 6);
        }
        return (word, 0b11, 5);
    }

    // Try 4 x 8-bit (fits -128..127), nibble=01
    if diffs.len() >= 4 && diffs[..4].iter().all(|&d| (-128..=127).contains(&d)) {
        let word = ((diffs[0] as u8 as u32) << 24)
            | ((diffs[1] as u8 as u32) << 16)
            | ((diffs[2] as u8 as u32) << 8)
            | (diffs[3] as u8 as u32);
        return (word, 0b01, 4);
    }

    // Try 3 x 10-bit (fits -512..511), nibble=10, dnib=11
    if diffs.len() >= 3 && diffs[..3].iter().all(|&d| (-512..=511).contains(&d)) {
        let mut word: u32 = 0b11 << 30; // dnib=11
        for (i, &d) in diffs[..3].iter().enumerate() {
            word |= ((d as u32) & 0x3FF) << (20 - i * 10);
        }
        return (word, 0b10, 3);
    }

    // Try 2 x 15-bit (fits -16384..16383), nibble=10, dnib=10
    if diffs.len() >= 2 && diffs[..2].iter().all(|&d| (-16384..=16383).contains(&d)) {
        let mut word: u32 = 0b10 << 30; // dnib=10
        for (i, &d) in diffs[..2].iter().enumerate() {
            word |= ((d as u32) & 0x7FFF) << (15 - i * 15);
        }
        return (word, 0b10, 2);
    }

    // Fallback: 1 x 30-bit, nibble=10, dnib=01
    let word = (0b01u32 << 30) | ((diffs[0] as u32) & 0x3FFF_FFFF);
    (word, 0b10, 1)
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
        use std::io::Read;
        let mut decoder = base64_decode_reader(s);
        let mut buf = Vec::new();
        decoder.read_to_end(&mut buf).unwrap();
        buf
    }

    fn base64_decode_reader(s: &str) -> impl std::io::Read + '_ {
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
                        if ch == b'\n' || ch == b'\r' || ch == b' ' {
                            continue;
                        }
                        quad[qi] = b64_val(ch);
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

        fn b64_val(ch: u8) -> u8 {
            match ch {
                b'A'..=b'Z' => ch - b'A',
                b'a'..=b'z' => ch - b'a' + 26,
                b'0'..=b'9' => ch - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                b'=' => 0xFF,
                _ => 0xFF,
            }
        }

        B64Reader {
            input: s.as_bytes(),
            pos: 0,
            buf: [0; 3],
            buf_len: 0,
            buf_pos: 0,
        }
    }

    #[test]
    fn test_steim1_decode_vectors() {
        let vectors = load_vectors("steim1_vectors.json");
        let arr = vectors.as_array().unwrap();

        for v in arr {
            let name = v["name"].as_str().unwrap();
            let num_samples = v["num_samples"].as_u64().unwrap() as usize;
            let compressed = decode_b64(v["compressed_data_b64"].as_str().unwrap());
            let expected: Vec<i32> = v["expected_samples"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect();

            let decoded =
                decode_steim1(&compressed, num_samples, ByteOrder::Big).unwrap_or_else(|e| {
                    panic!("steim1 decode failed for {name}: {e}");
                });

            assert_eq!(decoded, expected, "steim1 vector {name}: samples mismatch");
        }
    }

    #[test]
    fn test_steim2_decode_vectors() {
        let vectors = load_vectors("steim2_vectors.json");
        let arr = vectors.as_array().unwrap();

        for v in arr {
            let name = v["name"].as_str().unwrap();
            let num_samples = v["num_samples"].as_u64().unwrap() as usize;
            let compressed = decode_b64(v["compressed_data_b64"].as_str().unwrap());
            let expected: Vec<i32> = v["expected_samples"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect();

            let decoded =
                decode_steim2(&compressed, num_samples, ByteOrder::Big).unwrap_or_else(|e| {
                    panic!("steim2 decode failed for {name}: {e}");
                });

            assert_eq!(decoded, expected, "steim2 vector {name}: samples mismatch");
        }
    }

    #[test]
    fn test_steim1_roundtrip() {
        let samples: Vec<i32> = (0..100).collect();
        let encoded = encode_steim1(&samples, ByteOrder::Big).unwrap();
        let decoded = decode_steim1(&encoded, samples.len(), ByteOrder::Big).unwrap();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn test_steim2_roundtrip() {
        let samples: Vec<i32> = (0..100).collect();
        let encoded = encode_steim2(&samples, ByteOrder::Big).unwrap();
        let decoded = decode_steim2(&encoded, samples.len(), ByteOrder::Big).unwrap();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn test_steim1_roundtrip_random() {
        let mut rng_state: u32 = 42;
        let mut samples = Vec::with_capacity(200);
        let mut val: i32 = 0;
        for _ in 0..200 {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let diff = ((rng_state >> 16) as i32 % 1000) - 500;
            val = val.wrapping_add(diff);
            samples.push(val);
        }

        let encoded = encode_steim1(&samples, ByteOrder::Big).unwrap();
        let decoded = decode_steim1(&encoded, samples.len(), ByteOrder::Big).unwrap();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn test_steim2_roundtrip_random() {
        let mut rng_state: u32 = 42;
        let mut samples = Vec::with_capacity(200);
        let mut val: i32 = 0;
        for _ in 0..200 {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let diff = ((rng_state >> 16) as i32 % 1000) - 500;
            val = val.wrapping_add(diff);
            samples.push(val);
        }

        let encoded = encode_steim2(&samples, ByteOrder::Big).unwrap();
        let decoded = decode_steim2(&encoded, samples.len(), ByteOrder::Big).unwrap();
        assert_eq!(decoded, samples);
    }
}
