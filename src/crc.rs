//! CRC-32C (Castagnoli) for miniSEED v3 record integrity.
//!
//! Uses the Castagnoli polynomial `0x82F63B78`. The CRC field in a v3
//! record is zeroed before computation, then the computed CRC is written
//! back into the record.

/// CRC-32C lookup table (Castagnoli polynomial 0x82F63B78).
const CRC32C_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0x82F6_3B78;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
};

/// Compute CRC-32C (Castagnoli) over the given data.
pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32C_TABLE[index];
    }
    crc ^ 0xFFFF_FFFF
}

/// Compute CRC-32C for a miniSEED v3 record.
///
/// The CRC field (bytes 28-31) must be zeroed before computation.
/// This function takes a mutable slice, zeros the CRC field, computes
/// the CRC over the entire record, and writes it back.
pub fn compute_v3_crc(record: &mut [u8]) -> u32 {
    // Zero the CRC field (bytes 28-31) before computing
    if record.len() >= 32 {
        record[28] = 0;
        record[29] = 0;
        record[30] = 0;
        record[31] = 0;
    }
    let crc = crc32c(record);
    // Write CRC back (little-endian)
    if record.len() >= 32 {
        record[28..32].copy_from_slice(&crc.to_le_bytes());
    }
    crc
}

/// Verify the CRC-32C of a miniSEED v3 record.
///
/// Returns `true` if the CRC is valid.
pub fn verify_v3_crc(record: &[u8]) -> bool {
    if record.len() < 32 {
        return false;
    }
    // Read the stored CRC
    let stored_crc = u32::from_le_bytes([record[28], record[29], record[30], record[31]]);
    // Compute CRC with field zeroed
    let mut buf = record.to_vec();
    buf[28] = 0;
    buf[29] = 0;
    buf[30] = 0;
    buf[31] = 0;
    let computed = crc32c(&buf);
    stored_crc == computed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32c_empty() {
        assert_eq!(crc32c(&[]), 0x0000_0000);
    }

    #[test]
    fn test_crc32c_known_values() {
        // Well-known CRC-32C test vector
        let data = b"123456789";
        assert_eq!(crc32c(data), 0xE306_9283);
    }

    #[test]
    fn test_crc32c_single_byte() {
        // CRC-32C of a single zero byte
        let crc = crc32c(&[0]);
        assert_ne!(crc, 0); // Should be non-zero
    }

    #[test]
    fn test_crc32c_zeros() {
        // CRC of all-zero data should be deterministic
        let crc1 = crc32c(&[0; 10]);
        let crc2 = crc32c(&[0; 10]);
        assert_eq!(crc1, crc2);
    }

    #[test]
    fn test_compute_and_verify_v3_crc() {
        // Simulate a minimal v3-like record (at least 40 bytes)
        let mut record = vec![0u8; 64];
        record[0] = b'M';
        record[1] = b'S';
        record[2] = 3;

        let crc = compute_v3_crc(&mut record);
        assert_ne!(crc, 0);

        // Verify it
        assert!(verify_v3_crc(&record));

        // Corrupt one byte and verify failure
        record[10] ^= 0xFF;
        assert!(!verify_v3_crc(&record));
    }
}
