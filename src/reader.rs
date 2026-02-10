//! Iterator-based reader for multi-record miniSEED data.
//!
//! Use [`MseedReader`] to iterate over concatenated records in a byte slice.

use crate::Result;
use crate::decode::{self, MseedRecord};

/// Iterator over miniSEED v2 records in a byte slice.
///
/// Each call to `next()` decodes the next record and advances past it.
/// Iteration stops when the data is exhausted or a decode error occurs.
///
/// # Example
///
/// ```
/// use miniseed::{encode, MseedRecord, MseedReader, Samples};
///
/// let record = MseedRecord::new()
///     .with_nslc("XX", "TEST", "00", "BHZ")
///     .with_samples(Samples::Int(vec![1, 2, 3]));
/// let data = encode(&record).unwrap();
///
/// let records: Vec<_> = MseedReader::new(&data)
///     .collect::<Result<Vec<_>, _>>()
///     .unwrap();
/// assert_eq!(records.len(), 1);
/// ```
pub struct MseedReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> MseedReader<'a> {
    /// Create a new reader over the given byte slice.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }
}

impl Iterator for MseedReader<'_> {
    type Item = Result<MseedRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.data.len() {
            return None;
        }

        let remaining = &self.data[self.offset..];

        // Need at least 48 bytes for the fixed header to read record_length
        if remaining.len() < 48 {
            return None;
        }

        // Peek at blockette 1000 to determine record length
        let record_length = match peek_record_length(remaining) {
            Ok(len) => len,
            Err(e) => {
                // Move offset to end to stop iteration
                self.offset = self.data.len();
                return Some(Err(e));
            }
        };

        if remaining.len() < record_length {
            return None;
        }

        let record_data = &remaining[..record_length];
        match decode::decode(record_data) {
            Ok(record) => {
                self.offset += record_length;
                Some(Ok(record))
            }
            Err(e) => {
                self.offset = self.data.len();
                Some(Err(e))
            }
        }
    }
}

/// Peek at a record's blockette 1000 to determine the record length.
fn peek_record_length(data: &[u8]) -> Result<usize> {
    let first_blockette = u16::from_be_bytes([data[46], data[47]]) as usize;

    // Walk blockettes to find blockette 1000
    let mut offset = first_blockette;
    loop {
        if offset + 4 > data.len() {
            return Err(crate::MseedError::MissingBlockette1000);
        }
        let blockette_type = u16::from_be_bytes([data[offset], data[offset + 1]]);
        let next_offset = u16::from_be_bytes([data[offset + 2], data[offset + 3]]) as usize;

        if blockette_type == 1000 {
            if offset + 7 > data.len() {
                return Err(crate::MseedError::MissingBlockette1000);
            }
            let record_length_power = data[offset + 6];
            return Ok(1usize << record_length_power);
        }

        if next_offset == 0 {
            return Err(crate::MseedError::MissingBlockette1000);
        }
        offset = next_offset;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{BTime, Samples};
    use crate::encode;
    use crate::types::{ByteOrder, EncodingFormat};

    fn make_test_record(station: &str, samples: Vec<i32>) -> MseedRecord {
        MseedRecord {
            sequence_number: "000001".into(),
            quality: 'D',
            station: station.into(),
            location: "00".into(),
            channel: "BHZ".into(),
            network: "XX".into(),
            start_time: BTime {
                year: 2025,
                day: 1,
                hour: 0,
                minute: 0,
                second: 0,
                fract: 0,
            },
            sample_rate: 20.0,
            encoding: EncodingFormat::Int32,
            byte_order: ByteOrder::Big,
            record_length: 512,
            samples: Samples::Int(samples),
        }
    }

    #[test]
    fn test_reader_single_record() {
        let record = make_test_record("STA1", vec![1, 2, 3]);
        let data = encode::encode(&record).unwrap();

        let records: Vec<_> = MseedReader::new(&data).collect();
        assert_eq!(records.len(), 1);
        let decoded = records[0].as_ref().unwrap();
        assert_eq!(decoded.station, "STA1");
        assert_eq!(decoded.samples, Samples::Int(vec![1, 2, 3]));
    }

    #[test]
    fn test_reader_multiple_records() {
        let r1 = make_test_record("STA1", vec![10, 20, 30]);
        let r2 = make_test_record("STA2", vec![40, 50, 60]);
        let r3 = make_test_record("STA3", vec![70, 80, 90]);

        let mut data = Vec::new();
        data.extend_from_slice(&encode::encode(&r1).unwrap());
        data.extend_from_slice(&encode::encode(&r2).unwrap());
        data.extend_from_slice(&encode::encode(&r3).unwrap());

        let records: Vec<_> = MseedReader::new(&data)
            .collect::<Vec<_>>()
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(records.len(), 3);
        assert_eq!(records[0].station, "STA1");
        assert_eq!(records[1].station, "STA2");
        assert_eq!(records[2].station, "STA3");
    }

    #[test]
    fn test_reader_empty_data() {
        let records: Vec<_> = MseedReader::new(&[]).collect();
        assert!(records.is_empty());
    }
}
