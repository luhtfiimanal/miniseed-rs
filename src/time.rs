//! Nanosecond-precision timestamps for miniSEED v2 and v3.
//!
//! [`NanoTime`] is the primary timestamp type, supporting nanosecond resolution
//! as required by miniSEED v3. The legacy [`BTime`] type is preserved for
//! backward compatibility with v2 records.

use std::fmt;

/// Nanosecond-precision timestamp (year + day-of-year + time).
///
/// Used for both miniSEED v2 and v3 records. For v2, the sub-second
/// precision is limited to 0.1 ms (100 µs), stored in the `nanosecond`
/// field as a multiple of 100,000.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NanoTime {
    pub year: u16,
    pub day: u16,        // 1-366
    pub hour: u8,        // 0-23
    pub minute: u8,      // 0-59
    pub second: u8,      // 0-60 (60 for leap second in v3)
    pub nanosecond: u32, // 0-999_999_999
}

impl NanoTime {
    /// Create a NanoTime with default epoch (1970-001 00:00:00.000000000).
    pub fn epoch() -> Self {
        Self {
            year: 1970,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            nanosecond: 0,
        }
    }

    /// Create a NanoTime from a legacy [`BTime`] value.
    ///
    /// Converts the 0.0001-second fractional field to nanoseconds.
    pub fn from_btime(bt: &BTime) -> Self {
        Self {
            year: bt.year,
            day: bt.day,
            hour: bt.hour,
            minute: bt.minute,
            second: bt.second,
            nanosecond: bt.fract as u32 * 100_000, // 0.0001s = 100µs = 100_000ns
        }
    }

    /// Convert to a legacy [`BTime`] value.
    ///
    /// Nanosecond precision is truncated to 0.0001-second units.
    pub fn to_btime(self) -> BTime {
        BTime {
            year: self.year,
            day: self.day,
            hour: self.hour,
            minute: self.minute,
            second: self.second,
            fract: (self.nanosecond / 100_000) as u16,
        }
    }
}

impl Default for NanoTime {
    fn default() -> Self {
        Self::epoch()
    }
}

impl fmt::Display for NanoTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04}-{:03} {:02}:{:02}:{:02}.{:09}",
            self.year, self.day, self.hour, self.minute, self.second, self.nanosecond
        )
    }
}

impl From<BTime> for NanoTime {
    fn from(bt: BTime) -> Self {
        Self::from_btime(&bt)
    }
}

impl From<NanoTime> for BTime {
    fn from(nt: NanoTime) -> Self {
        nt.to_btime()
    }
}

/// Legacy BTIME timestamp (10 bytes in the miniSEED v2 fixed header).
///
/// Preserved for backward compatibility. Prefer [`NanoTime`] for new code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BTime {
    pub year: u16,
    pub day: u16,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub fract: u16, // 0.0001 second units
}

impl fmt::Display for BTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04}-{:03} {:02}:{:02}:{:02}.{:04}",
            self.year, self.day, self.hour, self.minute, self.second, self.fract
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nanotime_epoch() {
        let nt = NanoTime::epoch();
        assert_eq!(nt.year, 1970);
        assert_eq!(nt.day, 1);
        assert_eq!(nt.nanosecond, 0);
    }

    #[test]
    fn test_btime_to_nanotime_roundtrip() {
        let bt = BTime {
            year: 2024,
            day: 15,
            hour: 10,
            minute: 30,
            second: 45,
            fract: 1234,
        };
        let nt = NanoTime::from_btime(&bt);
        assert_eq!(nt.year, 2024);
        assert_eq!(nt.day, 15);
        assert_eq!(nt.hour, 10);
        assert_eq!(nt.minute, 30);
        assert_eq!(nt.second, 45);
        assert_eq!(nt.nanosecond, 1234 * 100_000); // 123_400_000

        let bt2 = nt.to_btime();
        assert_eq!(bt, bt2);
    }

    #[test]
    fn test_nanotime_nanosecond_precision() {
        let nt = NanoTime {
            year: 2025,
            day: 100,
            hour: 12,
            minute: 0,
            second: 0,
            nanosecond: 123_456_789,
        };
        // Converting to BTime truncates to 0.1ms
        let bt = nt.to_btime();
        assert_eq!(bt.fract, 1234); // 123_456_789 / 100_000 = 1234

        // Round-trip loses nanosecond precision
        let nt2 = NanoTime::from_btime(&bt);
        assert_eq!(nt2.nanosecond, 123_400_000);
    }

    #[test]
    fn test_from_trait_conversions() {
        let bt = BTime {
            year: 2024,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            fract: 5000,
        };
        let nt: NanoTime = bt.into();
        assert_eq!(nt.nanosecond, 500_000_000);

        let bt2: BTime = nt.into();
        assert_eq!(bt2.fract, 5000);
    }

    #[test]
    fn test_nanotime_display() {
        let nt = NanoTime {
            year: 2024,
            day: 15,
            hour: 10,
            minute: 30,
            second: 0,
            nanosecond: 500_000_000,
        };
        assert_eq!(format!("{nt}"), "2024-015 10:30:00.500000000");
    }

    #[test]
    fn test_nanotime_leap_second() {
        // v3 supports second=60 for leap seconds
        let nt = NanoTime {
            year: 2016,
            day: 366,
            hour: 23,
            minute: 59,
            second: 60,
            nanosecond: 0,
        };
        assert_eq!(nt.second, 60);
        assert_eq!(format!("{nt}"), "2016-366 23:59:60.000000000");
    }
}
