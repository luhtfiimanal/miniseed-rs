# Changelog

## [0.2.0] - 2026-02-11

miniSEED v3 support. Breaking changes from 0.1.0.

### Added

- **miniSEED v3** decode and encode with automatic format detection
- `FormatVersion` enum (`V2`, `V3`) on `MseedRecord`
- `NanoTime` timestamp with nanosecond precision (replaces `BTime` as primary type)
- `SourceId` for FDSN Source Identifiers (`FDSN:NET_STA_LOC_B_H_Z`)
- CRC-32C (Castagnoli) for v3 record integrity (`crc` module)
- `MseedRecord::new_v3()` builder for v3 records
- `MseedReader` now supports mixed v2+v3 record streams
- `decode()` auto-detects v2 vs v3 format
- `encode()` dispatches to v2 or v3 based on `format_version`
- New modules: `record`, `time`, `sid`, `crc`, `decode_v3`, `encode_v3`

### Changed

- `MseedRecord` is now a unified struct for both v2 and v3 (moved to `record` module)
- `start_time` changed from `BTime` to `NanoTime` (breaking)
- `record_length` changed from `u16` to `u32` (breaking)
- `Samples` moved from `decode` module to `record` module
- TDD oracle migrated from ObsPy to pymseed

### Deprecated

- `BTime` is still available but `NanoTime` is the primary timestamp type

## [0.1.0] - 2025-02-11

Initial release.

- Decode miniSEED v2 records (any power-of-2 record length: 256-4096+)
- Encode MseedRecord to miniSEED bytes
- Steim1 and Steim2 compression/decompression
- Uncompressed formats: INT16, INT32, FLOAT32, FLOAT64
- MseedReader iterator for multi-record byte slices
- Builder pattern: MseedRecord::new().with_*()
- Typed EncodingFormat enum
- Zero unsafe, zero C dependencies
