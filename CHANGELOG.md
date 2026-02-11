# Changelog

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
