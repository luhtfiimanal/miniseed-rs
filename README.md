# miniseed-rs

Pure Rust miniSEED v2 and v3 decoder and encoder. Zero `unsafe`, zero C dependencies.

[![Crates.io](https://img.shields.io/crates/v/miniseed-rs.svg)](https://crates.io/crates/miniseed-rs)
[![docs.rs](https://docs.rs/miniseed-rs/badge.svg)](https://docs.rs/miniseed-rs)
[![CI](https://github.com/luhtfiimanal/miniseed-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/luhtfiimanal/miniseed-rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024_edition-orange.svg)](https://www.rust-lang.org/)

## What is miniSEED?

[miniSEED](http://www.fdsn.org/pdf/SEEDManual_V2.4.pdf) is the standard binary format for seismological waveform data, used by earthquake monitoring networks worldwide (IRIS, FDSN, BMKG, etc.). Each record is a self-contained packet containing station metadata, timestamps, and compressed sample data.

This crate provides idiomatic Rust access to both miniSEED v2 and v3 records with full encode/decode support and automatic format detection.

## Features

- **miniSEED v2 and v3** with automatic format detection
- **Decode** records of any size (v2: power-of-2 lengths; v3: variable-length)
- **Encode** `MseedRecord` structs back to valid miniSEED bytes
- **Steim1/Steim2** compression and decompression
- **Uncompressed** formats: INT16, INT32, FLOAT32, FLOAT64
- **Iterator-based reader** for multi-record files (`MseedReader`), supports mixed v2+v3 streams
- **Builder pattern** for constructing records (`MseedRecord::new().with_*()`)
- **FDSN Source Identifiers** (`SourceId`) with NSLC conversion
- **Nanosecond timestamps** (`NanoTime`) with v2 `BTime` compatibility
- **CRC-32C** integrity checking for v3 records
- **Zero unsafe** -- no FFI, no transmute, no raw pointers
- **Zero C dependencies** -- pure Rust, compiles anywhere `rustc` runs

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
miniseed-rs = "0.2"
```

### Decode a record (auto-detects v2 or v3)

```rust
use miniseed_rs::{decode, encode, MseedRecord, Samples};

// Build a v2 record, encode, then decode
let record = MseedRecord::new()
    .with_nslc("IU", "ANMO", "00", "BHZ")
    .with_sample_rate(20.0)
    .with_samples(Samples::Int(vec![100, 200, 300]));

let bytes = encode(&record).unwrap();
let decoded = decode(&bytes).unwrap();

assert_eq!(decoded.network, "IU");
assert_eq!(decoded.station, "ANMO");
assert_eq!(decoded.samples.len(), 3);
```

### Read a multi-record file (v2, v3, or mixed)

```rust
use miniseed_rs::MseedReader;

let data = std::fs::read("waveform.mseed").unwrap();
for result in MseedReader::new(&data) {
    let record = result.unwrap();
    println!("{}", record);
    // Output: IU.ANMO.00.BHZ | 2025-100 12:30:45.000000000 | 20 Hz | 399 samples (Steim2) [miniSEED v2]
}
```

### Build a v3 record

```rust
use miniseed_rs::{MseedRecord, Samples, EncodingFormat, NanoTime, encode, decode};

let record = MseedRecord::new_v3()
    .with_nslc("IU", "ANMO", "00", "BHZ")
    .with_start_time(NanoTime {
        year: 2025, day: 100, hour: 12,
        minute: 30, second: 45, nanosecond: 500_000_000,
    })
    .with_sample_rate(20.0)
    .with_encoding(EncodingFormat::Steim2)
    .with_samples(Samples::Int(vec![1, -2, 3, -4, 100, -50]));

let bytes = encode(&record).unwrap();
let decoded = decode(&bytes).unwrap();
assert_eq!(decoded.start_time.nanosecond, 500_000_000);
```

## API Overview

```rust
// Top-level imports -- everything you need
use miniseed_rs::{
    decode, encode,                    // functions
    MseedRecord, MseedReader,         // core types
    NanoTime, BTime, Samples,         // data types
    SourceId,                          // FDSN Source Identifier
    ByteOrder, EncodingFormat,         // enums
    FormatVersion,                     // v2 or v3
    MseedError, Result,                // error handling
};
```

| Type | Description |
|------|-------------|
| `MseedRecord` | Unified record for v2 and v3 with header metadata and sample data |
| `MseedReader` | Iterator over multi-record byte slices (v2, v3, or mixed) |
| `NanoTime` | Timestamp with nanosecond precision (year, day, hour, min, sec, ns) |
| `BTime` | Legacy v2 timestamp (year, day, hour, min, sec, 0.0001s fract) |
| `SourceId` | FDSN Source Identifier with NSLC conversion |
| `Samples` | Enum: `Int(Vec<i32>)`, `Float(Vec<f32>)`, `Double(Vec<f64>)` |
| `EncodingFormat` | `Int16`, `Int32`, `Float32`, `Float64`, `Steim1`, `Steim2` |
| `FormatVersion` | `V2`, `V3` |

## Supported Encoding Formats

| Format | Code | Type | Description |
|--------|------|------|-------------|
| INT16 | 1 | `Samples::Int` | 16-bit signed integer |
| INT32 | 3 | `Samples::Int` | 32-bit signed integer |
| FLOAT32 | 4 | `Samples::Float` | 32-bit IEEE float |
| FLOAT64 | 5 | `Samples::Double` | 64-bit IEEE double |
| Steim1 | 10 | `Samples::Int` | Steim-1 differential compression |
| Steim2 | 11 | `Samples::Int` | Steim-2 differential compression |

## Architecture

```
src/
  lib.rs         -- crate root, re-exports, doc examples
  types.rs       -- ByteOrder, EncodingFormat, FormatVersion
  record.rs      -- MseedRecord (unified v2+v3), Samples
  time.rs        -- NanoTime, BTime
  sid.rs         -- SourceId (FDSN Source Identifier)
  crc.rs         -- CRC-32C (Castagnoli) for v3
  decode.rs      -- auto-detect decoder (v2/v3 dispatcher)
  decode_v3.rs   -- v3-specific decode
  encode.rs      -- version-dispatched encoder
  encode_v3.rs   -- v3-specific encode
  steim.rs       -- Steim1/2 compress + decompress (shared)
  reader.rs      -- MseedReader iterator (v2+v3+mixed)
  error.rs       -- MseedError enum (thiserror)
```

### Design Decisions

- **Unified struct**: single `MseedRecord` for both v2 and v3 (like libmseed's MS3Record)
- **Auto-detect**: `decode()` automatically identifies v2 vs v3 format
- **Iterator over callback**: `MseedReader` implements `Iterator<Item = Result<MseedRecord>>`
- **Enum over magic number**: `EncodingFormat::Steim1` instead of `10u8`
- **Builder over boilerplate**: `MseedRecord::new().with_nslc(...)` with sensible defaults
- **`&[u8]` over `Read` trait**: user calls `std::fs::read()` themselves -- simple, predictable, no hidden I/O

### TDD with pymseed

Test vectors are generated by Python/pymseed scripts, ensuring bit-exact compatibility with libmseed for both v2 and v3:

```bash
cd pyscripts && uv run python -m pyscripts.generate_vectors
cargo test
```

## Development

```bash
cargo build                     # build
cargo test                      # all tests
cargo clippy -- -D warnings     # lint (strict)
cargo fmt -- --check            # format check
cargo doc --no-deps --open      # browse docs locally
```

## References

- [FDSN SEED Manual v2.4](http://www.fdsn.org/pdf/SEEDManual_V2.4.pdf) -- v2 specification
- [FDSN miniSEED v3 specification](https://docs.fdsn.org/projects/miniseed3/) -- v3 specification
- [libmseed](https://github.com/EarthScope/libmseed) -- C reference implementation
- [pymseed](https://github.com/EarthScope/pymseed) -- Python wrapper (used for test vector generation)

## License

Apache-2.0
