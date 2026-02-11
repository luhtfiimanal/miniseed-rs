# miniseed

Pure Rust miniSEED v2 decoder and encoder. Zero `unsafe`, zero C dependencies.

[![CI](https://github.com/luhtfiimanal/miniseed-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/luhtfiimanal/miniseed-rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024_edition-orange.svg)](https://www.rust-lang.org/)

## What is miniSEED?

[miniSEED](http://www.fdsn.org/pdf/SEEDManual_V2.4.pdf) is the standard binary format for seismological waveform data, used by earthquake monitoring networks worldwide (IRIS, FDSN, BMKG, etc.). Each record is a self-contained packet (typically 512 or 4096 bytes, any power-of-2 length) containing station metadata, timestamps, and compressed sample data.

This crate provides idiomatic Rust access to miniSEED v2 records with full encode/decode support.

## Features

- **Decode** miniSEED v2 records (any power-of-2 length: 256, 512, 4096, etc.)
- **Encode** `MseedRecord` structs back to valid miniSEED bytes
- **Steim1/Steim2** compression and decompression (differential integer encoding)
- **Uncompressed** formats: INT16, INT32, FLOAT32, FLOAT64
- **Iterator-based reader** for multi-record files (`MseedReader`)
- **Builder pattern** for constructing records (`MseedRecord::new().with_*()`)
- **Typed encoding** with `EncodingFormat` enum (no magic numbers)
- **Zero unsafe** -- no FFI, no transmute, no raw pointers
- **Zero C dependencies** -- pure Rust, compiles anywhere `rustc` runs

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
miniseed-rs = "0.1"
```

### Decode a record

```rust
use miniseed_rs::{decode, encode, MseedRecord, Samples};

// Build a record, encode it, then decode
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

### Read a multi-record file

```rust
use miniseed_rs::MseedReader;

let data = std::fs::read("waveform.mseed").unwrap();
for result in MseedReader::new(&data) {
    let record = result.unwrap();
    println!("{}", record);
    // Output: IU.ANMO.00.BHZ | 2025-100 12:30:45.0000 | 20 Hz | 399 samples (Steim2)
}
```

### Build a record from scratch

```rust
use miniseed_rs::{MseedRecord, Samples, EncodingFormat, BTime, encode};

let record = MseedRecord::new()
    .with_nslc("XX", "TEST", "00", "BHZ")
    .with_start_time(BTime {
        year: 2025, day: 100, hour: 12,
        minute: 30, second: 45, fract: 0,
    })
    .with_sample_rate(20.0)
    .with_encoding(EncodingFormat::Steim1)
    .with_samples(Samples::Int(vec![1, -2, 3, -4, 100, -50]));

let bytes = encode(&record).unwrap();
assert_eq!(bytes.len(), 512);
```

## API Overview

```rust
// Top-level imports -- everything you need
use miniseed_rs::{
    decode, encode,           // functions
    MseedRecord, MseedReader, // core types
    BTime, Samples,           // data types
    ByteOrder, EncodingFormat, // enums
    MseedError, Result,       // error handling
};
```

| Type | Description |
|------|-------------|
| `MseedRecord` | Decoded record with header metadata and sample data |
| `MseedReader` | Iterator over multi-record byte slices |
| `BTime` | BTIME timestamp (year, day-of-year, hour, min, sec, fract) |
| `Samples` | Enum: `Int(Vec<i32>)`, `Float(Vec<f32>)`, `Double(Vec<f64>)` |
| `EncodingFormat` | `Int16`, `Int32`, `Float32`, `Float64`, `Steim1`, `Steim2` |
| `ByteOrder` | `Big`, `Little` |

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
  lib.rs       -- crate root, re-exports, runnable doc examples
  types.rs     -- ByteOrder, EncodingFormat enums
  decode.rs    -- decode(&[u8]) -> MseedRecord
  encode.rs    -- encode(&MseedRecord) -> Vec<u8>
  steim.rs     -- Steim1/2 compress + decompress
  reader.rs    -- MseedReader iterator
  error.rs     -- MseedError enum (thiserror)
```

### Design Decisions

- **Iterator over callback**: `MseedReader` implements `Iterator<Item = Result<MseedRecord>>` -- compose with standard iterator adapters
- **Enum over magic number**: `EncodingFormat::Steim1` instead of `10u8`
- **Builder over boilerplate**: `MseedRecord::new().with_nslc(...)` with sensible defaults
- **`&[u8]` over `Read` trait**: user calls `std::fs::read()` themselves -- simple, predictable, no hidden I/O

### TDD with ObsPy

Test vectors are generated by Python/ObsPy scripts, ensuring bit-exact compatibility with the reference seismological toolchain:

```bash
cd pyscripts && uv run python -m pyscripts.generate_vectors
cargo test
```

## Development

```bash
cargo build                     # build
cargo test                      # 19 unit + 1 integration + 4 doc tests
cargo clippy -- -D warnings     # lint (strict)
cargo fmt -- --check            # format check
cargo doc --no-deps --open      # browse docs locally
```

## References

- [FDSN SEED Manual v2.4](http://www.fdsn.org/pdf/SEEDManual_V2.4.pdf) -- the specification
- [libmseed](https://github.com/EarthScope/libmseed) -- C reference implementation
- [ObsPy](https://github.com/obspy/obspy) -- Python seismology framework (used for test vector generation)

## License

Apache-2.0
