# CLAUDE.md — miniseed-rs

Pure Rust miniSEED v2 and v3 decoder and encoder. Zero unsafe, zero C dependency. Apache 2.0.

## CRITICAL

- **Diskusi dulu sebelum implementasi** — investigasi, jelaskan, diskusikan, baru code
- **Jangan push tanpa persetujuan user**
- **stdout workaround**: `script -q -c "cargo test" /dev/null` (Claude Code bug)
- **Zero unsafe** — no FFI, no transmute, no raw pointers

## Scope

**miniSEED v2 + v3** with automatic format detection:

- **Decode**: Auto-detect v2/v3, parse headers, decode data
- **Encode**: MseedRecord → miniSEED bytes (v2 or v3 based on format_version)
- **Compression**: Steim1, Steim2 (shared between v2/v3, always BE)
- **Uncompressed**: INT16, INT32, FLOAT32, FLOAT64 (v2: configurable endian; v3: always LE)
- **v3 extras**: CRC-32C, FDSN Source Identifiers, NanoTime, variable-length records

## Module Structure

```
src/
  lib.rs         -- re-exports + crate doc
  error.rs       -- MseedError enum (thiserror)
  record.rs      -- MseedRecord (unified v2+v3) + Samples
  time.rs        -- NanoTime + BTime
  sid.rs         -- SourceId (FDSN Source Identifier)
  crc.rs         -- CRC-32C (Castagnoli) for v3
  decode.rs      -- auto-detect dispatcher + v2 decode
  decode_v3.rs   -- v3 decode
  encode.rs      -- version dispatcher + v2 encode
  encode_v3.rs   -- v3 encode
  steim.rs       -- Steim1/2 compress + decompress (shared)
  reader.rs      -- MseedReader iterator (v2+v3+mixed)
```

## Commands

```bash
cargo build                          # build
cargo test                           # test all
cargo test steim::                   # test single module
cargo clippy -- -D warnings          # lint (strict)
cargo fmt -- --check                 # format check

# pyscripts (TDD vector generation)
cd pyscripts && uv sync
cd pyscripts && uv run python -m pyscripts.generate_vectors
cd pyscripts && uv run ruff check src
cd pyscripts && uv run basedpyright src
```

## TDD Strategy

Python/pymseed generates test vectors → Rust tests assert against them.

1. `cd pyscripts && uv run python -m pyscripts.generate_vectors`
2. Write Rust test loading `test_vectors/*.json` — RED
3. Implement Rust code — GREEN
4. Validate: decoded samples exactly match pymseed/libmseed output

Test vectors saved as JSON in `pyscripts/test_vectors/` (gitignored, regenerate locally).

## Code Quality

- `cargo fmt` + `cargo clippy -- -D warnings` — pre-commit enforced
- `thiserror` for all error types
- No `unsafe` anywhere
- pyscripts: `basedpyright` strict + `ruff`

## miniSEED v2 Record Format

```
Bytes 0-47:   Fixed header (big-endian)
              ├─ [0..6]    Sequence number (ASCII digits)
              ├─ [6]       Data quality indicator (D, R, Q, M)
              ├─ [8..13]   Station code
              ├─ [13..15]  Location code
              ├─ [15..18]  Channel code
              ├─ [18..20]  Network code
              ├─ [20..30]  Start time (BTIME)
              ├─ [30..32]  Number of samples (u16 BE)
              ├─ [32..36]  Sample rate factor + multiplier
              ├─ [44..46]  Data offset (u16 BE)
              └─ [46..48]  First blockette offset (u16 BE)

Bytes 48+:    Blockette 1000 (encoding, byte order, record length power)
Data section: Steim1/2 or uncompressed samples
```

## miniSEED v3 Record Format

```
Bytes 0-39:   Fixed header (little-endian)
              ├─ [0..2]    "MS" magic
              ├─ [2]       Version = 3
              ├─ [3]       Flags
              ├─ [4..8]    Nanosecond (u32 LE)
              ├─ [8..10]   Year (u16 LE)
              ├─ [10..12]  Day of year (u16 LE)
              ├─ [12..15]  Hour, Minute, Second
              ├─ [15]      Encoding format
              ├─ [16..24]  Sample rate (f64 LE)
              ├─ [24..28]  Number of samples (u32 LE)
              ├─ [28..32]  CRC-32C (u32 LE)
              ├─ [32]      Publication version
              ├─ [33]      SID length
              ├─ [34..36]  Extra headers length (u16 LE)
              └─ [36..40]  Data payload length (u32 LE)

Variable:     [SID][Extra headers JSON][Data payload]
```

## References

- FDSN SEED Manual v2.4: http://www.fdsn.org/pdf/SEEDManual_V2.4.pdf
- FDSN miniSEED v3: https://docs.fdsn.org/projects/miniseed3/
- libmseed (C reference impl): https://github.com/EarthScope/libmseed
- pymseed: https://github.com/EarthScope/pymseed
