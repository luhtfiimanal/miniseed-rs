# CLAUDE.md — miniseed-rs

Pure Rust miniSEED v2 decoder and encoder. Zero unsafe, zero C dependency. Apache 2.0.

## CRITICAL

- **Diskusi dulu sebelum implementasi** — investigasi, jelaskan, diskusikan, baru code
- **Jangan push tanpa persetujuan user**
- **stdout workaround**: `script -q -c "cargo test" /dev/null` (Claude Code bug)
- **Zero unsafe** — no FFI, no transmute, no raw pointers

## Scope

**miniSEED v2** (FDSN SEED Manual, 512-byte records):

- **Decode**: Fixed header (48 bytes) + Blockette 1000 + data section
- **Encode**: Struct → miniSEED record bytes
- **Compression**: Steim1, Steim2 (decode + encode)
- **Uncompressed**: INT16, INT32, FLOAT32, FLOAT64
- **No miniSEED v3** (different format, separate scope)

## Module Structure

```
src/
  lib.rs        -- re-exports + crate doc
  error.rs      -- MseedError enum (thiserror)
  decode.rs     -- decode(&[u8]) → MseedRecord
  encode.rs     -- encode(MseedRecord) → Vec<u8>
  steim.rs      -- Steim1/2 compress + decompress
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

Python/ObsPy generates test vectors → Rust tests assert against them.

1. `cd pyscripts && uv run python -m pyscripts.generate_vectors`
2. Write Rust test loading `test_vectors/*.json` — RED
3. Implement Rust code — GREEN
4. Validate: decoded samples exactly match ObsPy output

### Test Vector Approach

ObsPy can:
- Read real miniSEED files and expose raw bytes + decoded samples
- Create synthetic miniSEED records with known content
- Encode with Steim1/Steim2 and provide raw compressed bytes

Test vectors saved as JSON in `pyscripts/test_vectors/` (gitignored, regenerate locally).

## Code Quality

- `cargo fmt` + `cargo clippy -- -D warnings` — pre-commit enforced
- `thiserror` for all error types
- No `unsafe` anywhere
- pyscripts: `basedpyright` strict + `ruff`

## miniSEED v2 Record Format (512 bytes)

```
Bytes 0-47:   Fixed header
              ├─ [0..6]    Sequence number (ASCII digits)
              ├─ [6]       Data quality indicator (D, R, Q, M)
              ├─ [7]       Reserved
              ├─ [8..13]   Station code (ASCII, right-padded spaces)
              ├─ [13..15]  Location code
              ├─ [15..18]  Channel code
              ├─ [18..20]  Network code
              ├─ [20..30]  Start time (BTIME: year, day, hour, min, sec, frac)
              ├─ [30..32]  Number of samples (u16 BE)
              ├─ [32..34]  Sample rate factor (i16 BE)
              ├─ [34..36]  Sample rate multiplier (i16 BE)
              ├─ [36]      Activity flags
              ├─ [37]      I/O and clock flags
              ├─ [38]      Data quality flags
              ├─ [39]      Number of blockettes that follow
              ├─ [40..44]  Time correction (i32 BE)
              ├─ [44..46]  Beginning of data (u16 BE)
              └─ [46..48]  First blockette (u16 BE)

Bytes 48+:    Blockette 1000 (8 bytes)
              ├─ [0..2]    Blockette type = 1000 (u16 BE)
              ├─ [2..4]    Next blockette offset (u16 BE)
              ├─ [4]       Encoding format
              ├─ [5]       Byte order (0=little, 1=big)
              ├─ [6]       Record length (power of 2, e.g. 9 = 512)
              └─ [7]       Reserved

Data section: Steim1/2 compressed or uncompressed samples
```

### Encoding Formats

| Code | Format | Sample size |
|------|--------|-------------|
| 1 | INT16 | 2 bytes |
| 3 | INT32 | 4 bytes |
| 4 | FLOAT32 | 4 bytes |
| 5 | FLOAT64 | 8 bytes |
| 10 | Steim1 | variable |
| 11 | Steim2 | variable |

### BTIME (10 bytes)

```
[0..2]  year (u16 BE)
[2..4]  day of year (u16 BE, 1-366)
[4]     hour
[5]     minute
[6]     second
[7]     unused
[8..10] 0.0001 seconds (u16 BE)
```

## References

- FDSN SEED Manual v2.4: http://www.fdsn.org/pdf/SEEDManual_V2.4.pdf
- Steim compression: Appendix B of SEED Manual
- libmseed (C reference impl): https://github.com/EarthScope/libmseed
