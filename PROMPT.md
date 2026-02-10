# miniseed-rs — Agent Bootstrap Prompt

## What is this?

`miniseed-rs` is a pure Rust library for decoding and encoding miniSEED v2 seismic data records. It targets the FDSN SEED Manual v2.4 format — 512-byte records containing seismic waveform data, typically Steim1/Steim2 compressed.

**Goal**: Publishable to crates.io. Zero unsafe, zero C dependency. Apache 2.0 licensed.

Read `CLAUDE.md` for full crate instructions, byte layout, and code quality rules.

## Implementation Roadmap

### Phase 1: Steim1 Decode
1. Generate test vectors: Python/ObsPy creates known Steim1-compressed bytes + expected decoded samples
2. Implement `steim::decode_steim1(data: &[u8], num_samples: usize, byte_order: ByteOrder) -> Result<Vec<i32>>`
3. Validate against test vectors

### Phase 2: Steim2 Decode
1. Same TDD approach — ObsPy test vectors for Steim2
2. Implement `steim::decode_steim2(...)`
3. Steim2 adds more bit-width options (5, 6, 7-bit diffs) vs Steim1 (8, 16, 32-bit)

### Phase 3: Fixed Header + Blockette 1000 Parse
1. Parse the 48-byte fixed header: station/channel codes, timestamps, sample count, sample rate
2. Parse Blockette 1000: encoding format, byte order, record length
3. Struct: `MseedRecord { network, station, location, channel, start_time, sample_rate, samples, ... }`

### Phase 4: Full Record Decode
1. Integrate header parse + Steim decode into `decode::decode(&[u8]) -> Result<MseedRecord>`
2. Support uncompressed formats: INT16, INT32, FLOAT32, FLOAT64
3. End-to-end test: raw miniSEED bytes → decoded record → validate all fields

### Phase 5: Encode
1. Steim1 encode (reverse of decode)
2. Steim2 encode
3. Full record encode: `encode::encode(&MseedRecord) -> Result<Vec<u8>>`
4. Round-trip test: encode → decode → compare

## TDD Workflow

```bash
# 1. Generate test vectors (Python/ObsPy oracle)
cd pyscripts && uv sync
uv run python -m pyscripts.generate_vectors

# 2. Write Rust test that loads test_vectors/*.json — should FAIL (RED)
cargo test

# 3. Implement — should PASS (GREEN)
cargo test

# 4. Lint
cargo clippy -- -D warnings
cargo fmt -- --check
```

### Python Test Vector Generation

ObsPy can:
- **Read real .mseed files**: `obspy.read("file.mseed")` → access raw bytes and decoded samples
- **Create synthetic records**: `obspy.core.trace.Trace(data=np.array(...))` → write as miniSEED
- **Access raw compressed bytes**: Read file as bytes, extract data section after header

Test vectors are JSON files in `pyscripts/test_vectors/` (gitignored).

Example vector structure:
```json
{
  "description": "Steim1 decode - 3-component seismic",
  "encoding": 10,
  "byte_order": 1,
  "num_samples": 500,
  "compressed_bytes": [/* base64 or hex */],
  "expected_samples": [1234, 1235, 1237, ...]
}
```

## Key Steim Algorithm Summary

### Steim1
- Data organized in 64-byte frames (16 x 32-bit words)
- Word 0 of each frame = nibble header (16 x 2-bit codes)
- Nibble codes: 00=skip, 01=four 8-bit diffs, 10=two 16-bit diffs, 11=one 32-bit diff
- Frame 0, word 1 = forward integration constant (first sample value)
- Frame 0, word 2 = reverse integration constant (last sample value, for validation)
- Decode: accumulate diffs starting from forward integration constant

### Steim2
- Same frame structure as Steim1
- Nibble 01: same as Steim1 (four 8-bit diffs)
- Nibble 10: sub-coded via bits 31-30: one 30-bit, two 15-bit, or three 10-bit diffs
- Nibble 11: sub-coded via bits 31-30: five 6-bit, six 5-bit, or seven 4-bit diffs

## References

- FDSN SEED Manual v2.4: http://www.fdsn.org/pdf/SEEDManual_V2.4.pdf (Appendix B for Steim)
- libmseed source: https://github.com/EarthScope/libmseed (C reference)
- ObsPy miniSEED: https://docs.obspy.org/packages/autogen/obspy.io.mseed.html
