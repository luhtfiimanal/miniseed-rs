//! Compile-time smoke test: verify top-level re-exports work.

use miniseed_rs::{
    BTime, ByteOrder, EncodingFormat, FormatVersion, MseedError, MseedRecord, NanoTime, Result,
    Samples, SourceId, decode, encode,
};

#[test]
fn top_level_imports_compile() {
    // Just verify the types are usable from the crate root
    let _: fn(&[u8]) -> Result<MseedRecord> = decode;
    let _: fn(&MseedRecord) -> Result<Vec<u8>> = encode;

    let _bo = ByteOrder::Big;
    let _s = Samples::Int(vec![]);
    let _bt = BTime {
        year: 2025,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
        fract: 0,
    };
    let _nt = NanoTime::epoch();

    let _enc = EncodingFormat::Steim1;
    let _fv = FormatVersion::V2;

    let _sid = SourceId::from_nslc("IU", "ANMO", "00", "BHZ");

    // MseedError is accessible
    let _e: Option<MseedError> = None;
}
