"""Generate test vectors for miniseed-rs using ObsPy as oracle.

Produces JSON files in pyscripts/test_vectors/ that Rust tests load
to validate decode/encode against known-good ObsPy output.
"""

from __future__ import annotations

import base64
import io
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from obspy import Stream, Trace, UTCDateTime  # type: ignore[import-untyped]

VECTORS_DIR = Path(__file__).resolve().parent.parent.parent / "test_vectors"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _make_trace(
    data: NDArray[Any],
    *,
    network: str = "IU",
    station: str = "ANMO",
    location: str = "00",
    channel: str = "BHZ",
    sampling_rate: float = 20.0,
    starttime: UTCDateTime | None = None,
) -> Trace:  # type: ignore[type-arg]
    tr = Trace(data=data)  # type: ignore[no-untyped-call]
    tr.stats.network = network  # type: ignore[union-attr]
    tr.stats.station = station  # type: ignore[union-attr]
    tr.stats.location = location  # type: ignore[union-attr]
    tr.stats.channel = channel  # type: ignore[union-attr]
    tr.stats.sampling_rate = sampling_rate  # type: ignore[union-attr]
    tr.stats.starttime = starttime or UTCDateTime(2024, 1, 15, 10, 30, 0, 0)  # type: ignore[union-attr, no-untyped-call]
    return tr  # type: ignore[return-value]


def _trace_to_mseed_bytes(tr: Trace, *, encoding: str | int, reclen: int = 512) -> bytes:  # type: ignore[type-arg]
    buf = io.BytesIO()
    st = Stream([tr])  # type: ignore[no-untyped-call]
    st.write(buf, format="MSEED", encoding=encoding, reclen=reclen)  # type: ignore[no-untyped-call]
    buf.seek(0)
    return buf.read()


def _parse_record_metadata(raw: bytes) -> dict[str, Any]:
    """Extract metadata from raw miniSEED record for test vector."""
    seq = raw[0:6].decode("ascii")
    quality = chr(raw[6])
    station = raw[8:13].decode("ascii").rstrip()
    location = raw[13:15].decode("ascii").rstrip()
    channel = raw[15:18].decode("ascii").rstrip()
    network = raw[18:20].decode("ascii").rstrip()

    year: int = struct.unpack(">H", raw[20:22])[0]
    day: int = struct.unpack(">H", raw[22:24])[0]
    hour = raw[24]
    minute = raw[25]
    second = raw[26]
    fract: int = struct.unpack(">H", raw[28:30])[0]

    num_samples: int = struct.unpack(">H", raw[30:32])[0]
    sample_rate_factor: int = struct.unpack(">h", raw[32:34])[0]
    sample_rate_multiplier: int = struct.unpack(">h", raw[34:36])[0]

    num_blockettes = raw[39]
    data_offset: int = struct.unpack(">H", raw[44:46])[0]
    blockette_offset: int = struct.unpack(">H", raw[46:48])[0]

    # Parse blockette 1000
    b1000_offset = blockette_offset
    b1000_type: int = struct.unpack(">H", raw[b1000_offset : b1000_offset + 2])[0]
    assert b1000_type == 1000, f"Expected blockette 1000, got {b1000_type}"
    encoding_format: int = raw[b1000_offset + 4]
    byte_order: int = raw[b1000_offset + 5]
    record_length_power: int = raw[b1000_offset + 6]

    return {
        "sequence_number": seq,
        "quality": quality,
        "station": station,
        "location": location,
        "channel": channel,
        "network": network,
        "year": year,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
        "fract": fract,
        "num_samples": num_samples,
        "sample_rate_factor": sample_rate_factor,
        "sample_rate_multiplier": sample_rate_multiplier,
        "num_blockettes": num_blockettes,
        "data_offset": data_offset,
        "blockette_offset": blockette_offset,
        "encoding_format": encoding_format,
        "byte_order": byte_order,
        "record_length_power": record_length_power,
    }


# ---------------------------------------------------------------------------
# Steim1 vectors
# ---------------------------------------------------------------------------


def generate_steim1_vectors() -> None:
    vectors: list[dict[str, Any]] = []

    # Pattern 1: ramp
    ramp = np.arange(0, 100, dtype=np.int32)
    tr = _make_trace(ramp)
    raw = _trace_to_mseed_bytes(tr, encoding="STEIM1")
    meta = _parse_record_metadata(raw)
    data_offset: int = meta["data_offset"]
    data_section = raw[data_offset:512]

    vectors.append(
        {
            "name": "ramp_100",
            "num_samples": int(meta["num_samples"]),
            "byte_order": "big",
            "compressed_data_b64": _b64(data_section),
            "expected_samples": ramp.tolist(),
        }
    )

    # Pattern 2: sine wave (larger range)
    t = np.linspace(0, 4 * np.pi, 200)
    sine = (np.sin(t) * 10000).astype(np.int32)
    tr = _make_trace(sine)
    raw = _trace_to_mseed_bytes(tr, encoding="STEIM1")
    meta = _parse_record_metadata(raw)
    data_section = raw[meta["data_offset"] : 512]

    vectors.append(
        {
            "name": "sine_200",
            "num_samples": int(meta["num_samples"]),
            "byte_order": "big",
            "compressed_data_b64": _b64(data_section),
            "expected_samples": sine.tolist(),
        }
    )

    # Pattern 3: small random
    rng = np.random.default_rng(42)
    rand_data = rng.integers(-500, 500, size=80, dtype=np.int32)
    rand_data = rand_data.astype(np.int32)
    tr = _make_trace(rand_data)
    raw = _trace_to_mseed_bytes(tr, encoding="STEIM1")
    meta = _parse_record_metadata(raw)
    data_section = raw[meta["data_offset"] : 512]

    vectors.append(
        {
            "name": "random_80",
            "num_samples": int(meta["num_samples"]),
            "byte_order": "big",
            "compressed_data_b64": _b64(data_section),
            "expected_samples": rand_data.tolist(),
        }
    )

    # Pattern 4: constant value (all diffs = 0 except first)
    const_data = np.full(50, 12345, dtype=np.int32)
    tr = _make_trace(const_data)
    raw = _trace_to_mseed_bytes(tr, encoding="STEIM1")
    meta = _parse_record_metadata(raw)
    data_section = raw[meta["data_offset"] : 512]

    vectors.append(
        {
            "name": "constant_50",
            "num_samples": int(meta["num_samples"]),
            "byte_order": "big",
            "compressed_data_b64": _b64(data_section),
            "expected_samples": const_data.tolist(),
        }
    )

    _write_json("steim1_vectors.json", vectors)
    print(f"  steim1: {len(vectors)} vectors")


# ---------------------------------------------------------------------------
# Steim2 vectors
# ---------------------------------------------------------------------------


def generate_steim2_vectors() -> None:
    vectors: list[dict[str, Any]] = []

    # Pattern 1: ramp
    ramp = np.arange(0, 100, dtype=np.int32)
    tr = _make_trace(ramp)
    raw = _trace_to_mseed_bytes(tr, encoding="STEIM2")
    meta = _parse_record_metadata(raw)
    data_section = raw[meta["data_offset"] : 512]

    vectors.append(
        {
            "name": "ramp_100",
            "num_samples": int(meta["num_samples"]),
            "byte_order": "big",
            "compressed_data_b64": _b64(data_section),
            "expected_samples": ramp.tolist(),
        }
    )

    # Pattern 2: sine wave
    t = np.linspace(0, 4 * np.pi, 200)
    sine = (np.sin(t) * 10000).astype(np.int32)
    tr = _make_trace(sine)
    raw = _trace_to_mseed_bytes(tr, encoding="STEIM2")
    meta = _parse_record_metadata(raw)
    data_section = raw[meta["data_offset"] : 512]

    vectors.append(
        {
            "name": "sine_200",
            "num_samples": int(meta["num_samples"]),
            "byte_order": "big",
            "compressed_data_b64": _b64(data_section),
            "expected_samples": sine.tolist(),
        }
    )

    # Pattern 3: small diffs (to exercise 4/5/6-bit packing)
    small_diffs = np.cumsum(np.random.default_rng(99).integers(-7, 8, size=150)).astype(np.int32)
    tr = _make_trace(small_diffs)
    raw = _trace_to_mseed_bytes(tr, encoding="STEIM2")
    meta = _parse_record_metadata(raw)
    data_section = raw[meta["data_offset"] : 512]

    vectors.append(
        {
            "name": "small_diffs_150",
            "num_samples": int(meta["num_samples"]),
            "byte_order": "big",
            "compressed_data_b64": _b64(data_section),
            "expected_samples": small_diffs.tolist(),
        }
    )

    # Pattern 4: large range
    rng = np.random.default_rng(7)
    large = rng.integers(-100000, 100000, size=120, dtype=np.int32)
    tr = _make_trace(large)
    raw = _trace_to_mseed_bytes(tr, encoding="STEIM2")
    meta = _parse_record_metadata(raw)
    data_section = raw[meta["data_offset"] : 512]

    vectors.append(
        {
            "name": "large_range_120",
            "num_samples": int(meta["num_samples"]),
            "byte_order": "big",
            "compressed_data_b64": _b64(data_section),
            "expected_samples": large.tolist(),
        }
    )

    _write_json("steim2_vectors.json", vectors)
    print(f"  steim2: {len(vectors)} vectors")


# ---------------------------------------------------------------------------
# Header parsing vectors
# ---------------------------------------------------------------------------


def generate_header_vectors() -> None:
    vectors: list[dict[str, Any]] = []

    starttimes = [
        UTCDateTime(2024, 1, 15, 10, 30, 0, 0),  # type: ignore[no-untyped-call]
        UTCDateTime(2023, 6, 1, 0, 0, 0, 500000),  # type: ignore[no-untyped-call]
        UTCDateTime(2025, 12, 31, 23, 59, 59, 999900),  # type: ignore[no-untyped-call]
    ]

    cfgs = [
        ("IU", "ANMO", "00", "BHZ", 20.0, starttimes[0]),
        ("GE", "DAV", "10", "HHE", 100.0, starttimes[1]),
        ("JP", "TSK", "", "LHN", 1.0, starttimes[2]),
    ]

    for i, (net, sta, loc, cha, sr, st) in enumerate(cfgs):
        data = np.arange(0, 20, dtype=np.int32)
        tr = _make_trace(
            data,
            network=net,
            station=sta,
            location=loc,
            channel=cha,
            sampling_rate=sr,
            starttime=st,
        )
        raw = _trace_to_mseed_bytes(tr, encoding="STEIM1")
        meta = _parse_record_metadata(raw)

        vectors.append(
            {
                "name": f"header_{i}",
                "record_b64": _b64(raw[:512]),
                "expected": {
                    "network": net,
                    "station": sta,
                    "location": loc,
                    "channel": cha,
                    "sample_rate": sr,
                    "year": st.year,  # type: ignore[union-attr]
                    "day": st.julday,  # type: ignore[union-attr]
                    "hour": st.hour,  # type: ignore[union-attr]
                    "minute": st.minute,  # type: ignore[union-attr]
                    "second": st.second,  # type: ignore[union-attr]
                    "fract": int(st.microsecond / 100),  # type: ignore[union-attr]
                    "num_samples": len(data),
                    "encoding_format": 10,  # STEIM1
                    "byte_order": 1,
                    "record_length_power": 9,  # 2^9 = 512
                    **meta,
                },
            }
        )

    _write_json("header_vectors.json", vectors)
    print(f"  header: {len(vectors)} vectors")


# ---------------------------------------------------------------------------
# Uncompressed format vectors
# ---------------------------------------------------------------------------


def generate_uncompressed_vectors() -> None:
    vectors: list[dict[str, Any]] = []

    # INT16 (encoding code 1)
    int16_data = np.array([0, 1, -1, 32767, -32768, 100, -200], dtype=np.int16)
    tr = _make_trace(int16_data)
    raw = _trace_to_mseed_bytes(tr, encoding="INT16")
    meta = _parse_record_metadata(raw)
    vectors.append(
        {
            "name": "int16",
            "record_b64": _b64(raw[:512]),
            "encoding": 1,
            "num_samples": len(int16_data),
            "expected_samples": int16_data.tolist(),
            "data_offset": meta["data_offset"],
        }
    )

    # INT32 (encoding code 3)
    int32_data = np.array([0, 1, -1, 2147483647, -2147483648, 100000, -999999], dtype=np.int32)
    tr = _make_trace(int32_data)
    raw = _trace_to_mseed_bytes(tr, encoding="INT32")
    meta = _parse_record_metadata(raw)
    vectors.append(
        {
            "name": "int32",
            "record_b64": _b64(raw[:512]),
            "encoding": 3,
            "num_samples": len(int32_data),
            "expected_samples": int32_data.tolist(),
            "data_offset": meta["data_offset"],
        }
    )

    # FLOAT32 (encoding code 4)
    float32_data = np.array([0.0, 1.5, -1.5, 3.14159, 1e10, -1e-5], dtype=np.float32)
    tr = _make_trace(float32_data)
    raw = _trace_to_mseed_bytes(tr, encoding="FLOAT32")
    meta = _parse_record_metadata(raw)
    vectors.append(
        {
            "name": "float32",
            "record_b64": _b64(raw[:512]),
            "encoding": 4,
            "num_samples": len(float32_data),
            "expected_samples": float32_data.tolist(),
            "data_offset": meta["data_offset"],
        }
    )

    # FLOAT64 (encoding code 5)
    float64_data = np.array([0.0, 1.5, -1.5, 3.141592653589793, 1e100, -1e-15], dtype=np.float64)
    tr = _make_trace(float64_data)
    raw = _trace_to_mseed_bytes(tr, encoding="FLOAT64")
    meta = _parse_record_metadata(raw)
    vectors.append(
        {
            "name": "float64",
            "record_b64": _b64(raw[:512]),
            "encoding": 5,
            "num_samples": len(float64_data),
            "expected_samples": float64_data.tolist(),
            "data_offset": meta["data_offset"],
        }
    )

    _write_json("uncompressed_vectors.json", vectors)
    print(f"  uncompressed: {len(vectors)} vectors")


# ---------------------------------------------------------------------------
# Roundtrip vectors (full records for encode→decode testing)
# ---------------------------------------------------------------------------


def generate_roundtrip_vectors() -> None:
    vectors: list[dict[str, Any]] = []

    configs: list[tuple[str, str, NDArray[Any]]] = [
        ("steim1", "STEIM1", np.arange(0, 80, dtype=np.int32)),
        ("steim2", "STEIM2", np.arange(0, 80, dtype=np.int32)),
        ("int16", "INT16", np.array([0, 100, -100, 32767, -32768], dtype=np.int16)),
        ("int32", "INT32", np.array([0, 100000, -100000, 2147483647], dtype=np.int32)),
        ("float32", "FLOAT32", np.array([0.0, 1.5, -3.14], dtype=np.float32)),
        ("float64", "FLOAT64", np.array([0.0, 1.5e100, -3.14e-50], dtype=np.float64)),
    ]

    for name, enc, data in configs:
        tr = _make_trace(
            data,
            network="TS",
            station="TEST",
            location="00",
            channel="BHZ",
            sampling_rate=20.0,
            starttime=UTCDateTime(2024, 7, 4, 12, 0, 0, 0),  # type: ignore[no-untyped-call]
        )
        raw = _trace_to_mseed_bytes(tr, encoding=enc)
        meta = _parse_record_metadata(raw)

        vectors.append(
            {
                "name": name,
                "record_b64": _b64(raw[:512]),
                "encoding": meta["encoding_format"],
                "num_samples": int(meta["num_samples"]),
                "expected_samples": data.tolist(),
                "network": "TS",
                "station": "TEST",
                "location": "00",
                "channel": "BHZ",
                "sample_rate": 20.0,
            }
        )

    _write_json("roundtrip_vectors.json", vectors)
    print(f"  roundtrip: {len(vectors)} vectors")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _write_json(filename: str, data: object) -> None:
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    path = VECTORS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    print("Generating test vectors...")
    generate_steim1_vectors()
    generate_steim2_vectors()
    generate_header_vectors()
    generate_uncompressed_vectors()
    generate_roundtrip_vectors()
    print(f"Done. Vectors written to {VECTORS_DIR}")


if __name__ == "__main__":
    main()
