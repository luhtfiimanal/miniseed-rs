"""Generate test vectors for miniseed-rs using pymseed (libmseed) as oracle.

Produces JSON files in pyscripts/test_vectors/ that Rust tests load
to validate decode/encode against known-good libmseed output.

Supports both miniSEED v2 and v3 test vectors.
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pymseed import (
    DataEncoding,
    MS3TraceList,
    nslc2sourceid,
    timestr2nstime,
)

VECTORS_DIR = Path(__file__).resolve().parent.parent.parent / "test_vectors"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _make_v2_record(
    data: NDArray[Any],
    *,
    network: str = "IU",
    station: str = "ANMO",
    location: str = "00",
    channel: str = "BHZ",
    sampling_rate: float = 20.0,
    starttime: str = "2024-01-15T10:30:00.000000Z",
    encoding: DataEncoding = DataEncoding.STEIM1,
    reclen: int = 512,
) -> bytes:
    """Create a v2 miniSEED record and return raw bytes."""
    sid = nslc2sourceid(network, station, location, channel)
    sample_type = _numpy_to_sample_type(data)
    nstime = timestr2nstime(starttime)

    tl = MS3TraceList()
    tl.add_data(
        sourceid=sid,
        data_samples=data.tolist(),
        sample_type=sample_type,
        sample_rate=sampling_rate,
        start_time=nstime,
    )

    records: list[bytes] = []
    for rec_bytes in tl.generate(
        record_length=reclen,
        encoding=encoding,
        format_version=2,
    ):
        records.append(rec_bytes)

    if not records:
        msg = "No records generated"
        raise RuntimeError(msg)

    return records[0][:reclen]


def _make_v3_record(
    data: NDArray[Any],
    *,
    network: str = "IU",
    station: str = "ANMO",
    location: str = "00",
    channel: str = "BHZ",
    sampling_rate: float = 20.0,
    starttime: str = "2024-01-15T10:30:00.000000Z",
    encoding: DataEncoding = DataEncoding.STEIM1,
    reclen: int = 4096,
) -> bytes:
    """Create a v3 miniSEED record and return raw bytes."""
    sid = nslc2sourceid(network, station, location, channel)
    sample_type = _numpy_to_sample_type(data)
    nstime = timestr2nstime(starttime)

    tl = MS3TraceList()
    tl.add_data(
        sourceid=sid,
        data_samples=data.tolist(),
        sample_type=sample_type,
        sample_rate=sampling_rate,
        start_time=nstime,
    )

    records: list[bytes] = []
    for rec_bytes in tl.generate(
        record_length=reclen,
        encoding=encoding,
        format_version=3,
    ):
        records.append(rec_bytes)

    if not records:
        msg = "No records generated"
        raise RuntimeError(msg)

    return records[0]


def _numpy_to_sample_type(data: NDArray[Any]) -> str:
    """Map numpy dtype to pymseed sample_type."""
    if data.dtype == np.int16 or data.dtype == np.int32:
        return "i"
    if data.dtype == np.float32:
        return "f"
    if data.dtype == np.float64:
        return "d"
    msg = f"Unsupported dtype: {data.dtype}"
    raise ValueError(msg)


def _parse_v2_record_metadata(raw: bytes) -> dict[str, Any]:
    """Extract metadata from raw miniSEED v2 record for test vector."""
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


def _parse_v3_header(raw: bytes) -> dict[str, Any]:
    """Extract metadata from raw miniSEED v3 record header."""
    assert raw[0:2] == b"MS", f"Expected 'MS' magic, got {raw[0:2]!r}"
    assert raw[2] == 3, f"Expected version 3, got {raw[2]}"

    flags = raw[3]
    nanosecond: int = struct.unpack("<I", raw[4:8])[0]
    year: int = struct.unpack("<H", raw[8:10])[0]
    day: int = struct.unpack("<H", raw[10:12])[0]
    hour = raw[12]
    minute = raw[13]
    second = raw[14]
    encoding_format = raw[15]
    sample_rate: float = struct.unpack("<d", raw[16:24])[0]
    num_samples: int = struct.unpack("<I", raw[24:28])[0]
    crc: int = struct.unpack("<I", raw[28:32])[0]
    pub_version = raw[32]
    sid_length = raw[33]
    extra_length: int = struct.unpack("<H", raw[34:36])[0]
    data_length: int = struct.unpack("<I", raw[36:40])[0]

    sid = raw[40 : 40 + sid_length].decode("ascii")
    extra_offset = 40 + sid_length
    extra_headers = raw[extra_offset : extra_offset + extra_length].decode("utf-8")
    data_offset = extra_offset + extra_length

    return {
        "flags": flags,
        "nanosecond": nanosecond,
        "year": year,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
        "encoding_format": encoding_format,
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "crc": crc,
        "pub_version": pub_version,
        "sid_length": sid_length,
        "extra_length": extra_length,
        "data_length": data_length,
        "sid": sid,
        "extra_headers": extra_headers,
        "data_offset": data_offset,
        "record_length": len(raw),
    }


# ---------------------------------------------------------------------------
# v2 Steim1 vectors
# ---------------------------------------------------------------------------


def generate_steim1_vectors() -> None:
    vectors: list[dict[str, Any]] = []

    # Pattern 1: ramp
    ramp = np.arange(0, 100, dtype=np.int32)
    raw = _make_v2_record(ramp, encoding=DataEncoding.STEIM1)
    meta = _parse_v2_record_metadata(raw)
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

    # Pattern 2: sine wave (larger range)
    t = np.linspace(0, 4 * np.pi, 200)
    sine = (np.sin(t) * 10000).astype(np.int32)
    raw = _make_v2_record(sine, encoding=DataEncoding.STEIM1)
    meta = _parse_v2_record_metadata(raw)
    reclen = 2 ** meta["record_length_power"]
    data_section = raw[meta["data_offset"] : reclen]

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
    raw = _make_v2_record(rand_data, encoding=DataEncoding.STEIM1)
    meta = _parse_v2_record_metadata(raw)
    reclen = 2 ** meta["record_length_power"]
    data_section = raw[meta["data_offset"] : reclen]

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
    raw = _make_v2_record(const_data, encoding=DataEncoding.STEIM1)
    meta = _parse_v2_record_metadata(raw)
    reclen = 2 ** meta["record_length_power"]
    data_section = raw[meta["data_offset"] : reclen]

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
# v2 Steim2 vectors
# ---------------------------------------------------------------------------


def generate_steim2_vectors() -> None:
    vectors: list[dict[str, Any]] = []

    # Pattern 1: ramp
    ramp = np.arange(0, 100, dtype=np.int32)
    raw = _make_v2_record(ramp, encoding=DataEncoding.STEIM2)
    meta = _parse_v2_record_metadata(raw)
    reclen = 2 ** meta["record_length_power"]
    data_section = raw[meta["data_offset"] : reclen]

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
    raw = _make_v2_record(sine, encoding=DataEncoding.STEIM2)
    meta = _parse_v2_record_metadata(raw)
    reclen = 2 ** meta["record_length_power"]
    data_section = raw[meta["data_offset"] : reclen]

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
    raw = _make_v2_record(small_diffs, encoding=DataEncoding.STEIM2)
    meta = _parse_v2_record_metadata(raw)
    reclen = 2 ** meta["record_length_power"]
    data_section = raw[meta["data_offset"] : reclen]

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
    large = rng.integers(-100000, 100000, size=60, dtype=np.int32)
    raw = _make_v2_record(large, encoding=DataEncoding.STEIM2)
    meta = _parse_v2_record_metadata(raw)
    reclen = 2 ** meta["record_length_power"]
    data_section = raw[meta["data_offset"] : reclen]

    vectors.append(
        {
            "name": "large_range_60",
            "num_samples": int(meta["num_samples"]),
            "byte_order": "big",
            "compressed_data_b64": _b64(data_section),
            "expected_samples": large.tolist(),
        }
    )

    _write_json("steim2_vectors.json", vectors)
    print(f"  steim2: {len(vectors)} vectors")


# ---------------------------------------------------------------------------
# v2 Header parsing vectors
# ---------------------------------------------------------------------------


def generate_header_vectors() -> None:
    vectors: list[dict[str, Any]] = []

    starttimes = [
        "2024-01-15T10:30:00.000000Z",
        "2023-06-01T00:00:00.500000Z",
        "2025-12-31T23:59:59.999900Z",
    ]

    cfgs = [
        ("IU", "ANMO", "00", "BHZ", 20.0, starttimes[0]),
        ("GE", "DAV", "10", "HHE", 100.0, starttimes[1]),
        ("JP", "TSK", "", "LHN", 1.0, starttimes[2]),
    ]

    for i, (net, sta, loc, cha, sr, st) in enumerate(cfgs):
        data = np.arange(0, 20, dtype=np.int32)
        raw = _make_v2_record(
            data,
            network=net,
            station=sta,
            location=loc,
            channel=cha,
            sampling_rate=sr,
            starttime=st,
            encoding=DataEncoding.STEIM1,
        )
        meta = _parse_v2_record_metadata(raw)

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
                    "year": meta["year"],
                    "day": meta["day"],
                    "hour": meta["hour"],
                    "minute": meta["minute"],
                    "second": meta["second"],
                    "fract": meta["fract"],
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
# v2 Uncompressed format vectors
# ---------------------------------------------------------------------------


def generate_uncompressed_vectors() -> None:
    vectors: list[dict[str, Any]] = []

    # INT16 (encoding code 1)
    int16_data = np.array([0, 1, -1, 32767, -32768, 100, -200], dtype=np.int16)
    raw = _make_v2_record(int16_data, encoding=DataEncoding.INT16)
    meta = _parse_v2_record_metadata(raw)
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
    raw = _make_v2_record(int32_data, encoding=DataEncoding.INT32)
    meta = _parse_v2_record_metadata(raw)
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
    raw = _make_v2_record(float32_data, encoding=DataEncoding.FLOAT32)
    meta = _parse_v2_record_metadata(raw)
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
    raw = _make_v2_record(float64_data, encoding=DataEncoding.FLOAT64)
    meta = _parse_v2_record_metadata(raw)
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
# v2 Roundtrip vectors (full records for encode->decode testing)
# ---------------------------------------------------------------------------


def generate_roundtrip_vectors() -> None:
    vectors: list[dict[str, Any]] = []

    configs: list[tuple[str, DataEncoding, NDArray[Any]]] = [
        ("steim1", DataEncoding.STEIM1, np.arange(0, 80, dtype=np.int32)),
        ("steim2", DataEncoding.STEIM2, np.arange(0, 80, dtype=np.int32)),
        ("int16", DataEncoding.INT16, np.array([0, 100, -100, 32767, -32768], dtype=np.int16)),
        ("int32", DataEncoding.INT32, np.array([0, 100000, -100000, 2147483647], dtype=np.int32)),
        ("float32", DataEncoding.FLOAT32, np.array([0.0, 1.5, -3.14], dtype=np.float32)),
        ("float64", DataEncoding.FLOAT64, np.array([0.0, 1.5e100, -3.14e-50], dtype=np.float64)),
    ]

    for name, enc, data in configs:
        raw = _make_v2_record(
            data,
            network="TS",
            station="TEST",
            location="00",
            channel="BHZ",
            sampling_rate=20.0,
            starttime="2024-07-04T12:00:00.000000Z",
            encoding=enc,
        )
        meta = _parse_v2_record_metadata(raw)

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
# v2 4096-byte record vectors
# ---------------------------------------------------------------------------


def generate_4096_vectors() -> None:
    vectors: list[dict[str, Any]] = []

    # Steim1 4096-byte record (~900 samples fit)
    rng = np.random.default_rng(2024)
    steim1_data = np.cumsum(rng.integers(-500, 500, size=900)).astype(np.int32)
    raw = _make_v2_record(steim1_data, station="S4K1", encoding=DataEncoding.STEIM1, reclen=4096)
    meta = _parse_v2_record_metadata(raw)
    assert 2 ** meta["record_length_power"] == 4096, "record length must be 4096"

    vectors.append(
        {
            "name": "steim1_4096",
            "record_b64": _b64(raw[:4096]),
            "encoding": meta["encoding_format"],
            "num_samples": int(meta["num_samples"]),
            "expected_samples": steim1_data.tolist(),
            "record_length": 4096,
            "network": "IU",
            "station": "S4K1",
        }
    )

    # Steim2 4096-byte record
    steim2_data = np.cumsum(rng.integers(-100, 100, size=800)).astype(np.int32)
    raw = _make_v2_record(steim2_data, station="S4K2", encoding=DataEncoding.STEIM2, reclen=4096)
    meta = _parse_v2_record_metadata(raw)

    vectors.append(
        {
            "name": "steim2_4096",
            "record_b64": _b64(raw[:4096]),
            "encoding": meta["encoding_format"],
            "num_samples": int(meta["num_samples"]),
            "expected_samples": steim2_data.tolist(),
            "record_length": 4096,
            "network": "IU",
            "station": "S4K2",
        }
    )

    # INT32 4096-byte record
    int32_data = rng.integers(-1000000, 1000000, size=500, dtype=np.int32)
    raw = _make_v2_record(int32_data, station="S4K3", encoding=DataEncoding.INT32, reclen=4096)
    meta = _parse_v2_record_metadata(raw)

    vectors.append(
        {
            "name": "int32_4096",
            "record_b64": _b64(raw[:4096]),
            "encoding": meta["encoding_format"],
            "num_samples": int(meta["num_samples"]),
            "expected_samples": int32_data.tolist(),
            "record_length": 4096,
            "network": "IU",
            "station": "S4K3",
        }
    )

    _write_json("record_4096_vectors.json", vectors)
    print(f"  4096-byte: {len(vectors)} vectors")


def generate_mixed_record_stream() -> None:
    """Generate a stream with 512-byte + 4096-byte v2 records concatenated."""
    vectors: list[dict[str, Any]] = []

    # 512-byte record
    data_512 = np.arange(0, 50, dtype=np.int32)
    raw_512 = _make_v2_record(data_512, station="MIX1", encoding=DataEncoding.STEIM1, reclen=512)

    # 4096-byte record
    data_4096 = np.arange(0, 200, dtype=np.int32)
    raw_4096 = _make_v2_record(data_4096, station="MIX2", encoding=DataEncoding.STEIM1, reclen=4096)

    # Concatenated stream
    stream_bytes = raw_512[:512] + raw_4096[:4096]

    vectors.append(
        {
            "name": "mixed_512_4096",
            "stream_b64": _b64(stream_bytes),
            "records": [
                {
                    "station": "MIX1",
                    "num_samples": len(data_512),
                    "record_length": 512,
                    "expected_samples": data_512.tolist(),
                },
                {
                    "station": "MIX2",
                    "num_samples": len(data_4096),
                    "record_length": 4096,
                    "expected_samples": data_4096.tolist(),
                },
            ],
        }
    )

    _write_json("mixed_record_vectors.json", vectors)
    print(f"  mixed records: {len(vectors)} vectors")


# ---------------------------------------------------------------------------
# v3 vectors
# ---------------------------------------------------------------------------


def generate_v3_header_vectors() -> None:
    """Generate v3 header parsing test vectors."""
    vectors: list[dict[str, Any]] = []

    cfgs = [
        ("IU", "ANMO", "00", "BHZ", 20.0, "2024-01-15T10:30:00.000000Z"),
        ("GE", "DAV", "10", "HHE", 100.0, "2023-06-01T00:00:00.500000Z"),
        ("JP", "TSK", "", "LHN", 1.0, "2025-12-31T23:59:59.999900Z"),
    ]

    for i, (net, sta, loc, cha, sr, st) in enumerate(cfgs):
        data = np.arange(0, 20, dtype=np.int32)
        raw = _make_v3_record(
            data,
            network=net,
            station=sta,
            location=loc,
            channel=cha,
            sampling_rate=sr,
            starttime=st,
            encoding=DataEncoding.STEIM2,
        )
        meta = _parse_v3_header(raw)
        sid = nslc2sourceid(net, sta, loc, cha)

        vectors.append(
            {
                "name": f"v3_header_{i}",
                "record_b64": _b64(raw),
                "record_length": len(raw),
                "expected": {
                    "sid": sid,
                    "network": net,
                    "station": sta,
                    "location": loc,
                    "channel": cha,
                    "sample_rate": sr,
                    "year": meta["year"],
                    "day": meta["day"],
                    "hour": meta["hour"],
                    "minute": meta["minute"],
                    "second": meta["second"],
                    "nanosecond": meta["nanosecond"],
                    "encoding_format": meta["encoding_format"],
                    "num_samples": len(data),
                    "flags": meta["flags"],
                    "pub_version": meta["pub_version"],
                },
            }
        )

    _write_json("v3_header_vectors.json", vectors)
    print(f"  v3 header: {len(vectors)} vectors")


def generate_v3_steim_vectors() -> None:
    """Generate v3 Steim1 and Steim2 decode test vectors."""
    vectors_s1: list[dict[str, Any]] = []
    vectors_s2: list[dict[str, Any]] = []

    # Steim1 patterns
    patterns_s1 = [
        ("ramp_100", np.arange(0, 100, dtype=np.int32)),
        ("sine_200", (np.sin(np.linspace(0, 4 * np.pi, 200)) * 10000).astype(np.int32)),
        (
            "random_80",
            np.random.default_rng(42).integers(-500, 500, size=80, dtype=np.int32),
        ),
        ("constant_50", np.full(50, 12345, dtype=np.int32)),
    ]

    for name, data in patterns_s1:
        raw = _make_v3_record(data, encoding=DataEncoding.STEIM1)
        meta = _parse_v3_header(raw)
        data_section = raw[meta["data_offset"] :]

        vectors_s1.append(
            {
                "name": name,
                "record_b64": _b64(raw),
                "num_samples": int(meta["num_samples"]),
                "compressed_data_b64": _b64(data_section),
                "expected_samples": data.tolist(),
            }
        )

    _write_json("v3_steim1_vectors.json", vectors_s1)
    print(f"  v3 steim1: {len(vectors_s1)} vectors")

    # Steim2 patterns
    patterns_s2 = [
        ("ramp_100", np.arange(0, 100, dtype=np.int32)),
        ("sine_200", (np.sin(np.linspace(0, 4 * np.pi, 200)) * 10000).astype(np.int32)),
        (
            "small_diffs_150",
            np.cumsum(np.random.default_rng(99).integers(-7, 8, size=150)).astype(np.int32),
        ),
        (
            "large_range_60",
            np.random.default_rng(7).integers(-100000, 100000, size=60, dtype=np.int32),
        ),
    ]

    for name, data in patterns_s2:
        raw = _make_v3_record(data, encoding=DataEncoding.STEIM2)
        meta = _parse_v3_header(raw)
        data_section = raw[meta["data_offset"] :]

        vectors_s2.append(
            {
                "name": name,
                "record_b64": _b64(raw),
                "num_samples": int(meta["num_samples"]),
                "compressed_data_b64": _b64(data_section),
                "expected_samples": data.tolist(),
            }
        )

    _write_json("v3_steim2_vectors.json", vectors_s2)
    print(f"  v3 steim2: {len(vectors_s2)} vectors")


def generate_v3_uncompressed_vectors() -> None:
    """Generate v3 uncompressed format test vectors (INT16, INT32, FLOAT32, FLOAT64)."""
    vectors: list[dict[str, Any]] = []

    int16_data = np.array([0, 1, -1, 32767, -32768, 100, -200], dtype=np.int16)
    int32_data = np.array([0, 1, -1, 2147483647, -2147483648, 100000, -999999], dtype=np.int32)
    float32_data = np.array([0.0, 1.5, -1.5, 3.14159, 1e10, -1e-5], dtype=np.float32)
    float64_data = np.array([0.0, 1.5, -1.5, 3.141592653589793, 1e100, -1e-15], dtype=np.float64)

    cases: list[tuple[str, DataEncoding, int, NDArray[Any]]] = [
        ("int16", DataEncoding.INT16, 1, int16_data),
        ("int32", DataEncoding.INT32, 3, int32_data),
        ("float32", DataEncoding.FLOAT32, 4, float32_data),
        ("float64", DataEncoding.FLOAT64, 5, float64_data),
    ]

    for name, enc, code, data in cases:
        raw = _make_v3_record(data, encoding=enc)
        meta = _parse_v3_header(raw)
        assert meta["encoding_format"] == code, (
            f"Expected encoding {code}, got {meta['encoding_format']}"
        )

        vectors.append(
            {
                "name": name,
                "record_b64": _b64(raw),
                "record_length": len(raw),
                "encoding": code,
                "num_samples": len(data),
                "expected_samples": data.tolist(),
                "data_offset": meta["data_offset"],
            }
        )

    _write_json("v3_uncompressed_vectors.json", vectors)
    print(f"  v3 uncompressed: {len(vectors)} vectors")


def generate_v3_roundtrip_vectors() -> None:
    """Generate v3 encode/decode roundtrip vectors."""
    vectors: list[dict[str, Any]] = []

    configs: list[tuple[str, DataEncoding, NDArray[Any]]] = [
        ("steim1", DataEncoding.STEIM1, np.arange(0, 80, dtype=np.int32)),
        ("steim2", DataEncoding.STEIM2, np.arange(0, 80, dtype=np.int32)),
        ("int16", DataEncoding.INT16, np.array([0, 100, -100, 32767, -32768], dtype=np.int16)),
        ("int32", DataEncoding.INT32, np.array([0, 100000, -100000, 2147483647], dtype=np.int32)),
        ("float32", DataEncoding.FLOAT32, np.array([0.0, 1.5, -3.14], dtype=np.float32)),
        ("float64", DataEncoding.FLOAT64, np.array([0.0, 1.5e100, -3.14e-50], dtype=np.float64)),
    ]

    for name, enc, data in configs:
        raw = _make_v3_record(
            data,
            network="TS",
            station="TEST",
            location="00",
            channel="BHZ",
            sampling_rate=20.0,
            starttime="2024-07-04T12:00:00.000000Z",
            encoding=enc,
        )

        vectors.append(
            {
                "name": name,
                "record_b64": _b64(raw),
                "record_length": len(raw),
                "num_samples": len(data),
                "expected_samples": data.tolist(),
                "network": "TS",
                "station": "TEST",
                "location": "00",
                "channel": "BHZ",
                "sample_rate": 20.0,
            }
        )

    _write_json("v3_roundtrip_vectors.json", vectors)
    print(f"  v3 roundtrip: {len(vectors)} vectors")


def generate_v3_crc_vectors() -> None:
    """Generate v3 CRC-32C validation vectors."""
    vectors: list[dict[str, Any]] = []

    patterns: list[tuple[str, NDArray[Any], DataEncoding]] = [
        ("steim1_ramp", np.arange(0, 50, dtype=np.int32), DataEncoding.STEIM1),
        ("int32_simple", np.array([1, 2, 3, 4, 5], dtype=np.int32), DataEncoding.INT32),
        ("float64_pi", np.array([3.141592653589793], dtype=np.float64), DataEncoding.FLOAT64),
    ]

    for name, data, enc in patterns:
        raw = _make_v3_record(data, encoding=enc)
        meta = _parse_v3_header(raw)

        vectors.append(
            {
                "name": name,
                "record_b64": _b64(raw),
                "record_length": len(raw),
                "crc": meta["crc"],
            }
        )

    _write_json("v3_crc_vectors.json", vectors)
    print(f"  v3 crc: {len(vectors)} vectors")


def generate_mixed_v2v3_vectors() -> None:
    """Generate mixed v2+v3 stream for auto-detection testing."""
    vectors: list[dict[str, Any]] = []

    # v2 512-byte record
    data_v2 = np.arange(0, 30, dtype=np.int32)
    raw_v2 = _make_v2_record(data_v2, station="V2ST", encoding=DataEncoding.STEIM1, reclen=512)

    # v3 record
    data_v3 = np.arange(0, 30, dtype=np.int32)
    raw_v3 = _make_v3_record(data_v3, station="V3ST", encoding=DataEncoding.STEIM2)

    stream_bytes = raw_v2[:512] + raw_v3

    vectors.append(
        {
            "name": "mixed_v2_v3",
            "stream_b64": _b64(stream_bytes),
            "records": [
                {
                    "station": "V2ST",
                    "format_version": 2,
                    "num_samples": len(data_v2),
                    "record_length": 512,
                },
                {
                    "station": "V3ST",
                    "format_version": 3,
                    "num_samples": len(data_v3),
                    "record_length": len(raw_v3),
                },
            ],
        }
    )

    _write_json("mixed_v2v3_vectors.json", vectors)
    print(f"  mixed v2+v3: {len(vectors)} vectors")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _write_json(filename: str, data: object) -> None:
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    path = VECTORS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    print("Generating test vectors (pymseed oracle)...")

    # v2 vectors (regenerated from pymseed)
    generate_steim1_vectors()
    generate_steim2_vectors()
    generate_header_vectors()
    generate_uncompressed_vectors()
    generate_roundtrip_vectors()
    generate_4096_vectors()
    generate_mixed_record_stream()

    # v3 vectors (new)
    generate_v3_header_vectors()
    generate_v3_steim_vectors()
    generate_v3_uncompressed_vectors()
    generate_v3_roundtrip_vectors()
    generate_v3_crc_vectors()
    generate_mixed_v2v3_vectors()

    print(f"Done. Vectors written to {VECTORS_DIR}")


if __name__ == "__main__":
    main()
