use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use miniseed_rs::{EncodingFormat, MseedReader, MseedRecord, NanoTime, Samples, decode, encode};

/// Generate realistic seismic-like samples (smooth with small diffs, good for Steim).
fn seismic_samples(n: usize) -> Vec<i32> {
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        // Slow sinusoidal drift + small noise-like variation
        let drift = (i as f64 * 0.05).sin() * 50.0;
        let noise = ((i as f64 * 1.7).sin() * 10.0) as i32;
        v.push(1000 + drift as i32 + noise);
    }
    v
}

fn make_v2_record(encoding: EncodingFormat, samples: Samples) -> Vec<u8> {
    let record = MseedRecord::new()
        .with_nslc("IU", "ANMO", "00", "BHZ")
        .with_start_time(NanoTime {
            year: 2025,
            day: 100,
            hour: 12,
            minute: 0,
            second: 0,
            nanosecond: 0,
        })
        .with_sample_rate(100.0)
        .with_encoding(encoding)
        .with_samples(samples);
    encode(&record).unwrap()
}

fn make_v3_record(encoding: EncodingFormat, samples: Samples) -> Vec<u8> {
    let record = MseedRecord::new_v3()
        .with_nslc("IU", "ANMO", "00", "BHZ")
        .with_start_time(NanoTime {
            year: 2025,
            day: 100,
            hour: 12,
            minute: 0,
            second: 0,
            nanosecond: 0,
        })
        .with_sample_rate(100.0)
        .with_encoding(encoding)
        .with_samples(samples);
    encode(&record).unwrap()
}

fn bench_decode(c: &mut Criterion) {
    let samples_100 = seismic_samples(100);

    // Pre-encode records for decode benchmarks
    let v2_steim1 = make_v2_record(EncodingFormat::Steim1, Samples::Int(samples_100.clone()));
    let v2_steim2 = make_v2_record(EncodingFormat::Steim2, Samples::Int(samples_100.clone()));
    let v2_int32 = make_v2_record(EncodingFormat::Int32, Samples::Int(samples_100.clone()));
    let v3_steim1 = make_v3_record(EncodingFormat::Steim1, Samples::Int(samples_100.clone()));
    let v3_steim2 = make_v3_record(EncodingFormat::Steim2, Samples::Int(samples_100.clone()));
    let v3_int32 = make_v3_record(EncodingFormat::Int32, Samples::Int(samples_100.clone()));

    let mut group = c.benchmark_group("decode");

    group.throughput(Throughput::Elements(100));

    group.bench_function("v2/steim1/100samp", |b| {
        b.iter(|| decode(black_box(&v2_steim1)).unwrap())
    });
    group.bench_function("v2/steim2/100samp", |b| {
        b.iter(|| decode(black_box(&v2_steim2)).unwrap())
    });
    group.bench_function("v2/int32/100samp", |b| {
        b.iter(|| decode(black_box(&v2_int32)).unwrap())
    });
    group.bench_function("v3/steim1/100samp", |b| {
        b.iter(|| decode(black_box(&v3_steim1)).unwrap())
    });
    group.bench_function("v3/steim2/100samp", |b| {
        b.iter(|| decode(black_box(&v3_steim2)).unwrap())
    });
    group.bench_function("v3/int32/100samp", |b| {
        b.iter(|| decode(black_box(&v3_int32)).unwrap())
    });

    group.finish();
}

fn bench_encode(c: &mut Criterion) {
    let samples_100 = seismic_samples(100);

    let rec_v2_steim1 = MseedRecord::new()
        .with_nslc("IU", "ANMO", "00", "BHZ")
        .with_sample_rate(100.0)
        .with_encoding(EncodingFormat::Steim1)
        .with_samples(Samples::Int(samples_100.clone()));
    let rec_v2_steim2 = MseedRecord::new()
        .with_nslc("IU", "ANMO", "00", "BHZ")
        .with_sample_rate(100.0)
        .with_encoding(EncodingFormat::Steim2)
        .with_samples(Samples::Int(samples_100.clone()));
    let rec_v2_int32 = MseedRecord::new()
        .with_nslc("IU", "ANMO", "00", "BHZ")
        .with_sample_rate(100.0)
        .with_encoding(EncodingFormat::Int32)
        .with_samples(Samples::Int(samples_100.clone()));
    let rec_v3_steim1 = MseedRecord::new_v3()
        .with_nslc("IU", "ANMO", "00", "BHZ")
        .with_sample_rate(100.0)
        .with_encoding(EncodingFormat::Steim1)
        .with_samples(Samples::Int(samples_100.clone()));
    let rec_v3_steim2 = MseedRecord::new_v3()
        .with_nslc("IU", "ANMO", "00", "BHZ")
        .with_sample_rate(100.0)
        .with_encoding(EncodingFormat::Steim2)
        .with_samples(Samples::Int(samples_100.clone()));
    let rec_v3_int32 = MseedRecord::new_v3()
        .with_nslc("IU", "ANMO", "00", "BHZ")
        .with_sample_rate(100.0)
        .with_encoding(EncodingFormat::Int32)
        .with_samples(Samples::Int(samples_100.clone()));

    let mut group = c.benchmark_group("encode");

    group.throughput(Throughput::Elements(100));

    group.bench_function("v2/steim1/100samp", |b| {
        b.iter(|| encode(black_box(&rec_v2_steim1)).unwrap())
    });
    group.bench_function("v2/steim2/100samp", |b| {
        b.iter(|| encode(black_box(&rec_v2_steim2)).unwrap())
    });
    group.bench_function("v2/int32/100samp", |b| {
        b.iter(|| encode(black_box(&rec_v2_int32)).unwrap())
    });
    group.bench_function("v3/steim1/100samp", |b| {
        b.iter(|| encode(black_box(&rec_v3_steim1)).unwrap())
    });
    group.bench_function("v3/steim2/100samp", |b| {
        b.iter(|| encode(black_box(&rec_v3_steim2)).unwrap())
    });
    group.bench_function("v3/int32/100samp", |b| {
        b.iter(|| encode(black_box(&rec_v3_int32)).unwrap())
    });

    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let samples_100 = seismic_samples(100);

    let rec_v2 = MseedRecord::new()
        .with_nslc("IU", "ANMO", "00", "BHZ")
        .with_sample_rate(100.0)
        .with_encoding(EncodingFormat::Steim2)
        .with_samples(Samples::Int(samples_100.clone()));
    let rec_v3 = MseedRecord::new_v3()
        .with_nslc("IU", "ANMO", "00", "BHZ")
        .with_sample_rate(100.0)
        .with_encoding(EncodingFormat::Steim2)
        .with_samples(Samples::Int(samples_100.clone()));

    let mut group = c.benchmark_group("roundtrip");

    group.throughput(Throughput::Elements(100));

    group.bench_function("v2/steim2/100samp", |b| {
        b.iter(|| {
            let bytes = encode(black_box(&rec_v2)).unwrap();
            decode(black_box(&bytes)).unwrap()
        })
    });
    group.bench_function("v3/steim2/100samp", |b| {
        b.iter(|| {
            let bytes = encode(black_box(&rec_v3)).unwrap();
            decode(black_box(&bytes)).unwrap()
        })
    });

    group.finish();
}

fn bench_reader(c: &mut Criterion) {
    let samples = seismic_samples(100);

    // Build a stream of 10 records (mixed v2+v3)
    let mut stream = Vec::new();
    for i in 0..10 {
        let rec = if i % 2 == 0 {
            MseedRecord::new()
                .with_nslc("IU", "ANMO", "00", "BHZ")
                .with_sample_rate(100.0)
                .with_encoding(EncodingFormat::Steim2)
                .with_samples(Samples::Int(samples.clone()))
        } else {
            MseedRecord::new_v3()
                .with_nslc("IU", "ANMO", "00", "BHZ")
                .with_sample_rate(100.0)
                .with_encoding(EncodingFormat::Steim2)
                .with_samples(Samples::Int(samples.clone()))
        };
        stream.extend_from_slice(&encode(&rec).unwrap());
    }

    c.bench_function("reader/mixed_10rec", |b| {
        b.iter(|| {
            let records: Vec<_> = MseedReader::new(black_box(&stream))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(records.len(), 10);
        })
    });
}

criterion_group!(
    benches,
    bench_decode,
    bench_encode,
    bench_roundtrip,
    bench_reader
);
criterion_main!(benches);
