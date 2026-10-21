//! End-to-end throughput benches over the spec matrix.
//!
//! Every corpus input runs both directions at block sizes 1 MiB and
//! 16 MiB, the matrix from the port spec. Decode inputs are encoded
//! once during setup, so the timer covers decode only. The corpus is
//! built inline with fixed seeds, so results stay deterministic.
//!
//! Corpus: `text` (wordy input, friendly to the transform), `random`
//! (fixed-seed bytes, entropy-coder-heavy), `runs` (one long run,
//! RLE model), `periodic` (short-period repeat, LZP model). Each input
//! holds 1 MiB, one full block at 1 MiB and one block of a 16 MiB
//! state. Big cases use small sample counts to keep runtimes sane.
//!
//! Re-baseline note: Linux x86-64 is normative. Record the machine
//! when storing new baselines; re-baseline on machine change.

use std::hint::black_box;
use std::io;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;

use bzip3_rs::BlockSize;
use bzip3_rs::Jobs;
use bzip3_rs::compress;
use bzip3_rs::compress_with_jobs;
use bzip3_rs::decompress;
use bzip3_rs::decompress_with_jobs;
use bzip3_rs::read;
use bzip3_rs::stream;
use bzip3_rs::write;

/// One mebibyte in bytes.
const MIB: usize = 1024 * 1024;

/// Wordy paragraph repeated to `len` bytes.
fn text_bytes(len: usize) -> Vec<u8> {
    const PARA: &[u8] = b"This is the 100th Etext file presented by Project Gutenberg. \
        Shakespeare, William. The complete works in one file. \
        To be, or not to be, that is the question. \
        All the world's a stage, and all the men and women merely players. ";
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let take = (len - out.len()).min(PARA.len());
        out.extend_from_slice(&PARA[..take]);
    }
    out
}

/// Fixed-seed pseudo-random bytes.
fn random_bytes(len: usize) -> Vec<u8> {
    let mut seed: u32 = 0x00C0_FFEE;
    (0..len)
        .map(|_| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            u8::try_from((seed >> 16) & 0xFF).expect("masked byte fits")
        })
        .collect()
}

/// Short-period repeat (LZP model input).
fn periodic_bytes(len: usize) -> Vec<u8> {
    const PHRASE: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()";
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let take = (len - out.len()).min(PHRASE.len());
        out.extend_from_slice(&PHRASE[..take]);
    }
    out
}

/// Corpus inputs, each 1 MiB.
fn corpus() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("text", text_bytes(MIB)),
        ("random", random_bytes(MIB)),
        ("runs", vec![b'A'; MIB]),
        ("periodic", periodic_bytes(MIB)),
    ]
}

/// Block sizes from the matrix: 1 MiB and 16 MiB.
fn block_sizes() -> Vec<(&'static str, BlockSize)> {
    vec![
        ("mib1", BlockSize::mib(1).expect("1 MiB is valid")),
        ("mib16", BlockSize::mib(16).expect("16 MiB is valid")),
    ]
}

/// Encode every corpus input at every matrix block size.
fn encode_matrix(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("encode");
    group
        .sample_size(15)
        .measurement_time(Duration::from_secs(5));
    for (input_name, input) in corpus() {
        group.throughput(Throughput::Bytes(input.len() as u64));
        for (size_name, size) in block_sizes() {
            let id = BenchmarkId::new(input_name, size_name);
            group.bench_with_input(id, &(input.clone(), size), |bench, (data, block)| {
                bench.iter(|| black_box(compress(black_box(data), *block)));
            });
            let id = BenchmarkId::new(format!("{input_name}-parallel"), size_name);
            group.bench_with_input(id, &(input.clone(), size), |bench, (data, block)| {
                bench.iter(|| black_box(compress_with_jobs(black_box(data), *block, Jobs::Global)));
            });
        }
    }
    group.finish();
}

/// Decode every corpus encoding at every matrix block size.
fn decode_matrix(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("decode");
    group
        .sample_size(15)
        .measurement_time(Duration::from_secs(5));
    for (input_name, input) in corpus() {
        group.throughput(Throughput::Bytes(input.len() as u64));
        for (size_name, size) in block_sizes() {
            let coded = compress(&input, size).expect("bench input encodes");
            let id = BenchmarkId::new(input_name, size_name);
            group.bench_with_input(id, &coded, |bench, data| {
                bench.iter(|| black_box(decompress(black_box(data))));
            });
            let id = BenchmarkId::new(format!("{input_name}-parallel"), size_name);
            group.bench_with_input(id, &coded, |bench, data| {
                bench.iter(|| black_box(decompress_with_jobs(black_box(data), Jobs::Global)));
            });
        }
    }
    group.finish();
}

/// Build `blocks` parts of 1 MiB with unequal cost.
///
/// The parts are text, noise and short-period repeat in turn.
fn mixed_input(blocks: usize) -> Vec<u8> {
    let mut input = Vec::new();
    for part in 0..blocks {
        let piece = match part % 3 {
            0 => text_bytes(MIB),
            1 => random_bytes(MIB),
            _ => periodic_bytes(MIB),
        };
        input.extend_from_slice(&piece);
    }
    input
}

/// Compare serial and parallel one-shot calls on a 32 MiB input.
///
/// The input holds 32 blocks of 1 MiB with unequal cost: text,
/// noise and runs in turn.
fn parallel_matrix(criterion: &mut Criterion) {
    let size = BlockSize::mib(1).expect("1 MiB is valid");
    let input = mixed_input(32);
    let coded = compress(&input, size).expect("bench input encodes");
    let mut group = criterion.benchmark_group("parallel");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    group.throughput(Throughput::Bytes(input.len() as u64));
    for (name, jobs) in [("serial", Jobs::SERIAL), ("pool", Jobs::Global)] {
        group.bench_function(BenchmarkId::new("encode", name), |bench| {
            bench.iter(|| black_box(compress_with_jobs(black_box(&input), size, jobs)));
        });
        group.bench_function(BenchmarkId::new("decode", name), |bench| {
            bench.iter(|| black_box(decompress_with_jobs(black_box(&coded), jobs)));
        });
    }
    group.finish();
}

/// Compare the streaming adapters and the one-shot stream calls.
///
/// The input holds 96 blocks of 1 MiB. On 32 threads, the adapters
/// code three batches, so the wait for the slowest block in each
/// batch shows. The one-shot calls have no such wait.
fn streaming_matrix(criterion: &mut Criterion) {
    let size = BlockSize::mib(1).expect("1 MiB is valid");
    let input = mixed_input(96);
    let mut coded = Vec::new();
    stream::compress_slice(&input, &mut coded, size, Jobs::SERIAL).expect("bench input encodes");
    let mut group = criterion.benchmark_group("streaming");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    group.throughput(Throughput::Bytes(input.len() as u64));
    for (name, jobs) in [("serial", Jobs::SERIAL), ("pool", Jobs::Global)] {
        group.bench_function(BenchmarkId::new("read-encode", name), |bench| {
            bench.iter(|| {
                let mut adapter =
                    read::Encoder::with_jobs(&input[..], size, jobs).expect("adapter builds");
                let mut out = Vec::new();
                black_box(io::copy(&mut adapter, &mut out)).expect("adapter reads");
                out
            });
        });
        group.bench_function(BenchmarkId::new("write-encode", name), |bench| {
            bench.iter(|| {
                let mut adapter =
                    write::Encoder::with_jobs(Vec::new(), size, jobs).expect("adapter builds");
                io::copy(&mut &input[..], &mut adapter).expect("adapter writes");
                black_box(adapter.finish()).expect("adapter finishes")
            });
        });
        group.bench_function(BenchmarkId::new("oneshot-encode", name), |bench| {
            bench.iter(|| {
                let mut out = Vec::new();
                black_box(stream::compress_slice(&input, &mut out, size, jobs))
                    .expect("stream encodes");
                out
            });
        });
        group.bench_function(BenchmarkId::new("read-decode", name), |bench| {
            bench.iter(|| {
                let mut adapter =
                    read::Decoder::with_jobs(&coded[..], jobs).expect("adapter builds");
                let mut out = Vec::new();
                black_box(io::copy(&mut adapter, &mut out)).expect("adapter reads");
                out
            });
        });
        group.bench_function(BenchmarkId::new("write-decode", name), |bench| {
            bench.iter(|| {
                let mut adapter = write::Decoder::with_jobs(Vec::new(), jobs);
                io::copy(&mut &coded[..], &mut adapter).expect("adapter writes");
                black_box(adapter.finish()).expect("adapter finishes")
            });
        });
        group.bench_function(BenchmarkId::new("oneshot-decode", name), |bench| {
            bench.iter(|| {
                let mut out = Vec::new();
                black_box(stream::decompress(&coded[..], &mut out, jobs)).expect("stream decodes");
                out
            });
        });
    }
    group.finish();
}

criterion_group!(
    end_to_end,
    encode_matrix,
    decode_matrix,
    parallel_matrix,
    streaming_matrix
);
criterion_main!(end_to_end);
