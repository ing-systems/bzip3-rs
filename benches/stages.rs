//! Hot-stage benches for the suffix array and the entropy coder.
//!
//! Both stages run inside the public block codec (`sais` and `cm`
//! stay private), so these benches target each stage through inputs
//! that stress it: wordy text spends most time in the Burrows-Wheeler
//! transform, fixed-seed random bytes spend most time in the
//! context-mixing coder. Encode benches reuse one state; decode
//! benches decode one pre-encoded payload with a reused decoder.
//!
//! Inputs hold 256 KiB: large enough to dominate setup cost, small
//! enough for quick runs. Re-baseline note: Linux x86-64 normative.

use std::hint::black_box;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;

use bzip3_rs::BlockDecoder;
use bzip3_rs::BlockEncoder;
use bzip3_rs::BlockSize;

/// Bench input size in bytes (256 KiB).
const STAGE_LEN: usize = 256 * 1024;

/// Wordy paragraph repeated to `STAGE_LEN` bytes (transform-heavy).
fn stage_text() -> Vec<u8> {
    const PARA: &[u8] = b"This is the 100th Etext file presented by Project Gutenberg. \
        Shakespeare, William. The complete works in one file. \
        To be, or not to be, that is the question. \
        All the world's a stage, and all the men and women merely players. ";
    let mut out = Vec::with_capacity(STAGE_LEN);
    while out.len() < STAGE_LEN {
        let take = (STAGE_LEN - out.len()).min(PARA.len());
        out.extend_from_slice(&PARA[..take]);
    }
    out
}

/// Fixed-seed pseudo-random bytes (entropy-coder-heavy).
fn stage_random() -> Vec<u8> {
    let mut seed: u32 = 0x00C0_FFEE;
    (0..STAGE_LEN)
        .map(|_| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            u8::try_from((seed >> 16) & 0xFF).expect("masked byte fits")
        })
        .collect()
}

/// Fresh 1 MiB state for stage benches.
fn stage_state() -> BlockEncoder {
    BlockEncoder::new(BlockSize::mib(1).expect("1 MiB is valid")).expect("state builds")
}

/// Suffix-array path: encode wordy text through one reused state.
fn sais_encode(criterion: &mut Criterion) {
    let input = stage_text();
    let mut state = stage_state();
    let mut group = criterion.benchmark_group("sais");
    group.throughput(Throughput::Bytes(input.len() as u64));
    group.bench_function("encode-text", |bench| {
        bench.iter(|| black_box(state.encode_block(black_box(&input))));
    });
    group.finish();
}

/// Suffix-array path: decode wordy text through one reused decoder.
fn sais_decode(criterion: &mut Criterion) {
    let input = stage_text();
    let payload = stage_state()
        .encode_block(&input)
        .expect("bench input encodes");
    let size = u32::try_from(input.len()).expect("bench input fits u32");
    let mut decoder =
        BlockDecoder::new(BlockSize::mib(1).expect("1 MiB is valid")).expect("decoder builds");
    let mut group = criterion.benchmark_group("sais");
    group.throughput(Throughput::Bytes(input.len() as u64));
    group.bench_function("decode-text", |bench| {
        bench.iter(|| black_box(decoder.decode_block(black_box(&payload), size)));
    });
    group.finish();
}

/// Entropy-coder path: encode random bytes through one reused state.
fn cm_encode(criterion: &mut Criterion) {
    let input = stage_random();
    let mut state = stage_state();
    let mut group = criterion.benchmark_group("cm");
    group.throughput(Throughput::Bytes(input.len() as u64));
    group.bench_function("encode-random", |bench| {
        bench.iter(|| black_box(state.encode_block(black_box(&input))));
    });
    group.finish();
}

/// Entropy-coder path: decode random bytes through one reused decoder.
fn cm_decode(criterion: &mut Criterion) {
    let input = stage_random();
    let payload = stage_state()
        .encode_block(&input)
        .expect("bench input encodes");
    let size = u32::try_from(input.len()).expect("bench input fits u32");
    let mut decoder =
        BlockDecoder::new(BlockSize::mib(1).expect("1 MiB is valid")).expect("decoder builds");
    let mut group = criterion.benchmark_group("cm");
    group.throughput(Throughput::Bytes(input.len() as u64));
    group.bench_function("decode-random", |bench| {
        bench.iter(|| black_box(decoder.decode_block(black_box(&payload), size)));
    });
    group.finish();
}

criterion_group!(stages, sais_encode, sais_decode, cm_encode, cm_decode);
criterion_main!(stages);
