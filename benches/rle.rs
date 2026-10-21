//! Benches for the run-length stage.
//!
//! The stage is private, so the `bench-internals` feature exposes it.
//! Inputs hold 256 KiB with fixed seeds: `text` has no net gain,
//! `mixed` has text with short runs, `runs` has run lengths of 1 to
//! 600 bytes, `random` has no runs. Decode benches use the inputs
//! that shrink.
//!
//! Re-baseline note: Linux x86-64 normative.

use std::hint::black_box;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;

use bzip3_rs::rle_bench::decode_into;
use bzip3_rs::rle_bench::encode_into;

/// Bench input size in bytes (256 KiB).
const LEN: usize = 256 * 1024;

/// Next value of a fixed-seed generator.
fn next(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    *seed >> 16
}

/// Wordy paragraph repeated to `LEN` bytes.
fn text() -> Vec<u8> {
    const PARA: &[u8] = b"This is the 100th Etext file presented by Project Gutenberg. \
        Shakespeare, William. The complete works in one file. \
        To be, or not to be, that is the question. ";
    PARA.iter().copied().cycle().take(LEN).collect()
}

/// Text with a run of 20 equal signs after each sentence.
fn mixed() -> Vec<u8> {
    const PARA: &[u8] =
        b"This is the 100th Etext file presented by Project Gutenberg.==================== \
        Shakespeare, William. The complete works in one file.==================== ";
    PARA.iter().copied().cycle().take(LEN).collect()
}

/// Runs of 1 to 600 bytes over four values.
fn runs() -> Vec<u8> {
    let mut seed: u32 = 0x00C0_FFEE;
    let mut out = Vec::with_capacity(LEN + 600);
    while out.len() < LEN {
        let value = u8::try_from(next(&mut seed) & 3).expect("masked byte fits");
        let len = 1 + usize::try_from(next(&mut seed) % 600).expect("small value fits");
        out.resize(out.len() + len, value);
    }
    out.truncate(LEN);
    out
}

/// Fixed-seed pseudo-random bytes.
fn random() -> Vec<u8> {
    let mut seed: u32 = 0x00C0_FFEE;
    (0..LEN)
        .map(|_| u8::try_from(next(&mut seed) & 0xFF).expect("masked byte fits"))
        .collect()
}

/// Forward transform on every input.
fn rle_encode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("rle-encode");
    group.throughput(Throughput::Bytes(LEN as u64));
    for (name, input) in [
        ("text", text()),
        ("mixed", mixed()),
        ("runs", runs()),
        ("random", random()),
    ] {
        let mut out = vec![0_u8; input.len()];
        group.bench_function(name, |bench| {
            bench.iter(|| black_box(encode_into(black_box(&input), &mut out)));
        });
    }
    group.finish();
}

/// Inverse transform on the inputs that shrink.
fn rle_decode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("rle-decode");
    group.throughput(Throughput::Bytes(LEN as u64));
    for (name, input) in [("mixed", mixed()), ("runs", runs())] {
        let mut encoded = vec![0_u8; input.len()];
        let len = encode_into(&input, &mut encoded).expect("input shrinks");
        encoded.truncate(len);
        let mut out = vec![0_u8; input.len()];
        group.bench_function(name, |bench| {
            bench.iter(|| {
                decode_into(black_box(&encoded), &mut out).expect("valid payload");
                black_box(&out);
            });
        });
    }
    group.finish();
}

criterion_group!(rle, rle_encode, rle_decode);
criterion_main!(rle);
