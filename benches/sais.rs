//! Benches for the suffix-array stage.
//!
//! The stage is private, so the `bench-internals` feature exposes it.
//! Inputs hold 16 MiB with fixed seeds, so the sort needs several
//! levels and its tables do not fit in cache. `text` has words from a
//! small vocabulary in random order. `base64` has 64 random symbols.
//!
//! Re-baseline note: Linux x86-64 normative.

use std::hint::black_box;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;

use bzip3_rs::sais_bench::BwtScratch;
use bzip3_rs::sais_bench::UnbwtScratch;
use bzip3_rs::sais_bench::bwt_with;
use bzip3_rs::sais_bench::unbwt_with;

/// Bench input size in bytes (16 MiB).
const LEN: usize = 16 * 1024 * 1024;

/// Next value of a fixed-seed generator.
fn next(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    *seed >> 16
}

/// Words in random order, with random case and random punctuation.
fn text() -> Vec<u8> {
    const WORDS: &[&str] = &[
        "the", "of", "and", "to", "in", "is", "that", "it", "was", "for", "on", "are", "with",
        "as", "his", "they", "be", "at", "one", "have", "this", "from", "or", "had", "by", "hot",
        "but", "some", "what", "there", "we", "can", "out", "other", "were", "all", "your", "when",
        "up", "use", "word", "how", "said", "an", "each", "she", "which", "do", "their", "time",
        "if", "will", "way", "about", "many", "then", "them", "would", "write", "like", "so",
        "these", "her", "long", "make", "thing", "see", "him", "two", "has", "look", "more",
    ];
    let mut seed = 0x00C0_FFEE_u32;
    let mut out = Vec::with_capacity(LEN + 16);
    while out.len() < LEN {
        let word = WORDS[next(&mut seed) as usize % WORDS.len()];
        let mut bytes = word.bytes();
        if let Some(first) = bytes.next() {
            let upper = next(&mut seed).is_multiple_of(8);
            out.push(if upper {
                first.to_ascii_uppercase()
            } else {
                first
            });
        }
        out.extend(bytes);
        out.push(match next(&mut seed) % 16 {
            0 => b',',
            1 => b'.',
            _ => b' ',
        });
    }
    out.truncate(LEN);
    out
}

/// Random bytes from a 64-symbol alphabet.
fn base64() -> Vec<u8> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut seed = 0x0BAD_5EED_u32;
    (0..LEN)
        .map(|_| ALPHABET[(next(&mut seed) & 63) as usize])
        .collect()
}

/// Bench the forward and inverse transform of one input.
fn bench_input(criterion: &mut Criterion, name: &str, input: &[u8]) {
    let mut scratch = BwtScratch::default();
    let mut unbwt = UnbwtScratch::default();
    let mut coded = vec![0_u8; input.len()];
    let index = bwt_with(input, &mut coded, &mut scratch).expect("bench input transforms");
    let mut restored = vec![0_u8; input.len()];

    let mut group = criterion.benchmark_group("sais-stage");
    group.sample_size(10);
    group.throughput(Throughput::Bytes(input.len() as u64));
    group.bench_function(format!("forward-{name}"), |bench| {
        let mut out = vec![0_u8; input.len()];
        bench.iter(|| black_box(bwt_with(black_box(input), &mut out, &mut scratch)));
    });
    group.bench_function(format!("inverse-{name}"), |bench| {
        bench.iter(|| {
            black_box(unbwt_with(
                black_box(&coded),
                &mut restored,
                index,
                &mut unbwt,
            ))
        });
    });
    group.finish();
}

/// Words in random order.
fn sais_text(criterion: &mut Criterion) {
    bench_input(criterion, "text", &text());
}

/// Random bytes from 64 symbols.
fn sais_base64(criterion: &mut Criterion) {
    bench_input(criterion, "base64", &base64());
}

criterion_group!(sais, sais_text, sais_base64);
criterion_main!(sais);
