//! Parallel adapter tests: bytes and errors match the serial adapters.

use std::io::Cursor;
use std::io::Read;
use std::io::Write;

use bzip3_rs::BlockSize;
use bzip3_rs::Jobs;
use bzip3_rs::read::Decoder as ReadDecoder;
use bzip3_rs::read::Encoder as ReadEncoder;
use bzip3_rs::write::Decoder as WriteDecoder;
use bzip3_rs::write::Encoder as WriteEncoder;

const JOB_COUNTS: [Jobs; 5] = [
    Jobs::SERIAL,
    Jobs::from_count(2),
    Jobs::from_count(3),
    Jobs::from_count(32),
    Jobs::Global,
];

/// Build mixed data: text-like, noise and runs, in this order.
fn mixed(len: usize) -> Vec<u8> {
    let mut seed: u32 = 0x9E37_79B9;
    (0..len)
        .map(|index| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            match (index / 40_000) % 3 {
                0 => b"pack my box with five dozen liquor jugs "[index % 40],
                1 => (seed >> 24) as u8,
                _ => 3,
            }
        })
        .collect()
}

fn lengths() -> Vec<usize> {
    let block = BlockSize::MIN.get() as usize;
    vec![0, 1, block - 1, block, block + 1, 4 * block, 7 * block + 5]
}

fn write_encode(data: &[u8], jobs: Jobs, piece: usize) -> Vec<u8> {
    let mut encoder = WriteEncoder::with_jobs(Vec::new(), BlockSize::MIN, jobs).unwrap();
    for part in data.chunks(piece) {
        encoder.write_all(part).unwrap();
    }
    encoder.finish().unwrap()
}

fn read_encode(data: &[u8], jobs: Jobs) -> Vec<u8> {
    let mut encoder = ReadEncoder::with_jobs(Cursor::new(data), BlockSize::MIN, jobs).unwrap();
    let mut out = Vec::new();
    encoder.read_to_end(&mut out).unwrap();
    out
}

/// Feed `framed` to the write decoder. Return the plain bytes and the error text.
fn write_decode(framed: &[u8], jobs: Jobs, piece: usize) -> (Vec<u8>, Option<String>) {
    let mut decoder = WriteDecoder::with_jobs(Vec::new(), jobs);
    for part in framed.chunks(piece) {
        if let Err(e) = decoder.write_all(part) {
            return (decoder.get_ref().clone(), Some(e.to_string()));
        }
    }
    let kept = decoder.get_ref().clone();
    match decoder.finish() {
        Ok(out) => (out, None),
        Err(e) => (kept, Some(format!("{e:?}"))),
    }
}

/// Read all plain bytes. Return them with the error text, if any.
fn read_decode(framed: &[u8], jobs: Jobs) -> (Vec<u8>, Option<String>) {
    let mut decoder = match ReadDecoder::with_jobs(Cursor::new(framed), jobs) {
        Ok(decoder) => decoder,
        Err(e) => return (Vec::new(), Some(format!("{e:?}"))),
    };
    let mut out = Vec::new();
    let mut buf = vec![0_u8; 10_000];
    loop {
        match decoder.read(&mut buf) {
            Ok(0) => return (out, None),
            Ok(n) => out.extend_from_slice(&buf[..n]),
            Err(e) => return (out, Some(e.to_string())),
        }
    }
}

#[test]
fn encoders_match_serial_at_every_job_count() {
    for len in lengths() {
        let data = mixed(len);
        let serial = write_encode(&data, Jobs::SERIAL, 10_000);
        for jobs in JOB_COUNTS {
            assert_eq!(
                write_encode(&data, jobs, 10_000),
                serial,
                "write {len} {jobs:?}"
            );
            assert_eq!(
                write_encode(&data, jobs, 1 << 20),
                serial,
                "big {len} {jobs:?}"
            );
            assert_eq!(read_encode(&data, jobs), serial, "read {len} {jobs:?}");
        }
    }
}

#[test]
fn decoders_round_trip_at_every_job_count() {
    for len in lengths() {
        let data = mixed(len);
        let framed = write_encode(&data, Jobs::SERIAL, 10_000);
        for jobs in JOB_COUNTS {
            assert_eq!(
                write_decode(&framed, jobs, 10_000),
                (data.clone(), None),
                "w {len} {jobs:?}"
            );
            assert_eq!(
                write_decode(&framed, jobs, 1 << 20),
                (data.clone(), None),
                "wb {len} {jobs:?}"
            );
            assert_eq!(
                read_decode(&framed, jobs),
                (data.clone(), None),
                "r {len} {jobs:?}"
            );
        }
    }
}

#[test]
fn decoder_errors_match_serial() {
    let block = BlockSize::MIN.get() as usize;
    let data = mixed(6 * block + 9);
    let framed = write_encode(&data, Jobs::SERIAL, 10_000);
    let mut cases: Vec<Vec<u8>> = Vec::new();
    cases.push(framed[..framed.len() - 3].to_vec());
    cases.push(framed[..9 + 4].to_vec());
    cases.push(framed[..3].to_vec());
    for fraction in [4, 2] {
        let mut flipped = framed.clone();
        let at = flipped.len() / fraction;
        flipped[at] ^= 0xFF;
        cases.push(flipped);
    }
    let mut trailing = framed.clone();
    trailing.extend_from_slice(b"xyz");
    cases.push(trailing);
    for (index, case) in cases.iter().enumerate() {
        let serial_write = write_decode(case, Jobs::SERIAL, 10_000);
        let serial_read = read_decode(case, Jobs::SERIAL);
        for jobs in JOB_COUNTS {
            assert_eq!(
                write_decode(case, jobs, 10_000),
                serial_write,
                "w {index} {jobs:?}"
            );
            assert_eq!(
                write_decode(case, jobs, 1 << 20),
                write_decode(case, Jobs::SERIAL, 1 << 20),
                "wb {index} {jobs:?}"
            );
            assert_eq!(read_decode(case, jobs), serial_read, "r {index} {jobs:?}");
        }
    }
}
