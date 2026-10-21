//! One-shot and slice parallel tests: bytes match the serial calls.
//!
//! This covers the worker-slot helpers in `stream` (`encode_blocks`
//! and `decode_blocks`), which the adapter tests do not reach.

use std::io::Cursor;

use bzip3_rs::BlockSize;
use bzip3_rs::Jobs;
use bzip3_rs::compress;
use bzip3_rs::compress_with_jobs;
use bzip3_rs::decompress;
use bzip3_rs::decompress_with_jobs;
use bzip3_rs::stream;

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
    let block = BlockSize::MIN.as_usize();
    vec![0, 1, block - 1, block, block + 1, 4 * block, 7 * block + 5]
}

#[test]
fn oneshot_matches_serial_at_every_job_count() {
    let size = BlockSize::MIN;
    for len in lengths() {
        let data = mixed(len);
        let serial = compress(&data, size).unwrap();
        assert_eq!(decompress(&serial).unwrap(), data, "serial RT {len}");
        for jobs in JOB_COUNTS {
            assert_eq!(
                compress_with_jobs(&data, size, jobs).unwrap(),
                serial,
                "enc {len} {jobs:?}"
            );
            assert_eq!(
                decompress_with_jobs(&serial, jobs).unwrap(),
                data,
                "dec {len} {jobs:?}"
            );
        }
    }
}

#[test]
fn slice_matches_serial_at_every_job_count() {
    let size = BlockSize::MIN;
    for len in lengths() {
        let data = mixed(len);
        let mut serial = Vec::new();
        stream::compress_slice(&data, &mut serial, size, Jobs::SERIAL).unwrap();
        let mut plain = Vec::new();
        stream::decompress(Cursor::new(serial.as_slice()), &mut plain, Jobs::SERIAL).unwrap();
        assert_eq!(plain, data, "serial RT {len}");
        for jobs in JOB_COUNTS {
            let mut coded = Vec::new();
            stream::compress_slice(&data, &mut coded, size, jobs).unwrap();
            assert_eq!(coded, serial, "enc {len} {jobs:?}");
            let mut out = Vec::new();
            stream::decompress(Cursor::new(serial.as_slice()), &mut out, jobs).unwrap();
            assert_eq!(out, data, "dec {len} {jobs:?}");
        }
    }
}
