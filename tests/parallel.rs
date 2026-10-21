//! Parallel one-shot tests: output and errors match the serial path.

use bzip3_rs::BlockSize;
use bzip3_rs::Jobs;
use bzip3_rs::compress;
use bzip3_rs::compress_with_jobs;
use bzip3_rs::decompress;
use bzip3_rs::decompress_with_jobs;
use bzip3_rs::error::Error;

const JOB_COUNTS: [Jobs; 5] = [
    Jobs::SERIAL,
    Jobs::from_count(2),
    Jobs::from_count(3),
    Jobs::from_count(32),
    Jobs::Global,
];

/// Build mixed data: text-like, noise and runs, in this order.
fn mixed(len: usize) -> Vec<u8> {
    let mut seed: u32 = 0x1234_5678;
    (0..len)
        .map(|index| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            match (index / 50_000) % 3 {
                0 => b"the quick brown fox "[index % 20],
                1 => (seed >> 24) as u8,
                _ => 7,
            }
        })
        .collect()
}

#[test]
fn compress_matches_serial_at_every_job_count() {
    let size = BlockSize::MIN;
    let block = size.get() as usize;
    for len in [0, 1, block - 1, block, block + 1, 5 * block + 17] {
        let data = mixed(len);
        let serial = compress(&data, size).unwrap();
        for jobs in JOB_COUNTS {
            let coded = compress_with_jobs(&data, size, jobs).unwrap();
            assert_eq!(coded, serial, "len {len}, jobs {jobs:?}");
        }
    }
}

#[test]
fn decompress_matches_serial_at_every_job_count() {
    let size = BlockSize::MIN;
    let data = mixed(5 * size.get() as usize + 17);
    let coded = compress(&data, size).unwrap();
    for jobs in JOB_COUNTS {
        assert_eq!(
            decompress_with_jobs(&coded, jobs).unwrap(),
            data,
            "{jobs:?}"
        );
    }
}

#[test]
fn trailing_bytes_stay_ignored() {
    let size = BlockSize::MIN;
    let data = mixed(3 * size.get() as usize);
    let mut coded = compress(&data, size).unwrap();
    coded.extend_from_slice(b"trailing");
    for jobs in JOB_COUNTS {
        assert_eq!(
            decompress_with_jobs(&coded, jobs).unwrap(),
            data,
            "{jobs:?}"
        );
    }
}

#[test]
fn errors_match_serial_at_every_job_count() {
    let size = BlockSize::MIN;
    let block = size.get() as usize;
    let data = mixed(4 * block);
    let coded = compress(&data, size).unwrap();
    let mut cases: Vec<Vec<u8>> = Vec::new();
    // Cut inside the last payload.
    cases.push(coded[..coded.len() - 3].to_vec());
    // Cut inside a block header.
    cases.push(
        coded[..coded.len() - 3]
            .iter()
            .copied()
            .take(13 + 4)
            .collect(),
    );
    // Flip one byte in the second payload.
    let mut flipped = coded.clone();
    let at = flipped.len() / 2;
    flipped[at] ^= 0xFF;
    cases.push(flipped);
    // Block count above the real count.
    let mut more = coded.clone();
    more[9..13].copy_from_slice(&9_u32.to_le_bytes());
    cases.push(more);
    for case in &cases {
        let serial = decompress(case).map_err(|e| format!("{e:?}"));
        for jobs in JOB_COUNTS {
            let result = decompress_with_jobs(case, jobs).map_err(|e| format!("{e:?}"));
            assert_eq!(result, serial, "jobs {jobs:?}");
        }
    }
}

#[test]
fn block_fault_comes_before_later_header_fault() {
    let size = BlockSize::MIN;
    let block = size.get() as usize;
    let data = mixed(3 * block);
    let mut coded = compress(&data, size).unwrap();
    // Damage the first block payload, then cut the last payload.
    coded[13 + 8 + 12] ^= 0xFF;
    coded.truncate(coded.len() - 1);
    let serial = decompress(&coded);
    assert!(serial.is_err());
    assert!(!matches!(serial, Err(Error::TruncatedData)));
    for jobs in JOB_COUNTS {
        let result = decompress_with_jobs(&coded, jobs);
        assert_eq!(format!("{result:?}"), format!("{serial:?}"), "{jobs:?}");
    }
}
