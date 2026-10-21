//! Recover mode tests. A damaged block gives zero bytes or its
//! checksum-faulty bytes, the exit code is 2, and no file is removed.

use std::path::Path;

use crate::support::pseudo;
use crate::support::run;
use crate::support::scratch;
use crate::support::write;

/// Length of the stream header in bytes.
const STREAM_HEADER_LEN: usize = 9;
/// Length of a block header in bytes.
const BLOCK_HEADER_LEN: usize = 8;
/// Block size of the test input: 65 KiB.
const BLOCK: usize = 65 * 1024;

/// Build three full blocks of compressible plain bytes.
fn plain_blocks() -> Vec<u8> {
    pseudo(7, 3 * BLOCK).iter().map(|byte| byte % 8).collect()
}

/// Encode `plain` with one worker and 65 KiB blocks.
fn encode(dir: &Path, plain: &[u8]) -> Vec<u8> {
    let out = run(dir, &["-c", "-j", "1", "-b", "65K"], plain);
    assert!(out.status.success());
    out.stdout
}

/// Return the byte offset of the payload of each block.
fn payload_starts(coded: &[u8]) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut pos = STREAM_HEADER_LEN;
    while pos < coded.len() {
        let size = u32::from_le_bytes(coded[pos..pos + 4].try_into().unwrap());
        starts.push(pos + BLOCK_HEADER_LEN);
        pos += BLOCK_HEADER_LEN + size as usize;
    }
    starts
}

/// Return the damage line of block `index` (from 0) of `coded`.
fn damage_line(coded: &[u8], index: usize, fault: &str, state: &str) -> String {
    let offset = payload_starts(coded)[index] - BLOCK_HEADER_LEN;
    let start = index * BLOCK;
    let last = start + BLOCK - 1;
    format!(
        "Writing invalid block {} at input offset {offset}: {fault}. Output bytes {start} to {last} {state}.\n",
        index + 1,
    )
}

fn stderr_of(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn clean_file_matches_decode_and_keeps_input() {
    let s = scratch("rec-clean");
    let plain = plain_blocks();
    write(&s.dir, "a.bz3", &encode(&s.dir, &plain));
    let out = run(&s.dir, &["-r", "a.bz3"], &[]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr_of(&out));
    assert!(out.stderr.is_empty());
    assert_eq!(std::fs::read(s.dir.join("a")).unwrap(), plain);
    assert!(s.dir.join("a.bz3").exists(), "recover keeps the input");
}

#[test]
fn checksum_fault_keeps_decoded_bytes() {
    let s = scratch("rec-crc");
    let plain = plain_blocks();
    let mut coded = encode(&s.dir, &plain);
    // The first four payload bytes hold the checksum.
    let starts = payload_starts(&coded);
    coded[starts[1]] ^= 0x01;
    write(&s.dir, "a.bz3", &coded);
    let out = run(&s.dir, &["-r", "a.bz3"], &[]);
    assert_eq!(out.status.code(), Some(2));
    let text = stderr_of(&out);
    let line = damage_line(&coded, 1, "Checksum mismatch", "keep the decoded bytes");
    assert!(text.contains(&line), "{text}");
    assert!(
        text.ends_with("Recovery finished. Damaged blocks: 1.\n"),
        "{text}"
    );
    assert_eq!(std::fs::read(s.dir.join("a")).unwrap(), plain);
    assert!(s.dir.join("a.bz3").exists());
}

#[test]
fn stage_fault_writes_zeros() {
    let s = scratch("rec-zero");
    let plain = plain_blocks();
    let mut coded = encode(&s.dir, &plain);
    // Bytes 4 to 7 of a payload hold the index of the transform.
    let starts = payload_starts(&coded);
    coded[starts[1] + 4..starts[1] + 8].copy_from_slice(&i32::MAX.to_le_bytes());
    write(&s.dir, "a.bz3", &coded);
    let out = run(&s.dir, &["-r", "-k", "a.bz3"], &[]);
    assert_eq!(out.status.code(), Some(2));
    let line = damage_line(&coded, 1, "Invalid header", "are zero");
    assert!(stderr_of(&out).contains(&line), "{}", stderr_of(&out));
    let got = std::fs::read(s.dir.join("a")).unwrap();
    assert_eq!(got.len(), plain.len());
    assert_eq!(got[..BLOCK], plain[..BLOCK]);
    assert!(got[BLOCK..2 * BLOCK].iter().all(|&byte| byte == 0));
    assert_eq!(got[2 * BLOCK..], plain[2 * BLOCK..]);
}

#[test]
fn damaged_stream_gives_same_result_at_every_job_count() {
    let s = scratch("rec-jobs");
    let plain = plain_blocks();
    let mut coded = encode(&s.dir, &plain);
    let starts = payload_starts(&coded);
    coded[starts[0]] ^= 0x01;
    coded[starts[2] + 4..starts[2] + 8].copy_from_slice(&i32::MAX.to_le_bytes());
    write(&s.dir, "a.bz3", &coded);
    let one = run(&s.dir, &["-r", "-c", "-j", "1", "a.bz3"], &[]);
    let four = run(&s.dir, &["-r", "-c", "-j", "4", "a.bz3"], &[]);
    assert_eq!(one.status.code(), Some(2));
    assert_eq!(four.status.code(), Some(2));
    assert_eq!(one.stdout, four.stdout);
    assert_eq!(one.stderr, four.stderr);
    assert!(stderr_of(&one).ends_with("Damaged blocks: 2.\n"));
    assert_eq!(one.stdout.len(), plain.len());
}

#[test]
fn bad_header_block_size_falls_back() {
    let s = scratch("rec-size");
    let plain = plain_blocks();
    let mut coded = encode(&s.dir, &plain);
    coded[5..9].copy_from_slice(&0_u32.to_le_bytes());
    write(&s.dir, "a.bz3", &coded);

    let out = run(&s.dir, &["-r", "-c", "a.bz3"], &[]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr_of(&out));
    assert_eq!(stderr_of(&out), "Recovery mode: Proceeding.\n");
    assert_eq!(out.stdout, plain);

    let four = run(&s.dir, &["-r", "-c", "-j", "4", "a.bz3"], &[]);
    assert_eq!(four.status.code(), Some(0), "{}", stderr_of(&four));
    assert_eq!(four.stdout, out.stdout);
    assert_eq!(four.stderr, out.stderr);

    let out = run(&s.dir, &["-d", "-c", "a.bz3"], &[]);
    assert_eq!(out.status.code(), Some(2));
    assert!(out.stdout.is_empty());
}

#[test]
fn bad_header_block_size_and_damaged_block_report_the_same_at_every_job_count() {
    let s = scratch("rec-size-damage");
    let plain = plain_blocks();
    let mut coded = encode(&s.dir, &plain);
    let starts = payload_starts(&coded);
    coded[starts[1]] ^= 0x01;
    coded[5..9].copy_from_slice(&u32::MAX.to_le_bytes());
    write(&s.dir, "a.bz3", &coded);
    let one = run(&s.dir, &["-r", "-c", "-j", "1", "a.bz3"], &[]);
    let four = run(&s.dir, &["-r", "-c", "-j", "4", "a.bz3"], &[]);
    assert_eq!(one.status.code(), Some(2));
    assert_eq!(one.stdout, plain);
    assert_eq!(four.stdout, one.stdout);
    assert_eq!(four.stderr, one.stderr);
    let line = damage_line(&coded, 1, "Checksum mismatch", "keep the decoded bytes");
    assert!(stderr_of(&one).contains(&line), "{}", stderr_of(&one));
}

#[test]
fn truncated_data_keeps_earlier_blocks() {
    let s = scratch("rec-trunc");
    let plain = plain_blocks();
    let mut coded = encode(&s.dir, &plain);
    let starts = payload_starts(&coded);
    coded.truncate(starts[2] + 10);
    write(&s.dir, "a.bz3", &coded);
    let out = run(&s.dir, &["-r", "a.bz3"], &[]);
    assert_eq!(out.status.code(), Some(2));
    let stop = format!(
        "Recovery stopped at block 3, input offset {}. Output bytes written: {}. Damaged blocks: 0.\n",
        starts[2] - BLOCK_HEADER_LEN,
        2 * BLOCK,
    );
    assert_eq!(stderr_of(&out), format!("{stop}Error: Corrupt file\n"));
    assert_eq!(std::fs::read(s.dir.join("a")).unwrap(), plain[..2 * BLOCK]);
    assert!(s.dir.join("a.bz3").exists());
}

#[test]
fn bad_magic_stays_fatal() {
    let s = scratch("rec-magic");
    write(&s.dir, "a.bz3", b"NOPE!");
    let out = run(&s.dir, &["-r", "a.bz3"], &[]);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(stderr_of(&out), "Invalid signature.\n");
    assert!(s.dir.join("a.bz3").exists());
}

#[test]
fn standard_input_and_alias_recover() {
    let s = scratch("rec-stdin");
    let plain = plain_blocks();
    let mut coded = encode(&s.dir, &plain);
    let starts = payload_starts(&coded);
    coded[starts[1]] ^= 0x01;

    let out = run(&s.dir, &["-r"], &coded);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(out.stdout, plain);

    #[cfg(unix)]
    for name in ["bz3cat", "bunzip3"] {
        let link = s.dir.join(name);
        std::os::unix::fs::symlink(crate::support::bin(), &link).unwrap();
        let out = std::process::Command::new(&link)
            .args(["-r", "-j", "1"])
            .stdin(std::fs::File::open(write(&s.dir, "in.bz3", &coded)).unwrap())
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2), "{name}");
        assert_eq!(out.stdout, plain, "{name}");
    }
}

#[test]
fn last_mode_flag_wins() {
    let s = scratch("rec-order");
    let plain = plain_blocks();
    let mut coded = encode(&s.dir, &plain);
    let starts = payload_starts(&coded);
    coded[starts[1]] ^= 0x01;
    let decode = run(&s.dir, &["-r", "-d"], &coded);
    assert_eq!(decode.status.code(), Some(2));
    assert!(decode.stderr.starts_with(b"Failed to decode"));
    let recover = run(&s.dir, &["-d", "-r"], &coded);
    assert_eq!(recover.status.code(), Some(2));
    assert_eq!(recover.stdout, plain);
}
