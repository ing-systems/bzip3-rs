//! Batch-mode and job-count tests.

use crate::support::pseudo;
use crate::support::run;
use crate::support::scratch;
use crate::support::write;

fn stderr_of(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn batch_encodes_decodes_and_tests() {
    let s = scratch("batch");
    let first = pseudo(1, 300);
    let second = pseudo(2, 700);
    write(&s.dir, "a.txt", &first);
    write(&s.dir, "b.txt", &second);
    let out = run(&s.dir, &["-B", "-b", "1", "a.txt", "b.txt"], &[]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert!(s.dir.join("a.txt.bz3").exists());
    assert!(s.dir.join("b.txt.bz3").exists());

    let out = run(&s.dir, &["-B", "-d", "-k", "a.txt.bz3", "b.txt.bz3"], &[]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert_eq!(std::fs::read(s.dir.join("a.txt")).expect("read a"), first);
    assert_eq!(std::fs::read(s.dir.join("b.txt")).expect("read b"), second);

    let out = run(&s.dir, &["-B", "-t", "a.txt.bz3", "b.txt.bz3"], &[]);
    assert!(out.status.success(), "{}", stderr_of(&out));
}

#[test]
fn batch_flag_error_aborts_before_work() {
    // `-j` is validated once globally before any file runs:
    // one message, exit 1, nothing processed.
    let s = scratch("batchfail");
    write(&s.dir, "a.txt", b"data data data");
    write(&s.dir, "good.txt", b"good data good data");
    let out = run(
        &s.dir,
        &["-B", "-b", "1", "-j", "65", "a.txt", "good.txt"],
        &[],
    );
    assert_eq!(out.status.code(), Some(1), "{}", stderr_of(&out));
    assert_eq!(
        stderr_of(&out),
        "Number of workers must be between 0 and 64.\n"
    );
    assert!(!s.dir.join("a.txt.bz3").exists(), "no output written");
    assert!(!s.dir.join("good.txt.bz3").exists(), "no output written");
    assert!(s.dir.join("a.txt").exists(), "input untouched");
    assert!(s.dir.join("good.txt").exists(), "input untouched");
}

#[test]
fn batch_continues_past_missing_input() {
    let s = scratch("batchcont");
    let bytes = pseudo(43, 400);
    write(&s.dir, "good.bin", &bytes);
    let out = run(&s.dir, &["-B", "-b", "1", "gone.bin", "good.bin"], &[]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr_of(&out));
    assert!(
        stderr_of(&out).contains("failed to open input file `gone.bin'"),
        "{}",
        stderr_of(&out)
    );
    assert!(
        s.dir.join("good.bin.bz3").exists(),
        "good file still compresses"
    );
    assert!(!s.dir.join("good.bin").exists(), "input deleted");
}

#[test]
fn batch_decode_unknown_extension_fails() {
    let s = scratch("batchext");
    write(&s.dir, "plain.txt", b"data");
    let out = run(&s.dir, &["-B", "-d", "plain.txt"], &[]);
    assert_eq!(out.status.code(), Some(2), "{}", stderr_of(&out));
    assert_eq!(
        stderr_of(&out),
        "bzip3: Can't guess original name for plain.txt -- using plain.txt.out\nInvalid signature.\n"
    );
    assert!(!s.dir.join("plain.txt.out").exists());
}

#[test]
fn batch_without_files_uses_stdio() {
    let s = scratch("batchstdio");
    let input = pseudo(9, 200);
    let out = run(&s.dir, &["-B", "-b", "1", "-c"], &input);
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert_eq!(&out.stdout[..5], b"BZ3v1");
}

#[test]
fn job_counts_zero_through_64_work() {
    let s = scratch("jobs");
    let input = pseudo(13, 5_000);
    for jobs in ["0", "1", "2", "64"] {
        let out = run(&s.dir, &["-b", "1", "-j", jobs, "-c"], &input);
        assert!(out.status.success(), "jobs={jobs}: {}", stderr_of(&out));
    }
}

#[test]
fn last_mode_flag_wins() {
    let s = scratch("modewin");
    let bytes = pseudo(21, 200);
    write(&s.dir, "in.bin", &bytes);
    assert!(run(&s.dir, &["-b", "1", "in.bin"], &[]).status.success());
    // `-e -d`: decode wins, so the `.bz3` input decodes back.
    let out = run(&s.dir, &["-e", "-d", "in.bin.bz3"], &[]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert_eq!(
        std::fs::read(s.dir.join("in.bin")).expect("read out"),
        bytes
    );
}
