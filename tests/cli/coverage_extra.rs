//! Extra CLI tests for output handling and standard I/O paths.

use crate::support::pseudo;
use crate::support::run;
use crate::support::scratch;
use crate::support::write;

fn stderr_of(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn existing_output_needs_force() {
    let s = scratch("exists");
    write(&s.dir, "in.bin", b"data data data");
    assert!(run(&s.dir, &["-k", "in.bin"], &[]).status.success());
    assert!(s.dir.join("in.bin.bz3").exists());
    let out = run(&s.dir, &["-k", "in.bin"], &[]);
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr_of(&out).contains("already exists. Use -f to force overwrite."));
    let out = run(&s.dir, &["-k", "-f", "in.bin"], &[]);
    assert!(out.status.success(), "{}", stderr_of(&out));
}

#[test]
fn output_directory_fails() {
    let s = scratch("outdir");
    write(&s.dir, "in.bin", b"data data data");
    std::fs::create_dir(s.dir.join("in.bin.bz3")).expect("make output dir");
    let out = run(&s.dir, &["-k", "in.bin"], &[]);
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr_of(&out).contains("is a directory."));
}

#[test]
fn stdin_encode_writes_stdout() {
    let s = scratch("stdin-enc");
    let out = run(&s.dir, &["-c"], b"hello world");
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert!(
        out.stdout.starts_with(b"BZ3v1"),
        "{:?}",
        &out.stdout[..5.min(out.stdout.len())]
    );
}

#[test]
fn stdin_test_reads_stdin() {
    let s = scratch("stdin-test");
    let coded = run(&s.dir, &["-c"], &pseudo(5, 500)).stdout;
    assert!(coded.starts_with(b"BZ3v1"));
    let out = run(&s.dir, &["-t"], &coded);
    assert!(out.status.success(), "{}", stderr_of(&out));
}
