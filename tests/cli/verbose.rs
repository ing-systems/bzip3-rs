//! Verbose-output tests. Byte format matches the reference CLI:
//! `"<file>:\t<in> -> <out> bytes, <pct>%, <bpb> bpb"` with one
//! leading space before the file name, and an `OK, ` marker in test mode.

use crate::support::run;
use crate::support::scratch;
use crate::support::write;

fn stderr_of(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn encode_empty_reports_inf() {
    let s = scratch("vempty");
    write(&s.dir, "empty.bin", b"");
    let out = run(&s.dir, &["-v", "empty.bin"], &[]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert_eq!(
        stderr_of(&out),
        " empty.bin:\t0 -> 9 bytes, inf%, inf bpb\n"
    );
}

#[test]
fn decode_and_test_lines_match() {
    let s = scratch("vdec");
    let plain = std::fs::read(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/cli/data/hello.bin"),
    )
    .expect("read plain");
    let coded = std::fs::read(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/cli/data/hello.bin.bz3"),
    )
    .expect("read coded");
    write(&s.dir, "hello.bin.bz3", &coded);
    let out = run(&s.dir, &["-d", "-v", "-f", "-k", "hello.bin.bz3"], &[]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert_eq!(
        std::fs::read(s.dir.join("hello.bin")).expect("read out"),
        plain
    );
    assert_eq!(
        stderr_of(&out),
        " hello.bin.bz3:\t49 -> 24 bytes, 204.17%, 16.33 bpb\n"
    );

    std::fs::remove_file(s.dir.join("hello.bin")).expect("drop plain");
    let out = run(&s.dir, &["-t", "-v", "hello.bin.bz3"], &[]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert_eq!(
        stderr_of(&out),
        " hello.bin.bz3:\tOK, 49 -> 24 bytes, 204.17%, 16.33 bpb\n"
    );
}

#[test]
fn stdio_verbose_has_no_file_prefix() {
    let s = scratch("vstdio");
    let input = vec![b'x'; 1_000];
    let out = run(&s.dir, &["-b", "1", "-v", "-c"], &input);
    assert!(out.status.success(), "{}", stderr_of(&out));
    let coded = out.stdout.clone();
    let err = stderr_of(&out);
    assert!(err.starts_with('\t'), "{err}");
    assert!(err.contains("1000 -> "), "{err}");
    assert!(err.ends_with("bpb\n"), "{err}");
    let back = run(&s.dir, &["-d", "-v", "-c"], &coded);
    assert!(back.status.success(), "{}", stderr_of(&back));
    assert_eq!(back.stdout, input);
    let err = stderr_of(&back);
    assert!(err.starts_with('\t'), "{err}");
    assert!(!err.contains("OK"), "{err}");
}
