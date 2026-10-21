#![cfg(unix)]

//! `argv[0]` dispatch tests: `bunzip3` decodes by default,
//! `bz3cat` decodes to stdout. Explicit flags win.

#[path = "cli/support.rs"]
mod support;

use std::path::Path;
use std::path::PathBuf;
use std::process::Output;
use support::pseudo;
use support::run;
use support::scratch;
use support::write;

/// Run `bin` in `dir` with `args` and `stdin`. Return the output.
fn run_bin(bin: &Path, dir: &Path, args: &[&str], stdin: &[u8]) -> Output {
    use std::process::Stdio;
    let mut child = std::process::Command::new(bin)
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn alias");
    child
        .stdin
        .as_mut()
        .expect("open stdin")
        .write_all(stdin)
        .expect("write stdin");
    child.wait_with_output().expect("wait alias")
}

/// Link the built binary into `dir` under `name`. Return the link path.
fn link(dir: &Path, name: &str) -> PathBuf {
    let target = dir.join(name);
    std::os::unix::fs::symlink(support::bin(), &target).expect("symlink alias");
    target
}

#[test]
fn bunzip3_decodes_by_default() {
    let s = scratch("bunzip3dec");
    let bytes = pseudo(51, 600);
    write(&s.dir, "in.bin", &bytes);
    assert!(
        run(&s.dir, &["-b", "1", "-k", "in.bin"], &[])
            .status
            .success()
    );
    std::fs::remove_file(s.dir.join("in.bin")).expect("drop plain");
    let bunzip3 = link(&s.dir, "bunzip3");
    let out = run_bin(&bunzip3, &s.dir, &["in.bin.bz3"], &[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        std::fs::read(s.dir.join("in.bin")).expect("read out"),
        bytes
    );
    assert!(!s.dir.join("in.bin.bz3").exists(), "input deleted");
}

#[test]
fn bunzip3_explicit_encode_flag_wins() {
    let s = scratch("bunzip3enc");
    let bytes = pseudo(52, 600);
    write(&s.dir, "plain.bin", &bytes);
    let bunzip3 = link(&s.dir, "bunzip3");
    let out = run_bin(&bunzip3, &s.dir, &["-b", "1", "-e", "plain.bin"], &[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(s.dir.join("plain.bin.bz3").exists());
    assert!(!s.dir.join("plain.bin").exists(), "input deleted");
}

#[test]
fn bz3cat_decodes_to_stdout_and_keeps_input() {
    let s = scratch("bz3cat");
    let bytes = pseudo(53, 600);
    write(&s.dir, "in.bin", &bytes);
    assert!(
        run(&s.dir, &["-b", "1", "-k", "in.bin"], &[])
            .status
            .success()
    );
    let bz3cat = link(&s.dir, "bz3cat");
    let out = run_bin(&bz3cat, &s.dir, &["in.bin.bz3"], &[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, bytes);
    assert!(s.dir.join("in.bin.bz3").exists(), "input kept");
}

#[test]
fn quiet_suppresses_guess_notice_but_writes_out() {
    let s = scratch("quiet");
    let bytes = pseudo(54, 600);
    write(&s.dir, "in.bin", &bytes);
    assert!(
        run(&s.dir, &["-b", "1", "-k", "in.bin"], &[])
            .status
            .success()
    );
    let coded = std::fs::read(s.dir.join("in.bin.bz3")).expect("read coded");
    write(&s.dir, "weird.dat", &coded);
    write(&s.dir, "hushed.dat", &coded);

    let out = run(&s.dir, &["-d", "-k", "weird.dat"], &[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stderr).into_owned(),
        "bzip3: Can't guess original name for weird.dat -- using weird.dat.out\n"
    );
    assert_eq!(
        std::fs::read(s.dir.join("weird.dat.out")).expect("read out"),
        bytes
    );

    let out = run(&s.dir, &["-d", "-k", "-q", "hushed.dat"], &[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !String::from_utf8_lossy(&out.stderr).contains("Can't guess"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        std::fs::read(s.dir.join("hushed.dat.out")).expect("read out"),
        bytes
    );
}

use std::io::Write as _;
