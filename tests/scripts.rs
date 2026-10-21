#![cfg(unix)]

//! Integration tests for the `bz3*` helper scripts.
//!
//! Build a temporary `bindir` with the `bzip3` binary and the scripts.
//! Run each script with `PATH="bindir:$PATH"`.
//! Assert standard output bytes and exit codes.

use std::io::Write as _;
use std::os::unix::fs::PermissionsExt as _;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Output;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

static NEXT: AtomicU64 = AtomicU64::new(0);

/// Scratch directory for one test. Removed on drop.
struct Scratch {
    dir: PathBuf,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Make an empty scratch directory for `case`.
fn scratch(case: &str) -> Scratch {
    let id = NEXT.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("bz3scripts-{}-{id}-{case}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("make scratch dir");
    Scratch { dir }
}

/// Path of the built `bzip3` binary.
fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_bzip3"))
}

/// Repository root (holds `scripts/`).
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Write `bytes` to `name` inside `dir`. Return the full path.
fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, bytes).expect("write scratch file");
    path
}

/// Copy a script to `name` inside `bindir` and mark it executable.
fn install_script(bindir: &Path, src: &Path, name: &str) {
    let dest = bindir.join(name);
    std::fs::copy(src, &dest).expect("copy script");
    let mut perms = std::fs::metadata(&dest).expect("stat script").permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&dest, perms).expect("chmod script");
}

/// Write a fake pager named `name` into `bindir`.
/// The pager copies standard input to standard output.
fn install_fake_pager(bindir: &Path, name: &str) {
    let dest = bindir.join(name);
    std::fs::write(&dest, "#!/bin/sh\nexec cat\n").expect("write fake pager");
    let mut perms = std::fs::metadata(&dest).expect("stat pager").permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&dest, perms).expect("chmod pager");
}

/// Build a temporary `bindir` for `case`.
/// Links the binary as `bzip3`/`bunzip3`/`bz3cat` and copies each
/// script under all of its install names. Returns the directory
/// and a `PATH` value with the directory first.
fn bindir(case: &str) -> (PathBuf, String) {
    let id = NEXT.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("bz3bindir-{}-{id}-{case}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("make bindir");
    let exe = bin();
    for name in ["bzip3", "bunzip3", "bz3cat"] {
        std::os::unix::fs::symlink(&exe, dir.join(name)).expect("link binary");
    }
    let scripts = root().join("scripts");
    for name in ["bz3grep", "bz3egrep", "bz3fgrep"] {
        install_script(&dir, &scripts.join("bz3grep"), name);
    }
    for name in ["bz3more", "bz3less", "bz3most"] {
        install_script(&dir, &scripts.join("bz3more"), name);
    }
    for name in ["bz3diff", "bz3cmp"] {
        install_script(&dir, &scripts.join("bz3diff"), name);
    }
    let old = std::env::var("PATH").unwrap_or_default();
    let path = format!("{}:{old}", dir.display());
    (dir, path)
}

/// Run `prog` with `args` and `stdin` inside `dir` and `path`.
/// Returns the captured output.
fn run(dir: &Path, path: &str, prog: &str, args: &[&str], stdin: &[u8]) -> Output {
    use std::process::Stdio;
    let full = Path::new(path.split(':').next().unwrap_or("")).join(prog);
    let mut child = Command::new(&full)
        .args(args)
        .current_dir(dir)
        .env("PATH", path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn script");
    child
        .stdin
        .as_mut()
        .expect("open stdin")
        .write_all(stdin)
        .expect("write stdin");
    child.wait_with_output().expect("wait script")
}

/// Compress `bytes` with the `bzip3` binary. Return the `.bz3` bytes.
fn compress(bytes: &[u8]) -> Vec<u8> {
    use std::process::Stdio;
    let mut child = Command::new(bin())
        .arg("-c")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn bzip3");
    child
        .stdin
        .as_mut()
        .expect("open stdin")
        .write_all(bytes)
        .expect("write stdin");
    let out = child.wait_with_output().expect("wait bzip3");
    assert!(out.status.success(), "compress failed: {out:?}");
    out.stdout
}

#[test]
fn bz3grep_mixed_files_and_stdin() {
    let work = scratch("grep-mixed");
    let (bindir, path) = bindir("grep-mixed");
    let _keep = bindir;
    let plain = b"hello plain\nfoo bar\n";
    let zipped = b"hello zipped\nfoo bar\n";
    write(&work.dir, "plain.txt", plain);
    write(&work.dir, "zipped.bz3", &compress(zipped));
    // Mixed plain and compressed inputs print `file:match` lines.
    let out = run(
        &work.dir,
        &path,
        "bz3grep",
        &["hello", "plain.txt", "zipped.bz3"],
        b"",
    );
    assert!(out.status.success(), "bz3grep failed: {out:?}");
    let text = String::from_utf8(out.stdout).expect("stdout text");
    assert!(text.contains("plain.txt:hello plain"), "got: {text}");
    assert!(text.contains("zipped.bz3:hello zipped"), "got: {text}");
    // Plain bytes on standard input are searched directly.
    let out = run(&work.dir, &path, "bz3grep", &["hello"], b"hello stdin\n");
    assert!(out.status.success(), "bz3grep stdin failed: {out:?}");
    let text = String::from_utf8(out.stdout).expect("stdout text");
    assert!(text.contains("hello stdin"), "got: {text}");
    // Compressed bytes on standard input are decompressed first.
    let out = run(&work.dir, &path, "bz3grep", &["hello"], &compress(zipped));
    assert!(out.status.success(), "bz3grep stdin bz3 failed: {out:?}");
    let text = String::from_utf8(out.stdout).expect("stdout text");
    assert!(text.contains("hello zipped"), "got: {text}");
}

#[test]
fn bz3egrep_and_bz3fgrep_dispatch() {
    let work = scratch("grep-dispatch");
    let (bindir, path) = bindir("grep-dispatch");
    let _keep = bindir;
    let body = b"ac\naxb\na|b\n";
    write(&work.dir, "data.txt", body);
    // Extended match treats `|` as alternation.
    let out = run(&work.dir, &path, "bz3egrep", &["a|b", "data.txt"], b"");
    assert!(out.status.success(), "bz3egrep failed: {out:?}");
    let text = String::from_utf8(out.stdout).expect("stdout text");
    assert!(text.contains("ac"), "got: {text}");
    // Fixed match treats `|` as a plain byte.
    let out = run(&work.dir, &path, "bz3fgrep", &["a|b", "data.txt"], b"");
    assert!(out.status.success(), "bz3fgrep failed: {out:?}");
    let text = String::from_utf8(out.stdout).expect("stdout text");
    assert!(text.contains("a|b"), "got: {text}");
    assert!(!text.contains("ac"), "got: {text}");
}

#[test]
fn bz3diff_one_file_and_two_files() {
    let work = scratch("diff");
    let (bindir, path) = bindir("diff");
    let _keep = bindir;
    let body = b"same line\n";
    write(&work.dir, "same", body);
    write(&work.dir, "same.bz3", &compress(body));
    // One-file form compares `same.bz3` against `same`.
    let out = run(&work.dir, &path, "bz3diff", &["same"], b"");
    assert_eq!(out.status.code(), Some(0), "one-file same: {out:?}");
    write(&work.dir, "same", b"other line\n");
    let out = run(&work.dir, &path, "bz3diff", &["same"], b"");
    assert_ne!(out.status.code(), Some(0), "one-file differs: {out:?}");
    // Two-file form covers compressed, mixed, and plain pairs.
    write(&work.dir, "same", body);
    write(&work.dir, "a.bz3", &compress(body));
    write(&work.dir, "b.bz3", &compress(body));
    let out = run(&work.dir, &path, "bz3diff", &["a.bz3", "b.bz3"], b"");
    assert_eq!(out.status.code(), Some(0), "two bz3 same: {out:?}");
    write(&work.dir, "c.bz3", &compress(b"other line\n"));
    let out = run(&work.dir, &path, "bz3diff", &["a.bz3", "c.bz3"], b"");
    assert_ne!(out.status.code(), Some(0), "two bz3 differ: {out:?}");
    let out = run(&work.dir, &path, "bz3diff", &["a.bz3", "same"], b"");
    assert_eq!(out.status.code(), Some(0), "mixed same: {out:?}");
    let out = run(&work.dir, &path, "bz3diff", &["same", "same"], b"");
    assert_eq!(out.status.code(), Some(0), "plain same: {out:?}");
}

#[test]
fn bz3cmp_dispatch() {
    let work = scratch("cmp");
    let (bindir, path) = bindir("cmp");
    let _keep = bindir;
    let body = b"cmp bytes\n";
    write(&work.dir, "a.bz3", &compress(body));
    write(&work.dir, "b.bz3", &compress(body));
    let out = run(&work.dir, &path, "bz3cmp", &["a.bz3", "b.bz3"], b"");
    assert_eq!(out.status.code(), Some(0), "bz3cmp same: {out:?}");
    write(&work.dir, "c.bz3", &compress(b"other\n"));
    let out = run(&work.dir, &path, "bz3cmp", &["a.bz3", "c.bz3"], b"");
    assert_ne!(out.status.code(), Some(0), "bz3cmp differ: {out:?}");
}

#[test]
fn bz3more_with_fake_pagers() {
    let work = scratch("more");
    let (bindir, path) = bindir("more");
    install_fake_pager(&bindir, "more");
    install_fake_pager(&bindir, "less");
    install_fake_pager(&bindir, "most");
    let body = b"page this line\n";
    write(&work.dir, "doc.bz3", &compress(body));
    write(&work.dir, "doc.txt", body);
    for prog in ["bz3more", "bz3less", "bz3most"] {
        let out = run(&work.dir, &path, prog, &["doc.bz3"], b"");
        assert!(out.status.success(), "{prog} bz3 failed: {out:?}");
        let text = String::from_utf8(out.stdout).expect("stdout text");
        assert!(text.contains("------> doc.bz3 <------"), "got: {text}");
        assert!(text.contains("page this line"), "got: {text}");
        let out = run(&work.dir, &path, prog, &["doc.txt"], b"");
        assert!(out.status.success(), "{prog} plain failed: {out:?}");
        let text = String::from_utf8(out.stdout).expect("stdout text");
        assert!(text.contains("page this line"), "got: {text}");
    }
}
