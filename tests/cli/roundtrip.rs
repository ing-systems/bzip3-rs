//! Encode/decode round-trip tests through the CLI binary.

use crate::support::pseudo;
use crate::support::run;
use crate::support::scratch;
use crate::support::write;

fn data(name: &str) -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/cli/data")
        .join(name);
    std::fs::read(path).expect("read fixture")
}

/// Encode `name` then decode the result. Return the decoded bytes.
fn round_trip(bytes: &[u8], extra_encode: &[&str]) -> Vec<u8> {
    let s = scratch("roundtrip");
    write(&s.dir, "in.bin", bytes);
    let mut args = vec!["-b", "1"];
    args.extend_from_slice(extra_encode);
    args.push("in.bin");
    let out = run(&s.dir, &args, &[]);
    assert!(
        out.status.success(),
        "encode: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let coded = std::fs::read(s.dir.join("in.bin.bz3")).expect("read coded");
    assert_eq!(&coded[..5], b"BZ3v1");
    let out = run(&s.dir, &["-d", "-f", "in.bin.bz3"], &[]);
    assert!(
        out.status.success(),
        "decode: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::read(s.dir.join("in.bin")).expect("read plain")
}

#[test]
fn sizes_round_trip_at_1mib() {
    for len in [0_usize, 1, 63, 64, 65, 1_000, 100_000] {
        let bytes = pseudo(0x1234_5678_9ABC_DEF0, len);
        assert_eq!(round_trip(&bytes, &[]), bytes, "len {len}");
    }
}

#[test]
fn text_round_trips_and_shrinks() {
    // Text-like input: repeated words compress well.
    let words = [
        "the ", "quick ", "brown ", "fox ", "jumps ", "over ", "the ", "lazy ", "dog.\n",
    ];
    let mut bytes = Vec::new();
    for i in 0..20_000 {
        bytes.extend_from_slice(words[i % words.len()].as_bytes());
    }
    let plain = round_trip(&bytes, &[]);
    assert_eq!(plain, bytes);
    // Sanity: the coded form exists and starts with the magic.
    let s = scratch("shrink");
    write(&s.dir, "in.bin", &bytes);
    let out = run(&s.dir, &["-b", "1", "in.bin"], &[]);
    assert!(out.status.success());
    let coded = std::fs::read(s.dir.join("in.bin.bz3")).expect("read coded");
    assert!(coded.len() < bytes.len(), "text must shrink");
}

#[test]
fn oracle_fixtures_decode_to_plain() {
    for name in ["empty.bin", "hello.bin", "seq64.bin", "kilo.bin"] {
        let s = scratch("fixture");
        let plain = data(name);
        let coded = data(&format!("{name}.bz3"));
        write(&s.dir, name, &plain);
        write(&s.dir, &format!("{name}.bz3"), &coded);
        std::fs::remove_file(s.dir.join(name)).expect("drop plain");
        let out = run(&s.dir, &["-d", &format!("{name}.bz3")], &[]);
        assert!(
            out.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(std::fs::read(s.dir.join(name)).expect("read out"), plain);
    }
}

#[test]
fn multiple_positionals_compress_all() {
    let s = scratch("multi");
    let first = pseudo(31, 400);
    let second = pseudo(32, 800);
    write(&s.dir, "a", &first);
    write(&s.dir, "b", &second);
    let out = run(&s.dir, &["-b", "1", "a", "b"], &[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(s.dir.join("a.bz3").exists());
    assert!(s.dir.join("b.bz3").exists());
    assert!(!s.dir.join("a").exists(), "input deleted");
    assert!(!s.dir.join("b").exists(), "input deleted");
}

#[test]
fn stdout_round_trip_through_pipes() {
    let s = scratch("pipes");
    let bytes = pseudo(7, 5_000);
    let out = run(&s.dir, &["-b", "1", "-c"], &bytes);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(&out.stdout[..5], b"BZ3v1");
    let back = run(&s.dir, &["-d", "-c"], &out.stdout);
    assert!(
        back.status.success(),
        "{}",
        String::from_utf8_lossy(&back.stderr)
    );
    assert_eq!(back.stdout, bytes);
}

#[test]
fn force_overwrites_and_keep_controls_delete() {
    let s = scratch("force");
    // `-k` keeps the input so the second encode hits "already exists".
    write(&s.dir, "f.bin", b"data data data data");
    let first = run(&s.dir, &["-b", "1", "-k", "f.bin"], &[]);
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(s.dir.join("f.bin").exists(), "input stays with -k");
    assert!(s.dir.join("f.bin.bz3").exists());
    let second = run(&s.dir, &["-b", "1", "-k", "f.bin"], &[]);
    assert!(!second.status.success());
    assert!(String::from_utf8_lossy(&second.stderr).contains("already exists"));
    let third = run(&s.dir, &["-b", "1", "-k", "-f", "f.bin"], &[]);
    assert!(
        third.status.success(),
        "{}",
        String::from_utf8_lossy(&third.stderr)
    );
    // Without `-k` a successful file-to-file encode deletes the input.
    write(&s.dir, "g.bin", b"data data data data");
    let gone = run(&s.dir, &["-b", "1", "g.bin"], &[]);
    assert!(
        gone.status.success(),
        "{}",
        String::from_utf8_lossy(&gone.stderr)
    );
    assert!(!s.dir.join("g.bin").exists(), "input deleted");
    assert!(s.dir.join("g.bin.bz3").exists());
    // A second encode now hits the missing input, not the existing output.
    let missing = run(&s.dir, &["-b", "1", "g.bin"], &[]);
    assert!(!missing.status.success());
    assert!(
        String::from_utf8_lossy(&missing.stderr).contains("failed to open input file"),
        "{}",
        String::from_utf8_lossy(&missing.stderr)
    );
}

#[test]
fn failures_never_delete_input() {
    let s = scratch("nodelete");
    // A failed encode (output exists, no -f) keeps the input.
    write(&s.dir, "f.bin", b"data data data data");
    assert!(
        run(&s.dir, &["-b", "1", "-k", "f.bin"], &[])
            .status
            .success()
    );
    let failed = run(&s.dir, &["-b", "1", "-k", "f.bin"], &[]);
    assert!(!failed.status.success());
    assert!(s.dir.join("f.bin").exists(), "input stays on failure");
    // A failed decode removes its partial output but keeps the input.
    write(&s.dir, "bad.bz3", b"NOPE!");
    let out = run(&s.dir, &["-d", "bad.bz3"], &[]);
    assert!(!out.status.success());
    assert!(s.dir.join("bad.bz3").exists(), "input stays on failure");
    assert!(!s.dir.join("bad").exists(), "no output on failure");
}

#[test]
#[cfg(unix)]
fn output_inherits_mode_and_mtime() {
    use std::os::unix::fs::PermissionsExt as _;
    let s = scratch("meta");
    let path = write(&s.dir, "m.bin", b"data data data data");
    let mtime = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_600_000_000);
    std::fs::File::options()
        .write(true)
        .open(&path)
        .expect("open")
        .set_modified(mtime)
        .expect("set mtime");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).expect("set mode");
    let out = run(&s.dir, &["-b", "1", "-k", "m.bin"], &[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let coded = s.dir.join("m.bin.bz3");
    let coded_meta = std::fs::metadata(&coded).expect("stat coded");
    assert_eq!(coded_meta.permissions().mode() & 0o777, 0o640);
    assert_eq!(coded_meta.modified().expect("mtime"), mtime);
}

#[test]
fn test_mode_checks_without_writing() {
    let s = scratch("testmode");
    let bytes = pseudo(99, 300);
    write(&s.dir, "in.bin", &bytes);
    assert!(run(&s.dir, &["-b", "1", "in.bin"], &[]).status.success());
    let out = run(&s.dir, &["-t", "in.bin.bz3"], &[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!s.dir.join("in.bin").exists(), "test writes no file");
    assert!(out.stdout.is_empty());
}

#[test]
fn jobs_1_and_4_emit_identical_bytes() {
    let s = scratch("jobs");
    let bytes = pseudo(5, 50_000);
    write(&s.dir, "in.bin", &bytes);
    let one = run(&s.dir, &["-b", "1", "-j", "1", "-c"], &bytes);
    assert!(one.status.success());
    let four = run(&s.dir, &["-b", "1", "-j", "4", "-c"], &bytes);
    assert!(four.status.success());
    assert_eq!(one.stdout, four.stdout, "threads must not change bytes");
    let back = run(&s.dir, &["-d", "-c", "-j", "4"], &four.stdout);
    assert!(back.status.success());
    assert_eq!(back.stdout, bytes);
}

#[test]
fn decode_worker_counts_agree_on_multi_block() {
    // Three 1 MiB blocks: default, `-j0`, and `-j1` must decode alike.
    let s = scratch("decodejobs");
    let bytes = pseudo(77, 2_100_000);
    write(&s.dir, "in.bin", &bytes);
    let coded_out = run(&s.dir, &["-b", "1", "-k", "in.bin"], &[]);
    assert!(
        coded_out.status.success(),
        "{}",
        String::from_utf8_lossy(&coded_out.stderr)
    );
    let coded = std::fs::read(s.dir.join("in.bin.bz3")).expect("read coded");
    let default = run(&s.dir, &["-d", "-c"], &coded);
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );
    let auto = run(&s.dir, &["-d", "-c", "-j", "0"], &coded);
    assert!(
        auto.status.success(),
        "{}",
        String::from_utf8_lossy(&auto.stderr)
    );
    let serial = run(&s.dir, &["-d", "-c", "-j", "1"], &coded);
    assert!(
        serial.status.success(),
        "{}",
        String::from_utf8_lossy(&serial.stderr)
    );
    assert_eq!(default.stdout, bytes, "default decode must match");
    assert_eq!(auto.stdout, bytes, "auto decode must match");
    assert_eq!(serial.stdout, bytes, "serial decode must match");
}

/// A name that is not valid UTF-8 keeps its bytes through encode and decode.
#[cfg(unix)]
#[test]
fn non_utf8_name_round_trips() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    use std::process::Command;

    let s = scratch("non-utf8");
    let plain = OsStr::from_bytes(b"\xff.bin");
    let coded = OsStr::from_bytes(b"\xff.bin.bz3");
    std::fs::write(s.dir.join(plain), b"hello hello hello hello").expect("write input");
    for (flag, name) in [("-e", plain), ("-d", coded)] {
        let out = Command::new(crate::support::bin())
            .arg(flag)
            .arg(name)
            .current_dir(&s.dir)
            .output()
            .expect("run bzip3");
        assert!(
            out.status.success(),
            "{flag}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    assert_eq!(
        std::fs::read(s.dir.join(plain)).expect("read output"),
        b"hello hello hello hello"
    );
}
