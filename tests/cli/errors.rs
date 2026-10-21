//! Exit-code and stderr-text tests. Stderr texts match the reference CLI.

use crate::support::pseudo;
use crate::support::run;
use crate::support::scratch;
use crate::support::write;

fn stderr_of(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn help_and_version_exit_zero_on_stdout() {
    let s = scratch("help");
    let out = run(&s.dir, &["--help"], &[]);
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(text.contains("Usage: bzip3"), "{text}");
    assert!(text.contains("-z"), "{text}");
    for name in [
        "--encode",
        "--decode",
        "--test",
        "--recover",
        "--keep",
        "--quiet",
        "--fast",
        "--best",
    ] {
        assert!(text.contains(name), "{name}: {text}");
    }
    assert!(text.contains("-1   65 KiB"), "{text}");
    assert!(text.contains("-14  511 MiB"), "{text}");
    assert!(text.contains("--fast"), "{text}");
    assert!(text.contains("--best"), "{text}");
    assert!(!text.contains("-B"), "-B stays undocumented: {text}");
    assert!(!text.contains("--rm"), "destructive flag stays absent");
    assert!(out.stderr.is_empty());

    let out = run(&s.dir, &["--version"], &[]);
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(text.starts_with("bzip3 "), "{text}");
    assert!(text.contains("Kamila Szewczyk"));
}

#[test]
fn unknown_flag_fails_with_hint() {
    let s = scratch("unknown");
    for args in [&["--bogus"][..], &["--rm"][..]] {
        let out = run(&s.dir, args, &[]);
        assert_eq!(out.status.code(), Some(1), "{args:?}");
        assert!(
            stderr_of(&out).contains("Try 'bzip3 --help' for more information.\n"),
            "{args:?}: {}",
            stderr_of(&out)
        );
    }
}

#[test]
fn bad_signature_fails() {
    let s = scratch("sig");
    write(&s.dir, "bad.bz3", b"NOPE!");
    let out = run(&s.dir, &["-t", "bad.bz3"], &[]);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(stderr_of(&out), "Invalid signature.\n");
}

#[test]
fn truncated_header_is_corrupt() {
    let s = scratch("trunc");
    write(&s.dir, "short.bz3", b"BZ3v1");
    let out = run(&s.dir, &["-t", "short.bz3"], &[]);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(stderr_of(&out), "Error: Corrupt file\n");
}

#[test]
fn trailing_byte_is_corrupt() {
    let s = scratch("trail");
    let bytes = pseudo(3, 100);
    write(&s.dir, "in.bin", &bytes);
    assert!(run(&s.dir, &["-b", "1", "in.bin"], &[]).status.success());
    let mut coded = std::fs::read(s.dir.join("in.bin.bz3")).expect("read coded");
    coded.push(0x58);
    write(&s.dir, "in.bin.bz3", &coded);
    let out = run(&s.dir, &["-t", "in.bin.bz3"], &[]);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(stderr_of(&out), "Error: Corrupt file\n");
}

#[test]
fn bad_header_block_size_fails() {
    let s = scratch("hdrsize");
    write(&s.dir, "bad.bz3", b"BZ3v1\x00\x00\x00\x00");
    let out = run(&s.dir, &["-t", "bad.bz3"], &[]);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(
        stderr_of(&out),
        "The input file is corrupted. Reason: Invalid block size in the header.\n"
    );
}

#[test]
fn block_size_range_rejects() {
    let s = scratch("blockrange");
    write(&s.dir, "in.bin", b"data");
    for size in ["0", "512", "64K", "523265K", "512M"] {
        let out = run(&s.dir, &["-b", size, "in.bin"], &[]);
        assert_eq!(out.status.code(), Some(1), "size {size}");
        assert_eq!(
            stderr_of(&out),
            "Block size must be between 65 KiB and 511 MiB.\n"
        );
    }
}

#[test]
fn block_size_suffix_forms_reject_malformed() {
    let s = scratch("blocksfx");
    write(&s.dir, "in.bin", b"data");
    for size in ["16B", "K", "1.5M", "16MB"] {
        let out = run(&s.dir, &["-b", size, "in.bin"], &[]);
        assert_eq!(out.status.code(), Some(1), "size {size}");
        assert_eq!(
            stderr_of(&out),
            format!("bzip3: invalid block size: {size}\n")
        );
    }
}

#[test]
fn non_numeric_sizes_reject() {
    let s = scratch("numeric");
    write(&s.dir, "in.bin", b"data");
    let out = run(&s.dir, &["-b", "xx", "in.bin"], &[]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(stderr_of(&out), "bzip3: invalid block size: xx\n");
    let out = run(&s.dir, &["-j", "xx", "in.bin"], &[]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(stderr_of(&out), "bzip3: invalid amount of jobs: xx\n");
}

#[test]
fn too_many_jobs_rejects() {
    let s = scratch("jobs");
    write(&s.dir, "in.bin", b"data data data");
    let out = run(&s.dir, &["-b", "1", "-j", "65", "in.bin"], &[]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        stderr_of(&out),
        "Number of workers must be between 0 and 64.\n"
    );
}

#[test]
fn unknown_extension_decodes_via_out_fallback() {
    let s = scratch("ext");
    write(&s.dir, "plain.txt", b"data");
    let out = run(&s.dir, &["-d", "plain.txt"], &[]);
    assert_eq!(out.status.code(), Some(2), "{}", stderr_of(&out));
    assert_eq!(
        stderr_of(&out),
        "bzip3: Can't guess original name for plain.txt -- using plain.txt.out\nInvalid signature.\n"
    );
    assert!(
        !s.dir.join("plain.txt.out").exists(),
        "failed decode removes its output"
    );
    assert!(s.dir.join("plain.txt").exists(), "input stays");
}

#[test]
fn decode_uppercase_suffix_uses_out_fallback() {
    let s = scratch("casesfx");
    let bytes = pseudo(17, 300);
    write(&s.dir, "in.bin", &bytes);
    assert!(
        run(&s.dir, &["-b", "1", "-k", "in.bin"], &[])
            .status
            .success()
    );
    let coded = std::fs::read(s.dir.join("in.bin.bz3")).expect("read coded");
    // Case-sensitive strip: `.BZ3` must not strip.
    write(&s.dir, "arch.BZ3", &coded);
    let out = run(&s.dir, &["-d", "-k", "arch.BZ3"], &[]);
    assert!(out.status.success(), "{}", stderr_of(&out));
    assert_eq!(
        stderr_of(&out),
        "bzip3: Can't guess original name for arch.BZ3 -- using arch.BZ3.out\n"
    );
    assert_eq!(
        std::fs::read(s.dir.join("arch.BZ3.out")).expect("read out"),
        bytes
    );
}

#[test]
fn encode_bz3_suffix_rejects() {
    let s = scratch("sfx");
    write(&s.dir, "already.bz3", b"data data data");
    let out = run(&s.dir, &["already.bz3"], &[]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr_of(&out));
    assert_eq!(
        stderr_of(&out),
        "bzip3: Input file already.bz3 already has .bz3 suffix.\n"
    );
    assert!(s.dir.join("already.bz3").exists(), "input stays");
    assert!(
        !s.dir.join("already.bz3.bz3").exists(),
        "no output on reject"
    );
}

#[test]
fn missing_input_fails() {
    let s = scratch("missing");
    let out = run(&s.dir, &["gone.bin"], &[]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        stderr_of(&out),
        "Error: failed to open input file `gone.bin': No such file or directory\n"
    );
}

#[test]
fn directory_input_fails() {
    let s = scratch("dir");
    std::fs::create_dir(s.dir.join("sub")).expect("make subdir");
    let out = run(&s.dir, &["sub"], &[]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(stderr_of(&out), "Error: input `sub' is a directory.\n");
}

#[test]
fn corrupt_payload_names_block_error() {
    let s = scratch("corrupt");
    let bytes = pseudo(11, 500);
    write(&s.dir, "in.bin", &bytes);
    assert!(run(&s.dir, &["-b", "1", "in.bin"], &[]).status.success());
    let mut coded = std::fs::read(s.dir.join("in.bin.bz3")).expect("read coded");
    let last = coded.len() - 1;
    coded[last] ^= 0xFF;
    write(&s.dir, "in.bin.bz3", &coded);
    // Default decode runs in parallel, so it names the data.
    let out = run(&s.dir, &["-d", "-f", "in.bin.bz3"], &[]);
    assert_eq!(out.status.code(), Some(2), "{}", stderr_of(&out));
    assert!(
        stderr_of(&out).starts_with("Failed to decode data: "),
        "{}",
        stderr_of(&out)
    );
    // Serial decode keeps the per-block wording.
    let out = run(&s.dir, &["-d", "-f", "-j", "1", "in.bin.bz3"], &[]);
    assert_eq!(out.status.code(), Some(2), "{}", stderr_of(&out));
    assert!(
        stderr_of(&out).starts_with("Failed to decode a block: "),
        "{}",
        stderr_of(&out)
    );
}
