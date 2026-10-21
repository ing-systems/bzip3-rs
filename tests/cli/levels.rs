//! Compression level tests (`-1..-14`, `--fast`, `--best`).

use crate::support::pseudo;
use crate::support::run;
use crate::support::scratch;

/// Expected block size in bytes for each level 1-14.
fn expected(level: u8) -> u32 {
    match level {
        1 => 66_560,
        2 => 131_072,
        3 => 262_144,
        4 => 524_288,
        5 => 1_048_576,
        6 => 2_097_152,
        7 => 4_194_304,
        8 => 8_388_608,
        9 => 16_777_216,
        10 => 33_554_432,
        11 => 67_108_864,
        12 => 134_217_728,
        13 => 268_435_456,
        14 => 535_822_336,
        _ => unreachable!(),
    }
}

/// Read the streaming header block size from framed output.
fn header_size(coded: &[u8]) -> u32 {
    assert!(coded.len() >= 9, "short frame");
    assert_eq!(&coded[..5], b"BZ3v1");
    u32::from_le_bytes(coded[5..9].try_into().expect("header size"))
}

#[test]
fn levels_map_to_expected_block_sizes() {
    let s = scratch("levels");
    let bytes = pseudo(42, 200);
    for level in 1..=14_u8 {
        let flag = format!("-{level}");
        let out = run(&s.dir, &[&flag, "-j", "1", "-c"], &bytes);
        assert!(
            out.status.success(),
            "level {level}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(header_size(&out.stdout), expected(level), "level {level}");
        let back = run(&s.dir, &["-d", "-c"], &out.stdout);
        assert!(back.status.success(), "decode level {level}");
        assert_eq!(back.stdout, bytes, "round-trip level {level}");
    }
}

#[test]
fn fast_and_best_alias_levels() {
    let s = scratch("aliases");
    let bytes = pseudo(7, 500);
    let one = run(&s.dir, &["-1", "-j", "1", "-c"], &bytes);
    let fast = run(&s.dir, &["--fast", "-j", "1", "-c"], &bytes);
    assert!(one.status.success() && fast.status.success());
    assert_eq!(one.stdout, fast.stdout, "--fast must equal -1");
    let top = run(&s.dir, &["-14", "-j", "1", "-c"], &bytes);
    let best = run(&s.dir, &["--best", "-j", "1", "-c"], &bytes);
    assert!(top.status.success() && best.status.success());
    assert_eq!(top.stdout, best.stdout, "--best must equal -14");
}

#[test]
fn min_max_levels_match_block_size_limits() {
    let s = scratch("minmax");
    let bytes = pseudo(13, 200);
    let min = run(&s.dir, &["-1", "-j", "1", "-c"], &bytes);
    assert!(min.status.success());
    assert_eq!(header_size(&min.stdout), 65 * 1024);
    let max = run(&s.dir, &["-14", "-j", "1", "-c"], &bytes);
    assert!(max.status.success());
    assert_eq!(header_size(&max.stdout), 511 * 1024 * 1024);
    let def = run(&s.dir, &["-j", "1", "-c"], &bytes);
    assert!(def.status.success());
    assert_eq!(header_size(&def.stdout), 16 * 1024 * 1024);
}

#[test]
fn last_block_flag_wins() {
    let s = scratch("lastwins");
    let bytes = pseudo(9, 200);
    let cases: &[(&[&str], u32)] = &[
        (&["-b", "1", "-14"], expected(14)),
        (&["-14", "-b", "1"], 1_048_576),
        (&["--fast", "--best"], expected(14)),
        (&["--best", "--fast"], expected(1)),
        (&["-1", "-14"], expected(14)),
        (&["-14", "-1"], expected(1)),
        (&["-b", "512", "-14"], expected(14)),
    ];
    for (args, want) in cases {
        let mut full = args.to_vec();
        full.extend_from_slice(&["-j", "1", "-c"]);
        let out = run(&s.dir, &full, &bytes);
        assert!(
            out.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(header_size(&out.stdout), *want, "{args:?}");
    }
    // Numeric out-of-range `-b` fails only when last.
    let out = run(&s.dir, &["-14", "-b", "512", "-j", "1", "-c"], &bytes);
    assert!(!out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Block size must be between 65 KiB and 511 MiB.\n"
    );
    // Non-numeric `-b` poisons even when a level follows.
    let out = run(&s.dir, &["-b", "xx", "-14", "-c"], &bytes);
    assert!(!out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "bzip3: invalid block size: xx\n"
    );
}

#[test]
fn block_size_suffixes_map_to_bytes() {
    let s = scratch("suffixes");
    let bytes = pseudo(21, 200);
    // Same size through every spelling: 16 MiB.
    let mut headers = Vec::new();
    for size in [
        "16", "16M", "16m", "16MiB", "16mib", "16384K", "16384k", "16384KiB", "16384kib",
    ] {
        let out = run(&s.dir, &["-b", size, "-j", "1", "-c"], &bytes);
        assert!(
            out.status.success(),
            "size {size}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        headers.push(header_size(&out.stdout));
    }
    assert!(
        headers.iter().all(|&h| h == 16 * 1024 * 1024),
        "{headers:?}"
    );
    // Minimum through KiB spellings.
    for size in ["65K", "65k", "65KiB", "65kib"] {
        let out = run(&s.dir, &["-b", size, "-j", "1", "-c"], &bytes);
        assert!(
            out.status.success(),
            "size {size}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(header_size(&out.stdout), 65 * 1024, "{size}");
    }
    // Maximum through both units.
    for size in ["511M", "511mib", "523264K", "523264kib"] {
        let out = run(&s.dir, &["-b", size, "-j", "1", "-c"], &bytes);
        assert!(
            out.status.success(),
            "size {size}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(header_size(&out.stdout), 511 * 1024 * 1024, "{size}");
    }
    // Attached and `--block=` forms match the spaced form.
    let spaced = run(&s.dir, &["-b", "65K", "-j", "1", "-c"], &bytes);
    let attached = run(&s.dir, &["-b65K", "-j", "1", "-c"], &bytes);
    let equals = run(&s.dir, &["--block=65K", "-j", "1", "-c"], &bytes);
    assert!(spaced.status.success() && attached.status.success() && equals.status.success());
    assert_eq!(spaced.stdout, attached.stdout);
    assert_eq!(spaced.stdout, equals.stdout);
}

#[test]
fn digit_clusters_stay_unknown_flags() {
    let s = scratch("clusters");
    for args in [&["-19"][..], &["-91"][..], &["-0"][..], &["-15"][..]] {
        let out = run(&s.dir, args, b"hi");
        assert!(!out.status.success(), "{args:?}");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("Try 'bzip3 --help'"),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn zero_level_stays_unknown_flag() {
    let s = scratch("zero");
    let out = run(&s.dir, &["-0", "-c"], b"hi");
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("Try 'bzip3 --help'"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn decode_ignores_levels() {
    let s = scratch("decignore");
    let bytes = pseudo(11, 300);
    let coded = run(&s.dir, &["-1", "-j", "1", "-c"], &bytes);
    assert!(coded.status.success());
    for args in [&["-d", "-c", "-14"][..], &["-d", "-c", "--best"][..]] {
        let back = run(&s.dir, args, &coded.stdout);
        assert!(
            back.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&back.stderr)
        );
        assert_eq!(back.stdout, bytes);
    }
}
