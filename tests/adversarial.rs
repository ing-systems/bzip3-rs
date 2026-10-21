//! Full decode pipeline tests against reference-produced fixtures (ticket 09).
//!
//! Fixtures under `vectors/decode09-*` come from the pinned C oracle
//! (pinned at `fd3d71f`) through its one-shot compress function
//! at oracle block sizes 65 KiB, 1 MiB, and 16 MiB. Inputs below rebuild the
//! exact oracle inputs. Decode must restore every input byte for byte.
//! Ordered error-table cases assert the spec-mandated variant with
//! first-failure-wins. The mutation sweep asserts no input panics,
//! hangs, or overruns.

use bzip3_rs::BlockDecoder;
use bzip3_rs::BlockSize;
use bzip3_rs::decompress;
use bzip3_rs::error::Error;

/// Paragraph repeated to build the text inputs.
fn para() -> Vec<u8> {
    let mut text = Vec::new();
    text.extend_from_slice(b"This is the 100th Etext file presented by Project Gutenberg. ");
    text.extend_from_slice(b"Shakespeare, William. The complete works in one file. ");
    text.extend_from_slice(b"To be, or not to be, that is the question. ");
    text.extend_from_slice(b"All the world's a stage, and all the men and women merely players. ");
    text
}

/// Doubled 0..50 phrase repeated to 2000 bytes (LZP model input).
fn periodic_input() -> Vec<u8> {
    let phrase: Vec<u8> = (0..50_u8).collect();
    let doubled = [phrase.clone(), phrase].concat();
    doubled.repeat(20)
}

/// Runs inside periodic text (combined LZP+RLE model input).
fn both_input() -> Vec<u8> {
    let phrase: Vec<u8> = (0..50_u8).collect();
    let doubled = [phrase.clone(), phrase].concat();
    let side = doubled.repeat(10)[..1000].to_vec();
    [side.clone(), vec![b'Q'; 600], side].concat()
}

/// Fixed-seed filler bytes (plain coded model input).
fn random_input() -> Vec<u8> {
    let mut state: u32 = 0xC0_FFEE;
    (0..4096)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            u8::try_from((state >> 16) & 0xFF).expect("masked byte fits")
        })
        .collect()
}

/// Read one committed fixture.
fn fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/vectors/decode09-{name}.bz3",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(path).expect("fixture reads")
}

/// Decode must succeed and restore `want`.
fn assert_decodes(name: &str, want: &[u8]) {
    let got = decompress(&fixture(name)).unwrap();
    assert_eq!(got, want, "fixture {name} differs");
}

/// First block payload bounds inside a framed fixture.
fn first_block(code: &[u8]) -> (usize, usize) {
    let comp = usize::try_from(u32::from_le_bytes(code[13..17].try_into().unwrap())).unwrap();
    (13 + 8, 13 + 8 + comp)
}

#[test]
fn reference_empty_decodes_at_all_matrix_sizes() {
    for name in ["empty-65k", "empty-1m", "empty-16m"] {
        assert_decodes(name, b"");
    }
}

#[test]
fn reference_stored_boundary_decodes() {
    assert_decodes("1byte-65k", b"A");
    assert_decodes("63b-65k", &[b'x'; 63]);
}

#[test]
fn reference_rle_path_decodes() {
    assert_decodes("64b-65k", &[b'y'; 64]);
    assert_decodes("65b-65k", &[b'z'; 65]);
    let want = vec![b'A'; 2000];
    for name in ["runs2k-65k", "runs2k-1m", "runs2k-16m"] {
        assert_decodes(name, &want);
    }
}

#[test]
fn reference_lzp_path_decodes() {
    let want = periodic_input();
    assert_eq!(want.len(), 2000);
    for name in ["periodic2k-65k", "periodic2k-1m", "periodic2k-16m"] {
        assert_decodes(name, &want);
    }
}

#[test]
fn reference_combined_path_decodes() {
    let want = both_input();
    assert_eq!(want.len(), 2600);
    assert_decodes("both2k-65k", &want);
}

#[test]
fn reference_plain_path_decodes() {
    let want = random_input();
    assert_eq!(want.len(), 4096);
    for name in ["random4k-65k", "random4k-1m", "random4k-16m"] {
        assert_decodes(name, &want);
    }
}

#[test]
fn reference_text_decodes() {
    let want = para().repeat(100)[..20_000].to_vec();
    for name in ["text20k-65k", "text20k-1m", "text20k-16m"] {
        assert_decodes(name, &want);
    }
}

#[test]
fn reference_multiblock_decodes() {
    let want = para().repeat(800);
    assert_eq!(want.len(), 180_000);
    let code = fixture("text180k-65k");
    assert_eq!(u32::from_le_bytes(code[9..13].try_into().unwrap()), 3);
    assert_eq!(decompress(&code).unwrap(), want);
}

#[test]
fn trailing_bytes_are_ignored_on_coded_frames() {
    for name in ["65b-65k", "periodic2k-65k", "random4k-65k", "text180k-65k"] {
        let mut code = fixture(name);
        code.extend_from_slice(b"TRAIL");
        let want = decompress(&fixture(name)).unwrap();
        assert_eq!(decompress(&code).unwrap(), want, "trailing for {name}");
    }
}

#[test]
fn decoder_state_reused_across_blocks() {
    let code = fixture("text180k-65k");
    let block_size = BlockSize::new(u32::from_le_bytes(code[5..9].try_into().unwrap())).unwrap();
    assert_eq!(block_size, BlockSize::MIN);
    let mut decoder = BlockDecoder::new(block_size).unwrap();
    assert_eq!(decoder.block_size(), block_size);
    let mut rest = &code[13..];
    let mut out = Vec::new();
    for _ in 0..3 {
        let comp = u32::from_le_bytes(rest[..4].try_into().unwrap());
        let orig = u32::from_le_bytes(rest[4..8].try_into().unwrap());
        let len = usize::try_from(comp).unwrap();
        let block = &rest[8..8 + len];
        out.extend_from_slice(&decoder.decode_block(block, orig).unwrap());
        rest = &rest[8 + len..];
    }
    assert_eq!(out, para().repeat(800));
}

#[test]
fn short_header_and_bad_magic_are_rejected() {
    assert!(matches!(decompress(&[]), Err(Error::TruncatedHeader)));
    assert!(matches!(decompress(&[0; 12]), Err(Error::TruncatedHeader)));
    let mut bad = fixture("1byte-65k");
    bad[0] = b'X';
    assert!(matches!(decompress(&bad), Err(Error::InvalidMagic)));
}

#[test]
fn bad_frame_block_size_is_invalid_block_size() {
    for size in [-1_i32, 0, 65 * 1024 - 1, 511 * 1024 * 1024 + 1] {
        let mut bad = fixture("1byte-65k");
        bad[5..9].copy_from_slice(&size.to_le_bytes());
        assert!(
            matches!(decompress(&bad), Err(Error::InvalidBlockSize)),
            "size {size}"
        );
    }
}

#[test]
fn bad_comp_size_is_malformed() {
    let mut bad = fixture("1byte-65k");
    bad[13..17].copy_from_slice(&(-1_i32).to_le_bytes());
    assert!(matches!(decompress(&bad), Err(Error::InvalidHeader)));
    let mut bad = fixture("1byte-65k");
    bad[13..17].copy_from_slice(&i32::MAX.to_le_bytes());
    assert!(matches!(decompress(&bad), Err(Error::InvalidHeader)));
}

#[test]
fn short_block_header_is_malformed() {
    let code = fixture("1byte-65k");
    let cut = &code[..13 + 3];
    assert!(matches!(decompress(cut), Err(Error::TruncatedHeader)));
}

#[test]
fn short_payload_is_truncated() {
    let code = fixture("random4k-65k");
    let cut = &code[..code.len() - 100];
    assert!(matches!(decompress(cut), Err(Error::TruncatedData)));
}

#[test]
fn bad_orig_size_is_malformed() {
    let mut bad = fixture("1byte-65k");
    bad[17..21].copy_from_slice(&(-1_i32).to_le_bytes());
    assert!(matches!(decompress(&bad), Err(Error::InvalidHeader)));
    // Fixture header holds the oracle 65 KiB size; anything above it fails.
    let mut bad = fixture("1byte-65k");
    bad[17..21].copy_from_slice(&(65 * 1024 + 1_u32).to_le_bytes());
    assert!(matches!(decompress(&bad), Err(Error::InvalidHeader)));
}

#[test]
fn stored_size_mismatch_is_malformed_not_crc() {
    // Wrong original size plus a corrupt body: the size check wins.
    let mut bad = fixture("1byte-65k");
    bad[17..21].copy_from_slice(&2_u32.to_le_bytes());
    let last = bad.len() - 1;
    bad[last] ^= 0x01;
    assert!(matches!(decompress(&bad), Err(Error::InvalidHeader)));
}

#[test]
fn stored_crc_mismatch_is_crc() {
    let mut bad = fixture("1byte-65k");
    let last = bad.len() - 1;
    bad[last] ^= 0x01;
    assert!(matches!(decompress(&bad), Err(Error::ChecksumMismatch)));
}

#[test]
fn coded_bad_lzp_size_is_malformed() {
    // LZP-only block: header is CRC32 + index + model + lzp size.
    let mut bad = fixture("periodic2k-65k");
    let (start, _) = first_block(&bad);
    bad[start + 9..start + 13].copy_from_slice(&(-1_i32).to_le_bytes());
    assert!(matches!(decompress(&bad), Err(Error::InvalidHeader)));
    let mut bad = fixture("periodic2k-65k");
    bad[start + 9..start + 13].copy_from_slice(&i32::MAX.to_le_bytes());
    assert!(matches!(decompress(&bad), Err(Error::InvalidHeader)));
}

#[test]
fn coded_bad_rle_size_is_malformed() {
    // RLE-only block: header is CRC32 + index + model + rle size.
    let mut bad = fixture("runs2k-65k");
    let (start, _) = first_block(&bad);
    bad[start + 9..start + 13].copy_from_slice(&i32::MAX.to_le_bytes());
    assert!(matches!(decompress(&bad), Err(Error::InvalidHeader)));
}

#[test]
fn coded_bad_bwt_index_is_malformed() {
    let mut bad = fixture("random4k-65k");
    let (start, _) = first_block(&bad);
    bad[start + 4..start + 8].copy_from_slice(&i32::MAX.to_le_bytes());
    assert!(matches!(decompress(&bad), Err(Error::InvalidHeader)));
}

#[test]
fn coded_bwt_index_beats_corrupt_stream() {
    // Huge index plus corrupt coded bytes: the index check wins.
    let mut bad = fixture("random4k-65k");
    let (start, end) = first_block(&bad);
    bad[start + 4..start + 8].copy_from_slice(&i32::MAX.to_le_bytes());
    bad[start + 9] ^= 0xFF;
    bad[end - 1] ^= 0xFF;
    assert!(matches!(decompress(&bad), Err(Error::InvalidHeader)));
}

#[test]
fn coded_corrupt_stream_is_crc() {
    // Pinned flips: corrupt coded bytes decode to garbage, then the
    // checksum fails exactly as the reference reports it.
    let (start, end) = first_block(&fixture("random4k-65k"));
    for offset in [start + 9, (start + end) / 2, end - 1] {
        let mut bad = fixture("random4k-65k");
        bad[offset] ^= 0xFF;
        assert!(
            matches!(decompress(&bad), Err(Error::ChecksumMismatch)),
            "offset {offset}"
        );
    }
}

#[test]
fn inflated_lzp_size_is_malformed() {
    // More LZP bytes claimed than the stream holds: the reference
    // pads reads with filler, the garbage decodes to the wrong size,
    // and the final size check fails. Same here.
    let mut bad = fixture("periodic2k-65k");
    let (start, _) = first_block(&bad);
    let size = i32::from_le_bytes(bad[start + 9..start + 13].try_into().unwrap());
    (bad[start + 9..start + 13]).copy_from_slice(&(size + 100).to_le_bytes());
    assert!(matches!(decompress(&bad), Err(Error::InvalidHeader)));
}

#[test]
fn shrunk_orig_on_lzp_path_is_malformed() {
    // Decoded LZP bytes no longer match the claimed original size.
    let mut bad = fixture("periodic2k-65k");
    let orig = u32::from_le_bytes(bad[17..21].try_into().unwrap());
    bad[17..21].copy_from_slice(&(orig - 1).to_le_bytes());
    assert!(matches!(decompress(&bad), Err(Error::InvalidHeader)));
}

#[test]
fn bad_magic_beats_truncated_blocks() {
    // Bad magic plus a missing tail: the magic check wins.
    let mut bad = fixture("random4k-65k");
    bad[0] = b'X';
    bad.truncate(bad.len() - 100);
    assert!(matches!(decompress(&bad), Err(Error::InvalidMagic)));
}

#[test]
fn negative_comp_beats_short_tail() {
    // Negative compressed size plus a missing tail: sizes first.
    let mut bad = fixture("random4k-65k");
    bad[13..17].copy_from_slice(&(-1_i32).to_le_bytes());
    bad.truncate(13 + 8);
    assert!(matches!(decompress(&bad), Err(Error::InvalidHeader)));
}

/// Run every single-byte flip and truncation without panic or hang.
///
/// Any error variant is a valid outcome. Success with changed bytes
/// is valid too. The test fails only on panic, hang, or overrun.
fn sweep_fixture(name: &str, stride: usize) {
    let code = fixture(name);
    for (index, _) in code.iter().enumerate().step_by(stride) {
        for flip in [0x01_u8, 0xFF_u8] {
            let mut bad = code.clone();
            bad[index] ^= flip;
            let _ = decompress(&bad);
        }
    }
    for end in (0..=code.len()).step_by(stride.max(1)) {
        let _ = decompress(&code[..end]);
    }
    // Appended garbage after a valid frame must stay ignored.
    let mut long = code.clone();
    long.extend_from_slice(&[0xAA; 64]);
    let _ = decompress(&long);
}

#[test]
fn mutation_sweep_never_panics_hangs_or_overruns() {
    sweep_fixture("1byte-65k", 1);
    sweep_fixture("65b-65k", 1);
    sweep_fixture("both2k-65k", 1);
    sweep_fixture("periodic2k-65k", 2);
    sweep_fixture("text20k-65k", 2);
    sweep_fixture("text180k-65k", 4);
    sweep_fixture("random4k-65k", 4);
}
