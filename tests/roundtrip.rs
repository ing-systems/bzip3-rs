//! One-shot round-trip tests for the stored-raw path (ticket 02).
//!
//! Tiny inputs (< 64 bytes) travel as stored-raw blocks inside the 13-byte
//! one-shot frame. Fixtures under `vectors/stored-*` are C-oracle outputs:
//! `.bin` holds the input, `.bz3` holds the one-shot encoding of that input
//! with block size 1 MiB (header clamps it to 65 KiB).

use bzip3_rs::BlockSize;
use bzip3_rs::compress;
use bzip3_rs::decompress;
use bzip3_rs::error::Error;

const EMPTY_BIN: &[u8] = include_bytes!("vectors/stored-empty.bin");
const EMPTY_BZ3: &[u8] = include_bytes!("vectors/stored-empty.bz3");
const ONE_BIN: &[u8] = include_bytes!("vectors/stored-1byte.bin");
const ONE_BZ3: &[u8] = include_bytes!("vectors/stored-1byte.bz3");
const B63_BIN: &[u8] = include_bytes!("vectors/stored-63b.bin");
const B63_BZ3: &[u8] = include_bytes!("vectors/stored-63b.bz3");

fn default_size() -> BlockSize {
    BlockSize::DEFAULT
}

#[test]
fn empty_round_trips_through_zero_block_frame() {
    let encoded = compress(EMPTY_BIN, default_size()).unwrap();
    assert_eq!(encoded, EMPTY_BZ3);
    assert_eq!(encoded.len(), 13);
    assert_eq!(&encoded[0..5], b"BZ3v1");
    assert_eq!(
        u32::from_le_bytes(encoded[5..9].try_into().unwrap()),
        65 * 1024
    );
    assert_eq!(u32::from_le_bytes(encoded[9..13].try_into().unwrap()), 0);
    assert_eq!(decompress(&encoded).unwrap(), EMPTY_BIN);
}

#[test]
fn tiny_inputs_encode_to_oracle_bytes() {
    assert_eq!(compress(ONE_BIN, default_size()).unwrap(), ONE_BZ3);
    assert_eq!(compress(B63_BIN, default_size()).unwrap(), B63_BZ3);
}

#[test]
fn oracle_outputs_decode_byte_identically() {
    assert_eq!(decompress(EMPTY_BZ3).unwrap(), EMPTY_BIN);
    assert_eq!(decompress(ONE_BZ3).unwrap(), ONE_BIN);
    assert_eq!(decompress(B63_BZ3).unwrap(), B63_BIN);
}

#[test]
fn stored_payload_layout_is_crc_marker_raw() {
    // 1-byte payload: CRC32 LE + 0xFFFFFFFF marker + raw byte.
    let payload = &ONE_BZ3[13 + 8..];
    assert_eq!(payload.len(), 9);
    assert_eq!(
        u32::from_le_bytes(payload[0..4].try_into().unwrap()),
        0x417B_1DBC
    );
    assert_eq!(payload[4..8], [0xFF; 4]);
    assert_eq!(payload[8..], [b'A']);
}

#[test]
fn trailing_bytes_are_ignored() {
    for (raw, coded) in [
        (EMPTY_BIN, EMPTY_BZ3),
        (ONE_BIN, ONE_BZ3),
        (B63_BIN, B63_BZ3),
    ] {
        let mut with_trailing = coded.to_vec();
        with_trailing.extend_from_slice(b"TRAIL");
        assert_eq!(decompress(&with_trailing).unwrap(), raw);
    }
}

#[test]
fn decoder_rejects_short_input_and_bad_magic() {
    assert!(matches!(decompress(&[]), Err(Error::TruncatedHeader)));
    assert!(matches!(
        decompress(&EMPTY_BZ3[..5]),
        Err(Error::TruncatedHeader)
    ));
    let mut bad = EMPTY_BZ3.to_vec();
    bad[0] = b'X';
    assert!(matches!(decompress(&bad), Err(Error::InvalidMagic)));
}

#[test]
fn decoder_reports_truncation() {
    // Drop the last payload bytes: block claims 9 payload bytes, fewer follow.
    let cut = &ONE_BZ3[..ONE_BZ3.len() - 4];
    assert!(matches!(decompress(cut), Err(Error::TruncatedData)));
    // Drop inside the block-size pair: no full block header follows.
    let cut = &ONE_BZ3[..13 + 3];
    assert!(matches!(decompress(cut), Err(Error::TruncatedHeader)));
}

#[test]
fn decoder_reports_crc_mismatch() {
    let mut bad = ONE_BZ3.to_vec();
    let last = bad.len() - 1;
    bad[last] ^= 0x01;
    assert!(matches!(decompress(&bad), Err(Error::ChecksumMismatch)));
}

#[test]
fn decoder_reports_malformed_block_sizes() {
    // Negative compressed size.
    let mut bad = ONE_BZ3.to_vec();
    bad[13..17].copy_from_slice(&(-1_i32).to_le_bytes());
    assert!(matches!(decompress(&bad), Err(Error::InvalidHeader)));
    // Original size above the header block size.
    let mut bad = ONE_BZ3.to_vec();
    bad[17..21].copy_from_slice(&(65 * 1024 + 1_u32).to_le_bytes());
    assert!(matches!(decompress(&bad), Err(Error::InvalidHeader)));
}

#[test]
fn coded_blocks_encode_past_the_stored_limit() {
    // 64 bytes leave the stored-raw route for the full pipeline.
    // Our coded-block decode lands in ticket 09, so the C oracle
    // verifies these outputs (see tests/corpus).
    let input = [b'q'; 64];
    let encoded = compress(&input, default_size()).unwrap();
    assert_eq!(&encoded[0..5], b"BZ3v1");
    assert_eq!(u32::from_le_bytes(encoded[9..13].try_into().unwrap()), 1);
}
