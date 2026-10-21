//! Lempel-Ziv-plus-prediction stage, forward and inverse.
//!
//! This stage collapses long redundant runs before entropy coding. The
//! encoder replaces repeats of at least [`MIN_MATCH`] bytes with a marker
//! run. The decoder restores the repeats. A literal marker byte escapes as
//! marker plus [`ESCAPE`]. Both directions clear the hash table on entry,
//! exactly like the reference. The decoder clamps match copies to the free
//! output space, so untrusted data cannot force an overflow.

use crate::buffer::filled;
use crate::error::Error;
use crate::error::Result;

/// Bits of the hash table index. The table holds one slot per index value.
pub const DICTIONARY_BITS: u32 = 18;

/// Slots in the hash table. Matches the reference table byte size.
pub const TABLE_LEN: usize = 1_usize << DICTIONARY_BITS;

/// Shortest repeat the encoder replaces with a match run.
pub const MIN_MATCH: usize = 40;

/// Smallest input the encoder accepts. Shorter inputs return `None`.
pub const MIN_INPUT: usize = MIN_MATCH + 32;

/// Marker byte that starts a match run.
pub const MATCH_MARKER: u8 = 0xf2;

/// Escape byte. A literal marker encodes as marker plus escape.
pub const ESCAPE: u8 = 255;

/// Match lengths encode minus [`MIN_MATCH`], in base 254 digits.
const LENGTH_BASE: usize = 254;

/// Stored value of one full length digit.
const LENGTH_BYTE: u8 = 254;

/// Bytes of slack the encoder keeps at the end of the output.
const ENCODE_RESERVE: usize = 8;

/// Hash table of the two stages.
pub type HashTable = [i32; TABLE_LEN];

/// Build a fresh zeroed hash table for [`encode_into`] and [`decode_into`].
///
/// # Errors
///
/// Returns [`Error::OutOfMemory`] when the table allocation fails.
pub fn new_table() -> Result<Box<HashTable>> {
    filled(TABLE_LEN, 0)
        .and_then(|table| table.into_boxed_slice().try_into().ok())
        .ok_or(Error::OutOfMemory)
}

/// Hash the 4-byte context into a table index. Mirrors the reference.
#[inline]
const fn hash(ctx: u32) -> usize {
    let raw = (ctx >> 15) ^ ctx ^ (ctx >> 3);
    (raw & ((1_u32 << DICTIONARY_BITS) - 1)) as usize
}

/// Load 4 bytes as one word. Uses native byte order like the reference.
#[inline]
fn load(block: &[u8], at: usize) -> u32 {
    let word = &block[at..at + 4];
    u32::from_ne_bytes([word[0], word[1], word[2], word[3]])
}

/// Fold the 4 bytes before `end` into a context value. The newest byte
/// sits in the low bits.
#[inline]
fn context_at(block: &[u8], end: usize) -> u32 {
    let word = &block[end - 4..end];
    u32::from_be_bytes([word[0], word[1], word[2], word[3]])
}

/// Encode `input` into `output` with scratch table `table`.
///
/// Needs `input.len() >= MIN_INPUT` and `output.len() >= input.len()`.
/// The table is the full table. The reserve counts from the input size, so any
/// larger output stays byte-identical to the reference. Returns `None`
/// exactly where the reference returns `-1`. Clears the table on entry.
#[must_use]
pub fn encode_into(input: &[u8], output: &mut [u8], table: &mut HashTable) -> Option<usize> {
    encode_kernel(input, output, table)
}

// Reason: positions stay below 2 GiB through the block API, so the
// casts below wrap exactly where the reference relies on 32-bit slots.
// Reason: `ref_pos` derives from a positive slot, so it never goes negative.
// Reason: the final length byte truncates exactly like the reference store.
#[expect(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
/// Body of [`encode_into`].
fn encode_kernel(input: &[u8], output: &mut [u8], table: &mut HashTable) -> Option<usize> {
    if input.len() < MIN_INPUT || output.len() < input.len() {
        return None;
    }
    table.fill(0);
    output[..4].copy_from_slice(&input[..4]);
    let mut pos = 4_usize;
    let mut next_out = 4_usize;
    let end = input.len();
    let bound = end - MIN_INPUT;
    let out_eob = end - ENCODE_RESERVE;
    let mut ctx = context_at(input, 4);
    // Latest input offset covered by a rejected short match. Later match
    // tries must still agree with the rejected bytes at this offset.
    let mut heur = 0_usize;
    while pos < bound && next_out < out_eob {
        let slot = hash(ctx);
        let val = table[slot];
        table[slot] = pos as i32;
        if val > 0 {
            let ref_pos = val as usize;
            // Both windows sit inside the input: `pos + 40` stays below
            // `end - 32`, and `ref_pos` stays below `pos`.
            if load(input, pos + MIN_MATCH - 4) == load(input, ref_pos + MIN_MATCH - 4)
                && load(input, pos) == load(input, ref_pos)
                && !(heur > pos && load(input, heur) != load(input, ref_pos + (heur - pos)))
            {
                let mut len = extend_match(input, pos, ref_pos, bound);
                if len < MIN_MATCH {
                    // Short matches stay literal, but they fence later tries.
                    // `heur` stays below `end - 32`, so later reads stay safe.
                    if heur < pos + len {
                        heur = pos + len;
                    }
                    emit_literal(output, input, &mut pos, &mut next_out, &mut ctx, true);
                } else {
                    // Tail reads stay inside the input: the word loop leaves
                    // `pos + len` below `end - 66`, and `ref_pos` below `pos`.
                    for _ in 0..3 {
                        if input[pos + len] == input[ref_pos + len] {
                            len += 1;
                        }
                    }
                    pos += len;
                    ctx = context_at(input, pos);
                    output[next_out] = MATCH_MARKER;
                    next_out += 1;
                    write_match_len(output, &mut next_out, out_eob, len - MIN_MATCH);
                }
            } else {
                // Warm slot without usable match: same miss rule as above.
                emit_literal(output, input, &mut pos, &mut next_out, &mut ctx, true);
            }
        } else {
            // Cold slot: the decoder reads the marker as one literal byte,
            // so no escape byte follows.
            emit_literal(output, input, &mut pos, &mut next_out, &mut ctx, false);
        }
    }
    // The tail pass copies literals only. It still escapes a marker when
    // the decoder would take the match path for the same table state.
    ctx = context_at(input, pos);
    while pos < end && next_out < out_eob {
        let slot = hash(ctx);
        let val = table[slot];
        table[slot] = pos as i32;
        emit_literal(output, input, &mut pos, &mut next_out, &mut ctx, val > 0);
    }
    if next_out >= out_eob {
        return None;
    }
    Some(next_out)
}

/// Count the bytes that `pos` and `ref_pos` share, in steps of 4 bytes.
///
/// Start at 4, as the match test already compared the first word. Stop at
/// the first unequal word or when `pos + len` reaches `bound`. The result
/// matches a word-by-word loop. Each window reads [`WINDOW`] bytes: `pos + len`
/// stays below `bound`, which is 72 bytes before the end, and `ref_pos`
/// stays below `pos`.
fn extend_match(input: &[u8], pos: usize, ref_pos: usize, bound: usize) -> usize {
    let mut len = 4_usize;
    while pos + len < bound {
        let diff = first_diff(&input[pos + len..], &input[ref_pos + len..]);
        // Words left before `bound`, counted from `pos + len`. Never zero here.
        let room = (bound - pos - len).div_ceil(4);
        let words = (diff / 4).min(room);
        len += 4 * words;
        if words < WINDOW / 4 {
            break;
        }
    }
    len
}

/// Bytes in one window of the match extension.
const WINDOW: usize = 32;

/// Return the index of the first unequal byte of two windows.
///
/// Return [`WINDOW`] when all bytes are equal. Both slices need [`WINDOW`] bytes.
#[inline]
fn first_diff(left: &[u8], right: &[u8]) -> usize {
    let left: &[u8; WINDOW] = left[..WINDOW].try_into().unwrap();
    let right: &[u8; WINDOW] = right[..WINDOW].try_into().unwrap();
    if left == right {
        return WINDOW;
    }
    first_unequal(left, right)
}

/// Return the index of the first unequal byte of two windows that differ.
///
/// The call stays out of line. Inline, the compiler merges this search with
/// the equal test and builds a slow byte-by-byte vector body.
#[inline(never)]
fn first_unequal(left: &[u8; WINDOW], right: &[u8; WINDOW]) -> usize {
    left.iter()
        .zip(right)
        .position(|(a, b)| a != b)
        .unwrap_or(0)
}

/// Emit one literal, plus the escape after a marker when `warm` is set.
///
/// The decoder takes the match path for a warm slot, so a marker
/// literal needs its escape byte there. A cold slot needs none. Both
/// stores land inside `output`: the caller holds `next_out` below the
/// reserve, and the escape below the input size, which never exceeds
/// the output size.
fn emit_literal(
    output: &mut [u8],
    input: &[u8],
    pos: &mut usize,
    next_out: &mut usize,
    ctx: &mut u32,
    warm: bool,
) {
    let next = input[*pos];
    output[*next_out] = next;
    *next_out += 1;
    *pos += 1;
    *ctx = (*ctx << 8) | u32::from(next);
    if warm && next == MATCH_MARKER {
        output[*next_out] = ESCAPE;
        *next_out += 1;
    }
}

/// Store a match length minus [`MIN_MATCH`] as base 254 digits.
///
/// Stops early at the reserve, like the reference. The caller then reports
/// no saving, so the cut digits never reach a decoder.
// Reason: the caller keeps `next_out` at or below the reserve, so each
// store below lands inside the output. The final byte truncates exactly
// like the reference store when the digit run cuts short.
#[expect(clippy::cast_possible_truncation)]
const fn write_match_len(
    output: &mut [u8],
    next_out: &mut usize,
    out_eob: usize,
    mut extra: usize,
) {
    while extra >= LENGTH_BASE {
        extra -= LENGTH_BASE;
        output[*next_out] = LENGTH_BYTE;
        *next_out += 1;
        if *next_out >= out_eob {
            break;
        }
    }
    output[*next_out] = extra as u8;
    *next_out += 1;
}

// Reason: positions stay below 2 GiB through the block API, so the
// casts below wrap exactly where the reference relies on 32-bit slots.
// Reason: `ref_pos` and the copy end derive from positive lengths, so
// they never go negative. Small outputs clamp through `max_len`.
#[expect(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
/// Decode `input` into `output` with scratch table `table`.
///
/// Needs `input.len() >= 4`, `output.len() >= 4`, and `table.len() >=
/// TABLE_LEN`. Match copies stop at the end of `output`, like the clamped
/// reference. Returns the decoded size. Clears the table on entry.
///
/// # Errors
///
/// Returns [`Error::CorruptStageData`] when the payload ends inside a match run or the
/// buffers cannot hold a valid decode. The block layer maps every decode
/// failure to this variant, exactly like the reference.
pub fn decode_into(input: &[u8], output: &mut [u8], table: &mut HashTable) -> Result<usize> {
    if input.len() < 4 || output.len() < 4 {
        return Err(Error::CorruptStageData);
    }
    table.fill(0);
    output[..4].copy_from_slice(&input[..4]);
    let mut pos = 4_usize;
    let mut next_out = 4_usize;
    let end = input.len();
    let capacity = output.len();
    let mut ctx = context_at(output, 4);
    while pos < end && next_out < capacity {
        let slot = hash(ctx);
        let val = table[slot];
        table[slot] = next_out as i32;
        if input[pos] == MATCH_MARKER && val > 0 {
            pos += 1;
            if pos == end {
                return Err(Error::CorruptStageData);
            }
            if input[pos] == ESCAPE {
                pos += 1;
                output[next_out] = MATCH_MARKER;
                next_out += 1;
                ctx = (ctx << 8) | u32::from(MATCH_MARKER);
            } else {
                let max_len = capacity - next_out;
                let mut len = MIN_MATCH;
                loop {
                    if pos == end {
                        return Err(Error::CorruptStageData);
                    }
                    // `len` stays within `max_len` plus one digit, so it cannot wrap.
                    if len <= max_len {
                        len += usize::from(input[pos]);
                    }
                    let byte = input[pos];
                    pos += 1;
                    if byte != LENGTH_BYTE {
                        break;
                    }
                }
                // Stored slots only hold past output offsets, so `ref_pos`
                // stays below `next_out` and every read lands inside `output`.
                let ref_pos = val as usize;
                let stop = if len > max_len {
                    capacity
                } else {
                    next_out + len
                };
                copy_match(output, ref_pos, stop, &mut next_out);
                ctx = context_at(output, next_out);
            }
        } else {
            let next = input[pos];
            output[next_out] = next;
            next_out += 1;
            pos += 1;
            ctx = (ctx << 8) | u32::from(next);
        }
    }
    Ok(next_out)
}

/// Copy a match run, so overlaps repeat fresh bytes.
///
/// The source ends where the copied bytes begin, so each step copies
/// a block that does not overlap its source. A block doubles the
/// repeated span of the step before it.
fn copy_match(output: &mut [u8], ref_pos: usize, stop: usize, next_out: &mut usize) {
    // `ref_pos` is below `next_out`. The floor keeps the loop finite anyway.
    let dist = (*next_out - ref_pos).max(1);
    let count = stop - *next_out;
    let mut copied = 0;
    while copied < count {
        let chunk = (count - copied).min(dist + copied);
        output.copy_within(ref_pos..ref_pos + chunk, *next_out + copied);
        copied += chunk;
    }
    *next_out = stop;
}

#[cfg(test)]
mod tests {
    use super::HashTable;
    use super::decode_into;
    use super::encode_into;
    use super::new_table;
    use crate::error::Error;

    fn fresh_table() -> Box<HashTable> {
        new_table().expect("table allocates")
    }

    fn encode(input: &[u8]) -> Option<Vec<u8>> {
        let mut table = fresh_table();
        let mut out = vec![0_u8; input.len()];
        encode_into(input, &mut out, &mut table).map(|n| {
            out.truncate(n);
            out
        })
    }

    fn decode(enc: &[u8], capacity: usize) -> Result<Vec<u8>, Error> {
        let mut table = fresh_table();
        let mut out = vec![0_u8; capacity];
        decode_into(enc, &mut out, &mut table).map(|n| {
            out.truncate(n);
            out
        })
    }

    #[test]
    fn forward_matches_oracle_minmatch40() {
        let raw = include_bytes!("../tests/vectors/lzp_minmatch40.in");
        let want = include_bytes!("../tests/vectors/lzp_minmatch40.enc");
        assert_eq!(encode(raw), Some(want.to_vec()));
    }

    #[test]
    fn forward_matches_oracle_minmatch120() {
        let raw = include_bytes!("../tests/vectors/lzp_minmatch120.in");
        let want = include_bytes!("../tests/vectors/lzp_minmatch120.enc");
        assert_eq!(encode(raw), Some(want.to_vec()));
    }

    #[test]
    fn forward_matches_oracle_escape() {
        let raw = include_bytes!("../tests/vectors/lzp_escape.in");
        let want = include_bytes!("../tests/vectors/lzp_escape.enc");
        assert_eq!(encode(raw), Some(want.to_vec()));
    }

    #[test]
    fn forward_matches_oracle_longmatch() {
        let raw = include_bytes!("../tests/vectors/lzp_longmatch.in");
        let want = include_bytes!("../tests/vectors/lzp_longmatch.enc");
        assert_eq!(encode(raw), Some(want.to_vec()));
    }

    #[test]
    fn forward_matches_oracle_dictedge() {
        let raw = include_bytes!("../tests/vectors/lzp_dictedge.in");
        let want = include_bytes!("../tests/vectors/lzp_dictedge.enc");
        assert_eq!(encode(raw), Some(want.to_vec()));
    }

    #[test]
    fn forward_matches_oracle_text() {
        let raw = include_bytes!("../tests/vectors/lzp_text.in");
        let want = include_bytes!("../tests/vectors/lzp_text.enc");
        assert_eq!(encode(raw), Some(want.to_vec()));
    }

    #[test]
    fn forward_rejects_oracle_negatives() {
        let cases: [&[u8]; 5] = [
            include_bytes!("../tests/vectors/lzp_random.in"),
            include_bytes!("../tests/vectors/lzp_nomatch39.in"),
            include_bytes!("../tests/vectors/lzp_short71.in"),
            include_bytes!("../tests/vectors/lzp_short72.in"),
            include_bytes!("../tests/vectors/lzp_short73.in"),
        ];
        for raw in cases {
            assert!(encode(raw).is_none(), "input of {} bytes", raw.len());
        }
    }

    #[test]
    fn inverse_restores_oracle_vectors() {
        let pairs: [(&[u8], &[u8]); 6] = [
            (
                include_bytes!("../tests/vectors/lzp_minmatch40.enc"),
                include_bytes!("../tests/vectors/lzp_minmatch40.in"),
            ),
            (
                include_bytes!("../tests/vectors/lzp_minmatch120.enc"),
                include_bytes!("../tests/vectors/lzp_minmatch120.in"),
            ),
            (
                include_bytes!("../tests/vectors/lzp_escape.enc"),
                include_bytes!("../tests/vectors/lzp_escape.in"),
            ),
            (
                include_bytes!("../tests/vectors/lzp_longmatch.enc"),
                include_bytes!("../tests/vectors/lzp_longmatch.in"),
            ),
            (
                include_bytes!("../tests/vectors/lzp_dictedge.enc"),
                include_bytes!("../tests/vectors/lzp_dictedge.in"),
            ),
            (
                include_bytes!("../tests/vectors/lzp_text.enc"),
                include_bytes!("../tests/vectors/lzp_text.in"),
            ),
        ];
        for (enc, raw) in pairs {
            let got = decode(enc, raw.len()).expect("oracle payload decodes");
            assert_eq!(got, raw);
        }
    }

    #[test]
    fn roundtrip_without_fixture() {
        let mut raw = [0_u8; 300];
        let mut next = 0_u8;
        for cell in &mut raw {
            // Cycles every 64 bytes, so repeats span the match window.
            *cell = next;
            next = if next == 63 { 0 } else { next + 1 };
        }
        let enc = encode(&raw).expect("periodic input compresses");
        let back = decode(&enc, raw.len()).expect("own output decodes");
        assert_eq!(back, raw);
    }

    #[test]
    fn corrupt_payloads_yield_crc() {
        let trunc = include_bytes!("../tests/vectors/lzp_trunc.bad");
        let flood = include_bytes!("../tests/vectors/lzp_flood.bad");
        let empty: &[u8] = &[];
        let tiny: &[u8] = b"ABC";
        for bad in [trunc.as_slice(), flood.as_slice(), empty, tiny] {
            assert!(matches!(decode(bad, 4096), Err(Error::CorruptStageData)));
        }
    }

    #[test]
    fn long_match_clamps_to_output() {
        let flood = include_bytes!("../tests/vectors/lzp_floodterm.bad");
        let full = decode(flood, 4096).expect("roomy decode succeeds");
        assert!(full.len() > 100);
        let clamped = decode(flood, 100).expect("small decode clamps");
        assert_eq!(clamped, full[..100]);
    }

    #[test]
    fn undersized_buffers_fail_without_panic() {
        let raw = include_bytes!("../tests/vectors/lzp_escape.in");
        let mut table = fresh_table();
        let mut tiny = vec![0_u8; 8];
        assert!(encode_into(raw, &mut tiny, &mut table).is_none());
        let mut out = vec![0_u8; raw.len()];
        let enc = include_bytes!("../tests/vectors/lzp_escape.enc");
        let mut no_room = [0_u8; 2];
        assert!(matches!(
            decode_into(enc, &mut no_room, &mut fresh_table()),
            Err(Error::CorruptStageData)
        ));
        let empty: &[u8] = &[];
        assert!(matches!(
            decode_into(empty, &mut out, &mut fresh_table()),
            Err(Error::CorruptStageData)
        ));
    }

    #[test]
    fn table_state_does_not_leak_between_calls() {
        // The reference clears its table on every call. Reused tables
        // must give the same bytes as fresh ones in both directions.
        let text = include_bytes!("../tests/vectors/lzp_text.in");
        let esc = include_bytes!("../tests/vectors/lzp_escape.in");
        let room = text.len().max(esc.len());
        let mut table = fresh_table();
        let mut out = vec![0_u8; room];
        let first = encode_into(text, &mut out, &mut table).map(|n| out[..n].to_vec());
        let _ = encode_into(esc, &mut out, &mut table);
        let again = encode_into(text, &mut out, &mut table).map(|n| out[..n].to_vec());
        assert_eq!(first, again);
        let text_enc = include_bytes!("../tests/vectors/lzp_text.enc");
        let esc_enc = include_bytes!("../tests/vectors/lzp_escape.enc");
        let mut dec = vec![0_u8; room];
        let n1 = decode_into(text_enc, &mut dec, &mut table).expect("decodes");
        let first_dec = dec[..n1].to_vec();
        decode_into(esc_enc, &mut dec, &mut table).ok();
        let n2 = decode_into(text_enc, &mut dec, &mut table).expect("decodes");
        let again_dec = dec[..n2].to_vec();
        assert_eq!(first_dec, again_dec);
    }

    /// Word-by-word match extension, kept as the reference for the window one.
    fn extend_match_words(input: &[u8], pos: usize, ref_pos: usize, bound: usize) -> usize {
        let word = |at: usize| u32::from_ne_bytes(input[at..at + 4].try_into().unwrap());
        let mut len = 4;
        while pos + len < bound {
            if word(pos + len) != word(ref_pos + len) {
                break;
            }
            len += 4;
        }
        len
    }

    #[test]
    fn match_extension_matches_word_loop() {
        let mut seed = 0x2545_F491_u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            seed >> 16
        };
        for _ in 0..3000 {
            let total = 80 + next() as usize % 400;
            let mut input: Vec<u8> = (0..total).map(|_| (next() % 3) as u8).collect();
            // Copy a span forward, then flip one byte, so a match ends at any offset.
            let ref_pos = next() as usize % 20;
            let dist = 1 + next() as usize % 30;
            let pos = ref_pos + dist;
            for at in pos..total {
                input[at] = input[at - dist];
            }
            let flip = pos + next() as usize % (total - pos);
            input[flip] ^= 1;
            let bound = total - super::MIN_INPUT;
            if pos >= bound {
                continue;
            }
            assert_eq!(
                super::extend_match(&input, pos, ref_pos, bound),
                extend_match_words(&input, pos, ref_pos, bound),
                "total {total} pos {pos} ref {ref_pos} flip {flip}"
            );
        }
    }
}
