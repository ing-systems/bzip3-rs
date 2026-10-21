//! Suffix-array transforms (encode and decode sides).
//!
//! Subset port of the bundled upstream suffix-array code
//! (`include/libsais.h` in the C reference at `fd3d71f`).
//! This module holds `bwt` (encode) plus `unbwt`
//! (decode) and their helpers.
//!
//! The forward suffix sort is SA-IS (Nong, Zhang, and Chan), not a
//! transliteration of the libsais core. Both directions run in
//! `O(n)` time.
//!
//! Upstream license notice: the ported logic derives from libsais,
//! which is Apache-2.0, (c) 2021-2022 Ilya Grebnov, with a stability
//! patch (c) 2022 Kamila Szewczyk under the same license.
//!
//! ===== Forward (encode) transform =====

use crate::buffer::refill;
use crate::buffer::resize_keep;
use crate::error::Error;
use crate::error::Result;

/// Run the forward Burrows-Wheeler transform.
///
/// Write the transform of `input` into `output` and return the
/// primary index. `output` must hold at least `input.len()` bytes.
/// Extra tail bytes in `output` stay untouched. The output bytes
/// and the index match the reference `libsais_bwt` on all inputs,
/// from empty to multi-block sizes.
///
/// # Errors
///
/// Returns [`Error::BwtFailed`] when `output` is too small, when the input
/// exceeds `i32::MAX` bytes, or when working memory allocation fails.
#[cfg(test)]
pub fn bwt(input: &[u8], output: &mut [u8]) -> Result<i32> {
    bwt_with(input, output, &mut BwtScratch::default())
}

/// Run the forward Burrows-Wheeler transform with caller scratch.
///
/// Same contract as `bwt`. The scratch keeps its buffers
/// between calls, so repeated blocks do not allocate again.
///
/// # Errors
///
/// Returns [`Error::BwtFailed`] under the same conditions as `bwt`.
pub fn bwt_with(input: &[u8], output: &mut [u8], scratch: &mut BwtScratch) -> Result<i32> {
    // Port of the `libsais_bwt` front end: the guards and the
    // short-input fast paths run first, in reference order.
    if output.len() < input.len() {
        return Err(Error::BwtFailed);
    }
    if input.is_empty() {
        return Ok(0);
    }
    if input.len() == 1 {
        output[0] = input[0];
        return Ok(1);
    }

    let n = input.len();
    scratch.sort_suffixes(input)?;
    let sa = &scratch.sa[..n];
    // Assemble the last column. This matches the reference copy:
    // `U[0]` is the byte before suffix 0, hence `input[n - 1]`. The
    // byte before each other suffix follows in rank order. The slot
    // of suffix 0 is the primary slot and has no byte of its own.
    output[0] = input[n - 1];
    let mut slot = 1;
    let mut primary = None;
    for (rank, &suffix) in sa.iter().enumerate() {
        if let Some(prev) = (suffix as usize).checked_sub(1) {
            output[slot] = input[prev];
            slot += 1;
        } else {
            primary = Some(rank);
        }
    }
    let primary = primary.ok_or(Error::BwtFailed)?;
    // `primary + 1` fits `i32`: the sort rejects huge inputs.
    i32::try_from(primary + 1).map_err(|_| Error::BwtFailed)
}

// ===== Inverse (decode) transform (later ticket) =====
// The decode side appends here. Do not add inverse functions above.

/// Run the inverse Burrows-Wheeler transform.
///
/// Rebuild the original bytes from `transform` and its primary
/// `index`. `transform` must hold reference `libsais_bwt` output,
/// and `index` must hold its 1-based primary index. `output` must
/// hold at least `transform.len()` bytes. Extra tail bytes in
/// `output` stay untouched. The output bytes match the encoded
/// original on all inputs, from empty to multi-block sizes.
///
/// The core follows the reference `libsais_unbwt` single-index
/// path: bigram histogram over the stored order, in-place
/// transpose, fast lookup table, `biPSI` links, then a paired
/// walk emitting two bytes per step. The multi-block walks,
/// context variants, and parallel paths are excluded: the caller
/// always passes one index, so only the single walk runs.
///
/// # Errors
///
/// Returns [`Error::BwtFailed`] when `output` is too small, when `index`
/// lies outside its valid range, when the input exceeds `i32::MAX`
/// bytes, or when working memory allocation fails.
///
/// # Panics
///
/// Never panics. All table reads stay in range by construction,
/// and the one open-ended scan reports [`Error::BwtFailed`] instead of
/// running past its table.
#[cfg(test)]
pub fn unbwt(transform: &[u8], output: &mut [u8], index: i32) -> Result<()> {
    unbwt_with(transform, output, index, &mut UnbwtScratch::default())
}

/// Reusable working memory for the inverse transform.
///
/// The buffers keep their capacity between calls, so repeated blocks
/// do not allocate again.
#[derive(Debug, Default)]
pub struct UnbwtScratch {
    /// Bigram counts, then range ends.
    bigram: Vec<u32>,
    /// Lookup table from position to bigram.
    fast: Vec<u16>,
    /// Successor of each position in the walk.
    link: Vec<u32>,
}

/// Run the inverse Burrows-Wheeler transform with caller scratch.
///
/// Same contract as the plain inverse transform. The scratch keeps its
/// buffers between calls.
///
/// # Errors
///
/// Returns [`Error::BwtFailed`] under the same conditions as the plain
/// inverse transform.
pub fn unbwt_with(
    transform: &[u8],
    output: &mut [u8],
    index: i32,
    scratch: &mut UnbwtScratch,
) -> Result<()> {
    // Port of the `libsais_unbwt` front end (`libsais_unbwt_aux`
    // with one index): the guards run first, in reference order.
    if output.len() < transform.len() {
        return Err(Error::BwtFailed);
    }
    if transform.is_empty() {
        // The reference requires the single index to equal `n`.
        return if index == 0 {
            Ok(())
        } else {
            Err(Error::BwtFailed)
        };
    }
    if transform.len() == 1 {
        if index != 1 {
            return Err(Error::BwtFailed);
        }
        output[0] = transform[0];
        return Ok(());
    }
    // The reference requires `1 <= index <= n`.
    if index < 1 {
        return Err(Error::BwtFailed);
    }
    let n = transform.len();
    let slot = usize::try_from(index - 1).map_err(|_| Error::BwtFailed)?;
    if slot >= n {
        return Err(Error::BwtFailed);
    }
    if usize::try_from(i32::MAX).map_or(true, |limit| n > limit) {
        return Err(Error::BwtFailed);
    }
    // From here `index` is a valid 1-based primary position.
    let primary = slot + 1;

    // Count each byte, then turn counts into 1-based group starts.
    let mut count = [0_usize; 256];
    unbwt_histogram(transform, &mut count);
    let mut start = [0_usize; 256];
    let mut next_free = 1_usize;
    for (slot, &group) in start.iter_mut().zip(count.iter()) {
        *slot = next_free;
        next_free += group;
    }

    // Count bigrams in stored order, then transpose the table.
    let UnbwtScratch { bigram, fast, link } = scratch;
    refill_buffer(bigram, 65_536, 0)?;
    unbwt_bigram_counts(transform, &start, &count, primary, bigram);
    unbwt_transpose_bigrams(bigram);

    // Turn bigram counts into starts and build the lookup table.
    let last = transform[0];
    let mut shift = 0_usize;
    while (n >> shift) > (1_usize << 17) {
        shift += 1;
    }
    refill_buffer(fast, 1 + (n >> shift), 0)?;
    unbwt_fastbits(bigram, usize::from(last), shift, fast);

    // Link bigram positions, then walk the pairs into `output`.
    refill_buffer(link, n + 1, 0)?;
    unbwt_bipsi(transform, &start, bigram, primary, link);
    unbwt_emit_pairs(link, bigram, fast, shift, primary, last, output)
}

/// Count each byte value in `data`.
///
/// Matches the reference histogram totals. The reference counts
/// with an unrolled 4-way loop; this loop writes the same totals.
fn unbwt_histogram(data: &[u8], count: &mut [usize; 256]) {
    for &byte in data {
        count[usize::from(byte)] += 1;
    }
}

/// Count bigrams of the stored order into `bigram`.
///
/// For each first byte `c`, count the bytes in its group range,
/// skipping the primary slot exactly like the reference: the low
/// part covers the stored items before the slot, the high part
/// covers the items after it. Counts land at `(c << 8) + d`.
fn unbwt_bigram_counts(
    transform: &[u8],
    start: &[usize; 256],
    count: &[usize; 256],
    primary: usize,
    bigram: &mut [u32],
) {
    for c in 0..256 {
        let first = start[c];
        let total = first + count[c];
        let hi = total.min(primary);
        if let Some(window) = transform.get(first..hi) {
            for &byte in window {
                bigram[(c << 8) + usize::from(byte)] += 1;
            }
        }
        let lo = first.max(primary + 1);
        if let Some(window) = transform.get(lo - 1..total - 1) {
            for &byte in window {
                bigram[(c << 8) + usize::from(byte)] += 1;
            }
        }
    }
}

/// Transpose the bigram table in place.
///
/// Swap every `(x, y)` count with `(y, x)`, so each entry holds
/// the count of its bigram in first/second order. The reference
/// swaps in cache-sized blocks; this loop writes the same table.
fn unbwt_transpose_bigrams(bigram: &mut [u32]) {
    for x in 0..256 {
        for y in (x + 1)..256 {
            bigram.swap((x << 8) + y, (y << 8) + x);
        }
    }
}

/// Turn bigram counts into starts and build the lookup table.
///
/// Replace each count with its 1-based range start, leaving room
/// for the implicit primary bigram after `last`. Fill `fast` so
/// `fast[p >> shift]` names a bigram at or before position `p`.
/// Positions stay below `n + 2`, and `n` is at most `i32::MAX`, so
/// they fit `u32`.
fn unbwt_fastbits(bigram: &mut [u32], last: usize, shift: usize, fast: &mut [u16]) {
    let mut pos = 0_usize;
    let mut total = 1_u32;
    for pair in 0..=u16::MAX {
        if usize::from(pair) == (last << 8) {
            total += 1;
        }
        let slot = &mut bigram[usize::from(pair)];
        let prev = total;
        total += *slot;
        *slot = prev;
        if prev != total {
            while pos <= ((total - 1) as usize >> shift) {
                fast[pos] = pair;
                pos += 1;
            }
        }
    }
}

/// Link bigram positions into the walk table.
///
/// Scan the stored bytes in order and hand each occurrence its
/// next free group slot. Record the stored item number in the
/// slot of its bigram, skipping the primary slot like the
/// reference. On return `bigram` holds range ends and `link`
/// maps each position to its successor.
fn unbwt_bipsi(
    transform: &[u8],
    start: &[usize; 256],
    bigram: &mut [u32],
    primary: usize,
    link: &mut [u32],
) {
    let mut cursor = *start;
    // The caller rejects `n > i32::MAX`, so item numbers fit `u32`.
    let primary_item = u32::try_from(primary).unwrap_or(u32::MAX);
    for (index, &byte) in (0_u32..).zip(transform) {
        // Item numbers skip the primary slot: stored bytes at or after
        // `primary` belong to the item one higher.
        let item = index + u32::from(index >= primary_item);
        let key = usize::from(byte);
        let pos = cursor[key];
        cursor[key] = pos + 1;
        if pos == primary {
            continue;
        }
        let prev = if pos > primary {
            transform[pos - 1]
        } else {
            transform[pos]
        };
        let pair = (usize::from(prev) << 8) + key;
        link[bigram[pair] as usize] = item;
        bigram[pair] += 1;
    }
}

/// Walk the links and emit the original bytes.
///
/// Start at the primary position. Each step finds the bigram
/// holding the current position, follows its link, and emits
/// the two bigram bytes. The length `n` is one less than the
/// link table size. The final byte is the stored first byte,
/// which the paired walk cannot reach on odd lengths.
fn unbwt_emit_pairs(
    link: &[u32],
    end: &[u32],
    fast: &[u16],
    shift: usize,
    primary: usize,
    last: u8,
    output: &mut [u8],
) -> Result<()> {
    let n = link.len().checked_sub(1).ok_or(Error::BwtFailed)?;
    let mut pos = primary;
    // An even length fills all `n` bytes, then `last` replaces the final one.
    for chunk in output[..n - n % 2].as_chunks_mut::<2>().0 {
        let mut pair = fast[pos >> shift];
        while end[usize::from(pair)] as usize <= pos {
            pair = pair.checked_add(1).ok_or(Error::BwtFailed)?;
        }
        pos = link[pos] as usize;
        *chunk = pair.to_be_bytes();
    }
    output[n - 1] = last;
    Ok(())
}

#[cfg(test)]
mod inv_tests {
    use super::unbwt;
    use crate::error::Error;

    /// Case names for the captured inverse vectors. Each case holds
    /// reference-produced transform bytes (`.bwt`), the reference
    /// index (`.index`), and the original bytes (`.original`).
    const INV_CASES: &[&str] = &[
        "empty",
        "1byte_a",
        "1byte_00",
        "1byte_ff",
        "2byte_ab",
        "2byte_ba",
        "2byte_aa",
        "banana",
        "mississippi",
        "abab",
        "aaaa_4",
        "aaaa_1000",
        "alphabet_sorted",
        "alphabet_rev",
        "rand63",
        "rand64",
        "rand65",
        "rand1000",
    ];

    /// Case names for the captured forward vectors, reused here so
    /// the inverse also covers the large inputs (64 KiB textlike,
    /// 1 MiB random) without duplicating their bytes.
    pub(super) const FWD_CASES: &[&str] = &[
        "empty",
        "1byte_a",
        "1byte_00",
        "1byte_ff",
        "2byte_ab",
        "2byte_ba",
        "2byte_aa",
        "banana",
        "mississippi",
        "abab",
        "aaaa_4",
        "aaaa_1000",
        "alphabet_sorted",
        "alphabet_rev",
        "rand63",
        "rand64",
        "rand65",
        "rand1000",
        "textlike_64k",
        "rand_1mib",
    ];

    /// Read one captured inverse vector. Returns the reference
    /// index, the reference transform bytes, and the original bytes.
    fn load_inverse_vector(name: &str) -> (i32, Vec<u8>, Vec<u8>) {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors");
        let transform = std::fs::read(format!("{dir}/sais_inv_{name}.bwt")).unwrap();
        let text = std::fs::read_to_string(format!("{dir}/sais_inv_{name}.index")).unwrap();
        let index: i32 = text.trim().parse().unwrap();
        let original = std::fs::read(format!("{dir}/sais_inv_{name}.original")).unwrap();
        (index, transform, original)
    }

    /// Read one captured forward vector. Returns the original bytes,
    /// the reference index, and the reference transform bytes.
    pub(super) fn load_forward_vector(name: &str) -> (Vec<u8>, i32, Vec<u8>) {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors");
        let input = std::fs::read(format!("{dir}/sais_fwd_{name}.input")).unwrap();
        let text = std::fs::read_to_string(format!("{dir}/sais_fwd_{name}.expected")).unwrap();
        let mut lines = text.lines();
        let index: i32 = lines.next().unwrap().parse().unwrap();
        let hex: String = lines.collect();
        let mut transform = Vec::with_capacity(hex.len() / 2);
        let bytes = hex.as_bytes();
        for pair in bytes.chunks(2) {
            let s = std::str::from_utf8(pair).unwrap();
            transform.push(u8::from_str_radix(s, 16).unwrap());
        }
        (input, index, transform)
    }

    #[test]
    fn inverse_restores_reference_vectors() {
        for name in INV_CASES {
            let (index, transform, want) = load_inverse_vector(name);
            let mut got = vec![0_u8; transform.len()];
            unbwt(&transform, &mut got, index).unwrap();
            assert_eq!(got, want, "original for {name}");
        }
    }

    #[test]
    fn inverse_restores_large_reference_vectors() {
        for name in FWD_CASES {
            let (want, index, transform) = load_forward_vector(name);
            let mut got = vec![0_u8; transform.len()];
            unbwt(&transform, &mut got, index).unwrap();
            assert_eq!(got, want, "original for {name}");
        }
    }

    /// Next state of a deterministic 64-bit mixer. No external
    /// random crate is available, so tests mix their own bytes.
    pub(super) fn next_mix(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state = state.wrapping_mul(0x5851_F42D_4C95_7F2D);
        *state
    }

    #[test]
    fn inverse_roundtrips_forward() {
        let mut state = 0x243F_6A88_85A3_08D3_u64;
        for round in 0..200 {
            let len = (next_mix(&mut state) % 400) as usize;
            let mut input = vec![0_u8; len];
            // Mix textlike runs with binary noise so primary
            // indices spread across the full range.
            let alphabet = if round % 2 == 0 { 4_u64 } else { 256_u64 };
            for byte in &mut input {
                *byte = u8::try_from(next_mix(&mut state) % alphabet).unwrap();
            }
            let mut coded = vec![0_u8; len];
            let index = super::bwt(&input, &mut coded).unwrap();
            let mut got = vec![0_u8; len];
            unbwt(&coded, &mut got, index).unwrap();
            assert_eq!(got, input, "round trip for len {len}");
        }
    }

    #[test]
    fn inverse_empty_returns_ok_and_touches_nothing() {
        let mut out = [0xA5_u8; 4];
        unbwt(&[], &mut out, 0).unwrap();
        assert_eq!(out, [0xA5_u8; 4]);
    }

    #[test]
    fn inverse_empty_rejects_nonzero_index() {
        let mut out = [0xA5_u8; 4];
        let err = unbwt(&[], &mut out, 1).unwrap_err();
        assert!(matches!(err, Error::BwtFailed));
    }

    #[test]
    fn inverse_single_byte_copies() {
        for byte in [0x00_u8, 0x61, 0xff] {
            let mut out = [0_u8; 1];
            unbwt(&[byte], &mut out, 1).unwrap();
            assert_eq!(out, [byte]);
        }
    }

    #[test]
    fn inverse_single_byte_rejects_bad_index() {
        for index in [0, -1, 2, i32::MAX, i32::MIN] {
            let mut out = [0_u8; 1];
            let err = unbwt(&[0x61], &mut out, index).unwrap_err();
            assert!(matches!(err, Error::BwtFailed), "index {index}");
        }
    }

    #[test]
    fn inverse_rejects_bad_index() {
        let (_, transform, _) = load_inverse_vector("banana");
        let n = transform.len();
        for index in [0, -1, i32::MIN, i32::try_from(n + 1).unwrap(), i32::MAX] {
            let mut out = vec![0_u8; n];
            let err = unbwt(&transform, &mut out, index).unwrap_err();
            assert!(matches!(err, Error::BwtFailed), "index {index}");
        }
    }

    #[test]
    fn inverse_rejects_short_output() {
        let (_, transform, _) = load_inverse_vector("banana");
        let mut out = vec![0_u8; transform.len() - 1];
        let err = unbwt(&transform, &mut out, 4).unwrap_err();
        assert!(matches!(err, Error::BwtFailed));
    }

    #[test]
    fn inverse_rejects_empty_output_for_nonempty_transform() {
        let mut out = [];
        let err = unbwt(&[0x61], &mut out, 1).unwrap_err();
        assert!(matches!(err, Error::BwtFailed));
    }

    #[test]
    fn inverse_tolerates_mutated_bytes_without_panic() {
        // Any byte string inverts to some output; mutated bytes
        // must decode without panic and without error.
        let (_, transform, _) = load_inverse_vector("mississippi");
        let n = transform.len();
        let mut mutated = transform;
        mutated[0] ^= 0xff;
        mutated[n - 1] ^= 0x01;
        let mut out = vec![0_u8; n];
        unbwt(&mutated, &mut out, 5).unwrap();
        assert_eq!(out.len(), n);
    }
}

// ===== Forward-transform helpers =====

/// Reusable working memory for the forward suffix sort.
///
/// Every buffer holds `u32` values: the sort rejects inputs above
/// `i32::MAX` bytes, so positions, names, and counts always fit.
#[derive(Debug, Default)]
pub struct BwtScratch {
    /// Suffix start positions in sorted order. Lower SA-IS levels
    /// keep their text and their suffix array in slices of it.
    sa: Vec<u32>,
    /// Working memory of each SA-IS level. Index 0 is the top level.
    levels: Vec<LevelScratch>,
}

/// Working memory of one SA-IS level.
#[derive(Debug, Default)]
struct LevelScratch {
    /// One bit for each text position. A set bit marks an S-type suffix.
    stype: Vec<u64>,
    /// Count of each symbol in the level text.
    count: Vec<u32>,
    /// Next free slot of each bucket, made again from `count` before each pass.
    bucket: Vec<u32>,
}

/// Value of an `sa` slot that holds no suffix.
const EMPTY: u32 = u32::MAX;

/// Upper limit of SA-IS levels. Each level holds at most half the
/// symbols of the level above it, and the top level holds below 2^31.
const MAX_LEVELS: usize = 32;

/// Text symbol of one SA-IS level: bytes at the top level, LMS
/// substring names below it.
trait Symbol: Copy + Ord {
    /// Return the bucket of the symbol.
    fn bucket_index(self) -> usize;

    /// Mark the S-type positions of `text` in `stype`.
    ///
    /// Returns [`Error::BwtFailed`] when working memory allocation fails.
    fn classify_types(text: &[Self], stype: &mut Vec<u64>) -> Result<()>;
}

impl Symbol for u8 {
    #[inline]
    fn bucket_index(self) -> usize {
        usize::from(self)
    }

    fn classify_types(text: &[Self], stype: &mut Vec<u64>) -> Result<()> {
        classify_bytes(text, stype)
    }
}

impl Symbol for u32 {
    #[inline]
    fn bucket_index(self) -> usize {
        self as usize
    }

    fn classify_types(text: &[Self], stype: &mut Vec<u64>) -> Result<()> {
        classify_types_scalar(text, stype)
    }
}

impl BwtScratch {
    /// Sort the suffixes of `text` with SA-IS into `self.sa`.
    ///
    /// The sort is a full lexicographic suffix sort, so the
    /// Burrows-Wheeler transform built on it matches the reference on
    /// every input. It runs in `O(n)` time: each level does a fixed
    /// number of passes, and each lower level holds at most half the
    /// symbols of the level above it.
    ///
    /// Returns [`Error::BwtFailed`] when `text` exceeds `i32::MAX` bytes or
    /// when working memory allocation fails.
    fn sort_suffixes(&mut self, text: &[u8]) -> Result<()> {
        let n = text.len();
        if usize::try_from(i32::MAX).map_or(true, |limit| n > limit) {
            return Err(Error::BwtFailed);
        }
        // `sort_level` fills the whole array before it reads one item.
        resize_keep(&mut self.sa, n, EMPTY).ok_or(Error::BwtFailed)?;
        if self.levels.len() < MAX_LEVELS {
            self.levels.resize_with(MAX_LEVELS, LevelScratch::default);
        }
        sort_level(text, &mut self.sa, 256, &mut self.levels)
    }
}

/// Clear `buf` and fill it with `len` copies of `value`.
///
/// Returns [`Error::BwtFailed`] when working memory allocation fails.
fn refill_buffer<V: Copy>(buf: &mut Vec<V>, len: usize, value: V) -> Result<()> {
    refill(buf, len, value).ok_or(Error::BwtFailed)
}

/// Sort the suffixes of one SA-IS level into `sa`.
///
/// `text` holds symbols below `alphabet`. A virtual sentinel follows the
/// last symbol and sorts before all symbols. `levels[0]` is the
/// working memory of this level. The lower levels use the rest.
///
/// The steps are: sort the LMS substrings, name them, sort the
/// reduced text of names, then induce the full order from the
/// sorted LMS suffixes.
///
/// Returns [`Error::BwtFailed`] when working memory allocation fails or
/// when the levels run out.
// Reason: `n` stays below 2^31, so every position fits `u32`.
#[expect(clippy::cast_possible_truncation)]
fn sort_level<T: Symbol>(
    text: &[T],
    sa: &mut [u32],
    alphabet: usize,
    levels: &mut [LevelScratch],
) -> Result<()> {
    let n = text.len();
    if n == 0 {
        return Ok(());
    }
    let (level, deeper) = levels.split_first_mut().ok_or(Error::BwtFailed)?;
    T::classify_types(text, &mut level.stype)?;
    count_symbols(text, alphabet, &mut level.count)?;
    refill_buffer(&mut level.bucket, alphabet, 0)?;
    let stype = &level.stype;
    let count = &level.count;
    let bucket = &mut level.bucket;

    // Sort the LMS substrings: put each LMS position at the tail of
    // its bucket, then induce the L-type and S-type positions.
    sa.fill(EMPTY);
    find_bucket_tails(count, bucket);
    for_each_lms(stype, |i| {
        let key = text[i].bucket_index();
        bucket[key] -= 1;
        sa[bucket[key] as usize] = i as u32;
    });
    induce_l(text, count, bucket, sa);
    induce_s(text, count, bucket, sa);

    // Move the sorted LMS positions to the front.
    let mut n1 = 0;
    for i in 0..n {
        let p = sa[i];
        if is_lms(stype, p as usize) {
            sa[n1] = p;
            n1 += 1;
        }
    }

    // Name the LMS substrings in sorted order. LMS positions are at
    // least 2 apart and below `n - 1`, so `n1 + p / 2` is a unique
    // slot below `n`. First store the length of each substring in its
    // slot. A substring runs from its LMS position to the next LMS
    // position, both included. The last substring reaches the virtual
    // sentinel, so it is unique and gets length 0.
    //
    // Two substrings with equal length and equal symbols are equal:
    // both end on an S-type position, and the type of each position
    // follows from its symbol and the type after it. This replaces a
    // walk over the type bits with one slice compare.
    sa[n1..].fill(EMPTY);
    let mut last_lms = None;
    for_each_lms(stype, |i| {
        if let Some(q) = last_lms {
            sa[n1 + q / 2] = (i - q + 1) as u32;
        }
        last_lms = Some(i);
    });
    if let Some(q) = last_lms {
        sa[n1 + q / 2] = 0;
    }
    let mut names = 0_u32;
    let (mut prev, mut prev_len) = (0, 0);
    for i in 0..n1 {
        let p = sa[i] as usize;
        let len = sa[n1 + p / 2] as usize;
        if len == 0 || len != prev_len || text[p..p + len] != text[prev..prev + len] {
            names += 1;
        }
        (prev, prev_len) = (p, len);
        sa[n1 + p / 2] = names - 1;
    }

    // Move the names, in text order, to the reduced text `sa[n - n1..]`.
    // Each step writes below `next`, in a slot already read, and moves
    // `next` only for a name. This avoids a branch on the unpredictable test.
    let mut next = n;
    for i in (n1..n).rev() {
        let slot = sa[i];
        sa[next - 1] = slot;
        next -= usize::from(slot != EMPTY);
    }

    // Sort the reduced text into `sa[..n1]`. Unique names give the
    // order at once. `n1 <= n - n1`, so the two slices do not overlap.
    let (head, reduced) = sa.split_at_mut(n - n1);
    let reduced_sa = &mut head[..n1];
    let names = names as usize;
    if names < n1 {
        sort_level(&*reduced, reduced_sa, names, deeper)?;
    } else {
        for (i, &name) in reduced.iter().enumerate() {
            reduced_sa[name as usize] = i as u32;
        }
    }

    // Turn the reduced order into LMS positions: put the LMS
    // positions in text order over the reduced text, then look up.
    let lms = reduced;
    let mut next = 0;
    for_each_lms(stype, |i| {
        lms[next] = i as u32;
        next += 1;
    });
    for slot in reduced_sa.iter_mut() {
        *slot = lms[*slot as usize];
    }

    // Put the sorted LMS suffixes at their bucket tails, last first,
    // then induce the full order.
    sa[n1..].fill(EMPTY);
    find_bucket_tails(count, bucket);
    for i in (0..n1).rev() {
        let p = sa[i];
        sa[i] = EMPTY;
        let key = text[p as usize].bucket_index();
        bucket[key] -= 1;
        sa[bucket[key] as usize] = p;
    }
    induce_l(text, count, bucket, sa);
    induce_s(text, count, bucket, sa);
    Ok(())
}

/// Mark the S-type positions of `text` in `stype`, one position at a time.
///
/// The last position is L-type: the virtual sentinel after it sorts
/// first. Position `i` is S-type when its suffix sorts before the
/// suffix at `i + 1`.
///
/// Returns [`Error::BwtFailed`] when working memory allocation fails.
fn classify_types_scalar<T: Symbol>(text: &[T], stype: &mut Vec<u64>) -> Result<()> {
    refill_buffer(stype, text.len().div_ceil(64), 0)?;
    set_stypes(text, 0, stype);
    Ok(())
}

/// Mark the S-type positions of `text[from..]` in `stype`, one position
/// at a time. The bits at and above `from` must be clear. The last
/// position of `text` is L-type, so the pass starts with no S-type
/// position after it.
fn set_stypes<T: Symbol>(text: &[T], from: usize, stype: &mut [u64]) {
    let mut next_is_s = false;
    for (i, pair) in text[from..].windows(2).enumerate().rev() {
        let is_s = pair[0] < pair[1] || (pair[0] == pair[1] && next_is_s);
        if is_s {
            let i = from + i;
            stype[i / 64] |= 1 << (i % 64);
        }
        next_is_s = is_s;
    }
}

/// Mark the S-type positions of byte `text` in `stype`, 64 at a time.
///
/// Same result as [`classify_types_scalar`]. For each word, compare
/// every byte with the next byte. Bit `i` of `less` is set when
/// `text[i] < text[i + 1]`. Bit `i` of `same` is set when they are equal.
/// Then `is_s(i) = less(i) | (same(i) & is_s(i + 1))`. This is a carry
/// chain, so one addition on the bit-reversed masks solves a whole word.
///
/// Returns [`Error::BwtFailed`] when working memory allocation fails.
fn classify_bytes(text: &[u8], stype: &mut Vec<u64>) -> Result<()> {
    let n = text.len();
    refill_buffer(stype, n.div_ceil(64), 0)?;
    // A full word `w` needs the byte at `64 * w + 64`, so it needs `64 * w + 64 < n`.
    let full = n.saturating_sub(1) / 64;
    // The positions from `64 * full` up have no full word. Set them one by one.
    set_stypes(text, 64 * full, stype);
    let mut carry = stype.get(full).map_or(0, |&word| word & 1);
    for w in (0..full).rev() {
        let Some((less, same)) = compare_masks(&text[64 * w..], &text[64 * w + 1..]) else {
            return Err(Error::BwtFailed);
        };
        let generate = less.reverse_bits();
        let spread = generate | same.reverse_bits();
        let sum = spread.wrapping_add(generate).wrapping_add(carry);
        let word = ((spread & !sum) | generate).reverse_bits();
        stype[w] = word;
        carry = word & 1;
    }
    Ok(())
}

/// Compare the first 64 bytes of `here` with the first 64 bytes of `next`.
///
/// Bit `j` of the first mask is set when `here[j] < next[j]`. Bit `j`
/// of the second mask is set when they are equal. Both slices hold at
/// least 64 bytes. The compare loop writes one flag byte for each
/// position, which the compiler turns into vector compares.
fn compare_masks(here: &[u8], next: &[u8]) -> Option<(u64, u64)> {
    let here: &[u8; 64] = here.first_chunk()?;
    let next: &[u8; 64] = next.first_chunk()?;
    let mut less = [0_u8; 64];
    let mut same = [0_u8; 64];
    for j in 0..64 {
        less[j] = u8::from(here[j] < next[j]);
        same[j] = u8::from(here[j] == next[j]);
    }
    Some((pack_flags(&less), pack_flags(&same)))
}

/// Pack 64 flag bytes, each 0 or 1, into 64 bits. Flag `j` becomes bit `j`.
///
/// One multiplication gathers the low bit of eight bytes into the top byte.
#[inline]
fn pack_flags(flags: &[u8; 64]) -> u64 {
    let mut bits = 0_u64;
    for (k, group) in flags.as_chunks::<8>().0.iter().enumerate() {
        let group = u64::from_le_bytes(*group);
        bits |= (group.wrapping_mul(0x0102_0408_1020_4080) >> 56) << (8 * k);
    }
    bits
}

/// Return true when position `i` is S-type.
#[inline]
const fn is_s(stype: &[u64], i: usize) -> bool {
    (stype[i / 64] >> (i % 64)) & 1 != 0
}

/// Return true when position `i` starts an LMS suffix: an S-type
/// position with an L-type position before it.
#[inline]
const fn is_lms(stype: &[u64], i: usize) -> bool {
    i > 0 && is_s(stype, i) && !is_s(stype, i - 1)
}

/// Return the LMS positions of word `w` of `stype` as a bit mask.
///
/// A bit is set for an S-type position with an L-type position before
/// it. Position 0 has no position before it, so it is never LMS.
#[inline]
const fn lms_word(stype: &[u64], w: usize) -> u64 {
    let word = stype[w];
    if w == 0 {
        return word & !(word << 1) & !1;
    }
    word & !((word << 1) | (stype[w - 1] >> 63))
}

/// Call `visit` with each LMS position, from low to high.
///
/// Bits above the last text position are clear in `stype`, so no
/// position reaches the text length.
fn for_each_lms(stype: &[u64], mut visit: impl FnMut(usize)) {
    for w in 0..stype.len() {
        let mut bits = lms_word(stype, w);
        while bits != 0 {
            visit(w * 64 + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
}

/// Count each symbol of `text` into `count`, sized to `k` symbols.
///
/// Returns [`Error::BwtFailed`] when working memory allocation fails.
fn count_symbols<T: Symbol>(text: &[T], k: usize, count: &mut Vec<u32>) -> Result<()> {
    refill_buffer(count, k, 0)?;
    for &symbol in text {
        count[symbol.bucket_index()] += 1;
    }
    Ok(())
}

/// Set each bucket to the first slot of its symbol range.
fn find_bucket_heads(count: &[u32], bucket: &mut [u32]) {
    let mut total = 0;
    for (slot, &size) in bucket.iter_mut().zip(count) {
        *slot = total;
        total += size;
    }
}

/// Set each bucket to the slot after the end of its symbol range.
fn find_bucket_tails(count: &[u32], bucket: &mut [u32]) {
    let mut total = 0;
    for (slot, &size) in bucket.iter_mut().zip(count) {
        total += size;
        *slot = total;
    }
}

/// Flag bit of an `sa` entry. The flag is set when the item before the
/// suffix is S-type, or when the suffix is at position 0. `induce_l`
/// sets it, so `induce_s` finds its work without a read of `stype`.
/// Positions stay below 2^31, so a flagged position never equals [`EMPTY`].
const PREV_S: u32 = 1 << 31;

/// Induce the order of the L-type suffixes, from left to right.
///
/// The virtual sentinel comes first, so the last suffix goes to the
/// head of its bucket. Then each suffix `p` in `sa` puts `p - 1` at
/// the next free head slot of its bucket when `p - 1` is L-type.
/// That is the case for an entry without [`PREV_S`]. The pass sets
/// the flag in each entry that it writes. [`EMPTY`] has the flag.
// Reason: `n` stays below 2^31, so every position fits `u32`.
#[expect(clippy::cast_possible_truncation)]
fn induce_l<T: Symbol>(text: &[T], count: &[u32], bucket: &mut [u32], sa: &mut [u32]) {
    let n = text.len();
    find_bucket_heads(count, bucket);
    let mut put = |sa: &mut [u32], j: usize| {
        let c = text[j].bucket_index();
        let flag = if j == 0 || text[j - 1] < text[j] {
            PREV_S
        } else {
            0
        };
        sa[bucket[c] as usize] = j as u32 | flag;
        bucket[c] += 1;
    };
    put(sa, n - 1);
    for i in 0..n {
        let v = sa[i];
        if v & PREV_S == 0 && v != 0 {
            put(sa, v as usize - 1);
        }
    }
}

/// Induce the order of the S-type suffixes, from right to left.
///
/// Each entry `p` in `sa` with [`PREV_S`] puts `p - 1` at the next free
/// tail slot of its bucket. This pass writes every S-type suffix
/// again, so it also replaces the LMS seeds. It clears the flag in each
/// entry that it reads, so no flag stays in `sa`.
// Reason: `n` stays below 2^31, so every position fits `u32`.
#[expect(clippy::cast_possible_truncation)]
fn induce_s<T: Symbol>(text: &[T], count: &[u32], bucket: &mut [u32], sa: &mut [u32]) {
    find_bucket_tails(count, bucket);
    for i in (0..text.len()).rev() {
        let v = sa[i];
        if v == EMPTY || v & PREV_S == 0 {
            continue;
        }
        let p = v & !PREV_S;
        sa[i] = p;
        if p == 0 {
            continue;
        }
        let j = p as usize - 1;
        let c = text[j].bucket_index();
        let flag = if j > 0 && text[j - 1] <= text[j] {
            PREV_S
        } else {
            0
        };
        bucket[c] -= 1;
        sa[bucket[c] as usize] = j as u32 | flag;
    }
}

#[cfg(test)]
mod tests {
    use super::BwtScratch;
    use super::bwt;
    use super::classify_bytes;
    use super::classify_types_scalar;
    use super::for_each_lms;
    use super::inv_tests::FWD_CASES;
    use super::inv_tests::load_forward_vector;
    use super::inv_tests::next_mix;
    use super::is_lms;
    use crate::error::Error;

    #[test]
    fn forward_matches_reference_vectors() {
        for name in FWD_CASES {
            let (input, want_index, want) = load_forward_vector(name);
            let mut got = vec![0_u8; input.len()];
            let got_index = bwt(&input, &mut got).unwrap();
            assert_eq!(got_index, want_index, "index for {name}");
            assert_eq!(got, want, "bytes for {name}");
        }
    }

    #[test]
    fn forward_single_byte_matches_reference() {
        for byte in [0x00_u8, 0x61, 0xff] {
            let mut out = [0_u8; 1];
            assert_eq!(bwt(&[byte], &mut out).unwrap(), 1);
            assert_eq!(out, [byte]);
        }
    }

    #[test]
    fn forward_empty_returns_zero() {
        let mut out = [0xA5_u8; 4];
        assert_eq!(bwt(&[], &mut out).unwrap(), 0);
        assert_eq!(out, [0xA5_u8; 4]);
    }

    #[test]
    fn forward_rejects_short_output() {
        let mut out = [0_u8; 5];
        let err = bwt(b"banana", &mut out).unwrap_err();
        assert!(matches!(err, Error::BwtFailed));
    }

    #[test]
    fn forward_rejects_empty_output_for_nonempty_input() {
        let mut out = [];
        let err = bwt(b"a", &mut out).unwrap_err();
        assert!(matches!(err, Error::BwtFailed));
    }

    /// Sort the suffixes of `text` by direct comparison.
    // Reason: test inputs stay far below 2^32 bytes.
    #[expect(clippy::cast_possible_truncation)]
    fn naive_suffix_array(text: &[u8]) -> Vec<u32> {
        let mut sa: Vec<u32> = (0..text.len() as u32).collect();
        sa.sort_by(|&a, &b| text[a as usize..].cmp(&text[b as usize..]));
        sa
    }

    /// Sort `text` with `scratch` and return the suffix array.
    fn sais_suffix_array(scratch: &mut BwtScratch, text: &[u8]) -> Vec<u32> {
        scratch.sort_suffixes(text).unwrap();
        scratch.sa[..text.len()].to_vec()
    }

    /// Fibonacci word of at least `len` bytes. Its many repeats make
    /// the SA-IS recursion deep.
    fn fibonacci_word(len: usize) -> Vec<u8> {
        let (mut a, mut b) = (b"a".to_vec(), b"ab".to_vec());
        while b.len() < len {
            let next = [b.as_slice(), a.as_slice()].concat();
            a = b;
            b = next;
        }
        b
    }

    /// Inputs with fixed shapes that stress the SA-IS edge cases.
    fn shaped_inputs() -> Vec<Vec<u8>> {
        vec![
            Vec::new(),
            b"a".to_vec(),
            b"ba".to_vec(),
            b"mississippi".to_vec(),
            b"banana".to_vec(),
            vec![b'a'; 1000],
            b"ab".repeat(500),
            b"abc".repeat(333),
            (0..=255_u8).collect(),
            (0..=255_u8).rev().collect(),
            (0..=255_u8).cycle().take(3000).collect(),
            fibonacci_word(4000),
        ]
    }

    #[test]
    fn suffix_array_matches_naive_sort() {
        let mut scratch = BwtScratch::default();
        for text in shaped_inputs() {
            let want = naive_suffix_array(&text);
            assert_eq!(
                sais_suffix_array(&mut scratch, &text),
                want,
                "len {}",
                text.len()
            );
        }
        let mut state = 0x1319_8A2E_0370_7344_u64;
        for round in 0..300 {
            let len = usize::try_from(next_mix(&mut state) % 600).unwrap();
            let alphabet = [2_u64, 4, 256][round % 3];
            let text: Vec<u8> = (0..len)
                .map(|_| u8::try_from(next_mix(&mut state) % alphabet).unwrap())
                .collect();
            let want = naive_suffix_array(&text);
            assert_eq!(
                sais_suffix_array(&mut scratch, &text),
                want,
                "round {round}"
            );
        }
    }

    #[test]
    fn suffix_array_matches_naive_sort_with_many_names() {
        // A 64-symbol alphabet gives many equal LMS substrings, so the
        // lower levels name and sort a long reduced text.
        let mut state = 0x510E_527F_ADE6_82D1_u64;
        let text: Vec<u8> = (0..20_000)
            .map(|_| u8::try_from(next_mix(&mut state) % 64).unwrap())
            .collect();
        let want = naive_suffix_array(&text);
        assert_eq!(sais_suffix_array(&mut BwtScratch::default(), &text), want);
    }

    #[test]
    fn lms_iteration_matches_position_test() {
        let mut texts = shaped_inputs();
        let mut state = 0x3C6E_F372_FE94_F82B_u64;
        for round in 0..300 {
            let len = usize::try_from(next_mix(&mut state) % 600).unwrap();
            let alphabet = [2_u64, 4, 256][round % 3];
            texts.push(
                (0..len)
                    .map(|_| u8::try_from(next_mix(&mut state) % alphabet).unwrap())
                    .collect(),
            );
        }
        let mut stype = Vec::new();
        for text in texts {
            classify_types_scalar(&text, &mut stype).unwrap();
            let want: Vec<usize> = (1..text.len()).filter(|&i| is_lms(&stype, i)).collect();
            let mut got = Vec::new();
            for_each_lms(&stype, |i| got.push(i));
            assert_eq!(got, want, "len {}", text.len());
        }
    }

    #[test]
    fn byte_classification_matches_scalar() {
        let mut texts = shaped_inputs();
        for len in [63, 64, 65, 127, 128, 129, 130, 192, 193] {
            texts.push(vec![b'a'; len]);
            texts.push((0..len).map(|i| u8::try_from(i % 3).unwrap()).collect());
        }
        let mut state = 0x6A09_E667_F3BC_C908_u64;
        for round in 0..300 {
            let len = usize::try_from(next_mix(&mut state) % 600).unwrap();
            let alphabet = [2_u64, 4, 256][round % 3];
            texts.push(
                (0..len)
                    .map(|_| u8::try_from(next_mix(&mut state) % alphabet).unwrap())
                    .collect(),
            );
        }
        let (mut want, mut got) = (Vec::new(), Vec::new());
        for text in texts {
            classify_types_scalar(&text, &mut want).unwrap();
            classify_bytes(&text, &mut got).unwrap();
            assert_eq!(got, want, "len {}", text.len());
        }
    }

    #[test]
    fn scratch_reuse_gives_same_result() {
        let large = fibonacci_word(5000);
        let small = b"mississippi".to_vec();
        let mut reused = BwtScratch::default();
        for text in [&large, &small, &large, &small] {
            let fresh = sais_suffix_array(&mut BwtScratch::default(), text);
            assert_eq!(
                sais_suffix_array(&mut reused, text),
                fresh,
                "len {}",
                text.len()
            );
        }
    }
}
