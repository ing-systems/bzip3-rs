//! Order-0 context-mixing entropy coder.
//!
//! Port of the arithmetic coder in the reference `libbz3.c`
//! (`begin`, `encode_bytes`, `decode_bytes`). Each call starts from
//! the canonical initial state, exactly as the reference block codec
//! calls `begin` before every block. A short input ends the decode
//! with [`Error::TruncatedData`]. It never panics and never hangs.

// Hot-loop helpers below use plain `#[inline]`: the bit loop runs
// them millions of times per block, so call overhead shows in
// throughput. Visibility stays `pub(crate)` for the block codec.
// Reason: the parent module is private, so the
// nursery lint asks for plain `pub`, but `pub(crate)` states the
// crate-wide intent at each item.
#![expect(clippy::redundant_pub_crate)]

use crate::error::Error;
use crate::error::Result;

/// Size of the local output buffer of the encoder.
const OUT_BUF_LEN: usize = 4096;

/// Most bytes that one input byte can add to the output.
///
/// One coded bit adds at most 4 bytes, so 8 bits add at most 32.
const BYTE_OUT_MAX: usize = 32;

/// Initial mixer row: step `k` starts at `k << 12`.
///
/// The last step starts one less, so it matches the reference
/// `(k << 12) - (k == 16)` narrowed into `u16`.
const MIXER_INIT: [u16; 17] = {
    let mut row = [0_u16; 17];
    let mut step = 0_usize;
    while step < 17 {
        // Reason: `step << 12` stays below 2^17, and the
        // reference narrows it to 16 bits the same way.
        #[expect(clippy::cast_possible_truncation)]
        let value = ((step << 12) as u32).wrapping_sub((step == 16) as u32) as u16;
        row[step] = value;
        step += 1;
    }
    row
};

/// Mixing state (`C0`, `C1`, `C2` tables of the reference).
#[derive(Clone)]
pub(crate) struct CmCoder {
    /// Zero-order counter per bit context.
    c0: [u16; 256],
    /// First-order counters per previous byte and bit context.
    c1: [[u16; 256]; 256],
    /// Second-stage mixer per bit context, run flag, and probability step.
    c2: [[u16; 17]; 512],
}

/// Read the coded byte at `pos`, or a filler byte past the end.
///
/// # Errors
///
/// Returns [`Error::TruncatedData`] when `pos` is past the filler.
#[inline]
fn next_byte(input: &[u8], pos: usize, filler: usize) -> Result<u8> {
    match input.get(pos) {
        Some(&byte) => Ok(byte),
        None => fill_byte(pos - input.len(), filler),
    }
}

/// Return a filler byte when `offset` is inside the filler.
#[cold]
const fn fill_byte(offset: usize, filler: usize) -> Result<u8> {
    if offset < filler {
        Ok(0xFF)
    } else {
        Err(Error::TruncatedData)
    }
}

/// Mutable per-call state of the bit loops.
///
/// Bundles the range endpoints with the previous bytes and the run
/// count, so the per-byte helpers take few arguments.
#[derive(Debug)]
struct BitCursor {
    /// Low end of the current range.
    low: u32,
    /// High end of the current range.
    high: u32,
    /// Last coded byte.
    prev_first: u8,
    /// Byte before the last coded byte.
    prev_second: u8,
    /// Length of the current equal-byte run.
    run: u32,
}

impl BitCursor {
    /// Cursor for the canonical initial range with no history.
    #[inline]
    const fn fresh() -> Self {
        Self {
            low: 0,
            high: 0xFFFF_FFFF,
            prev_first: 0,
            prev_second: 0,
            run: 0,
        }
    }

    /// Update the run count and return the mixer flag.
    ///
    /// The flag is set for a run above two, exactly as the per-byte
    /// loop did inline.
    #[inline]
    fn flag(&mut self) -> usize {
        if self.prev_first == self.prev_second {
            self.run += 1;
        } else {
            self.run = 0;
        }
        usize::from(self.run > 2)
    }

    /// Move past one coded byte.
    #[inline]
    const fn advance(&mut self, byte: u8) {
        self.prev_second = self.prev_first;
        self.prev_first = byte;
    }
}

impl CmCoder {
    /// Fresh state in canonical initial form.
    ///
    /// The tables stay inline: the hot loop touches them on every
    /// bit, and the transient stack use stays far below limits.
    /// The state is built once per codec state and reused through
    /// [`CmCoder::reset`], as the reference heap-allocates its state
    /// once when it creates the codec state.
    #[expect(clippy::large_stack_arrays)]
    pub(crate) fn new() -> Self {
        let mut state = Self {
            c0: [0; 256],
            c1: [[0; 256]; 256],
            c2: [[0; 17]; 512],
        };
        state.reset();
        state
    }

    /// Reset tables to canonical initial form.
    ///
    /// Matches the reference `begin`: zero-order and first-order
    /// counters start at `1 << 15`. Mixer steps start at `step << 12`,
    /// except the last step which starts one less.
    pub(crate) fn reset(&mut self) {
        self.c0.fill(1 << 15);
        for row in &mut self.c1 {
            row.fill(1 << 15);
        }
        self.c2.fill(MIXER_INIT);
    }

    /// Append the coded form of `input` to `output`.
    ///
    /// Starts from the canonical initial state, exactly as the
    /// reference block codec calls `begin` before every block.
    pub(crate) fn encode(&mut self, input: &[u8], output: &mut Vec<u8>) {
        self.reset();
        let mut cursor = BitCursor::fresh();
        // Local buffer: a `Vec::push` in the bit loop adds a call
        // path that forces the loop state onto the stack.
        let mut out_buf = [0_u8; OUT_BUF_LEN];
        let mut used = 0_usize;

        for &byte in input {
            flush_full(output, &out_buf, &mut used);
            let flag = cursor.flag();
            self.encode_byte(byte, flag, &mut cursor, &mut out_buf, &mut used);
            cursor.advance(byte);
        }

        output.extend_from_slice(&out_buf[..used]);
        output.extend_from_slice(&cursor.low.to_be_bytes());
    }

    /// Decode exactly `output.len()` bytes from `input`.
    ///
    /// Extra trailing bytes are ignored. After the end of `input`, the
    /// decoder reads up to `filler` bytes of `0xFF`, as the reference
    /// reads `-1` past a short stream. Past the filler, the call fails
    /// with [`Error::TruncatedData`].
    pub(crate) fn decode(&mut self, input: &[u8], filler: usize, output: &mut [u8]) -> Result<()> {
        if output.is_empty() {
            return Ok(());
        }
        self.reset();
        // Single bounds-checked read per coded byte. `slice.get`
        // checks once; the filler path stays out of the hot loop.
        let mut pos = 0_usize;
        let mut cursor = BitCursor::fresh();
        let mut code = 0_u32;
        for _ in 0..4 {
            let byte = next_byte(input, pos, filler)?;
            pos += 1;
            code = (code << 8) | u32::from(byte);
        }

        for slot in output.iter_mut() {
            let flag = cursor.flag();
            let byte = self.decode_byte(input, filler, flag, &mut cursor, &mut code, &mut pos)?;
            cursor.advance(byte);
            *slot = byte;
        }
        Ok(())
    }

    /// Mixed split offset and mixer step for one bit context.
    ///
    /// Combines the zero-order, first-order, and mixer counters
    /// exactly as the reference does.
    #[inline]
    fn predict(&self, context: usize, first: u8, second: u8, flag: usize) -> (u32, usize) {
        let direct = u32::from(self.c0[context]) + u32::from(self.c1[usize::from(first)][context]);
        let mixed = direct * 7 + u32::from(self.c1[usize::from(second)][context]) * 2;
        let estimate = mixed >> 4;
        let step = (estimate >> 12) as usize;
        // All values fit in 32 bits: edges hold 0..65535, the
        // difference spans +-65535, and the fraction spans 0..4095.
        // The product stays below 2^29, so `i32` math matches the
        // reference exactly with cheaper 32-bit ops.
        let row = &self.c2[2 * context + flag];
        let low_edge = i32::from(row[step]);
        let high_edge = i32::from(row[step + 1]);
        let frac = {
            // Reason: the estimate stays below 2^16,
            // so the masked fraction always fits in `i32`.
            #[expect(clippy::cast_possible_wrap)]
            let frac = (estimate & 4095) as i32;
            frac
        };
        let point = low_edge + (((high_edge - low_edge) * frac) >> 12);
        // Reason: the interpolated point stays in
        // 0..65536, so the `u32` view matches the reference exactly.
        #[expect(clippy::cast_sign_loss)]
        let mix = (point as u32) * 3 + estimate;
        // The mix stays below 2^18 by the counter bounds above.
        (mix, step)
    }

    /// Code one byte through its eight bit contexts.
    ///
    /// Runs the per-bit predict, range select, shared-byte flush, and
    /// adapt steps. The caller leaves four spare bytes after `used`.
    #[inline]
    fn encode_byte(
        &mut self,
        byte: u8,
        flag: usize,
        cursor: &mut BitCursor,
        out_buf: &mut [u8; OUT_BUF_LEN],
        used: &mut usize,
    ) {
        let mut context = 1_usize;
        let mut bits = byte;
        while context < 256 {
            let (split, step) =
                self.predict(context, cursor.prev_first, cursor.prev_second, flag);
            let mid = cursor.low.wrapping_add(scaled_width(cursor.low, cursor.high, split));
            let bit = bits & 128 != 0;
            // Branchless range select. Random input makes the
            // bit unpredictable, so a branch mispredicts here.
            let keep = u32::from(bit).wrapping_neg();
            cursor.low = (cursor.low & keep) | (mid.wrapping_add(1) & !keep);
            cursor.high = (mid & keep) | (cursor.high & !keep);
            flush_shared_bytes(&mut cursor.low, &mut cursor.high, out_buf, used);
            self.adapt(context, cursor.prev_first, flag, step, bit);
            context = 2 * context + usize::from(bit);
            bits <<= 1;
        }
    }

    /// Decode one byte through its eight bit contexts.
    ///
    /// Predicts each bit, resolves it against `code`, renormalizes the
    /// range, and adapts the tables. Returns the decoded byte.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TruncatedData`] when `input` and filler end.
    #[inline]
    fn decode_byte(
        &mut self,
        input: &[u8],
        filler: usize,
        flag: usize,
        cursor: &mut BitCursor,
        code: &mut u32,
        pos: &mut usize,
    ) -> Result<u8> {
        let mut context = 1_usize;
        let mut current = self.predict(1, cursor.prev_first, cursor.prev_second, flag);
        while context < 256 {
            let (split, step) = current;
            // Both child contexts use counters that the current
            // bit does not change. Predict them now, so the
            // loads and the mix run while the bit resolves.
            let (zero, one) = if context < 128 {
                (
                    self.predict(2 * context, cursor.prev_first, cursor.prev_second, flag),
                    self.predict(2 * context + 1, cursor.prev_first, cursor.prev_second, flag),
                )
            } else {
                ((0, 0), (0, 0))
            };
            let mid = cursor.low.wrapping_add(scaled_width(cursor.low, cursor.high, split));
            let bit = *code <= mid;
            // Branchless range select. Random input makes the
            // bit unpredictable, so a branch mispredicts here.
            let keep = u32::from(bit).wrapping_neg();
            let over = mid.wrapping_add(1);
            cursor.low = (cursor.low & keep) | (over & !keep);
            cursor.high = (mid & keep) | (cursor.high & !keep);
            renormalize_after_bit(input, filler, cursor, code, pos)?;
            self.adapt(context, cursor.prev_first, flag, step, bit);
            context = 2 * context + usize::from(bit);
            let pick = usize::from(bit).wrapping_neg();
            current = (
                (one.0 & keep) | (zero.0 & !keep),
                (one.1 & pick) | (zero.1 & !pick),
            );
        }

        // Reason: the mask keeps 0..255, so the
        // byte view matches the reference exactly.
        #[expect(clippy::cast_possible_truncation)]
        let byte = (context & 255) as u8;
        Ok(byte)
    }

    /// Count one coded bit in the three counter tables.
    ///
    /// Branchless select: random input makes the bit unpredictable,
    /// so a data-dependent branch mispredicts here. Both directions
    /// compute, then the mask picks one. Values match the
    /// branch form bit for bit.
    #[inline]
    fn adapt(&mut self, context: usize, first: u8, flag: usize, step: usize, bit: bool) {
        let mask = u16::from(bit).wrapping_neg();
        Self::adapt_counter(&mut self.c0[context], 2, mask);
        Self::adapt_counter(&mut self.c1[usize::from(first)][context], 4, mask);
        // Borrow one row once, then adapt both mixer entries.
        let row = &mut self.c2[2 * context + flag];
        let (head, tail) = row.split_at_mut(step + 1);
        Self::adapt_counter(&mut head[step], 6, mask);
        Self::adapt_counter(&mut tail[0], 6, mask);
    }

    /// Move one counter toward one or zero as `mask` selects.
    ///
    /// `mask` holds all ones for a one bit else all zeros. Both
    /// directions compute, then the mask picks one. Values match
    /// the reference `update1`/`update0` bit for bit.
    #[inline]
    const fn adapt_counter(slot: &mut u16, rate: u32, mask: u16) {
        let value = *slot;
        let raised = value.wrapping_add((value ^ 0xFFFF) >> rate);
        let lowered = value.wrapping_sub(value >> rate);
        *slot = (raised & mask) | (lowered & !mask);
    }
}

/// Width of the taken range part for one split mix.
///
/// Matches `(u64)(high - low) * mix >> 18` in the reference.
#[inline]
const fn scaled_width(low: u32, high: u32, mix: u32) -> u32 {
    let spread = high.wrapping_sub(low) as u64;
    // The spread fits in `u32` and the mix stays below 2^18, so the
    // shifted product always fits in `u32`. The multiply wraps exactly
    // as the reference `u64` multiply does.
    #[expect(clippy::cast_possible_truncation)]
    let width = (spread.wrapping_mul(mix as u64) >> 18) as u32;
    width
}

/// Count of leading bytes that the range endpoints already share.
#[inline]
const fn shared_bytes(low: u32, high: u32) -> u32 {
    (low ^ high).leading_zeros() / 8
}

/// Shift both range endpoints left by `shift` bits.
///
/// New low bits of `high` become ones, as in the reference. A shift
/// of 32 bits clears `low` and sets all bits of `high`.
#[inline]
const fn shift_range(low: u32, high: u32, shift: u32) -> (u32, u32) {
    let wide_low = (low as u64) << shift;
    let wide_high = ((high as u64) << shift) | ((1_u64 << shift) - 1);
    // Reason: the reference keeps only the low 32 bits.
    #[expect(clippy::cast_possible_truncation)]
    (wide_low as u32, wide_high as u32)
}

/// Move the buffered bytes to `output` when fewer than one coded
/// byte fits in the buffer.
///
/// One coded bit adds at most 4 bytes, so one byte adds at most
/// [`BYTE_OUT_MAX`]. The caller runs this before each input byte.
#[inline]
fn flush_full(output: &mut Vec<u8>, out_buf: &[u8; OUT_BUF_LEN], used: &mut usize) {
    if *used > OUT_BUF_LEN - BYTE_OUT_MAX {
        output.extend_from_slice(&out_buf[..*used]);
        *used = 0;
    }
}
///
/// Emit the leading bytes the range endpoints already agree on.
///
/// Writes the four bytes of `low` and keeps only the shared ones.
/// The caller leaves four spare bytes after `used`.
#[inline]
fn flush_shared_bytes(
    low: &mut u32,
    high: &mut u32,
    out_buf: &mut [u8; OUT_BUF_LEN],
    used: &mut usize,
) {
    // The guard stays a branch: skewed input rarely passes it.
    if (*low ^ *high) < (1 << 24) {
        let count = shared_bytes(*low, *high);
        out_buf[*used..*used + 4].copy_from_slice(&low.to_be_bytes());
        *used += count as usize;
        (*low, *high) = shift_range(*low, *high, 8 * count);
    }
}

/// Shift the shared range bytes after one decode bit.
///
/// Fast path reads four bytes at once and keeps the shared count.
/// Falls back to the tail reader at the end of `input` and in the
/// filler.
///
/// # Errors
///
/// Returns [`Error::TruncatedData`] when `input` and filler end.
#[inline]
fn renormalize_after_bit(
    input: &[u8],
    filler: usize,
    cursor: &mut BitCursor,
    code: &mut u32,
    pos: &mut usize,
) -> Result<()> {
    // The guard stays a branch: skewed input rarely passes it.
    if (cursor.low ^ cursor.high) < (1 << 24) {
        if let Some(window) = input.get(*pos..*pos + 4) {
            // Fast path: read four bytes, keep the shared count.
            let count = shared_bytes(cursor.low, cursor.high);
            let shift = 8 * count;
            let next = u32::from_be_bytes([window[0], window[1], window[2], window[3]]);
            (cursor.low, cursor.high) = shift_range(cursor.low, cursor.high, shift);
            // Reason: the reference keeps only the low 32 bits.
            #[expect(clippy::cast_possible_truncation)]
            let merged = (u64::from(*code) << shift | u64::from(next) >> (32 - shift)) as u32;
            *code = merged;
            *pos += count as usize;
        } else {
            (cursor.low, cursor.high, *code, *pos) = renormalize_tail(
                cursor.low,
                cursor.high,
                *code,
                *pos,
                input,
                filler,
            )?;
        }
    }
    Ok(())
}

/// Read coded bytes until the range endpoints differ in the top byte.
///
/// Slow path for the last bytes of `input` and the filler.
/// Returns the new `(low, high, code, pos)`.
///
/// # Errors
///
/// Returns [`Error::TruncatedData`] when `input` and filler end.
#[cold]
fn renormalize_tail(
    mut low: u32,
    mut high: u32,
    mut code: u32,
    mut pos: usize,
    input: &[u8],
    filler: usize,
) -> Result<(u32, u32, u32, usize)> {
    while (low ^ high) < (1 << 24) {
        low <<= 8;
        high = (high << 8) + 255;
        code = (code << 8) | u32::from(next_byte(input, pos, filler)?);
        pos += 1;
    }
    Ok((low, high, code, pos))
}

#[cfg(test)]
mod tests {
    use super::CmCoder;
    use crate::error::Error;

    /// One oracle pair: name, raw input bytes, and reference coded bytes.
    macro_rules! vector {
        ($name:literal) => {
            (
                $name,
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/vectors/cm/",
                    $name,
                    ".bin"
                )) as &[u8],
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/vectors/cm/",
                    $name,
                    ".cm"
                )) as &[u8],
            )
        };
    }

    /// All oracle vectors: `(name, raw, coded)` triples.
    const VECTORS: &[(&str, &[u8], &[u8])] = &[
        vector!("empty"),
        vector!("zero1"),
        vector!("ff1"),
        vector!("hello"),
        vector!("inc16"),
        vector!("all256"),
        vector!("zeros64"),
        vector!("zeros65"),
        vector!("zeros300"),
        vector!("rand63"),
        vector!("rand64"),
        vector!("rand65"),
        vector!("rand1k"),
        vector!("text2k"),
        vector!("runs"),
        vector!("abalt"),
    ];

    fn encode_all(input: &[u8]) -> Vec<u8> {
        let mut coder = CmCoder::new();
        let mut out = Vec::new();
        coder.encode(input, &mut out);
        out
    }

    fn decode_all(coded: &[u8], len: usize) -> crate::error::Result<Vec<u8>> {
        let mut state = CmCoder::new();
        let mut out = vec![0; len];
        state.decode(coded, 0, &mut out).map(|()| out)
    }

    /// Deterministic filler bytes from a fixed seed.
    fn pseudo_random(seed: u32, len: usize) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                u8::try_from((state >> 16) & 0xFF).expect("masked byte fits")
            })
            .collect()
    }

    #[test]
    fn encode_matches_oracle_byte_for_byte() {
        for (name, raw, coded) in VECTORS {
            assert_eq!(encode_all(raw), coded.to_vec(), "encode differs for {name}");
        }
    }

    #[test]
    fn decode_restores_oracle_originals() {
        for (name, raw, coded) in VECTORS {
            let got = decode_all(coded, raw.len())
                .unwrap_or_else(|_| panic!("oracle stream decodes for {name}"));
            assert_eq!(got, raw.to_vec(), "decode differs for {name}");
        }
    }

    #[test]
    fn roundtrip_on_synthetic_inputs() {
        let mut inputs: Vec<Vec<u8>> = vec![
            vec![],
            vec![0],
            vec![255],
            vec![0, 255],
            vec![0; 1000],
            vec![255; 1000],
            vec![b'A'; 300],
            b"AB".repeat(200),
            (0..=255u8).cycle().take(1024).collect(),
        ];
        for len in [1usize, 2, 3, 7, 8, 63, 64, 65, 100, 1000] {
            inputs.push(pseudo_random(0x00C0_FFEE, len));
        }
        for raw in &inputs {
            let coded = encode_all(raw);
            let got = decode_all(&coded, raw.len()).expect("roundtrip decodes");
            assert_eq!(&got, raw, "roundtrip differs for len {}", raw.len());
        }
    }

    #[test]
    fn every_truncation_fails_with_truncated_data() {
        for (name, raw, coded) in VECTORS {
            // Empty output decodes vacuously; see
            // `empty_output_needs_no_input`.
            if raw.is_empty() {
                continue;
            }
            for end in 0..coded.len() {
                let mut state = CmCoder::new();
                let mut out = vec![0; raw.len()];
                let err = state
                    .decode(&coded[..end], 0, &mut out)
                    .expect_err(&format!("short stream of vector {name} must fail"));
                assert!(
                    matches!(err, Error::TruncatedData),
                    "wrong error for short stream of {name}"
                );
            }
        }
    }

    #[test]
    fn empty_output_needs_no_input() {
        let mut state = CmCoder::new();
        state.decode(&[], 0, &mut []).expect("empty decode is free");
        let (_, _, coded) = VECTORS[4];
        state
            .decode(&coded[..3.min(coded.len())], 0, &mut [])
            .expect("empty output ignores input");
    }

    #[test]
    fn trailing_bytes_are_ignored() {
        for (name, raw, coded) in VECTORS {
            let mut padded = coded.to_vec();
            padded.extend_from_slice(&[0xAA, 0x55, 0x00, 0xFF]);
            let got = decode_all(&padded, raw.len())
                .unwrap_or_else(|_| panic!("padded stream decodes for {name}"));
            assert_eq!(got, raw.to_vec(), "padded decode differs for {name}");
        }
    }

    #[test]
    fn state_resets_between_calls() {
        let first = encode_all(b"hello world, hello world");
        let mut coder = CmCoder::new();
        let mut again = Vec::new();
        coder.encode(b"something else entirely", &mut again);
        let mut second = Vec::new();
        coder.encode(b"hello world, hello world", &mut second);
        assert_eq!(second, first, "coder state leaked between calls");
        let got = decode_all(&second, 24).expect("decodes");
        assert_eq!(got, b"hello world, hello world");
    }
}
