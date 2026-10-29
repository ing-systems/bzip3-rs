//! Run-length stage of the bzip3 port.
//!
//! The forward transform collapses runs with a net gain and stores
//! the choice per byte value in a 32-byte bitmap. The inverse
//! transform restores the original bytes. Both match the reference
//! `mrlec`/`mrled` byte for byte. All decode failures map to
//! [`Error::CorruptStageData`], which has the reference checksum code.

use crate::error::Error;
use crate::error::Result;

/// Length of the bitmap that opens the output.
const BITMAP_LEN: usize = 32;

/// Bytes in one block.
const STEP: usize = 32;

/// Bytes that the member search checks one by one before it uses blocks.
const HEAD: usize = 8;

/// Bytes in one block of the table search.
const TABLE_STEP: usize = 16;

/// Largest set that the compare search handles.
const FEW: usize = 4;

/// Run the forward run-length transform into a fixed buffer.
///
/// Bytes with a net run gain collapse into a value plus count bytes.
/// A 32-byte bitmap records which values collapse. Output matches the
/// reference `mrlec` byte for byte. Return the output length when it
/// is shorter than `input`, else `None`. The write stops early in the
/// second case, so `out` needs only `input.len()` bytes.
pub fn encode_into(input: &[u8], out: &mut [u8]) -> Option<usize> {
    let gain = count_gain(input);
    // With no net gain, every byte stays literal. The output is then
    // `32 + input.len()` bytes, which is never shorter than `input`.
    if gain.iter().all(|&g| g <= 0) {
        return None;
    }
    let limit = input.len().min(out.len());
    encode_runs(input, &gain, out, limit)
}

/// Run the forward run-length transform into a new vector.
///
/// Same output as [`encode_into`] without the length limit.
///
/// # Errors
///
/// Returns [`Error::OutOfMemory`] when the output buffer cannot grow.
#[cfg(test)]
pub fn encode(input: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let capacity = input.len().saturating_mul(2).saturating_add(BITMAP_LEN);
    out.try_reserve_exact(capacity)
        .map_err(|_| Error::OutOfMemory)?;
    out.resize(capacity, 0);
    let gain = count_gain(input);
    let len = encode_runs(input, &gain, &mut out, capacity + 1).ok_or(Error::OutOfMemory)?;
    out.truncate(len);
    Ok(out)
}

/// Count the run of equal bytes that opens `data`.
///
/// `data` must not be empty. The loop reduces each block with no early
/// exit, so the compiler turns it into vector compares.
#[inline]
fn run_len(data: &[u8]) -> usize {
    let first = data[0];
    let (blocks, tail) = data.as_chunks::<STEP>();
    for (index, block) in blocks.iter().enumerate() {
        if block.iter().fold(0_u8, |acc, &byte| acc | (byte ^ first)) != 0 {
            let at = block.iter().position(|&byte| byte != first).unwrap_or(0);
            return index * STEP + at;
        }
    }
    blocks.len() * STEP + tail.iter().take_while(|&&byte| byte == first).count()
}

/// Count the net gain of each byte value.
///
/// A value collapses when its gain is above zero. A run of `n` bytes
/// costs one, then gains one for each repeat except every 255th.
fn count_gain(input: &[u8]) -> [i64; 256] {
    let mut gain = [0_i64; 256];
    let mut start = 0;
    while start < input.len() {
        let value = input[start];
        // A run of one byte is the common case. It costs one.
        if input.get(start + 1) != Some(&value) {
            gain[usize::from(value)] -= 1;
            start += 1;
            continue;
        }
        let len = run_len(&input[start..]);
        let repeats = len - 1;
        let net = i64::try_from(repeats - repeats / 255).unwrap_or(i64::MAX) - 1;
        gain[usize::from(value)] += net;
        start += len;
    }
    gain
}

/// Core of the forward transform. Write the output into `out`.
///
/// Return the output length, or `None` when the output would reach
/// `limit` bytes.
fn encode_runs(input: &[u8], gain: &[i64; 256], out: &mut [u8], limit: usize) -> Option<usize> {
    if limit <= BITMAP_LEN {
        return None;
    }
    write_bitmap(out, gain);
    let collapse = ByteSet::new(gain.map(|g| g > 0));
    let mut pos = BITMAP_LEN;
    let mut start = 0;
    while start < input.len() {
        // Copy the span of bytes that never collapse in one step.
        let rest = &input[start..];
        let span = collapse.first_member(rest);
        if pos + span >= limit {
            return None;
        }
        out[pos..pos + span].copy_from_slice(&rest[..span]);
        pos += span;
        start += span;
        if start == input.len() {
            break;
        }
        // The value, then `repeats / 255` full count bytes, then the rest.
        let len = run_len(&input[start..]);
        let repeats = len - 1;
        if pos + repeats / 255 + 2 >= limit {
            return None;
        }
        pos = emit_run(out, pos, input[start], repeats);
        start += len;
    }
    Some(pos)
}

/// Write the 32-byte collapse bitmap of `gain` at the start of `out`.
///
/// `out` holds at least [`BITMAP_LEN`] bytes: the caller checks the
/// limit before this runs.
#[inline]
fn write_bitmap(out: &mut [u8], gain: &[i64; 256]) {
    for (slot, chunk) in out.iter_mut().zip(gain.as_chunks::<8>().0) {
        *slot = chunk
            .iter()
            .enumerate()
            .fold(0_u8, |acc, (bit, &g)| acc | (u8::from(g > 0) << bit));
    }
}

/// Append one collapsed run of `value` with `repeats` extra bytes.
///
/// Writes the value, then `repeats / 255` full count bytes, then the
/// rest. Returns the position after the run. The caller checks room.
#[inline]
fn emit_run(out: &mut [u8], pos: usize, value: u8, repeats: usize) -> usize {
    let full = repeats / 255;
    out[pos] = value;
    out[pos + 1..pos + 1 + full].fill(255);
    out[pos + 1 + full] = u8::try_from(repeats % 255).unwrap_or(u8::MAX);
    pos + full + 2
}

/// Run the inverse run-length transform into a new vector.
///
/// Same output as [`decode_into`].
///
/// # Errors
///
/// Returns [`Error::CorruptStageData`] under the same conditions as [`decode_into`].
/// Returns [`Error::OutOfMemory`] when the output buffer cannot grow.
#[cfg(test)]
pub fn decode(input: &[u8], out_len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.try_reserve_exact(out_len)
        .map_err(|_| Error::OutOfMemory)?;
    out.resize(out_len, 0);
    decode_into(input, &mut out)?;
    Ok(out)
}

/// Run the inverse run-length transform into a fixed buffer.
///
/// `input` holds the forward form (32-byte bitmap plus payload).
/// `out` takes the decoded bytes and its length is the expected
/// decoded length. Output matches the reference `mrled`. Overlong
/// runs stop at the end of `out`, as the reference does.
///
/// # Errors
///
/// Returns [`Error::CorruptStageData`] when the payload is short, corrupt, or
/// decodes to a length other than `out.len()`.
pub fn decode_into(input: &[u8], out: &mut [u8]) -> Result<()> {
    if input.len() < 32 {
        return Err(Error::CorruptStageData);
    }
    let out_len = out.len();
    let collapse = ByteSet::new(parse_bitmap(input));
    let mut op: usize = 0;
    let mut ip: usize = 32;
    while op < out_len && ip < input.len() {
        // Copy the span of literal bytes in one step.
        let limit = (out_len - op).min(input.len() - ip);
        let span = collapse.first_member(&input[ip..ip + limit]);
        out[op..op + span].copy_from_slice(&input[ip..ip + span]);
        op += span;
        ip += span;
        if span == limit {
            continue;
        }
        let byte = input[ip];
        ip += 1;
        let run = read_run_count(input, &mut ip, out_len)?;
        // The reference caps the fill at `out_len` and drops the rest.
        let fill = run.min(out_len - op);
        out[op..op + fill].fill(byte);
        op += fill;
    }
    if op != out_len {
        return Err(Error::CorruptStageData);
    }
    Ok(())
}

/// Read the 32-byte collapse bitmap at the start of `input`.
///
/// `input` holds at least 32 bytes: the caller checks this first.
#[inline]
fn parse_bitmap(input: &[u8]) -> [bool; 256] {
    let mut gain = [false; 256];
    for (chunk, &byte) in input.iter().take(32).enumerate() {
        for bit in 0..8_usize {
            gain[chunk * 8 + bit] = (byte >> bit) & 1 == 1;
        }
    }
    gain
}

/// Read the count bytes of one collapsed run.
///
/// Advances `ip` past the terminator. The total stops growing past
/// `out_len`, but the read still consumes every count byte, as the
/// reference does.
///
/// # Errors
///
/// Returns [`Error::CorruptStageData`] when the payload ends inside
/// the run.
#[inline]
fn read_run_count(input: &[u8], ip: &mut usize, out_len: usize) -> Result<usize> {
    let mut run: usize = 1;
    loop {
        let Some(&count) = input.get(*ip) else {
            return Err(Error::CorruptStageData);
        };
        *ip += 1;
        // The reference stops adding past `out_len`
        // but still reads count bytes to the end.
        if run <= out_len {
            run += usize::from(count);
        }
        if count != 255 {
            break;
        }
    }
    Ok(run)
}

/// Set of byte values with a fast search for the first member.
///
/// The searches reduce each block with no early exit, so the compiler
/// turns them into vector compares.
struct ByteSet {
    /// Membership of each byte value.
    member: [bool; 256],
    /// The members when the set has at most [`FEW`], padded with the first.
    few: [u8; FEW],
    /// Number of members.
    count: usize,
}

impl ByteSet {
    /// Build the set from the membership of each byte value.
    fn new(member: [bool; 256]) -> Self {
        let mut few = [0_u8; FEW];
        let mut count = 0;
        for value in (0..=u8::MAX).filter(|&value| member[usize::from(value)]) {
            if count < FEW {
                few[count] = value;
            }
            count += 1;
        }
        if count > 0 {
            let first = few[0];
            few[count.min(FEW)..].fill(first);
        }
        Self { member, few, count }
    }

    /// Return the index of the first member of `data`, or `data.len()`.
    fn first_member(&self, data: &[u8]) -> usize {
        if self.count == 0 {
            return data.len();
        }
        // Spans are often short, and the block setup costs more than it saves.
        let head = data.len().min(HEAD);
        if let Some(found) = self.position(&data[..head]) {
            return found;
        }
        let rest = &data[head..];
        let found = if self.count <= FEW {
            self.search_few(rest)
        } else {
            self.search_table(rest)
        };
        head + found
    }

    /// Return the index of the first member of `data` one byte at a time.
    #[inline]
    fn position(&self, data: &[u8]) -> Option<usize> {
        data.iter().position(|&byte| self.member[usize::from(byte)])
    }

    /// Return the first member of `data`, or its length, for a set of at most [`FEW`].
    fn search_few(&self, data: &[u8]) -> usize {
        let [a, b, c, d] = self.few;
        let (blocks, tail) = data.as_chunks::<STEP>();
        for (index, block) in blocks.iter().enumerate() {
            let hit = block.iter().fold(0_u8, |acc, &byte| {
                acc | u8::from(byte == a)
                    | u8::from(byte == b)
                    | u8::from(byte == c)
                    | u8::from(byte == d)
            });
            if hit != 0 {
                return index * STEP + self.position(block).unwrap_or(0);
            }
        }
        blocks.len() * STEP + self.position(tail).unwrap_or(tail.len())
    }

    /// Return the first member of `data`, or its length, with one lookup for each byte.
    fn search_table(&self, data: &[u8]) -> usize {
        let (blocks, tail) = data.as_chunks::<TABLE_STEP>();
        for (index, block) in blocks.iter().enumerate() {
            let hit = block
                .iter()
                .fold(false, |acc, &byte| acc | self.member[usize::from(byte)]);
            if hit {
                return index * TABLE_STEP + self.position(block).unwrap_or(0);
            }
        }
        blocks.len() * TABLE_STEP + self.position(tail).unwrap_or(tail.len())
    }
}

#[cfg(test)]
mod tests {
    use super::ByteSet;
    use super::decode;
    use super::decode_into;
    use super::encode;
    use super::encode_into;
    use crate::error::Error;

    const EMPTY_INPUT: &[u8] = include_bytes!("../tests/vectors/rle_empty_input.bin");
    const EMPTY_ENCODED: &[u8] = include_bytes!("../tests/vectors/rle_empty_encoded.bin");
    const MIXED_INPUT: &[u8] = include_bytes!("../tests/vectors/rle_mixed_input.bin");
    const MIXED_ENCODED: &[u8] = include_bytes!("../tests/vectors/rle_mixed_encoded.bin");
    const MAXRUN_INPUT: &[u8] = include_bytes!("../tests/vectors/rle_maxrun_input.bin");
    const MAXRUN_ENCODED: &[u8] = include_bytes!("../tests/vectors/rle_maxrun_encoded.bin");
    const BINARY_INPUT: &[u8] = include_bytes!("../tests/vectors/rle_binary_input.bin");
    const BINARY_ENCODED: &[u8] = include_bytes!("../tests/vectors/rle_binary_encoded.bin");
    const NOGAIN_INPUT: &[u8] = include_bytes!("../tests/vectors/rle_nogain_input.bin");
    const NOGAIN_ENCODED: &[u8] = include_bytes!("../tests/vectors/rle_nogain_encoded.bin");

    #[test]
    fn fixture_encode_matches_oracle() {
        let pairs: &[(&[u8], &[u8])] = &[
            (EMPTY_INPUT, EMPTY_ENCODED),
            (MIXED_INPUT, MIXED_ENCODED),
            (MAXRUN_INPUT, MAXRUN_ENCODED),
            (BINARY_INPUT, BINARY_ENCODED),
            (NOGAIN_INPUT, NOGAIN_ENCODED),
        ];
        for (input, want) in pairs {
            assert_eq!(&encode(input).unwrap(), want, "len {}", input.len());
        }
    }

    #[test]
    fn fixture_decode_restores_oracle_input() {
        let pairs: &[(&[u8], &[u8])] = &[
            (EMPTY_ENCODED, EMPTY_INPUT),
            (MIXED_ENCODED, MIXED_INPUT),
            (MAXRUN_ENCODED, MAXRUN_INPUT),
            (BINARY_ENCODED, BINARY_INPUT),
            (NOGAIN_ENCODED, NOGAIN_INPUT),
        ];
        for (encoded, want) in pairs {
            assert_eq!(&decode(encoded, want.len()).unwrap(), want);
        }
    }

    #[test]
    fn empty_round_trip() {
        assert_eq!(encode(&[]).unwrap(), &[0_u8; 32]);
        assert_eq!(decode(&[0_u8; 32], 0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn zero_gain_run_of_two_stays_literal() {
        // One pair gives a gain score of exactly zero, so no bit is set.
        let encoded = encode(b"BB").unwrap();
        let mut want = vec![0_u8; 32];
        want.extend_from_slice(b"BB");
        assert_eq!(encoded, want);
        assert_eq!(decode(&encoded, 2).unwrap(), b"BB");
    }

    #[test]
    fn positive_gain_run_of_three_collapses() {
        // One triple gives a gain score of one, so the bit is set.
        let encoded = encode(b"CCC").unwrap();
        let mut want = vec![0_u8; 32];
        want[8] = 0x08;
        want.extend_from_slice(&[b'C', 2]);
        assert_eq!(encoded, want);
        assert_eq!(decode(&encoded, 3).unwrap(), b"CCC");
    }

    #[test]
    fn run_of_256_uses_two_count_bytes() {
        let input = vec![0x51_u8; 256];
        let encoded = encode(&input).unwrap();
        let mut want = vec![0_u8; 32];
        want[10] = 0x02;
        want.extend_from_slice(&[0x51, 255, 0]);
        assert_eq!(encoded, want);
        assert_eq!(decode(&encoded, 256).unwrap(), input);
    }

    #[test]
    fn maximal_runs_restore() {
        for len in [255, 256, 257, 511, 512, 513, 65_535, 65_536] {
            let input = vec![0x5A_u8; len];
            let encoded = encode(&input).unwrap();
            assert_eq!(decode(&encoded, len).unwrap(), input, "len {len}");
        }
    }

    #[test]
    fn single_bytes_round_trip() {
        assert_eq!(decode(&encode(&[0xAB]).unwrap(), 1).unwrap(), &[0xAB]);
        let input: Vec<u8> = (0..=255_u8).collect();
        let encoded = encode(&input).unwrap();
        assert_eq!(&encoded[..32], &[0_u8; 32]);
        assert_eq!(decode(&encoded, 256).unwrap(), input);
    }

    #[test]
    fn encode_into_matches_encode_when_it_shrinks() {
        let mut out = vec![0_u8; MIXED_INPUT.len()];
        let len = encode_into(MIXED_INPUT, &mut out);
        let full = encode(MIXED_INPUT).unwrap();
        if full.len() < MIXED_INPUT.len() {
            assert_eq!(&out[..len.unwrap()], &full[..]);
        } else {
            assert_eq!(len, None);
        }
        let input = vec![0x51_u8; 1000];
        let mut out = vec![0_u8; input.len()];
        let len = encode_into(&input, &mut out).unwrap();
        assert_eq!(&out[..len], &encode(&input).unwrap()[..]);
    }

    #[test]
    fn encode_into_rejects_input_without_gain() {
        let mut out = vec![0_u8; 512];
        let no_runs: Vec<u8> = (0..=255_u8).collect();
        assert_eq!(encode_into(&no_runs, &mut out), None);
        assert_eq!(encode_into(b"BB", &mut out), None);
        assert_eq!(encode_into(&[], &mut out), None);
        for input in [NOGAIN_INPUT, BINARY_INPUT] {
            let mut out = vec![0_u8; input.len()];
            let want = encode(input).unwrap();
            let got = encode_into(input, &mut out);
            assert_eq!(got.is_some(), want.len() < input.len());
        }
    }

    #[test]
    fn encode_is_deterministic() {
        assert_eq!(encode(MIXED_INPUT).unwrap(), encode(MIXED_INPUT).unwrap());
    }

    #[test]
    fn truncated_input_rejected() {
        // Header shorter than the 32-byte bitmap.
        assert!(matches!(
            decode(&MIXED_ENCODED[..31], MIXED_INPUT.len()),
            Err(Error::CorruptStageData)
        ));
        // Payload cut by one byte.
        assert!(matches!(
            decode(&MIXED_ENCODED[..MIXED_ENCODED.len() - 1], MIXED_INPUT.len()),
            Err(Error::CorruptStageData)
        ));
        // Maximal run cut mid-count.
        assert!(matches!(
            decode(&MAXRUN_ENCODED[..40], MAXRUN_INPUT.len()),
            Err(Error::CorruptStageData)
        ));
        // Empty payload cannot yield one byte.
        assert!(matches!(decode(&[], 1), Err(Error::CorruptStageData)));
        assert!(matches!(
            decode(EMPTY_ENCODED, 1),
            Err(Error::CorruptStageData)
        ));
    }

    #[test]
    fn corrupt_payloads_map_to_crc() {
        // Shorter run count underfills the output.
        let mut short = MAXRUN_ENCODED.to_vec();
        let last = short.len() - 1;
        short[last] -= 1;
        assert!(matches!(
            decode(&short, MAXRUN_INPUT.len()),
            Err(Error::CorruptStageData)
        ));
        // Flipped bitmap bit changes the fill.
        let mut bitmap = MIXED_ENCODED.to_vec();
        bitmap[0] ^= 0xFF;
        assert!(matches!(
            decode(&bitmap, MIXED_INPUT.len()),
            Err(Error::CorruptStageData)
        ));
        // Wrong expected length mismatches the fill.
        assert!(matches!(
            decode(MIXED_ENCODED, MIXED_INPUT.len() + 1),
            Err(Error::CorruptStageData)
        ));
        // Garbage bytes never decode to real data.
        assert!(matches!(
            decode(&[0xFF_u8; 33], 1),
            Err(Error::CorruptStageData)
        ));
    }

    #[test]
    fn short_output_yields_prefix() {
        // The reference stops once the shorter buffer fills.
        let prefix_len = MIXED_INPUT.len() - 1;
        let got = decode(MIXED_ENCODED, prefix_len).unwrap();
        assert_eq!(&got, &MIXED_INPUT[..prefix_len]);
    }

    /// Slow forward transform, kept as the reference for the fast one.
    fn oracle_encode_into(input: &[u8], out: &mut [u8]) -> Option<usize> {
        let mut gain = [0_i64; 256];
        for run in input.chunk_by(|a, b| a == b) {
            let repeats = run.len() - 1;
            let net = i64::try_from(repeats - repeats / 255).unwrap() - 1;
            gain[usize::from(run[0])] += net;
        }
        if gain.iter().all(|&g| g <= 0) {
            return None;
        }
        let limit = input.len().min(out.len());
        if limit <= 32 {
            return None;
        }
        for (slot, chunk) in out.iter_mut().zip(gain.chunks(8)) {
            *slot = chunk
                .iter()
                .enumerate()
                .fold(0_u8, |acc, (bit, &g)| acc | (u8::from(g > 0) << bit));
        }
        let mut pos = 32;
        for run in input.chunk_by(|a, b| a == b) {
            let value = run[0];
            if gain[usize::from(value)] > 0 {
                let repeats = run.len() - 1;
                let full = repeats / 255;
                if pos + full + 2 >= limit {
                    return None;
                }
                out[pos] = value;
                out[pos + 1..pos + 1 + full].fill(255);
                out[pos + 1 + full] = u8::try_from(repeats % 255).unwrap();
                pos += full + 2;
            } else {
                if pos + run.len() >= limit {
                    return None;
                }
                out[pos..pos + run.len()].fill(value);
                pos += run.len();
            }
        }
        Some(pos)
    }

    /// Slow inverse transform, kept as the reference for the fast one.
    fn oracle_decode_into(input: &[u8], out: &mut [u8]) -> Result<(), Error> {
        if input.len() < 32 {
            return Err(Error::CorruptStageData);
        }
        let out_len = out.len();
        let mut gain = [false; 256];
        for (chunk, &byte) in input.iter().take(32).enumerate() {
            for bit in 0..8_usize {
                gain[chunk * 8 + bit] = (byte >> bit) & 1 == 1;
            }
        }
        let mut op = 0;
        let mut ip = 32;
        while op < out_len && ip < input.len() {
            let byte = input[ip];
            ip += 1;
            if gain[usize::from(byte)] {
                let mut run: usize = 1;
                loop {
                    let Some(&count) = input.get(ip) else {
                        return Err(Error::CorruptStageData);
                    };
                    ip += 1;
                    if run <= out_len {
                        run += usize::from(count);
                    }
                    if count != 255 {
                        break;
                    }
                }
                let fill = run.min(out_len - op);
                out[op..op + fill].fill(byte);
                op += fill;
            } else {
                out[op] = byte;
                op += 1;
            }
        }
        if op == out_len {
            Ok(())
        } else {
            Err(Error::CorruptStageData)
        }
    }

    /// Next value of a fixed-seed generator.
    fn next(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        *seed >> 8
    }

    /// Inputs with mixed run lengths, value counts and sizes.
    fn generated_inputs() -> Vec<Vec<u8>> {
        let mut seed = 0x1234_5678;
        let mut inputs = Vec::new();
        for values in [1_u32, 2, 4, 16, 256] {
            for max_run in [1_u32, 3, 9, 40, 600] {
                for target in [0_usize, 1, 31, 33, 255, 256, 257, 1000, 5000] {
                    let mut data = Vec::new();
                    while data.len() < target {
                        let value = u8::try_from(next(&mut seed) % values).unwrap();
                        let len = 1 + usize::try_from(next(&mut seed) % max_run).unwrap();
                        data.resize(data.len() + len, value);
                    }
                    inputs.push(data);
                }
            }
        }
        for len in [254, 255, 256, 510, 511, 512, 513, 765, 766, 767] {
            inputs.push(vec![7_u8; len]);
            let mut data = vec![7_u8; len];
            data.extend_from_slice(&[1, 2, 3]);
            inputs.push(data);
        }
        inputs
    }

    #[test]
    fn encode_into_matches_oracle() {
        for input in generated_inputs() {
            let limits = [
                0,
                1,
                32,
                33,
                40,
                input.len() / 2,
                input.len(),
                input.len() + 70,
            ];
            for out_len in limits {
                let mut got = vec![0xAA_u8; out_len];
                let mut want = vec![0xAA_u8; out_len];
                let got_len = encode_into(&input, &mut got);
                let want_len = oracle_encode_into(&input, &mut want);
                assert_eq!(got_len, want_len, "len {} out {out_len}", input.len());
                if let Some(len) = got_len {
                    assert_eq!(&got[..len], &want[..len]);
                }
            }
        }
    }

    #[test]
    fn decode_into_matches_oracle() {
        let mut seed = 0x0BAD_CAFE;
        for input in generated_inputs() {
            let mut encoded = vec![0_u8; input.len() + 70];
            let Some(len) = oracle_encode_into(&input, &mut encoded) else {
                continue;
            };
            encoded.truncate(len);
            let mut payloads = vec![encoded.clone()];
            for cut in [1, 2, 33, 40] {
                if encoded.len() > cut {
                    payloads.push(encoded[..encoded.len() - cut].to_vec());
                }
            }
            for _ in 0..4 {
                let mut bad = encoded.clone();
                let at = usize::try_from(next(&mut seed)).unwrap() % bad.len();
                bad[at] ^= u8::try_from(1 + next(&mut seed) % 255).unwrap();
                payloads.push(bad);
            }
            for payload in &payloads {
                for out_len in [
                    0,
                    1,
                    input.len() / 2,
                    input.len().saturating_sub(1),
                    input.len(),
                    input.len() + 1,
                ] {
                    let mut got = vec![0x55_u8; out_len];
                    let mut want = vec![0x55_u8; out_len];
                    let got_res = decode_into(payload, &mut got);
                    let want_res = oracle_decode_into(payload, &mut want);
                    assert_eq!(got_res.is_ok(), want_res.is_ok(), "out {out_len}");
                    if got_res.is_ok() {
                        assert_eq!(got, want);
                    }
                }
            }
        }
    }

    /// Inputs whose runs end at every offset modulo the block size,
    /// with collapse sets over all high nibbles.
    fn edge_inputs() -> Vec<Vec<u8>> {
        let mut inputs = Vec::new();
        for shift in 0..40_usize {
            for run in [2_usize, 3, 31, 32, 33, 64, 65, 300] {
                for value in [0x00_u8, 0x07, 0x08, 0x7F, 0x80, 0xF2, 0xFF] {
                    let mut data: Vec<u8> = (0..shift)
                        .map(|i| 1 + u8::try_from(i % 5).unwrap())
                        .collect();
                    data.resize(data.len() + run, value);
                    data.extend((0..shift).map(|i| 200 + u8::try_from(i % 7).unwrap()));
                    inputs.push(data);
                }
            }
        }
        inputs
    }

    /// Every size of collapse set, so both set searches run.
    #[test]
    fn set_search_matches_scalar_search() {
        let mut seed = 0x9E37_79B9_u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            seed >> 16
        };
        for size in [0_usize, 1, 2, 3, 4, 5, 6, 40, 256] {
            for _ in 0..40 {
                let mut member = [false; 256];
                let mut chosen = 0;
                while chosen < size {
                    let value = next() as usize % 256;
                    if !member[value] {
                        member[value] = true;
                        chosen += 1;
                    }
                }
                let set = ByteSet::new(member);
                for len in [0_usize, 1, 7, 8, 9, 39, 40, 41, 71, 72, 73, 200] {
                    let data: Vec<u8> = (0..len)
                        .map(|_| u8::try_from(next() % 256).unwrap())
                        .collect();
                    let want = data
                        .iter()
                        .position(|&byte| member[usize::from(byte)])
                        .unwrap_or(len);
                    assert_eq!(set.first_member(&data), want, "size {size} len {len}");
                }
            }
        }
    }

    #[test]
    fn encode_matches_oracle_on_edges() {
        for input in edge_inputs() {
            let mut want = vec![0_u8; input.len() + 70];
            let mut got = vec![0_u8; input.len() + 70];
            let want_len = oracle_encode_into(&input, &mut want);
            assert_eq!(encode_into(&input, &mut got), want_len);
            if let Some(len) = want_len {
                assert_eq!(&got[..len], &want[..len]);
                let mut back = vec![0_u8; input.len()];
                decode_into(&got[..len], &mut back).unwrap();
                assert_eq!(back, input);
            }
        }
    }
}
