//! Inverse block decode pipeline with check order, fallback, and checksum last.

use std::fmt::Debug;
use std::fmt::Formatter;

use crate::BlockSize;
use crate::buffer::filled;
use crate::cm::CmCoder;
use crate::crc32;
use crate::error::Error;
use crate::error::Result;
use crate::frame::read_i32;
use crate::frame::read_u32;
use crate::lzp;
use crate::rle;
use crate::sais;
use crate::sais::UnbwtScratch;

use super::FILLER_CAP;
use super::MODEL_LEN;
use super::PREFIX_LEN;
use super::PayloadPrefix;
use super::STORED_MARKER;
use super::STORED_MAX_BODY;
use super::STORED_PREFIX_LEN;
use super::model::Model;

/// State of the bytes that [`BlockDecoder::recover_block_into`] appended.
#[derive(Debug)]
pub enum BlockStatus {
    /// The block decoded with no fault.
    Intact,
    /// Only the checksum is bad. The appended bytes are the decoded bytes.
    BadChecksum,
    /// The block has another fault. The appended bytes are zero.
    Zeroed(Error),
}

/// Block decoder with constructor-owned working memory.
///
/// This type mirrors [`BlockEncoder`](super::BlockEncoder) on the decode side and never touches
/// the encoder. The hash table and the entropy coder state persist
/// across calls but reset per block, exactly like the reference. One
/// call decodes one block payload through the inverse pipeline:
/// entropy decode, inverse Burrows-Wheeler transform, then inverse
/// Lempel-Ziv prediction and inverse run-length transform as the model
/// byte selects. Every check runs in reference order, first failure
/// wins. Like the reference, a short entropy stream reads filler bytes
/// past its end; the port retries with equivalent filler and fails
/// downstream exactly as the reference does.
///
/// # Examples
///
/// ```
/// use bzip3_rs::{BlockDecoder, BlockEncoder, BlockSize};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let data = b"hello world ".repeat(100);
/// let size = BlockSize::MIN;
/// let payload = BlockEncoder::new(size)?.encode_block(&data)?;
///
/// let mut decoder = BlockDecoder::new(size)?;
/// let mut plain = Vec::new();
/// decoder.decode_block_into(&payload, u32::try_from(data.len())?, &mut plain)?;
/// assert_eq!(plain, data);
/// # Ok(())
/// # }
/// ```
pub struct BlockDecoder {
    /// Block size recorded in the frame header.
    block_size: BlockSize,
    /// LZP hash table, cleared by the stage on every call.
    lzp_table: Box<lzp::HashTable>,
    /// Entropy coder state, reset by the stage on every call.
    /// Boxed like the reference heap state, so the decoder struct
    /// stays small on the stack.
    coder: Box<CmCoder>,
    /// First ping-pong stage buffer, [`compress_bound`](crate::compress_bound) bytes.
    buf_a: Vec<u8>,
    /// Second ping-pong stage buffer, [`compress_bound`](crate::compress_bound) bytes.
    buf_b: Vec<u8>,
    /// Inverse transform scratch, kept between blocks.
    unbwt_scratch: UnbwtScratch,
}

impl Debug for BlockDecoder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockDecoder")
            .field("block_size", &self.block_size)
            .finish_non_exhaustive()
    }
}

impl BlockDecoder {
    /// Build a decoder for `block_size`, allocating working memory once.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] when any working buffer allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::{BlockDecoder, BlockSize};
    ///
    /// # fn main() -> Result<(), bzip3_rs::error::Error> {
    /// let decoder = BlockDecoder::new(BlockSize::MIN)?;
    /// assert_eq!(decoder.block_size(), BlockSize::MIN);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(block_size: BlockSize) -> Result<Self> {
        let capacity = crate::compress_bound(block_size.as_usize());
        Ok(Self {
            block_size,
            lzp_table: lzp::new_table()?,
            coder: Box::new(CmCoder::new()),
            buf_a: filled(capacity, 0).ok_or(Error::OutOfMemory)?,
            buf_b: filled(capacity, 0).ok_or(Error::OutOfMemory)?,
            unbwt_scratch: UnbwtScratch::default(),
        })
    }

    /// Return the block size this decoder was built for.
    #[must_use]
    #[inline]
    pub const fn block_size(&self) -> BlockSize {
        self.block_size
    }

    /// Decode one block payload into a new vector.
    ///
    /// Same result as [`BlockDecoder::decode_block_into`]. Use that call
    /// to reuse one output vector across blocks.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`BlockDecoder::decode_block_into`].
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::{BlockDecoder, BlockEncoder, BlockSize};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let data = b"hello world ".repeat(100);
    /// let payload = BlockEncoder::new(BlockSize::MIN)?.encode_block(&data)?;
    /// let mut decoder = BlockDecoder::new(BlockSize::MIN)?;
    /// let block = decoder.decode_block(&payload, u32::try_from(data.len())?)?;
    /// assert_eq!(block, data);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "decoded output must be written to the frame"]
    #[inline]
    pub fn decode_block(&mut self, payload: &[u8], orig_size: u32) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.decode_block_into(payload, orig_size, &mut out)?;
        Ok(out)
    }

    /// Decode one block payload and append its original bytes to `out`.
    ///
    /// On an error, `out` keeps its length. The call checks the sizes
    /// before it grows `out`, so a hostile `orig_size` cannot force a
    /// large allocation.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`BlockDecoder::decode_block_to`], and
    /// [`Error::OutOfMemory`] when `out` cannot grow.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::{BlockDecoder, BlockEncoder, BlockSize};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let first = b"first block ".repeat(50);
    /// let second = b"second block ".repeat(50);
    /// let mut encoder = BlockEncoder::new(BlockSize::MIN)?;
    /// let mut decoder = BlockDecoder::new(BlockSize::MIN)?;
    ///
    /// let mut plain = Vec::new();
    /// for block in [&first, &second] {
    ///     let payload = encoder.encode_block(block)?;
    ///     decoder.decode_block_into(&payload, u32::try_from(block.len())?, &mut plain)?;
    /// }
    /// assert_eq!(plain, [first, second].concat());
    /// # Ok(())
    /// # }
    /// ```
    pub fn decode_block_into(
        &mut self,
        payload: &[u8],
        orig_size: u32,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        let head = self.read_prefix(payload, orig_size)?;
        let old_len = out.len();
        out.try_reserve(head.original_len)
            .map_err(|_| Error::OutOfMemory)?;
        out.resize(old_len + head.original_len, 0);
        let result = self.decode_payload(payload, &head, &mut out[old_len..]);
        if result.is_err() {
            out.truncate(old_len);
        }
        result
    }

    /// Decode one block payload and always append `orig_size` bytes to `out`.
    ///
    /// Use this call to recover the data of a damaged stream. A checksum
    /// fault keeps the decoded bytes, because the checksum is the last
    /// check. Any other block fault appends zero bytes, because the
    /// decode stages leave partial data. The returned [`BlockStatus`]
    /// tells which case happened. A size above the block size appends the
    /// block size.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] when `out` cannot grow.
    /// Then `out` keeps its length. Block faults are not errors. They
    /// come back in the [`BlockStatus`] value.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::error::Error;
    /// use bzip3_rs::{BlockDecoder, BlockEncoder, BlockStatus, BlockSize};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let data = b"hello world ".repeat(100);
    /// let orig_size = u32::try_from(data.len())?;
    /// let mut payload = BlockEncoder::new(BlockSize::MIN)?.encode_block(&data)?;
    /// let mut decoder = BlockDecoder::new(BlockSize::MIN)?;
    ///
    /// // A wrong checksum keeps the decoded bytes.
    /// payload[0] ^= 1;
    /// let mut plain = Vec::new();
    /// let result = decoder.recover_block_into(&payload, orig_size, &mut plain)?;
    /// assert!(matches!(result, BlockStatus::BadChecksum));
    /// assert_eq!(plain, data);
    ///
    /// // A short payload gives zero bytes.
    /// let mut plain = Vec::new();
    /// let result = decoder.recover_block_into(&payload[..4], orig_size, &mut plain)?;
    /// assert!(matches!(result, BlockStatus::Zeroed(Error::TruncatedHeader)));
    /// assert_eq!(plain, vec![0_u8; data.len()]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn recover_block_into(
        &mut self,
        payload: &[u8],
        orig_size: u32,
        out: &mut Vec<u8>,
    ) -> Result<BlockStatus> {
        let fill = usize::try_from(orig_size)
            .unwrap_or(usize::MAX)
            .min(self.block_size.as_usize());
        let old_len = out.len();
        out.try_reserve(fill).map_err(|_| Error::OutOfMemory)?;
        out.resize(old_len + fill, 0);
        let plain = &mut out[old_len..];
        let decoded = self
            .read_prefix(payload, orig_size)
            .and_then(|head| self.decode_body(payload, &head, plain).map(|()| head));
        Ok(match decoded {
            Ok(head) if check_crc(plain, &head).is_ok() => BlockStatus::Intact,
            Ok(_) => BlockStatus::BadChecksum,
            Err(fault) => {
                plain.fill(0);
                BlockStatus::Zeroed(fault)
            }
        })
    }

    /// Decode one block payload into `out`.
    ///
    /// `payload` holds the block bytes (`compressed_size` long) and
    /// `orig_size` holds the claimed original size. `out` must be
    /// exactly `orig_size` bytes long. The call runs the reference
    /// check order: buffer fit, compressed-size range, original-size
    /// range, stored-raw path, header-size fit, stage sizes,
    /// intermediate-size fit, entropy decode, index range, inverse
    /// transform, inverse stages, final-size match, checksum. The call
    /// does not allocate after the first block, except on a starved
    /// entropy stream.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferTooSmall`] when a size exceeds the decode buffer,
    /// [`Error::OutputSizeMismatch`] when `out` has the wrong length,
    /// [`Error::TruncatedHeader`] on a short payload,
    /// [`Error::InvalidHeader`] on bad sizes,
    /// [`Error::BwtFailed`] when the inverse transform fails,
    /// [`Error::ChecksumMismatch`] on checksum mismatch,
    /// [`Error::CorruptStageData`] on corrupt RLE or LZP data,
    /// [`Error::OutOfMemory`] when working memory allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::error::Error;
    /// use bzip3_rs::{BlockDecoder, BlockEncoder, BlockSize};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let data = b"hello world ".repeat(100);
    /// let orig_size = u32::try_from(data.len())?;
    /// let payload = BlockEncoder::new(BlockSize::MIN)?.encode_block(&data)?;
    /// let mut decoder = BlockDecoder::new(BlockSize::MIN)?;
    ///
    /// let mut out = vec![0_u8; data.len()];
    /// decoder.decode_block_to(&payload, orig_size, &mut out)?;
    /// assert_eq!(out, data);
    ///
    /// // A wrong output length fails.
    /// let mut short = vec![0_u8; data.len() - 1];
    /// let result = decoder.decode_block_to(&payload, orig_size, &mut short);
    /// assert!(matches!(result, Err(Error::OutputSizeMismatch)));
    /// # Ok(())
    /// # }
    /// ```
    pub fn decode_block_to(
        &mut self,
        payload: &[u8],
        orig_size: u32,
        out: &mut [u8],
    ) -> Result<()> {
        let head = self.read_prefix(payload, orig_size)?;
        if out.len() != head.original_len {
            return Err(Error::OutputSizeMismatch);
        }
        self.decode_payload(payload, &head, out)
    }

    /// Run the first reference checks and read the fixed prefix.
    fn read_prefix(&self, payload: &[u8], orig_size: u32) -> Result<PayloadPrefix> {
        let block = self.block_size.as_usize();
        // The buffer holds `compress_bound(block)` bytes, which exceeds every
        // fixed header. Buffer fit runs before any header read.
        let limit = crate::compress_bound(block);
        if limit < payload.len() {
            return Err(Error::BufferTooSmall);
        }
        let Some(prefix) = payload.get(..PREFIX_LEN) else {
            return Err(Error::TruncatedHeader);
        };
        let crc = read_u32(prefix);
        let bwt_index = read_i32(&prefix[4..]);
        let original_len = usize::try_from(orig_size).unwrap_or(usize::MAX);
        if original_len > block {
            return Err(Error::InvalidHeader);
        }
        Ok(PayloadPrefix {
            original_len,
            crc,
            bwt_index,
            limit,
        })
    }

    /// Decode a payload that passed [`BlockDecoder::read_prefix`].
    #[inline]
    fn decode_payload(
        &mut self,
        payload: &[u8],
        head: &PayloadPrefix,
        out: &mut [u8],
    ) -> Result<()> {
        self.decode_body(payload, head, out)?;
        check_crc(out, head)
    }

    /// Decode a payload through every stage, without the checksum.
    ///
    /// `out` holds the full block when this call succeeds.
    #[inline]
    fn decode_body(&mut self, payload: &[u8], head: &PayloadPrefix, out: &mut [u8]) -> Result<()> {
        if head.bwt_index == STORED_MARKER {
            return decode_stored(payload, head, out);
        }
        self.decode_stages(payload, head, out)
    }

    /// Decode a non-stored block payload through every inverse stage.
    ///
    /// `limit` is `compress_bound` of the block size. The reference passes the
    /// same value as both the compressed-size bound and the decode
    /// buffer size. The buffers here have exactly that size, so a
    /// stage size inside `0..=limit` always fits.
    fn decode_stages(
        &mut self,
        payload: &[u8],
        head: &PayloadPrefix,
        out: &mut [u8],
    ) -> Result<()> {
        let PayloadPrefix {
            original_len,
            bwt_index,
            limit,
            ..
        } = *head;
        let Some(&raw) = payload.get(PREFIX_LEN) else {
            return Err(Error::TruncatedHeader);
        };
        let model = Model(raw);
        let header_len = model.header_len();
        let Some(header) = payload.get(..header_len) else {
            return Err(Error::TruncatedHeader);
        };
        let mut sizes = &header[PREFIX_LEN + MODEL_LEN..];
        let lzp_size = if model.has_lzp() {
            let size = read_stage_size(sizes, limit)?;
            sizes = &sizes[4..];
            Some(size)
        } else {
            None
        };
        let rle_size = if model.has_rle() {
            Some(read_stage_size(sizes, limit)?)
        } else {
            None
        };
        let bwt_len = lzp_size.or(rle_size).unwrap_or(original_len);
        let coded = &payload[header_len..];
        self.decode_entropy(coded, bwt_len, bwt_index, model)?;
        if usize::try_from(bwt_index).is_ok_and(|index| index > bwt_len) {
            return Err(Error::InvalidHeader);
        }

        // Inverse Burrows-Wheeler transform. A block without LZP and RLE
        // has `bwt_len == original_len`, so it goes straight into `out`.
        if !model.has_lzp() && !model.has_rle() {
            sais::unbwt_with(
                &self.buf_a[..bwt_len],
                out,
                bwt_index,
                &mut self.unbwt_scratch,
            )?;
        } else {
            sais::unbwt_with(
                &self.buf_a[..bwt_len],
                &mut self.buf_b[..bwt_len],
                bwt_index,
                &mut self.unbwt_scratch,
            )?;
            let mut current = &self.buf_b[..bwt_len];
            if model.has_lzp() {
                let size =
                    lzp::decode_into(current, &mut self.buf_a[..limit], &mut self.lzp_table)?;
                current = &self.buf_a[..size];
            }
            if model.has_rle() {
                rle::decode_into(current, out)?;
            } else {
                if current.len() != original_len {
                    return Err(Error::InvalidHeader);
                }
                out.copy_from_slice(current);
            }
        }
        Ok(())
    }

    /// Entropy-decode `coded` into the first `bwt_len` bytes of `buf_a`,
    /// emulating filler.
    ///
    /// Runs before the index check, exactly like the reference. A
    /// short stream reads filler bytes past its end in the reference,
    /// and `0xFF` bytes feed the decoder state exactly as the
    /// reference `-1` reads do. The decoder therefore reads the same
    /// filler and reproduces the reference garbage byte for byte while
    /// the filler lasts. When even the capped filler runs out, the
    /// claimed sizes dwarf the stream, so the fallback keeps the
    /// reference order on what is known: index range first, then the
    /// modal downstream outcome.
    ///
    /// # Errors
    ///
    /// Returns the ordered fallback variant on a starved stream.
    fn decode_entropy(
        &mut self,
        coded: &[u8],
        bwt_len: usize,
        bwt_index: i32,
        model: Model,
    ) -> Result<()> {
        let filler = bwt_len
            .saturating_mul(32)
            .saturating_add(64)
            .min(FILLER_CAP);
        match self.coder.decode(coded, filler, &mut self.buf_a[..bwt_len]) {
            Err(Error::TruncatedData) => {
                let index = i64::from(bwt_index);
                let span = i64::try_from(bwt_len).unwrap_or(i64::MAX);
                if index > span {
                    return Err(Error::InvalidHeader);
                }
                if index < 1 {
                    return Err(Error::BwtFailed);
                }
                if model.has_lzp() && !model.has_rle() {
                    return Err(Error::InvalidHeader);
                }
                if model.has_rle() {
                    return Err(Error::CorruptStageData);
                }
                Err(Error::ChecksumMismatch)
            }
            outcome => outcome,
        }
    }
}

/// Check the checksum of a decoded block. This is the last check of a decode.
///
/// # Errors
///
/// Returns [`Error::ChecksumMismatch`] when the sums differ.
#[inline]
fn check_crc(plain: &[u8], head: &PayloadPrefix) -> Result<()> {
    if crc32::sum(1, plain) == head.crc {
        Ok(())
    } else {
        Err(Error::ChecksumMismatch)
    }
}

/// Read one stage size field.
///
/// # Errors
///
/// Returns [`Error::InvalidHeader`] when the size is outside `0..=limit`.
#[inline]
fn read_stage_size(bytes: &[u8], limit: usize) -> Result<usize> {
    usize::try_from(read_i32(bytes))
        .ok()
        .filter(|&size| size <= limit)
        .ok_or(Error::InvalidHeader)
}

/// Decode a stored-raw block payload.
///
/// The body must hold exactly `orig` bytes within the 64-byte small
/// block bound. The caller checks the checksum, so size faults win over it.
#[inline]
fn decode_stored(payload: &[u8], head: &PayloadPrefix, out: &mut [u8]) -> Result<()> {
    let original_len = head.original_len;
    let Some(body) = payload.get(STORED_PREFIX_LEN..) else {
        return Err(Error::TruncatedHeader);
    };
    if body.len() > STORED_MAX_BODY || body.len() != original_len {
        return Err(Error::InvalidHeader);
    }
    out.copy_from_slice(body);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::BlockDecoder;
    use super::BlockStatus;
    use super::decode_stored;
    use crate::BlockSize;
    use crate::block::BlockEncoder;
    use crate::compress_bound;
    use crate::error::Error;

    /// Smallest decoder used by the tests (65 KiB block).
    fn small_decoder() -> BlockDecoder {
        BlockDecoder::new(BlockSize::MIN).unwrap()
    }

    /// Smallest state used by the tests (65 KiB block).
    fn small_state() -> BlockEncoder {
        BlockEncoder::new(BlockSize::MIN).unwrap()
    }

    #[test]
    fn decoder_restores_stored_blocks() {
        let mut encoder = small_state();
        let mut decoder = small_decoder();
        let sixty_three = b"x".repeat(63);
        for input in [b"A".as_slice(), b"".as_slice(), sixty_three.as_slice()] {
            let payload = encoder.encode_block(input).unwrap();
            let len = u32::try_from(input.len()).unwrap();
            assert_eq!(decoder.decode_block(&payload, len).unwrap(), input);
        }
    }

    #[test]
    fn decoder_restores_every_model() {
        let mut seed = 0x00C0_FFEE_u32;
        let random: Vec<u8> = (0..1000)
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                u8::try_from((seed >> 16) & 0xFF).expect("masked byte fits")
            })
            .collect();
        let phrase: Vec<u8> = (0..50_u8).collect();
        let mut periodic = Vec::new();
        for _ in 0..20 {
            periodic.extend_from_slice(&phrase);
            periodic.extend_from_slice(&phrase);
        }
        let inputs: Vec<Vec<u8>> = vec![
            vec![b'A'; 1000],
            periodic,
            random,
            [vec![b'B'; 300], (0..50_u8).collect::<Vec<u8>>().repeat(10)].concat(),
        ];
        let mut models = std::collections::HashSet::new();
        for input in &inputs {
            let payload = small_state().encode_block(input).unwrap();
            models.insert(payload[8]);
            let len = u32::try_from(input.len()).unwrap();
            let got = small_decoder().decode_block(&payload, len).unwrap();
            assert_eq!(&got, input);
        }
        // Runs set RLE, periodic text sets LZP, filler stays plain.
        assert!(models.contains(&0x04));
        assert!(models.contains(&0x02));
        assert!(models.contains(&0x00));
    }

    #[test]
    fn decoder_enforces_check_order() {
        let mut decoder = small_decoder();
        let payload = small_state().encode_block(b"A").unwrap();
        assert_eq!(decoder.decode_block(&payload, 1).unwrap(), b"A");
        // Short payload and body/size mismatch are malformed.
        assert!(matches!(
            decoder.decode_block(&payload[..4], 1),
            Err(crate::error::Error::TruncatedHeader)
        ));
        assert!(matches!(
            decoder.decode_block(&payload, 2),
            Err(crate::error::Error::InvalidHeader)
        ));
        // Body above 64 bytes is malformed even with matching size.
        let mut big = payload.clone();
        big.extend_from_slice(&[0; 64]);
        assert!(matches!(
            decoder.decode_block(&big, 65),
            Err(crate::error::Error::InvalidHeader)
        ));
        // Corrupt body fails the checksum.
        let mut bad = payload.clone();
        bad[8] ^= 0x01;
        assert!(matches!(
            decoder.decode_block(&bad, 1),
            Err(crate::error::Error::ChecksumMismatch)
        ));
        // Original size above the block size is malformed.
        assert!(matches!(
            decoder.decode_block(&payload, 65 * 1024 + 1),
            Err(crate::error::Error::InvalidHeader)
        ));
    }

    #[test]
    fn recover_keeps_bytes_on_checksum_fault() {
        let data = [b'A'; 1000];
        let mut decoder = small_decoder();
        // A coded block and a stored block.
        for input in [data.as_slice(), b"short".as_slice()] {
            let mut payload = small_state().encode_block(input).unwrap();
            payload[0] ^= 0x01;
            let len = u32::try_from(input.len()).unwrap();
            let mut out = b"head".to_vec();
            let result = decoder.recover_block_into(&payload, len, &mut out);
            assert!(matches!(result, Ok(BlockStatus::BadChecksum)));
            assert_eq!(out[..4], *b"head");
            assert_eq!(out[4..], *input);
        }
    }

    #[test]
    fn recover_writes_zeros_on_stage_fault() {
        let payload = small_state().encode_block(&[b'A'; 1000]).unwrap();
        let mut bad = payload.clone();
        bad[13] ^= 0xFF;
        let mut out = Vec::new();
        let result = small_decoder().recover_block_into(&bad, 1000, &mut out);
        assert!(matches!(
            result,
            Ok(BlockStatus::Zeroed(crate::error::Error::CorruptStageData))
        ));
        assert_eq!(out, vec![0_u8; 1000]);

        // A short payload and a size above the block size.
        let mut out = Vec::new();
        let result = small_decoder().recover_block_into(&payload[..4], 10, &mut out);
        assert!(matches!(
            result,
            Ok(BlockStatus::Zeroed(crate::error::Error::TruncatedHeader))
        ));
        assert_eq!(out, vec![0_u8; 10]);
        let mut out = Vec::new();
        let result = small_decoder().recover_block_into(&payload, u32::MAX, &mut out);
        assert!(matches!(
            result,
            Ok(BlockStatus::Zeroed(crate::error::Error::InvalidHeader))
        ));
        assert_eq!(out.len(), BlockSize::MIN.as_usize());

        // A bad index of the transform gives zero bytes.
        let mut bad = payload.clone();
        bad[4..8].copy_from_slice(&i32::MAX.to_le_bytes());
        let mut out = Vec::new();
        let result = small_decoder().recover_block_into(&bad, 1000, &mut out);
        assert!(matches!(
            result,
            Ok(BlockStatus::Zeroed(crate::error::Error::InvalidHeader))
        ));
        assert_eq!(out, vec![0_u8; 1000]);
    }

    #[test]
    fn recover_matches_decode_on_good_blocks() {
        let mut decoder = small_decoder();
        for input in [b"A".as_slice(), &[b'B'; 1000], &[]] {
            let payload = small_state().encode_block(input).unwrap();
            let len = u32::try_from(input.len()).unwrap();
            let mut out = Vec::new();
            let result = decoder.recover_block_into(&payload, len, &mut out);
            assert!(matches!(result, Ok(BlockStatus::Intact)));
            assert_eq!(out, input);
        }
    }

    #[test]
    fn decoder_enforces_coded_check_order() {
        // RLE-model payload exercises the coded path.
        let payload = small_state().encode_block(&[b'A'; 1000]).unwrap();
        assert_eq!(payload[8] & 0x04, 0x04);
        let mut decoder = small_decoder();
        assert_eq!(decoder.decode_block(&payload, 1000).unwrap().len(), 1000);
        // Bad stage size beats the checksum.
        let mut bad = payload.clone();
        bad[9..13].copy_from_slice(&i32::MAX.to_le_bytes());
        assert!(matches!(
            decoder.decode_block(&bad, 1000),
            Err(crate::error::Error::InvalidHeader)
        ));
        // Bad index beats the corrupt stream.
        let mut bad = payload.clone();
        bad[4..8].copy_from_slice(&i32::MAX.to_le_bytes());
        let last = bad.len() - 1;
        bad[last] ^= 0xFF;
        assert!(matches!(
            decoder.decode_block(&bad, 1000),
            Err(crate::error::Error::InvalidHeader)
        ));
        // Corrupt stream fails the RLE stage or the checksum. The first coded byte
        // feeds the entropy decoder; the tail holds flush bytes the
        // decoder may never read, so the flip targets the head.
        let mut bad = payload.clone();
        bad[13] ^= 0xFF;
        assert!(matches!(
            decoder.decode_block(&bad, 1000),
            Err(crate::error::Error::CorruptStageData)
        ));
    }

    #[test]
    fn decoder_final_size_mismatch_is_malformed() {
        // LZP-only payload: shrinking the claim breaks the final match.
        let phrase: Vec<u8> = (0..50_u8).collect();
        let mut input = Vec::new();
        for _ in 0..4 {
            input.extend_from_slice(&phrase);
        }
        let payload = small_state().encode_block(&input).unwrap();
        assert_eq!(payload[8] & 0x06, 0x02);
        let short = u32::try_from(input.len() - 1).unwrap();
        assert!(matches!(
            small_decoder().decode_block(&payload, short),
            Err(crate::error::Error::InvalidHeader)
        ));
    }

    #[test]
    fn decoder_reuse_gives_identical_bytes() {
        let payload = small_state().encode_block(&[b'A'; 1000]).unwrap();
        let mut decoder = small_decoder();
        let first = decoder.decode_block(&payload, 1000).unwrap();
        let other = small_state().encode_block(&[b'B'; 500]).unwrap();
        let _ = decoder.decode_block(&other, 500).unwrap();
        let again = decoder.decode_block(&payload, 1000).unwrap();
        assert_eq!(first, again);
    }

    #[test]
    fn decoder_block_size_accessor_matches_constructor() {
        assert_eq!(small_decoder().block_size(), BlockSize::MIN);
    }

    #[test]
    fn debug_names_type_and_block_size() {
        let encoder_text = format!("{:?}", small_state());
        assert!(encoder_text.contains("BlockEncoder"));
        assert!(encoder_text.contains("block_size"));
        let decoder_text = format!("{:?}", small_decoder());
        assert!(decoder_text.contains("BlockDecoder"));
        assert!(decoder_text.contains("block_size"));
    }

    #[test]
    fn decode_block_to_rejects_wrong_output_length() {
        let data = [b'A'; 1000];
        let payload = small_state().encode_block(&data).unwrap();
        let mut exact = vec![0_u8; data.len()];
        small_decoder()
            .decode_block_to(&payload, 1000, &mut exact)
            .unwrap();
        assert_eq!(exact, data);
        for len in [0, data.len() - 1, data.len() + 1] {
            let mut out = vec![0_u8; len];
            assert!(matches!(
                small_decoder().decode_block_to(&payload, 1000, &mut out),
                Err(Error::OutputSizeMismatch)
            ));
        }
    }

    #[test]
    fn oversize_payload_fails_buffer_too_small() {
        let limit = compress_bound(BlockSize::MIN.as_usize());
        let big = vec![0_u8; limit + 1];
        assert!(matches!(
            small_decoder().decode_block(&big, 1),
            Err(Error::BufferTooSmall)
        ));
        let mut out = Vec::new();
        assert!(matches!(
            small_decoder().recover_block_into(&big, 1, &mut out),
            Ok(BlockStatus::Zeroed(Error::BufferTooSmall))
        ));
    }

    #[test]
    fn eight_byte_prefix_fails_truncated_header() {
        // The prefix passes the fixed-size check, but no model byte follows.
        let mut payload = vec![0_u8; 8];
        payload[4..8].copy_from_slice(&0_i32.to_le_bytes());
        assert!(matches!(
            small_decoder().decode_block(&payload, 100),
            Err(Error::TruncatedHeader)
        ));
    }

    #[test]
    fn short_stage_header_fails_truncated_header() {
        // A set stage bit claims a 4-byte size that the payload lacks.
        for model in [0x02_u8, 0x04_u8] {
            let mut payload = vec![0_u8; 9];
            payload[4..8].copy_from_slice(&0_i32.to_le_bytes());
            payload[8] = model;
            assert!(matches!(
                small_decoder().decode_block(&payload, 100),
                Err(Error::TruncatedHeader)
            ));
        }
    }

    #[test]
    fn stored_prefix_short_fails_truncated_header() {
        // `read_prefix` always passes at least 8 bytes, so this arm runs
        // only through a direct call.
        let head = crate::block::PayloadPrefix {
            original_len: 4,
            crc: 0,
            bwt_index: -1,
            limit: 64,
        };
        let mut out = [0_u8; 4];
        assert!(matches!(
            decode_stored(&[0_u8; 4], &head, &mut out),
            Err(Error::TruncatedHeader)
        ));
    }
}
