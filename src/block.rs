//! Stateful block encoder for the bzip3 port.
//!
//! `BlockEncoder` owns all working memory sized from the block size at
//! construction and reuses it across blocks. One call encodes one
//! block through the full pipeline: checksum, run-length transform,
//! Lempel-Ziv prediction, Burrows-Wheeler transform, then the
//! order-0 context-mixing entropy coder. Model selection matches the
//! reference exactly. Each of the first two stages applies only when
//! it shrinks the block. Inputs below [`MIN_CODED_LEN`] bytes take the
//! stored-raw layout: checksum, stored marker, raw bytes.
//! Decoding enforces the reference check order, first failure wins.

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
use crate::sais::BwtScratch;
use crate::sais::UnbwtScratch;

/// Minimum input size that leaves the stored-raw route.
///
/// The reference takes the stored path when `data_size < 64`.
pub const MIN_CODED_LEN: usize = 64;

/// Marker stored in place of the BWT index for raw blocks.
const STORED_MARKER: i32 = -1;

/// Length of the stored prefix (CRC32 plus marker).
const STORED_PREFIX_LEN: usize = 8;

/// Largest raw body a stored block carries (`64` bytes).
const STORED_MAX_BODY: usize = 64;

/// Model byte of one coded block.
///
/// Bit `0x02` selects the LZP stage. Bit `0x04` selects the RLE
/// stage. Other bits stay clear on encode. The decoder reads any
/// wire value but only these bits select stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Model(u8);

impl Model {
    /// Bit set when the LZP stage applies.
    const LZP: u8 = 0x02;
    /// Bit set when the RLE stage applies.
    const RLE: u8 = 0x04;

    /// True when the LZP stage applies.
    const fn has_lzp(self) -> bool {
        self.0 & Self::LZP != 0
    }

    /// True when the RLE stage applies.
    const fn has_rle(self) -> bool {
        self.0 & Self::RLE != 0
    }

    /// Mark the LZP stage as applied.
    const fn set_lzp(&mut self) {
        self.0 |= Self::LZP;
    }

    /// Mark the RLE stage as applied.
    const fn set_rle(&mut self) {
        self.0 |= Self::RLE;
    }

    /// Full header length: prefix, model byte, kept stage sizes.
    const fn header_len(self) -> usize {
        let mut len = PREFIX_LEN + MODEL_LEN;
        if self.has_lzp() {
            len += 4;
        }
        if self.has_rle() {
            len += 4;
        }
        len
    }
}

/// Prefix bytes before the model byte (CRC32 plus BWT index).
const PREFIX_LEN: usize = 8;

/// Maximum filler bytes appended on entropy-stream retry.
///
/// A short stream reads filler in the reference. Each output byte
/// consumes at most 32 stream bytes in the pathological case. A retry
/// with `min(32 * bwt_len + 64, FILLER_CAP)` filler matches the
/// reference exactly on small blocks. It falls back in order on huge
/// claims. The cap bounds time and memory on hostile inputs.
const FILLER_CAP: usize = 1 << 20;

/// Length of the model byte.
const MODEL_LEN: usize = 1;

/// Block encoder with constructor-owned working memory.
///
/// The two swap buffers each hold [`compress_bound`](crate::compress_bound) of the block
/// size, so every intermediate stage result fits. The hash table and
/// the entropy coder state persist across calls but reset per block,
/// exactly like the reference. Slice-based calls borrow the input and
/// return an owned payload.
///
/// # Examples
///
/// Build a streaming frame with one block, then decode it with
/// [`stream::decompress`](crate::stream::decompress).
///
/// ```
/// use bzip3_rs::{BlockEncoder, BlockHeader, BlockSize, Jobs, StreamHeader, stream};
///
/// # fn main() -> Result<(), bzip3_rs::error::Error> {
/// let input = b"hello world ".repeat(1000);
/// let size = BlockSize::MIN;
///
/// let mut encoder = BlockEncoder::new(size)?;
/// let payload = encoder.encode_block(&input)?;
///
/// let mut frame = StreamHeader::new(size).to_bytes().to_vec();
/// frame.extend_from_slice(&BlockHeader::for_block(payload.len(), input.len())?.to_bytes());
/// frame.extend_from_slice(&payload);
///
/// let mut plain = Vec::new();
/// stream::decompress(frame.as_slice(), &mut plain, Jobs::SERIAL)?;
/// assert_eq!(plain, input);
/// # Ok(())
/// # }
/// ```
pub struct BlockEncoder {
    /// Maximum input bytes accepted by [`BlockEncoder::encode_block`].
    block_size: BlockSize,
    /// First ping-pong stage buffer.
    buf_a: Vec<u8>,
    /// Second ping-pong stage buffer.
    buf_b: Vec<u8>,
    /// LZP hash table, cleared by the stage on every call.
    lzp_table: Box<lzp::HashTable>,
    /// Entropy coder state, reset by the stage on every call.
    /// Boxed like the reference heap state, so the encoder struct
    /// stays small on the stack.
    coder: Box<CmCoder>,
    /// Suffix sort scratch, kept between blocks.
    bwt_scratch: BwtScratch,
}

impl Debug for BlockEncoder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockEncoder")
            .field("block_size", &self.block_size)
            .finish_non_exhaustive()
    }
}

impl BlockEncoder {
    /// Build a block encoder for `block_size`, allocating working memory once.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] when any working buffer allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::{BlockEncoder, BlockSize};
    ///
    /// # fn main() -> Result<(), bzip3_rs::error::Error> {
    /// let encoder = BlockEncoder::new(BlockSize::MIN)?;
    /// assert_eq!(encoder.block_size(), BlockSize::MIN);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(block_size: BlockSize) -> Result<Self> {
        let capacity = crate::compress_bound(block_size.as_usize());
        Ok(Self {
            block_size,
            buf_a: filled(capacity, 0).ok_or(Error::OutOfMemory)?,
            buf_b: filled(capacity, 0).ok_or(Error::OutOfMemory)?,
            lzp_table: lzp::new_table()?,
            coder: Box::new(CmCoder::new()),
            bwt_scratch: BwtScratch::default(),
        })
    }

    /// Return the block size this block encoder was built for.
    #[must_use]
    #[inline]
    pub const fn block_size(&self) -> BlockSize {
        self.block_size
    }

    /// Encode one block into a new vector.
    ///
    /// Same output as [`BlockEncoder::encode_block_into`]. Use that call
    /// to reuse one output vector across blocks.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`BlockEncoder::encode_block_into`].
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::{BlockDecoder, BlockEncoder, BlockSize};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let data = b"hello world ".repeat(100);
    /// let mut encoder = BlockEncoder::new(BlockSize::MIN)?;
    /// let payload = encoder.encode_block(&data)?;
    ///
    /// let mut decoder = BlockDecoder::new(BlockSize::MIN)?;
    /// assert_eq!(decoder.decode_block(&payload, u32::try_from(data.len())?)?, data);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "encode output must be written to the frame"]
    #[inline]
    pub fn encode_block(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.encode_block_into(data, &mut out)?;
        Ok(out)
    }

    /// Encode one block through the full pipeline.
    ///
    /// Append the block to `out`. On an error, `out` keeps its length.
    /// Inputs below 64 bytes take the stored-raw layout.
    /// Larger inputs run RLE, then LZP, then BWT, then entropy coding,
    /// with each of the first two stages kept only when it shrinks the
    /// block. The header matches the reference byte for byte: CRC32,
    /// BWT index, model byte, then the kept stage sizes in LZP, RLE
    /// order, then the coded stream.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BlockTooLarge`] when `data` exceeds the block size,
    /// [`Error::OutOfMemory`] when the RLE stage cannot grow its output,
    /// [`Error::BwtFailed`] when the transform fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::{BlockEncoder, BlockSize};
    ///
    /// # fn main() -> Result<(), bzip3_rs::error::Error> {
    /// let mut encoder = BlockEncoder::new(BlockSize::MIN)?;
    /// let mut out = Vec::new();
    /// for chunk in [b"first block ".repeat(50), b"second block ".repeat(50)] {
    ///     out.clear(); // Reuse one vector for every block.
    ///     encoder.encode_block_into(&chunk, &mut out)?;
    ///     assert!(!out.is_empty());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn encode_block_into(&mut self, data: &[u8], out: &mut Vec<u8>) -> Result<()> {
        let limit = self.block_size.as_usize();
        if data.len() > limit {
            return Err(Error::BlockTooLarge);
        }
        if data.len() < MIN_CODED_LEN {
            return encode_stored(data, out);
        }
        let crc = crc32::sum(1, data);
        // Each kept stage output becomes the next stage input. The first
        // stage reads `data`, so the input needs no copy. `src` names the
        // buffer that holds the current stage output.
        let mut src = StageBuffer::Input;
        let mut cur_len = data.len();
        let mut model = Model(0);
        let mut lzp_size: u32 = 0;
        let mut rle_size: u32 = 0;

        // RLE forward transform, kept only when it shrinks the block.
        let (input, stage_out, next) =
            stage_buffers(src, cur_len, data, &mut self.buf_a, &mut self.buf_b);
        if let Some(len) = rle::encode_into(input, stage_out) {
            rle_size = u32::try_from(len).map_err(|_| Error::BlockTooLarge)?;
            src = next;
            cur_len = len;
            model.set_rle();
        }

        // LZP forward transform, kept only when it shrinks the block.
        // `None` is the reference `-1`: no saving, keep the input.
        let (input, stage_out, next) =
            stage_buffers(src, cur_len, data, &mut self.buf_a, &mut self.buf_b);
        if let Some(len) = lzp::encode_into(input, stage_out, &mut self.lzp_table) {
            lzp_size = u32::try_from(len).map_err(|_| Error::BlockTooLarge)?;
            src = next;
            cur_len = len;
            model.set_lzp();
        }

        // Burrows-Wheeler transform into the free buffer.
        let (input, bwt_out, _) =
            stage_buffers(src, cur_len, data, &mut self.buf_a, &mut self.buf_b);
        let bwt_idx = sais::bwt_with(input, bwt_out, &mut self.bwt_scratch)?;

        // Write the header, then entropy-code the transform from the
        // canonical start state straight after it.
        out.try_reserve(model.header_len() + crate::compress_bound(cur_len))
            .map_err(|_| Error::OutOfMemory)?;
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&bwt_idx.to_le_bytes());
        out.push(model.0);
        if model.has_lzp() {
            out.extend_from_slice(&lzp_size.to_le_bytes());
        }
        if model.has_rle() {
            out.extend_from_slice(&rle_size.to_le_bytes());
        }
        self.coder.encode(&bwt_out[..cur_len], out);
        Ok(())
    }
}

/// Buffer that holds the current stage output.
#[derive(Clone, Copy)]
enum StageBuffer {
    /// The caller block, not yet transformed.
    Input,
    /// The first stage buffer.
    A,
    /// The second stage buffer.
    B,
}

/// Pick the input slice of the next stage and its free output buffer.
///
/// Return the buffer tag of that output too. The output never aliases
/// the input.
fn stage_buffers<'a>(
    src: StageBuffer,
    len: usize,
    data: &'a [u8],
    a: &'a mut [u8],
    b: &'a mut [u8],
) -> (&'a [u8], &'a mut [u8], StageBuffer) {
    match src {
        StageBuffer::Input => (data, a, StageBuffer::A),
        StageBuffer::A => (&a[..len], b, StageBuffer::B),
        StageBuffer::B => (&b[..len], a, StageBuffer::A),
    }
}

/// Encode one block through the stored-raw layout.
///
/// The caller routes inputs below [`MIN_CODED_LEN`] bytes here.
/// Append the block to `out`.
fn encode_stored(data: &[u8], out: &mut Vec<u8>) -> Result<()> {
    out.try_reserve(data.len() + STORED_PREFIX_LEN)
        .map_err(|_| Error::OutOfMemory)?;
    out.extend_from_slice(&crc32::sum(1, data).to_le_bytes());
    out.extend_from_slice(&STORED_MARKER.to_le_bytes());
    out.extend_from_slice(data);
    Ok(())
}

/// Fields read from the fixed prefix of one block payload.
#[derive(Debug, Clone, Copy)]
struct PayloadPrefix {
    /// Claimed original size, at most the block size.
    original_len: usize,
    /// CRC32 stored in the payload.
    crc: u32,
    /// BWT index, or [`STORED_MARKER`] for a raw block.
    bwt_index: i32,
    /// Bound of the block size, the decode buffer size.
    limit: usize,
}

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
/// This type mirrors [`BlockEncoder`] on the decode side and never touches
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
    use super::BlockEncoder;
    use super::BlockStatus;
    use super::MIN_CODED_LEN;
    use crate::BlockSize;
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
    fn stored_layout_matches_reference() {
        let out = small_state().encode_block(b"A").unwrap();
        assert_eq!(out.len(), 9);
        assert_eq!(
            u32::from_le_bytes(out[..4].try_into().unwrap()),
            0x417B_1DBC
        );
        assert_eq!(out[4..8], [0xFF; 4]);
        assert_eq!(out[8..], [b'A']);
    }

    #[test]
    fn small_block_boundary_routes_stored_vs_coded() {
        let mut state = small_state();
        for len in 0..MIN_CODED_LEN {
            let out = state.encode_block(&vec![b'q'; len]).unwrap();
            assert_eq!(out.len(), len + 8, "stored len {len}");
            assert_eq!(out[4..8], [0xFF; 4], "stored marker len {len}");
        }
        // `MIN_CODED_LEN` itself leaves the stored route: the RLE stage
        // collapses the run, so a model byte replaces the raw layout.
        let out = state.encode_block(&[b'q'; MIN_CODED_LEN]).unwrap();
        assert_eq!(out[8], 0x04, "RLE-only model {}", out[8]);
        assert_ne!(out[4..8], [0xFF; 4]);
    }

    #[test]
    fn oversize_input_fails_data_too_big() {
        let mut state = small_state();
        let big = vec![0_u8; usize::try_from(BlockSize::MIN.get()).unwrap() + 1];
        assert!(matches!(
            state.encode_block(&big),
            Err(crate::error::Error::BlockTooLarge)
        ));
    }

    #[test]
    fn repeated_bytes_set_rle_model_bit() {
        let mut state = small_state();
        let out = state.encode_block(&[b'A'; 1000]).unwrap();
        assert_eq!(out[8] & 0x04, 0x04, "RLE bit set, model {}", out[8]);
    }

    #[test]
    fn periodic_text_sets_lzp_without_rle() {
        // A long-period repeat compresses via LZP but has no runs for RLE.
        let phrase: Vec<u8> = (0..50_u8).collect();
        let mut input = Vec::new();
        for _ in 0..4 {
            input.extend_from_slice(&phrase);
        }
        assert!(input.len() >= 72);
        let out = small_state().encode_block(&input).unwrap();
        assert_eq!(out[8] & 0x06, 0x02, "LZP-only model {}", out[8]);
    }

    #[test]
    fn random_bytes_take_plain_coded_path() {
        // Fixed-seed filler with no runs and no long matches.
        let mut seed = 0x00C0_FFEE_u32;
        let input: Vec<u8> = (0..1000)
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                u8::try_from((seed >> 16) & 0xFF).expect("masked byte fits")
            })
            .collect();
        let out = small_state().encode_block(&input).unwrap();
        assert_eq!(out[8], 0, "plain model {}", out[8]);
        // Model zero means no size fields: the coded stream starts at 9.
        assert!(out.len() > 9);
    }

    #[test]
    fn state_reuse_gives_identical_bytes() {
        let mut state = small_state();
        let first = state.encode_block(&[b'A'; 1000]).unwrap();
        let _ = state.encode_block(&[b'B'; 500]).unwrap();
        let again = state.encode_block(&[b'A'; 1000]).unwrap();
        assert_eq!(first, again);
    }

    #[test]
    fn block_size_accessor_matches_constructor() {
        assert_eq!(small_state().block_size(), BlockSize::MIN);
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
        let head = super::PayloadPrefix {
            original_len: 4,
            crc: 0,
            bwt_index: -1,
            limit: 64,
        };
        let mut out = [0_u8; 4];
        assert!(matches!(
            super::decode_stored(&[0_u8; 4], &head, &mut out),
            Err(Error::TruncatedHeader)
        ));
    }
}
