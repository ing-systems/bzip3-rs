//! Forward block encode pipeline with stage selection and stored-raw routing.

use std::fmt::Debug;
use std::fmt::Formatter;

use crate::BlockSize;
use crate::buffer::filled;
use crate::cm::CmCoder;
use crate::crc32;
use crate::error::Error;
use crate::error::Result;
use crate::lzp;
use crate::rle;
use crate::sais;
use crate::sais::BwtScratch;

use super::MIN_CODED_LEN;
use super::STORED_MARKER;
use super::STORED_PREFIX_LEN;
use super::model::Model;

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
/// The caller routes inputs below [`MIN_CODED_LEN`](super::MIN_CODED_LEN) bytes here.
/// Append the block to `out`.
fn encode_stored(data: &[u8], out: &mut Vec<u8>) -> Result<()> {
    out.try_reserve(data.len() + STORED_PREFIX_LEN)
        .map_err(|_| Error::OutOfMemory)?;
    out.extend_from_slice(&crc32::sum(1, data).to_le_bytes());
    out.extend_from_slice(&STORED_MARKER.to_le_bytes());
    out.extend_from_slice(data);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::BlockEncoder;
    use super::MIN_CODED_LEN;
    use crate::BlockSize;

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
}
