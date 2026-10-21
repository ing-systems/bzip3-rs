//! One-shot 13-byte framed format plus streaming 9-byte framed format.
//!
//! The one-shot header holds the magic, the effective block size, and
//! the block count. Each block holds its compressed size, its original
//! size, and the payload. Empty input yields zero blocks. Trailing
//! bytes after the last block are ignored on decode.
//!
//! The streaming header holds only the magic and the configured block
//! size. Blocks follow until the input ends. The streaming decoder
//! rejects trailing bytes that do not form a full valid block.

use crate::BlockSize;
use crate::Jobs;
use crate::MAGIC_NUMBER;
use crate::block::BlockDecoder;
use crate::block::BlockEncoder;
use crate::compress_bound;
use crate::error::Error;
use crate::error::Result;
use crate::stream::decode_blocks;
use crate::stream::encode_blocks;
use crate::stream::run_in_pool;

/// Length of the frame header (magic plus two sizes).
pub const ONE_SHOT_HEADER_LEN: usize = 13;

/// Sizes of one framed block, shared by both framings.
///
/// Each block on the wire holds the compressed size, the original
/// size, and the payload, in that order.
///
/// # Examples
///
/// ```
/// use bzip3_rs::BlockHeader;
///
/// let header = BlockHeader::new(10, 100);
/// let bytes = header.to_bytes();
/// assert_eq!(bytes.len(), BlockHeader::LEN);
/// assert_eq!(BlockHeader::from_bytes(&bytes), header);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHeader {
    /// Compressed payload bytes after this header.
    pub compressed_size: u32,
    /// Original bytes the payload decodes to.
    pub original_size: u32,
}

impl BlockHeader {
    /// Length of one block header on the wire.
    pub const LEN: usize = 8;

    /// Build a block header from both sizes.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::BlockHeader;
    ///
    /// let header = BlockHeader::new(10, 100);
    /// assert_eq!(header.compressed_size, 10);
    /// assert_eq!(header.original_size, 100);
    /// ```
    #[must_use]
    #[inline]
    pub const fn new(compressed_size: u32, original_size: u32) -> Self {
        Self {
            compressed_size,
            original_size,
        }
    }

    /// Build the header for one encoded block.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BlockTooLarge`] when a size does not fit 32 bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::BlockHeader;
    ///
    /// # fn main() -> Result<(), bzip3_rs::error::Error> {
    /// let header = BlockHeader::for_block(10, 100)?;
    /// assert_eq!(header, BlockHeader::new(10, 100));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn for_block(payload_len: usize, original_len: usize) -> Result<Self> {
        let comp = u32::try_from(payload_len).map_err(|_| Error::BlockTooLarge)?;
        let orig = u32::try_from(original_len).map_err(|_| Error::BlockTooLarge)?;
        Ok(Self::new(comp, orig))
    }

    /// Encode the header as 8 little-endian bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::BlockHeader;
    ///
    /// let bytes = BlockHeader::new(1, 2).to_bytes();
    /// assert_eq!(bytes, [1, 0, 0, 0, 2, 0, 0, 0]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn to_bytes(self) -> [u8; 8] {
        let comp = self.compressed_size.to_le_bytes();
        let orig = self.original_size.to_le_bytes();
        [
            comp[0], comp[1], comp[2], comp[3], orig[0], orig[1], orig[2], orig[3],
        ]
    }

    /// Decode a header from 8 wire bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::BlockHeader;
    ///
    /// let header = BlockHeader::from_bytes(&[1, 0, 0, 0, 2, 0, 0, 0]);
    /// assert_eq!(header, BlockHeader::new(1, 2));
    /// ```
    #[must_use]
    #[inline]
    pub const fn from_bytes(bytes: &[u8; 8]) -> Self {
        Self {
            compressed_size: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            original_size: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        }
    }
}

/// One parsed streaming block borrowing its payload.
pub struct FramedBlock<'a> {
    /// Compressed and original sizes from the wire.
    pub header: BlockHeader,
    /// Payload bytes (`compressed_size` long).
    pub payload: &'a [u8],
}

/// Streaming frame header: the magic bytes and the block size.
///
/// The header is 9 bytes long. It records the requested block size
/// verbatim. No clamp applies: short inputs still encode under the
/// configured size. Blocks follow it until the input ends.
///
/// # Examples
///
/// ```
/// use bzip3_rs::{BlockSize, MAGIC_NUMBER, StreamHeader};
///
/// # fn main() -> Result<(), bzip3_rs::error::Error> {
/// let header = StreamHeader::new(BlockSize::MIN);
/// let bytes = header.to_bytes();
/// assert_eq!(bytes.len(), StreamHeader::LEN);
/// assert!(bytes.starts_with(&MAGIC_NUMBER));
/// assert_eq!(StreamHeader::from_bytes(&bytes)?, header);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamHeader {
    /// Largest original block size in the stream.
    pub block_size: BlockSize,
}

impl StreamHeader {
    /// Length of the header in bytes.
    pub const LEN: usize = 9;

    /// Build a header for `block_size`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::{BlockSize, StreamHeader};
    ///
    /// let header = StreamHeader::new(BlockSize::DEFAULT);
    /// assert_eq!(header.block_size, BlockSize::DEFAULT);
    /// ```
    #[must_use]
    #[inline]
    pub const fn new(block_size: BlockSize) -> Self {
        Self { block_size }
    }

    /// Encode the header into its 9 bytes.
    ///
    /// The first 5 bytes are the magic number. The last 4 bytes are the
    /// block size as a little-endian integer.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::{BlockSize, StreamHeader};
    ///
    /// let bytes = StreamHeader::new(BlockSize::MIN).to_bytes();
    /// assert_eq!(&bytes[..5], b"BZ3v1");
    /// assert_eq!(bytes[5..], BlockSize::MIN.get().to_le_bytes());
    /// ```
    #[must_use]
    pub const fn to_bytes(self) -> [u8; Self::LEN] {
        let size = self.block_size.get().to_le_bytes();
        [
            MAGIC_NUMBER[0],
            MAGIC_NUMBER[1],
            MAGIC_NUMBER[2],
            MAGIC_NUMBER[3],
            MAGIC_NUMBER[4],
            size[0],
            size[1],
            size[2],
            size[3],
        ]
    }

    /// Parse the 9 header bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidMagic`] on bad magic,
    /// [`Error::InvalidBlockSize`] on a block size outside the valid range.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::error::Error;
    /// use bzip3_rs::{BlockSize, StreamHeader};
    ///
    /// # fn main() -> Result<(), Error> {
    /// let bytes = StreamHeader::new(BlockSize::MIN).to_bytes();
    /// assert_eq!(StreamHeader::from_bytes(&bytes)?.block_size, BlockSize::MIN);
    ///
    /// let bad = [0_u8; StreamHeader::LEN];
    /// assert!(matches!(StreamHeader::from_bytes(&bad), Err(Error::InvalidMagic)));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn from_bytes(bytes: &[u8; Self::LEN]) -> Result<Self> {
        if bytes[..5] != MAGIC_NUMBER {
            return Err(Error::InvalidMagic);
        }
        Ok(Self::new(block_size_from_field(&bytes[5..9])?))
    }

    /// Parse the header at the start of `data`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TruncatedHeader`] on a short header,
    /// [`Error::InvalidMagic`] on bad magic,
    /// [`Error::InvalidBlockSize`] on a block size outside the valid range.
    pub(crate) fn parse(data: &[u8]) -> Result<Self> {
        let Some(bytes) = data.first_chunk::<{ Self::LEN }>() else {
            return Err(Error::TruncatedHeader);
        };
        Self::from_bytes(bytes)
    }
}

/// Read a block size field. Any invalid value fails with [`Error::InvalidBlockSize`].
fn block_size_from_field(bytes: &[u8]) -> Result<BlockSize> {
    let value = u32::try_from(read_i32(bytes)).map_err(|_| Error::InvalidBlockSize)?;
    BlockSize::new(value).map_err(|_| Error::InvalidBlockSize)
}

/// Parse and check the streaming block header at the start of `bytes`.
///
/// `bytes` must hold at least [`BlockHeader::LEN`] bytes.
///
/// # Errors
///
/// Returns [`Error::InvalidHeader`] on a payload above `compress_bound` or an
/// original size above the block size. A negative wire size reads as a
/// value above both limits, so it fails the same way.
pub fn parse_block_header(bytes: &[u8], block_size: BlockSize) -> Result<BlockHeader> {
    let header = BlockHeader::new(read_u32(bytes), read_u32(&bytes[4..]));
    if header.compressed_size as usize > compress_bound(block_size.as_usize())
        || header.original_size > block_size.get()
    {
        return Err(Error::InvalidHeader);
    }
    Ok(header)
}

/// Walk the blocks of streaming input, one at a time.
///
/// Blocks borrow their payloads, so no copy occurs. The walk stops
/// after its first error.
pub struct StreamBlocks<'a> {
    /// Bytes after the last block that the walk returned.
    rest: &'a [u8],
    /// Block size from the stream header.
    block_size: BlockSize,
}

impl<'a> StreamBlocks<'a> {
    /// Check the 9-byte stream header and start the walk after it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TruncatedHeader`] on a short header.
    /// Returns [`Error::InvalidMagic`] on bad magic.
    /// Returns [`Error::InvalidBlockSize`] on an invalid header block size.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let block_size = StreamHeader::parse(data)?.block_size;
        Ok(Self {
            rest: &data[StreamHeader::LEN..],
            block_size,
        })
    }

    /// Return the block size from the stream header.
    #[must_use]
    pub const fn block_size(&self) -> BlockSize {
        self.block_size
    }

    /// Split the next block off the input.
    fn split(&self) -> Result<(FramedBlock<'a>, &'a [u8])> {
        let Some(block_header) = self.rest.get(..BlockHeader::LEN) else {
            return Err(Error::TruncatedHeader);
        };
        let header = parse_block_header(block_header, self.block_size)?;
        let end = BlockHeader::LEN + header.compressed_size as usize;
        let Some(payload) = self.rest.get(BlockHeader::LEN..end) else {
            return Err(Error::TruncatedData);
        };
        Ok((FramedBlock { header, payload }, &self.rest[end..]))
    }
}

impl<'a> Iterator for StreamBlocks<'a> {
    type Item = Result<FramedBlock<'a>>;

    /// Return the next block, or the error that ends the walk.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidHeader`] on bad sizes.
    /// Returns [`Error::TruncatedHeader`] on trailing bytes
    /// below a full block header. Returns [`Error::TruncatedData`] on a
    /// short payload.
    fn next(&mut self) -> Option<Self::Item> {
        if self.rest.is_empty() {
            return None;
        }
        match self.split() {
            Ok((block, rest)) => {
                self.rest = rest;
                Some(Ok(block))
            }
            Err(fault) => {
                self.rest = &[];
                Some(Err(fault))
            }
        }
    }
}

/// Split streaming input into its header block size and blocks.
///
/// # Errors
///
/// Returns [`Error::TruncatedHeader`] on a short header or trailing bytes
/// below a full block header. Returns [`Error::InvalidMagic`] on bad magic.
/// Returns [`Error::InvalidHeader`] on bad sizes.
/// Returns [`Error::TruncatedData`] on a short payload.
/// Returns [`Error::InvalidBlockSize`] on an invalid header block size.
pub fn split_stream(data: &[u8]) -> Result<(BlockSize, Vec<FramedBlock<'_>>)> {
    let walk = StreamBlocks::new(data)?;
    let block_size = walk.block_size();
    Ok((block_size, walk.collect::<Result<_>>()?))
}

/// Encode `chunk` through `state` and append header plus payload to `out`.
///
/// The payload goes straight into `out`. The block header takes its
/// place after the payload size is known, so no copy occurs.
///
/// # Errors
///
/// Returns the codec fault of [`BlockEncoder::encode_block_into`], or
/// [`Error::BlockTooLarge`] when a size does not fit 32 bits. On an error,
/// `out` keeps its length.
pub fn append_framed_block(
    state: &mut BlockEncoder,
    chunk: &[u8],
    out: &mut Vec<u8>,
) -> Result<()> {
    let at = out.len();
    out.extend_from_slice(&[0_u8; BlockHeader::LEN]);
    let outcome = state
        .encode_block_into(chunk, out)
        .and_then(|()| BlockHeader::for_block(out.len() - at - BlockHeader::LEN, chunk.len()));
    match outcome {
        Ok(header) => {
            out[at..at + BlockHeader::LEN].copy_from_slice(&header.to_bytes());
            Ok(())
        }
        Err(e) => {
            out.truncate(at);
            Err(e)
        }
    }
}

/// Copy the first four bytes of a slice. Panics on a shorter slice.
const fn first_four(bytes: &[u8]) -> [u8; 4] {
    [bytes[0], bytes[1], bytes[2], bytes[3]]
}

/// Read a little-endian `i32` from a slice prefix.
///
/// # Panics
///
/// Panics when `bytes` holds fewer than four bytes.
pub const fn read_i32(bytes: &[u8]) -> i32 {
    i32::from_le_bytes(first_four(bytes))
}

/// Read a little-endian `u32` from a slice prefix.
///
/// # Panics
///
/// Panics when `bytes` holds fewer than four bytes.
pub const fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(first_four(bytes))
}

/// Mirror the block-size clamp for one-shot encode.
///
/// When the requested size exceeds the input, encode with
/// `compress_bound(input)` clamped to a minimum of 65 KiB and record that value.
fn clamp_block_size(requested: BlockSize, input_len: usize) -> Result<BlockSize> {
    let size = if requested.as_usize() > input_len {
        compress_bound(input_len).max(BlockSize::MIN.as_usize())
    } else {
        requested.as_usize()
    };
    BlockSize::new(u32::try_from(size).map_err(|_| Error::InvalidBlockSize)?)
}

/// Encode one-shot framed output for `data`.
///
/// `jobs` follows the rule of [`crate::stream::compress_slice`]:
/// `Jobs::SERIAL` runs serially, with one state that encodes every block
/// and reuses its working memory. Other counts encode blocks on a
/// pool. Output bytes are identical at every count. Blocks at or
/// above 64 bytes run the full pipeline.
///
/// # Errors
///
/// Returns [`Error::InvalidBlockSize`] when the clamped size leaves the valid range,
/// [`Error::OutOfMemory`] when the encoder state allocation fails,
/// [`Error::BlockTooLarge`] when a block exceeds the effective size,
/// [`Error::BwtFailed`] when a transform fails,
/// [`Error::ThreadPoolFailed`] when the pool build fails.
pub fn encode_frame(data: &[u8], block_size: BlockSize, jobs: Jobs) -> Result<Vec<u8>> {
    let effective = clamp_block_size(block_size, data.len())?;
    let width = effective.as_usize();
    let block_count = data.len().div_ceil(width);
    let count_field = u32::try_from(block_count).map_err(|_| Error::BlockTooLarge)?;

    let mut out =
        Vec::with_capacity(ONE_SHOT_HEADER_LEN + block_count * BlockHeader::LEN + data.len() + 8);
    out.extend_from_slice(&StreamHeader::new(effective).to_bytes());
    out.extend_from_slice(&count_field.to_le_bytes());

    if jobs.is_serial() || block_count < 2 {
        let mut state = BlockEncoder::new(effective)?;
        for chunk in data.chunks(width) {
            append_framed_block(&mut state, chunk, &mut out)?;
        }
    } else {
        let parts = run_in_pool(jobs, block_count, || encode_blocks(data, effective))?;
        for part in &parts {
            out.extend_from_slice(part);
        }
    }
    Ok(out)
}

/// Decode one-shot framed input, ignoring trailing bytes.
///
/// `jobs` follows the rule of [`crate::stream::compress_slice`].
/// Output bytes and the reported error are identical at every count.
/// A block fault comes before a header fault of a later block, as in
/// the serial path.
///
/// # Errors
///
/// Returns [`Error::TruncatedHeader`] on a short header,
/// [`Error::InvalidMagic`] on bad magic,
/// [`Error::InvalidHeader`] on bad block sizes, [`Error::InvalidBlockSize`] on a bad header block size,
/// [`Error::TruncatedData`] on a short payload,
/// [`Error::OutOfMemory`] when a decoder state fails to build,
/// [`Error::ThreadPoolFailed`] when the pool build fails,
/// and the block error when a payload fails to decode.
pub fn decode_frame(data: &[u8], jobs: Jobs) -> Result<Vec<u8>> {
    let Some(header) = data.get(..ONE_SHOT_HEADER_LEN) else {
        return Err(Error::TruncatedHeader);
    };
    let block_size = StreamHeader::parse(header)?.block_size;
    let block_count = read_u32(&header[StreamHeader::LEN..]);
    let limit = compress_bound(block_size.as_usize());

    // The serial path decodes each block after its split. A block fault
    // still comes before a header fault of a later block.
    if jobs.is_serial() {
        let mut decoder = BlockDecoder::new(block_size)?;
        let mut out = Vec::new();
        let mut rest = &data[ONE_SHOT_HEADER_LEN..];
        for _ in 0..block_count {
            let (block, next) = split_block(rest, block_size, limit)?;
            decoder.decode_block_into(block.payload, block.header.original_size, &mut out)?;
            rest = next;
        }
        return Ok(out);
    }

    // Walk the headers first. A header fault stops the walk, but the
    // blocks before it still decode before the fault shows.
    let mut rest = &data[ONE_SHOT_HEADER_LEN..];
    let mut blocks = Vec::new();
    let mut walk_fault = None;
    for _ in 0..block_count {
        match split_block(rest, block_size, limit) {
            Ok((block, next)) => {
                blocks.push(block);
                rest = next;
            }
            Err(fault) => {
                walk_fault = Some(fault);
                break;
            }
        }
    }

    let mut out = Vec::new();
    if blocks.len() < 2 {
        let mut decoder = BlockDecoder::new(block_size)?;
        for block in &blocks {
            decoder.decode_block_into(block.payload, block.header.original_size, &mut out)?;
        }
    } else {
        out = run_in_pool(jobs, blocks.len(), || decode_blocks(&blocks, block_size))?;
    }
    walk_fault.map_or(Ok(out), Err)
}

/// Split the first one-shot block off `rest`.
///
/// Returns the block and the bytes after it. The checks keep the
/// reference order.
///
/// # Errors
///
/// Returns [`Error::TruncatedHeader`] on a short header,
/// [`Error::InvalidHeader`] on bad sizes,
/// [`Error::TruncatedData`] on a short payload.
fn split_block(
    rest: &[u8],
    block_size: BlockSize,
    limit: usize,
) -> Result<(FramedBlock<'_>, &[u8])> {
    let Some(block_header) = rest.get(..BlockHeader::LEN) else {
        return Err(Error::TruncatedHeader);
    };
    // A negative wire size reads as a value above both limits.
    let header = BlockHeader::new(read_u32(block_header), read_u32(&block_header[4..]));
    let comp_len = header.compressed_size as usize;
    if comp_len > limit {
        return Err(Error::InvalidHeader);
    }
    let Some(payload) = rest.get(BlockHeader::LEN..BlockHeader::LEN + comp_len) else {
        return Err(Error::TruncatedData);
    };
    if header.original_size > block_size.get() {
        return Err(Error::InvalidHeader);
    }
    Ok((
        FramedBlock { header, payload },
        &rest[BlockHeader::LEN + comp_len..],
    ))
}

#[cfg(test)]
mod tests {
    use super::BlockHeader;
    use super::StreamBlocks;
    use super::StreamHeader;
    use super::append_framed_block;
    use super::decode_frame;
    use super::encode_frame;
    use super::parse_block_header;
    use super::split_stream;
    use crate::BlockSize;
    use crate::Jobs;
    use crate::block::BlockEncoder;
    use crate::compress_bound;
    use crate::error::Error;
    use crate::stream;

    #[test]
    fn empty_yields_zero_block_header() {
        let out = encode_frame(b"", BlockSize::DEFAULT, Jobs::SERIAL).unwrap();
        assert_eq!(out.len(), 13);
        assert_eq!(&out[..5], b"BZ3v1");
        assert_eq!(u32::from_le_bytes(out[5..9].try_into().unwrap()), 65 * 1024);
        assert_eq!(u32::from_le_bytes(out[9..13].try_into().unwrap()), 0);
        assert_eq!(decode_frame(&out, Jobs::SERIAL).unwrap(), b"");
    }

    #[test]
    fn short_inputs_use_one_stored_block() {
        let out = encode_frame(b"A", BlockSize::DEFAULT, Jobs::SERIAL).unwrap();
        assert_eq!(u32::from_le_bytes(out[9..13].try_into().unwrap()), 1);
        assert_eq!(decode_frame(&out, Jobs::SERIAL).unwrap(), b"A");
    }

    #[test]
    fn truncated_payload_reports_truncation() {
        let mut out = encode_frame(b"A", BlockSize::DEFAULT, Jobs::SERIAL).unwrap();
        out.truncate(out.len() - 2);
        assert!(matches!(
            decode_frame(&out, Jobs::SERIAL),
            Err(Error::TruncatedData)
        ));
    }

    #[test]
    fn negative_block_size_fails_invalid_block_size() {
        let mut out = encode_frame(b"A", BlockSize::DEFAULT, Jobs::SERIAL).unwrap();
        out[5..9].copy_from_slice(&(-1_i32).to_le_bytes());
        assert!(matches!(
            decode_frame(&out, Jobs::SERIAL),
            Err(Error::InvalidBlockSize)
        ));
    }

    #[test]
    fn for_block_rejects_sizes_above_u32() {
        let huge = u32::MAX as usize + 1;
        assert!(matches!(
            BlockHeader::for_block(huge, 0),
            Err(Error::BlockTooLarge)
        ));
        assert!(matches!(
            BlockHeader::for_block(0, huge),
            Err(Error::BlockTooLarge)
        ));
        assert!(BlockHeader::for_block(10, 100).is_ok());
    }

    #[test]
    fn parse_block_header_rejects_oversize() {
        let size = BlockSize::MIN;
        let limit = compress_bound(size.as_usize());
        let mut bad_comp = Vec::new();
        bad_comp.extend_from_slice(&u32::try_from(limit + 1).unwrap().to_le_bytes());
        bad_comp.extend_from_slice(&0_u32.to_le_bytes());
        assert!(matches!(
            parse_block_header(&bad_comp, size),
            Err(Error::InvalidHeader)
        ));
        let mut bad_orig = Vec::new();
        bad_orig.extend_from_slice(&0_u32.to_le_bytes());
        bad_orig.extend_from_slice(&(size.get() + 1).to_le_bytes());
        assert!(matches!(
            parse_block_header(&bad_orig, size),
            Err(Error::InvalidHeader)
        ));
        let mut ok = Vec::new();
        ok.extend_from_slice(&0_u32.to_le_bytes());
        ok.extend_from_slice(&size.get().to_le_bytes());
        assert!(parse_block_header(&ok, size).is_ok());
    }

    #[test]
    fn split_stream_rejects_short_inputs() {
        assert!(matches!(split_stream(&[]), Err(Error::TruncatedHeader)));
        assert!(matches!(
            split_stream(b"BZ3v1"),
            Err(Error::TruncatedHeader)
        ));
        let mut coded = Vec::new();
        stream::compress_slice(b"hi", &mut coded, BlockSize::MIN, Jobs::SERIAL).unwrap();
        let (size, blocks) = split_stream(&coded).unwrap();
        assert_eq!(size, BlockSize::MIN);
        assert_eq!(blocks.len(), 1);
        let mut cut = coded.clone();
        cut.truncate(StreamHeader::LEN + BlockHeader::LEN - 1);
        assert!(matches!(split_stream(&cut), Err(Error::TruncatedHeader)));
        let header = BlockHeader::new(10, 2).to_bytes();
        let mut short = StreamHeader::new(BlockSize::MIN).to_bytes().to_vec();
        short.extend_from_slice(&header);
        short.extend_from_slice(&[0_u8; 2]);
        assert!(matches!(split_stream(&short), Err(Error::TruncatedData)));
        let mut bad = StreamHeader::new(BlockSize::MIN).to_bytes().to_vec();
        let limit = compress_bound(BlockSize::MIN.as_usize()) + 1;
        bad.extend_from_slice(&u32::try_from(limit).unwrap().to_le_bytes());
        bad.extend_from_slice(&0_u32.to_le_bytes());
        assert!(matches!(split_stream(&bad), Err(Error::InvalidHeader)));
    }

    #[test]
    fn stream_blocks_empty_rest_yields_none() {
        let mut coded = Vec::new();
        stream::compress_slice(b"", &mut coded, BlockSize::MIN, Jobs::SERIAL).unwrap();
        let mut walk = StreamBlocks::new(&coded).unwrap();
        assert!(walk.next().is_none());
        assert!(walk.next().is_none());
    }

    #[test]
    fn append_framed_block_rolls_back_on_error() {
        let mut state = BlockEncoder::new(BlockSize::MIN).unwrap();
        let big = vec![0_u8; BlockSize::MIN.as_usize() + 1];
        let mut out = vec![9_u8, 9, 9];
        assert!(matches!(
            append_framed_block(&mut state, &big, &mut out),
            Err(Error::BlockTooLarge)
        ));
        assert_eq!(out, vec![9_u8, 9, 9]);
        let mut ok_out = Vec::new();
        append_framed_block(&mut state, b"hi", &mut ok_out).unwrap();
        assert!(ok_out.len() > BlockHeader::LEN);
    }
}
