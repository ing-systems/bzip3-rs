//! Streaming block iteration and batch decode with damage reporting.
//!
//! This module walks the block list of a streaming frame, decodes
//! batches of blocks with one reused decoder per slot, and reports
//! damaged blocks in recover mode. It depends only on [`crate::cli_args`]
//! for the shared error type.

// Visibility stays `pub(crate)` for the run module.
// Reason: the parent module is private, so the
// nursery lint asks for plain `pub`, but `pub(crate)` states the
// crate-wide intent at each item.
#![expect(clippy::redundant_pub_crate)]

use std::ops::Range;

use bzip3_rs::BlockDecoder;
use bzip3_rs::BlockHeader;
use bzip3_rs::BlockSize;
use bzip3_rs::BlockStatus;
use bzip3_rs::StreamHeader;
use bzip3_rs::error::Error;

use crate::cli_args::ExitError;

/// Parse the frame header. Returns the block size.
/// Corrupt shapes (bad signature, truncated header, bad block size)
/// use code 2, like the reference data-error path. Recover mode
/// replaces a bad block size with a size from the block headers.
/// The reference uses the largest block size.
pub(crate) fn parse_stream_header(data: &[u8], recover: bool) -> Result<BlockSize, ExitError> {
    if !data.starts_with(&bzip3_rs::MAGIC_NUMBER) {
        return Err(ExitError::corrupt("Invalid signature."));
    }
    let size = data
        .get(bzip3_rs::MAGIC_NUMBER.len()..)
        .and_then(<[u8]>::first_chunk::<4>)
        .ok_or_else(|| ExitError::corrupt("Error: Corrupt file"))?;
    resolve_header_size(u32::from_le_bytes(*size), data, recover)
}

/// Resolve a raw frame size to a block size.
///
/// A negative wire size reads as a value above the largest block size.
/// Recover mode infers a size from the block headers instead of failing.
fn resolve_header_size(raw: u32, data: &[u8], recover: bool) -> Result<BlockSize, ExitError> {
    match BlockSize::new(raw) {
        Ok(block_size) => Ok(block_size),
        Err(_) if recover => {
            eprintln!("Recovery mode: Proceeding.");
            Ok(infer_block_size(data))
        }
        Err(_) => Err(ExitError::corrupt(
            "The input file is corrupted. Reason: Invalid block size in the header.",
        )),
    }
}

/// Find a block size that fits every block header before the first fault.
///
/// The size is the largest original size or payload size of those
/// blocks, limited to the valid range. One pass reads the headers.
/// `data` must hold the frame header.
fn infer_block_size(data: &[u8]) -> BlockSize {
    let largest = BlockSpans::new(data, BlockSize::MAX)
        .map_while(Result::ok)
        .map(|span| {
            let payload = u32::try_from(span.payload.len()).unwrap_or(u32::MAX);
            span.original_size.max(payload)
        })
        .max()
        .unwrap_or(0);
    let size = largest.clamp(BlockSize::MIN.get(), BlockSize::MAX.get());
    BlockSize::new(size).unwrap_or(BlockSize::MAX)
}

/// One parsed block: claimed original size plus payload range.
pub(crate) struct BlockSpan {
    /// Offset of the block header inside the input.
    pub(crate) offset: usize,
    /// Claimed original size.
    original_size: u32,
    /// Byte range of the payload inside the input.
    pub(crate) payload: Range<usize>,
}

/// Walk the block list of a frame, one `BlockSpan` per block.
///
/// A fault ends the walk after it yields the error once.
pub(crate) struct BlockSpans<'a> {
    /// Whole input, frame header included.
    data: &'a [u8],
    /// Offset of the next block header.
    pos: usize,
    /// Largest payload size of one block.
    max_payload: u32,
    /// Largest original size of one block.
    max_original: u32,
    /// Offset of the block header that caused the fault.
    pub(crate) fault_at: usize,
}

impl<'a> BlockSpans<'a> {
    /// Start the walk behind the frame header.
    pub(crate) fn new(data: &'a [u8], block_size: BlockSize) -> Self {
        Self {
            data,
            pos: StreamHeader::LEN,
            max_payload: u32::try_from(bzip3_rs::compress_bound(block_size.as_usize()))
                .unwrap_or(u32::MAX),
            max_original: block_size.get(),
            fault_at: 0,
        }
    }

    /// Parse the block header at `self.pos`. Truncated shapes and
    /// inconsistent sizes use code 2, like the reference data-error path.
    fn parse(&self) -> Result<BlockSpan, ExitError> {
        let corrupt = || ExitError::corrupt("Error: Corrupt file");
        let rest = &self.data[self.pos..];
        let raw = rest
            .first_chunk::<{ BlockHeader::LEN }>()
            .ok_or_else(corrupt)?;
        // A negative wire size reads as a value above both limits.
        let header = BlockHeader::from_bytes(raw);
        if header.compressed_size > self.max_payload || header.original_size > self.max_original {
            return Err(ExitError::corrupt(
                "Failed to decode a block: Inconsistent headers.",
            ));
        }
        let start = self.pos + BlockHeader::LEN;
        let end = start + header.compressed_size as usize;
        if end > self.data.len() {
            return Err(corrupt());
        }
        Ok(BlockSpan {
            offset: self.pos,
            original_size: header.original_size,
            payload: start..end,
        })
    }
}

impl Iterator for BlockSpans<'_> {
    type Item = Result<BlockSpan, ExitError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos == self.data.len() {
            return None;
        }
        let item = self.parse();
        if item.is_err() {
            self.fault_at = self.pos;
        }
        self.pos = item
            .as_ref()
            .map_or(self.data.len(), |span| span.payload.end);
        Some(item)
    }
}

/// One decoder state with the plain bytes and the result of its last block.
pub(crate) struct DecodeSlot {
    /// Decoder of this slot.
    decoder: BlockDecoder,
    /// Plain bytes of the last block.
    pub(crate) output: Vec<u8>,
    /// Result of the last block.
    pub(crate) result: Result<BlockStatus, Error>,
}

impl DecodeSlot {
    /// Build a slot with a new decoder.
    fn new(block_size: BlockSize) -> Result<Self, Error> {
        Ok(Self {
            decoder: BlockDecoder::new(block_size)?,
            output: Vec::new(),
            result: Ok(BlockStatus::Intact),
        })
    }

    /// Decode one block into `self.output` and record the result.
    ///
    /// Recover mode keeps the bytes of a damaged block in `self.output`.
    fn decode(&mut self, data: &[u8], span: &BlockSpan, recover: bool) {
        self.output.clear();
        let payload = &data[span.payload.clone()];
        self.result = if recover {
            self.decoder
                .recover_block_into(payload, span.original_size, &mut self.output)
        } else {
            self.decoder
                .decode_block_into(payload, span.original_size, &mut self.output)
                .map(|()| BlockStatus::Intact)
        };
    }
}

/// Decode batches of blocks. Each slot reuses its decoder, so uneven
/// blocks cost no new state.
pub(crate) struct BatchDecoder {
    /// Block size of the frame.
    block_size: BlockSize,
    /// Thread count of the pool.
    workers: usize,
    /// True when a damaged block keeps its bytes.
    recover: bool,
    /// One slot for each block of the largest batch.
    slots: Vec<DecodeSlot>,
    /// Pool that the first batch of two blocks or more builds.
    pool: Option<rayon::ThreadPool>,
}

impl BatchDecoder {
    /// Start with no slots and no pool.
    pub(crate) const fn new(block_size: BlockSize, workers: usize, recover: bool) -> Self {
        Self {
            block_size,
            workers,
            recover,
            slots: Vec::new(),
            pool: None,
        }
    }

    /// Grow the slots so one slot exists for each span.
    fn ensure_slots(&mut self, count: usize) -> Result<(), Error> {
        for _ in self.slots.len()..count {
            self.slots.push(DecodeSlot::new(self.block_size)?);
        }
        Ok(())
    }

    /// Take the pool or build it with the worker count.
    fn pool_or_build(&mut self) -> Result<rayon::ThreadPool, Error> {
        match self.pool.take() {
            Some(pool) => Ok(pool),
            None => rayon::ThreadPoolBuilder::new()
                .num_threads(self.workers)
                .build()
                .map_err(|_| Error::ThreadPoolFailed),
        }
    }

    /// Decode `spans` on the pool and keep the pool for the next batch.
    fn decode_parallel(&mut self, data: &[u8], spans: &[BlockSpan]) -> Result<(), Error> {
        use rayon::prelude::*;
        let pool = self.pool_or_build()?;
        let recover = self.recover;
        let slots = &mut self.slots[..spans.len()];
        pool.install(|| {
            slots
                .par_iter_mut()
                .zip(spans.par_iter())
                .for_each(|(slot, span)| slot.decode(data, span, recover));
        });
        self.pool = Some(pool);
        Ok(())
    }

    /// Decode `spans` and return one slot for each span, in span order.
    ///
    /// One span runs on the caller thread. More spans run on the pool.
    /// A slot holds its own failure, so the caller can write the blocks
    /// before it.
    pub(crate) fn decode(
        &mut self,
        data: &[u8],
        spans: &[BlockSpan],
    ) -> Result<&[DecodeSlot], Error> {
        if spans.is_empty() {
            return Ok(&[]);
        }
        self.ensure_slots(spans.len())?;
        if spans.len() == 1 {
            let recover = self.recover;
            self.slots[0].decode(data, &spans[0], recover);
            return Ok(&self.slots[..1]);
        }
        self.decode_parallel(data, spans)?;
        Ok(&self.slots[..spans.len()])
    }
}

/// Print one line about a damaged block in recover mode.
///
/// `start` is the output offset of the block and `len` is its length.
pub(crate) fn report_damage(
    number: u64,
    offset: usize,
    fault: &Error,
    kept: bool,
    start: u64,
    len: usize,
) {
    let range = if len == 0 {
        String::new()
    } else {
        let last = start + len as u64 - 1;
        let state = if kept {
            "keep the decoded bytes"
        } else {
            "are zero"
        };
        format!(" Output bytes {start} to {last} {state}.")
    };
    eprintln!("Writing invalid block {number} at input offset {offset}: {fault}.{range}");
}

#[cfg(test)]
mod tests {
    use super::BatchDecoder;
    use super::BlockSpans;
    use super::infer_block_size;
    use super::parse_stream_header;
    use super::report_damage;
    use bzip3_rs::BlockSize;
    use bzip3_rs::BlockStatus;
    use bzip3_rs::Jobs;

    #[test]
    fn infer_block_size_follows_block_headers() {
        let plain = vec![b'A'; 3 * BlockSize::MIN.as_usize() / 2];
        let coded = bzip3_rs::compress(&plain, BlockSize::MIN).unwrap();
        assert_eq!(infer_block_size(&coded), BlockSize::MIN);
        // A frame header with no blocks.
        assert_eq!(infer_block_size(&coded[..9]), BlockSize::MIN);
        // A block header with an original size above the minimum.
        let mut big = coded[..9].to_vec();
        big.extend_from_slice(&0_u32.to_le_bytes());
        big.extend_from_slice(&(1_u32 << 20).to_le_bytes());
        assert_eq!(infer_block_size(&big), BlockSize::new(1 << 20).unwrap());
        // A block header that claims more bytes than the input holds.
        let mut cut = coded[..9].to_vec();
        cut.extend_from_slice(&u32::MAX.to_le_bytes());
        cut.extend_from_slice(&(1_u32 << 20).to_le_bytes());
        assert_eq!(infer_block_size(&cut), BlockSize::MIN);
    }

    #[test]
    fn block_spans_reports_all_faults() {
        use bzip3_rs::StreamHeader;
        let input = b"hello world ".repeat(200);
        let mut coded = Vec::new();
        bzip3_rs::stream::compress_slice(&input, &mut coded, BlockSize::MIN, Jobs::SERIAL).unwrap();
        let spans = BlockSpans::new(&coded, BlockSize::MIN)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(spans.len(), 1);
        let head_only = &coded[..StreamHeader::LEN];
        assert!(BlockSpans::new(head_only, BlockSize::MIN).next().is_none());
        let cut = &coded[..StreamHeader::LEN + 3];
        let mut walk = BlockSpans::new(cut, BlockSize::MIN);
        assert!(matches!(walk.next(), Some(Err(_))));
        assert!(walk.next().is_none());
        let mut bad_comp = coded.clone();
        bad_comp[StreamHeader::LEN..StreamHeader::LEN + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut walk = BlockSpans::new(&bad_comp, BlockSize::MIN);
        assert!(matches!(walk.next(), Some(Err(_))));
        let mut bad_orig = coded.clone();
        bad_orig[StreamHeader::LEN + 4..StreamHeader::LEN + 8]
            .copy_from_slice(&(BlockSize::MIN.get() + 1).to_le_bytes());
        let mut walk = BlockSpans::new(&bad_orig, BlockSize::MIN);
        assert!(matches!(walk.next(), Some(Err(_))));
        let mut short = coded.clone();
        let keep = short.len() - 1;
        short.truncate(keep);
        let mut walk = BlockSpans::new(&short, BlockSize::MIN);
        assert!(matches!(walk.next(), Some(Err(_))));
        assert!(walk.next().is_none());
    }

    #[test]
    fn batch_decoder_covers_single_empty_and_reuse() {
        let input = vec![b'q'; 3 * BlockSize::MIN.as_usize() + 100];
        let mut coded = Vec::new();
        bzip3_rs::stream::compress_slice(&input, &mut coded, BlockSize::MIN, Jobs::SERIAL).unwrap();
        let spans = BlockSpans::new(&coded, BlockSize::MIN)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(spans.len() >= 3);
        let mut solo = BatchDecoder::new(BlockSize::MIN, 1, false);
        let empty = solo.decode(&coded, &[]).unwrap();
        assert_eq!(empty.len(), 0);
        let one = solo.decode(&coded, &spans[..1]).unwrap();
        assert_eq!(one.len(), 1);
        assert!(matches!(one[0].result, Ok(BlockStatus::Intact)));
        assert_eq!(one[0].output.len(), BlockSize::MIN.as_usize());
        let mut pool = BatchDecoder::new(BlockSize::MIN, 2, false);
        let first = pool.decode(&coded, &spans[..2]).unwrap().len();
        assert_eq!(first, 2);
        let second = pool.decode(&coded, &spans[2..]).unwrap().len();
        assert_eq!(second, spans.len() - 2);
        let mut damaged = coded.clone();
        damaged[spans[0].payload.start] ^= 0xFF;
        let hurt = BlockSpans::new(&damaged, BlockSize::MIN)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let mut plain = BatchDecoder::new(BlockSize::MIN, 1, false);
        let slots = plain.decode(&damaged, &hurt[..1]).unwrap();
        assert!(!matches!(
            slots[0].result,
            Ok(bzip3_rs::BlockStatus::Intact)
        ));
        let mut kept = BatchDecoder::new(BlockSize::MIN, 1, true);
        let slots = kept.decode(&damaged, &hurt[..1]).unwrap();
        assert!(!matches!(slots[0].result, Ok(BlockStatus::Intact)));
    }

    #[test]
    fn report_damage_handles_empty_range() {
        use bzip3_rs::error::Error;
        report_damage(1, 0, &Error::InvalidHeader, true, 0, 0);
        report_damage(2, 9, &Error::ChecksumMismatch, true, 0, 10);
        report_damage(3, 9, &Error::TruncatedData, false, 10, 5);
    }

    #[test]
    fn infer_block_size_clamps_small() {
        use bzip3_rs::StreamHeader;
        let mut data = StreamHeader::new(BlockSize::MIN).to_bytes().to_vec();
        data.extend_from_slice(&0_u32.to_le_bytes());
        data.extend_from_slice(&0_u32.to_le_bytes());
        assert_eq!(infer_block_size(&data), BlockSize::MIN);
    }

    #[test]
    fn stream_header_parse_covers_branches() {
        assert!(parse_stream_header(b"NOPE!", false).is_err());
        assert!(parse_stream_header(b"BZ3v1", false).is_err());
        let mut bad = b"BZ3v1".to_vec();
        bad.extend_from_slice(&0_u32.to_le_bytes());
        assert!(parse_stream_header(&bad, false).is_err());
        let input = b"hello world ".repeat(50);
        let mut coded = Vec::new();
        bzip3_rs::stream::compress_slice(&input, &mut coded, BlockSize::MIN, Jobs::SERIAL).unwrap();
        assert_eq!(parse_stream_header(&coded, false).unwrap(), BlockSize::MIN);
        let mut broken = coded.clone();
        broken[5..9].copy_from_slice(&0_u32.to_le_bytes());
        assert!(parse_stream_header(&broken, true).is_ok());
    }
}
