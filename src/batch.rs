//! Batches of codec states for the streaming adapters.
//!
//! A batch holds one codec state for each worker. The adapter fills
//! the batch with whole blocks, the workers code the blocks together,
//! and the adapter reads the results in block order. The bytes match
//! the serial path at every worker count. States build on first use,
//! so a short stream does not pay for the idle workers.

use std::io::Write;
use std::ops::Range;

use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

use crate::BlockSize;
use crate::Jobs;
use crate::block::BlockDecoder;
use crate::block::BlockEncoder;
use crate::error::Error;
use crate::error::Result;
use crate::frame::BlockHeader;
use crate::frame::append_framed_block;
use crate::frame::parse_block_header;

/// Select the pool and the worker count for `jobs`.
///
/// [`Jobs::Global`] selects the global pool. A count above one
/// builds a private pool. The callers handle the serial count on their
/// own path, so it never reaches this function.
///
/// # Errors
///
/// Returns [`Error::ThreadPoolFailed`] when the private pool fails to build.
pub fn build_pool(jobs: Jobs) -> Result<(Option<ThreadPool>, usize)> {
    match jobs {
        Jobs::Threads(count) if count.get() > 1 => {
            let pool = ThreadPoolBuilder::new()
                .num_threads(count.get())
                .build()
                .map_err(|_| Error::ThreadPoolFailed)?;
            Ok((Some(pool), count.get()))
        }
        _ => Ok((None, rayon::current_num_threads().max(1))),
    }
}

/// Run `work` on the private pool, or on the current pool when `pool` is empty.
#[inline]
pub fn run_on<T: Send>(pool: Option<&ThreadPool>, work: impl FnOnce() -> T + Send) -> T {
    match pool {
        Some(pool) => pool.install(work),
        None => work(),
    }
}

/// One encoder state with its output.
struct EncodeSlot {
    /// Encoder state of this slot.
    encoder: BlockEncoder,
    /// Header plus payload of the last block.
    block_bytes: Vec<u8>,
    /// Outcome of the last block.
    result: Result<()>,
}

/// Encoder states that code up to one block for each worker.
pub struct EncodeBatch {
    /// Private pool, or none for the current pool.
    pool: Option<ThreadPool>,
    /// States built so far.
    slots: Vec<EncodeSlot>,
    /// Block size of every state.
    block_size: BlockSize,
    /// Worker count.
    workers: usize,
}

impl EncodeBatch {
    /// Build an empty batch for `jobs`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ThreadPoolFailed`] when the private pool fails to build.
    pub fn new(block_size: BlockSize, jobs: Jobs) -> Result<Self> {
        let (pool, workers) = build_pool(jobs)?;
        Ok(Self {
            pool,
            slots: Vec::new(),
            block_size,
            workers,
        })
    }

    /// Return the block size of every state.
    #[inline]
    pub const fn block_size(&self) -> BlockSize {
        self.block_size
    }

    /// Return the most plain bytes that one `encode` call takes.
    #[inline]
    pub const fn input_len(&self) -> usize {
        let width = self.block_size.as_usize();
        width.saturating_mul(self.workers)
    }

    /// Encode `data` as up to one block for each worker.
    ///
    /// Return the block count. Read the results with `take_block_bytes`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] when a state fails to build.
    fn encode(&mut self, data: &[u8]) -> Result<usize> {
        let width = self.block_size.as_usize();
        let count = data.len().div_ceil(width);
        debug_assert!(count <= self.workers);
        while self.slots.len() < count {
            self.slots.push(EncodeSlot {
                encoder: BlockEncoder::new(self.block_size)?,
                block_bytes: Vec::new(),
                result: Ok(()),
            });
        }
        let slots = &mut self.slots[..count];
        run_on(self.pool.as_ref(), || {
            slots
                .par_iter_mut()
                .zip(data.par_chunks(width))
                .for_each(|(slot, chunk)| {
                    slot.block_bytes.clear();
                    slot.result =
                        append_framed_block(&mut slot.encoder, chunk, &mut slot.block_bytes);
                });
        });
        Ok(count)
    }

    /// Take the framed bytes of block `index` from the last `encode`.
    ///
    /// # Errors
    ///
    /// Returns the codec fault of that block.
    #[inline]
    fn take_block_bytes(&mut self, index: usize) -> Result<&[u8]> {
        let slot = &mut self.slots[index];
        std::mem::replace(&mut slot.result, Ok(()))?;
        Ok(&slot.block_bytes)
    }

    /// Encode `data` and write the framed blocks to `inner` in order.
    ///
    /// # Errors
    ///
    /// Returns the codec fault of the first failing block, or the
    /// I/O fault.
    pub fn write_to<W: Write>(&mut self, data: &[u8], inner: &mut W) -> Result<()> {
        let count = self.encode(data)?;
        for index in 0..count {
            inner.write_all(self.take_block_bytes(index)?)?;
        }
        Ok(())
    }

    /// Encode `data` and append the framed blocks to `out` in order.
    ///
    /// # Errors
    ///
    /// Returns the codec fault of the first failing block.
    pub fn append_to(&mut self, data: &[u8], out: &mut Vec<u8>) -> Result<()> {
        let count = self.encode(data)?;
        for index in 0..count {
            out.extend_from_slice(self.take_block_bytes(index)?);
        }
        Ok(())
    }
}

/// Encoder states of one adapter: one serial state, or one batch.
pub enum EncoderSet {
    /// One state that codes on the caller thread.
    Serial {
        /// Encoder state.
        state: BlockEncoder,
        /// Framed bytes of one block, reused across blocks.
        block_bytes: Vec<u8>,
    },
    /// One state for each worker.
    Parallel(EncodeBatch),
}

impl EncoderSet {
    /// Build the states for `jobs`. The serial count selects the serial state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] when a state allocation fails or
    /// [`Error::ThreadPoolFailed`] when the pool build fails.
    pub fn new(block_size: BlockSize, jobs: Jobs) -> Result<Self> {
        if jobs.is_serial() {
            Ok(Self::Serial {
                state: BlockEncoder::new(block_size)?,
                block_bytes: Vec::new(),
            })
        } else {
            Ok(Self::Parallel(EncodeBatch::new(block_size, jobs)?))
        }
    }

    /// Return the block size of the states.
    #[inline]
    pub const fn block_size(&self) -> BlockSize {
        match self {
            Self::Serial { state, .. } => state.block_size(),
            Self::Parallel(batch) => batch.block_size(),
        }
    }

    /// Return the most plain bytes that one `write_to` call takes.
    ///
    /// This is one block in the serial mode and one block for each
    /// worker in the parallel mode.
    #[inline]
    pub const fn input_len(&self) -> usize {
        match self {
            Self::Serial { state, .. } => state.block_size().as_usize(),
            Self::Parallel(batch) => batch.input_len(),
        }
    }

    /// Encode `data` (at most `input_len` bytes) and write its framed blocks to `inner`.
    ///
    /// # Errors
    ///
    /// Returns the codec fault of the first failing block, or the
    /// I/O fault.
    pub fn write_to<W: Write>(&mut self, data: &[u8], inner: &mut W) -> Result<()> {
        match self {
            Self::Serial { state, block_bytes } => {
                block_bytes.clear();
                append_framed_block(state, data, block_bytes)?;
                Ok(inner.write_all(block_bytes)?)
            }
            Self::Parallel(batch) => batch.write_to(data, inner),
        }
    }

    /// Encode `data` (at most `input_len` bytes) and append its framed blocks to `out`.
    ///
    /// # Errors
    ///
    /// Returns the codec fault of the first failing block.
    #[inline]
    pub fn append_to(&mut self, data: &[u8], out: &mut Vec<u8>) -> Result<()> {
        match self {
            Self::Serial { state, .. } => append_framed_block(state, data, out),
            Self::Parallel(batch) => batch.append_to(data, out),
        }
    }
}

/// One decoder state with its job and output.
struct DecodeSlot {
    /// Decoder state of this slot.
    decoder: BlockDecoder,
    /// Plain bytes of the last block.
    plain: Vec<u8>,
    /// Payload range of the queued block.
    range: Range<usize>,
    /// Original size of the queued block.
    original_size: u32,
    /// Outcome of the last block.
    result: Result<()>,
}

/// Decoder states that decode up to one block for each worker.
pub struct DecodeBatch {
    /// Private pool, or none for the current pool.
    pool: Option<ThreadPool>,
    /// States built so far.
    slots: Vec<DecodeSlot>,
    /// Block size of every state.
    block_size: BlockSize,
    /// Worker count.
    workers: usize,
    /// Blocks queued now.
    queued: usize,
}

impl DecodeBatch {
    /// Build an empty batch for `jobs`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ThreadPoolFailed`] when the private pool fails to build.
    pub fn new(block_size: BlockSize, jobs: Jobs) -> Result<Self> {
        let (pool, workers) = build_pool(jobs)?;
        Ok(Self {
            pool,
            slots: Vec::new(),
            block_size,
            workers,
            queued: 0,
        })
    }

    /// Return the block size of every state.
    #[inline]
    pub const fn block_size(&self) -> BlockSize {
        self.block_size
    }

    /// Return the most blocks that one batch takes.
    #[inline]
    pub const fn max_blocks(&self) -> usize {
        self.workers
    }

    /// Queue the full blocks of `tail` that start at `start`.
    ///
    /// Stop at the batch capacity, at a partial block, or at a bad
    /// block header. Return the offset after the last queued block
    /// and the header fault, if any. The fault comes after the
    /// queued blocks in stream order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] when a state fails to build.
    pub fn queue_blocks(&mut self, tail: &[u8], start: usize) -> Result<(usize, Option<Error>)> {
        self.queued = 0;
        let mut pos = start;
        while self.queued < self.workers && tail.len() - pos >= BlockHeader::LEN {
            let rest = &tail[pos..];
            let header = match parse_block_header(rest, self.block_size) {
                Ok(header) => header,
                Err(fault) => return Ok((pos, Some(fault))),
            };
            let comp = header.compressed_size as usize;
            if rest.len() - BlockHeader::LEN < comp {
                break;
            }
            let range = pos + BlockHeader::LEN..pos + BlockHeader::LEN + comp;
            self.queue(range, header.original_size)?;
            pos += BlockHeader::LEN + comp;
        }
        Ok((pos, None))
    }

    /// Queue one block.
    fn queue(&mut self, range: Range<usize>, original_size: u32) -> Result<()> {
        if self.slots.len() == self.queued {
            self.slots.push(DecodeSlot {
                decoder: BlockDecoder::new(self.block_size)?,
                plain: Vec::new(),
                range: 0..0,
                original_size: 0,
                result: Ok(()),
            });
        }
        let slot = &mut self.slots[self.queued];
        slot.range = range;
        slot.original_size = original_size;
        self.queued += 1;
        Ok(())
    }

    /// Return the count of queued blocks.
    #[inline]
    pub const fn queued(&self) -> usize {
        self.queued
    }

    /// Decode the queued blocks of `tail` together.
    pub fn decode_queued(&mut self, tail: &[u8]) {
        let slots = &mut self.slots[..self.queued];
        run_on(self.pool.as_ref(), || {
            slots.par_iter_mut().for_each(|slot| {
                slot.plain.clear();
                slot.result = slot.decoder.decode_block_into(
                    &tail[slot.range.clone()],
                    slot.original_size,
                    &mut slot.plain,
                );
            });
        });
    }

    /// Return the start offset of queued block `index` in the tail.
    #[inline]
    pub fn block_start(&self, index: usize) -> usize {
        self.slots[index].range.start - BlockHeader::LEN
    }

    /// Return the end offset of queued block `index` in the tail.
    #[inline]
    pub fn block_end(&self, index: usize) -> usize {
        self.slots[index].range.end
    }

    /// Take the plain bytes of queued block `index` from the last `decode_queued`.
    ///
    /// # Errors
    ///
    /// Returns the codec fault of that block.
    #[inline]
    pub fn take_plain(&mut self, index: usize) -> Result<&[u8]> {
        let slot = &mut self.slots[index];
        std::mem::replace(&mut slot.result, Ok(()))?;
        Ok(&slot.plain)
    }

    /// Decode every full block of `tail[*start..]` and write them to `inner`.
    ///
    /// Move `*start` past each written block. On a fault, `*start`
    /// stays at the failed block, so the caller can drain the written
    /// bytes. A block fault comes before a later header fault.
    ///
    /// # Errors
    ///
    /// Returns the exact codec fault or the I/O fault.
    pub fn write_blocks<W: Write>(
        &mut self,
        tail: &[u8],
        start: &mut usize,
        inner: &mut W,
    ) -> Result<()> {
        loop {
            let (_, fault) = self.queue_blocks(tail, *start)?;
            let queued = self.queued;
            self.decode_queued(tail);
            for index in 0..queued {
                let end = self.block_end(index);
                inner.write_all(self.take_plain(index)?)?;
                *start = end;
            }
            if let Some(fault) = fault {
                return Err(fault);
            }
            if queued < self.workers {
                return Ok(());
            }
        }
    }
}

/// Decoder states of one adapter: one serial state, or one batch.
pub enum DecoderSet {
    /// One state that decodes on the caller thread.
    Serial(BlockDecoder),
    /// One state for each worker.
    Parallel(DecodeBatch),
}

impl DecoderSet {
    /// Build the states for `jobs`. The serial count selects the serial state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] when a state allocation fails or
    /// [`Error::ThreadPoolFailed`] when the pool build fails.
    pub fn new(block_size: BlockSize, jobs: Jobs) -> Result<Self> {
        if jobs.is_serial() {
            Ok(Self::Serial(BlockDecoder::new(block_size)?))
        } else {
            Ok(Self::Parallel(DecodeBatch::new(block_size, jobs)?))
        }
    }

    /// Return the block size of the states.
    #[inline]
    pub const fn block_size(&self) -> BlockSize {
        match self {
            Self::Serial(decoder) => decoder.block_size(),
            Self::Parallel(batch) => batch.block_size(),
        }
    }

    /// Decode every full block of `tail[*start..]` and write the plain bytes to `inner`.
    ///
    /// Move `*start` past each written block. On a fault, `*start`
    /// stays at the failed block, so the caller can drain the written
    /// bytes. A block fault comes before a later header fault. `plain`
    /// is scratch for the serial mode and keeps its capacity.
    ///
    /// # Errors
    ///
    /// Returns the exact codec fault or the I/O fault.
    pub fn write_blocks<W: Write>(
        &mut self,
        tail: &[u8],
        start: &mut usize,
        plain: &mut Vec<u8>,
        inner: &mut W,
    ) -> Result<()> {
        let decoder = match self {
            Self::Serial(decoder) => decoder,
            Self::Parallel(batch) => return batch.write_blocks(tail, start, inner),
        };
        while tail.len() - *start >= BlockHeader::LEN {
            let rest = &tail[*start..];
            let header = parse_block_header(rest, decoder.block_size())?;
            let end = BlockHeader::LEN + header.compressed_size as usize;
            if rest.len() < end {
                break;
            }
            plain.clear();
            decoder.decode_block_into(&rest[BlockHeader::LEN..end], header.original_size, plain)?;
            inner.write_all(plain)?;
            *start += end;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::DecodeBatch;
    use super::DecoderSet;
    use super::EncodeBatch;
    use super::EncoderSet;
    use super::build_pool;
    use super::run_on;
    use crate::BlockSize;
    use crate::Jobs;
    use crate::block::BlockEncoder;
    use crate::error::Error;
    use crate::frame::BlockHeader;
    use crate::frame::append_framed_block;

    // Build one framed block for `data`.
    fn framed_block(data: &[u8]) -> Vec<u8> {
        let mut state = BlockEncoder::new(BlockSize::MIN).expect("state builds");
        let mut out = Vec::new();
        append_framed_block(&mut state, data, &mut out).expect("block codes");
        out
    }

    // Build a header that fails the size check.
    fn bad_header_bytes() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&u32::MAX.to_le_bytes());
        out.extend_from_slice(&0_u32.to_le_bytes());
        out
    }

    #[test]
    fn build_pool_serial_has_no_pool() {
        let (pool, workers) = build_pool(Jobs::SERIAL).expect("pool builds");
        assert!(pool.is_none());
        assert!(workers >= 1);
    }

    #[test]
    fn build_pool_threads_two_builds_pool() {
        let (pool, workers) = build_pool(Jobs::from_count(2)).expect("pool builds");
        assert!(pool.is_some());
        assert_eq!(workers, 2);
    }

    #[test]
    fn build_pool_global_has_no_pool() {
        let (pool, workers) = build_pool(Jobs::Global).expect("pool builds");
        assert!(pool.is_none());
        assert!(workers >= 1);
    }

    #[test]
    fn run_on_without_pool_runs_work() {
        assert_eq!(run_on(None, || 7), 7);
    }

    #[test]
    fn run_on_with_pool_runs_work() {
        let (pool, _) = build_pool(Jobs::from_count(2)).expect("pool builds");
        assert_eq!(run_on(pool.as_ref(), || 7), 7);
    }

    #[test]
    fn encode_batch_reports_block_size_and_input_len() {
        let serial = EncodeBatch::new(BlockSize::MIN, Jobs::SERIAL).expect("batch builds");
        assert_eq!(serial.block_size(), BlockSize::MIN);
        assert!(serial.input_len() >= BlockSize::MIN.as_usize());
        let parallel = EncodeBatch::new(BlockSize::MIN, Jobs::from_count(2)).expect("batch builds");
        assert_eq!(parallel.block_size(), BlockSize::MIN);
        assert_eq!(parallel.input_len(), BlockSize::MIN.as_usize() * 2);
    }

    #[test]
    fn encode_batch_encode_empty_returns_zero() {
        let mut batch = EncodeBatch::new(BlockSize::MIN, Jobs::SERIAL).expect("batch builds");
        assert_eq!(batch.encode(&[]).expect("encode runs"), 0);
    }

    #[test]
    fn encode_batch_write_empty_writes_nothing() {
        let mut batch =
            EncodeBatch::new(BlockSize::MIN, Jobs::from_count(2)).expect("batch builds");
        let mut out = Vec::new();
        batch.write_to(&[], &mut out).expect("write runs");
        assert_eq!(out, [] as [u8; 0]);
    }

    #[test]
    fn encode_batch_append_empty_keeps_out_empty() {
        let mut batch =
            EncodeBatch::new(BlockSize::MIN, Jobs::from_count(2)).expect("batch builds");
        let mut out = vec![1_u8];
        batch.append_to(&[], &mut out).expect("append runs");
        assert_eq!(out, vec![1_u8]);
    }

    #[test]
    fn encode_batch_roundtrip_preserves_bytes() {
        let data = b"hello world ".repeat(500);
        let mut coded = Vec::new();
        let mut batch =
            EncodeBatch::new(BlockSize::MIN, Jobs::from_count(2)).expect("batch builds");
        batch.append_to(&data, &mut coded).expect("encode runs");
        assert_ne!(coded, [] as [u8; 0]);
        let mut plain = Vec::new();
        let mut decode =
            DecodeBatch::new(BlockSize::MIN, Jobs::from_count(2)).expect("batch builds");
        let mut start = 0;
        decode
            .write_blocks(&coded, &mut start, &mut plain)
            .expect("decode runs");
        assert_eq!(start, coded.len());
        assert_eq!(plain, data);
    }

    #[test]
    fn decode_batch_reports_block_size_max_blocks_and_empty_queue() {
        let batch = DecodeBatch::new(BlockSize::MIN, Jobs::from_count(2)).expect("batch builds");
        assert_eq!(batch.block_size(), BlockSize::MIN);
        assert_eq!(batch.max_blocks(), 2);
        assert_eq!(batch.queued(), 0);
    }

    #[test]
    fn decode_batch_block_start_and_end_mark_framed_range() {
        let first = framed_block(b"first block");
        let second = framed_block(b"second block");
        let tail = [first.clone(), second].concat();
        let mut batch =
            DecodeBatch::new(BlockSize::MIN, Jobs::from_count(2)).expect("batch builds");
        let (pos, fault) = batch.queue_blocks(&tail, 0).expect("queue runs");
        assert!(fault.is_none());
        assert_eq!(batch.queued(), 2);
        assert_eq!(batch.block_start(0), 0);
        assert_eq!(batch.block_end(0), first.len());
        assert_eq!(batch.block_start(1), first.len());
        assert_eq!(batch.block_end(1), tail.len());
        assert_eq!(pos, tail.len());
        batch.decode_queued(&tail);
        assert_eq!(batch.take_plain(0).expect("plain reads"), b"first block");
        assert_eq!(batch.take_plain(1).expect("plain reads"), b"second block");
    }

    #[test]
    fn queue_blocks_partial_header_queues_nothing() {
        let mut batch =
            DecodeBatch::new(BlockSize::MIN, Jobs::from_count(2)).expect("batch builds");
        let (pos, fault) = batch.queue_blocks(&[1_u8, 2, 3], 0).expect("queue runs");
        assert!(fault.is_none());
        assert_eq!(pos, 0);
        assert_eq!(batch.queued(), 0);
    }

    #[test]
    fn queue_blocks_short_payload_stops_without_fault() {
        let mut tail = BlockHeader::new(100, 10).to_bytes().to_vec();
        tail.extend_from_slice(&[9_u8; 5]);
        let mut batch =
            DecodeBatch::new(BlockSize::MIN, Jobs::from_count(2)).expect("batch builds");
        let (pos, fault) = batch.queue_blocks(&tail, 0).expect("queue runs");
        assert!(fault.is_none());
        assert_eq!(pos, 0);
        assert_eq!(batch.queued(), 0);
    }

    #[test]
    fn queue_blocks_bad_header_returns_fault() {
        let tail = bad_header_bytes();
        let mut batch =
            DecodeBatch::new(BlockSize::MIN, Jobs::from_count(2)).expect("batch builds");
        let (pos, fault) = batch.queue_blocks(&tail, 0).expect("queue runs");
        assert!(matches!(fault, Some(Error::InvalidHeader)));
        assert_eq!(pos, 0);
        assert_eq!(batch.queued(), 0);
    }

    #[test]
    fn write_blocks_fault_follows_good_blocks() {
        let good = framed_block(b"good bytes");
        let tail = [good.clone(), bad_header_bytes()].concat();
        let mut batch =
            DecodeBatch::new(BlockSize::MIN, Jobs::from_count(2)).expect("batch builds");
        let mut plain = Vec::new();
        let mut start = 0;
        let result = batch.write_blocks(&tail, &mut start, &mut plain);
        assert!(matches!(result, Err(Error::InvalidHeader)));
        assert_eq!(start, good.len());
        assert_eq!(plain, b"good bytes");
    }

    #[test]
    fn write_blocks_partial_tail_returns_ok() {
        let good = framed_block(b"good bytes");
        let tail = [good.clone(), vec![1_u8, 2, 3]].concat();
        let mut batch =
            DecodeBatch::new(BlockSize::MIN, Jobs::from_count(2)).expect("batch builds");
        let mut plain = Vec::new();
        let mut start = 0;
        batch
            .write_blocks(&tail, &mut start, &mut plain)
            .expect("decode runs");
        assert_eq!(start, good.len());
        assert_eq!(plain, b"good bytes");
    }

    #[test]
    fn encoder_set_serial_and_parallel_report_sizes() {
        let serial = EncoderSet::new(BlockSize::MIN, Jobs::SERIAL).expect("set builds");
        assert_eq!(serial.block_size(), BlockSize::MIN);
        assert_eq!(serial.input_len(), BlockSize::MIN.as_usize());
        let parallel = EncoderSet::new(BlockSize::MIN, Jobs::from_count(2)).expect("set builds");
        assert_eq!(parallel.block_size(), BlockSize::MIN);
        assert_eq!(parallel.input_len(), BlockSize::MIN.as_usize() * 2);
    }

    #[test]
    fn encoder_set_parallel_write_matches_serial() {
        let data = b"hello world ".repeat(100);
        let mut serial_out = Vec::new();
        let mut serial = EncoderSet::new(BlockSize::MIN, Jobs::SERIAL).expect("set builds");
        serial
            .append_to(&data, &mut serial_out)
            .expect("encode runs");
        let mut parallel_out = Vec::new();
        let mut parallel =
            EncoderSet::new(BlockSize::MIN, Jobs::from_count(2)).expect("set builds");
        parallel
            .append_to(&data, &mut parallel_out)
            .expect("encode runs");
        assert_eq!(serial_out, parallel_out);
        let mut writer_out = Vec::new();
        parallel
            .write_to(&data, &mut writer_out)
            .expect("write runs");
        assert_eq!(writer_out, parallel_out);
    }

    #[test]
    fn decoder_set_serial_and_parallel_report_block_size() {
        let serial = DecoderSet::new(BlockSize::MIN, Jobs::SERIAL).expect("set builds");
        assert_eq!(serial.block_size(), BlockSize::MIN);
        let parallel = DecoderSet::new(BlockSize::MIN, Jobs::from_count(2)).expect("set builds");
        assert_eq!(parallel.block_size(), BlockSize::MIN);
    }

    #[test]
    fn decoder_set_write_blocks_roundtrips() {
        let data = b"hello world ".repeat(100);
        let mut coded = Vec::new();
        let mut encoders =
            EncoderSet::new(BlockSize::MIN, Jobs::from_count(2)).expect("set builds");
        encoders.append_to(&data, &mut coded).expect("encode runs");
        for jobs in [Jobs::SERIAL, Jobs::from_count(2)] {
            let mut set = DecoderSet::new(BlockSize::MIN, jobs).expect("set builds");
            let mut plain = Vec::new();
            let mut scratch = Vec::new();
            let mut start = 0;
            set.write_blocks(&coded, &mut start, &mut scratch, &mut plain)
                .expect("decode runs");
            assert_eq!(start, coded.len());
            assert_eq!(plain, data);
        }
    }

    #[test]
    fn decoder_set_write_blocks_reports_bad_header() {
        let tail = bad_header_bytes();
        let mut set = DecoderSet::new(BlockSize::MIN, Jobs::from_count(2)).expect("set builds");
        let mut plain = Vec::new();
        let mut scratch = Vec::new();
        let mut start = 0;
        let result = set.write_blocks(&tail, &mut start, &mut scratch, &mut plain);
        assert!(matches!(result, Err(Error::InvalidHeader)));
        assert_eq!(start, 0);
    }
}
