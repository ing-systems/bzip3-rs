//! Parallel streaming compression over the 9-byte frame.
//!
//! These functions move bulk bytes through rayon and keep output
//! byte-identical to the serial path at any job count. Each block
//! runs under its own codec state, and blocks join in order. Codec
//! states share no mutable data, so `Send` holds by safe construction
//! under `forbid(unsafe_code)`.
//!
//! The functions read and write the streaming frame. See
//! [Frame formats](crate#frame-formats) for the difference from the one-shot frame.

use std::io::Read;
use std::io::Write;
use std::ops::Range;

use rayon::prelude::*;

use crate::BlockSize;
use crate::Jobs;
use crate::batch::build_pool;
use crate::batch::run_on;
use crate::block::BlockDecoder;
use crate::block::BlockEncoder;
use crate::buffer::filled;
use crate::error::Error;
use crate::error::Result;
use crate::frame::BlockHeader;
use crate::frame::FramedBlock;
use crate::frame::StreamBlocks;
use crate::frame::StreamHeader;
use crate::frame::append_framed_block;
use crate::frame::split_stream;

/// Read one reader fully into memory.
///
/// # Errors
///
/// Returns [`Error::Io`] when the read fails.
#[inline]
fn read_fully<R: Read>(mut source: R) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    source.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Read up to `width` bytes of `source` into `buf`, after it clears `buf`.
///
/// A result shorter than `width` means the source ended.
///
/// # Errors
///
/// Returns [`Error::Io`] when the read fails.
#[inline]
fn read_chunk<R: Read>(source: &mut R, width: usize, buf: &mut Vec<u8>) -> Result<()> {
    buf.clear();
    source.take(width as u64).read_to_end(buf)?;
    Ok(())
}

/// Write one slice fully and report its length.
///
/// # Errors
///
/// Returns [`Error::Io`] when the write fails.
#[inline]
fn write_counted<W: Write>(dest: &mut W, bytes: &[u8]) -> Result<u64> {
    dest.write_all(bytes)?;
    Ok(bytes.len() as u64)
}

/// Split `len` items into at most `parts` contiguous non-empty ranges.
///
/// Workers take whole ranges, so uneven blocks still keep every worker
/// busy while states stay worker-local.
fn partition(len: usize, parts: usize) -> Vec<Range<usize>> {
    let parts = parts.max(1).min(len.max(1));
    let width = len.div_ceil(parts);
    (0..parts)
        .map(|index| index * width..((index + 1) * width).min(len))
        .filter(|range| !range.is_empty())
        .collect()
}

/// Encode one input chunk with a reused state into a new framed block.
fn encode_one(state: &mut BlockEncoder, chunk: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.try_reserve(crate::compress_bound(chunk.len()) + BlockHeader::LEN)
        .map_err(|_| Error::OutOfMemory)?;
    append_framed_block(state, chunk, &mut out)?;
    Ok(out)
}

/// Encode each block of `data` on the current pool.
///
/// Each worker owns one state built on the worker itself, so workers
/// share no lock and no state crosses threads. Each output holds one
/// block header plus payload. Workers take whole ranges, so uneven
/// blocks keep all workers busy. The outputs join in order, so the
/// bytes match the serial path.
///
/// # Errors
///
/// Returns the codec fault of the first failing block in block order,
/// [`Error::OutOfMemory`] when a state or output allocation fails.
pub(crate) fn encode_blocks(data: &[u8], block_size: BlockSize) -> Result<Vec<Vec<u8>>> {
    let width = block_size.as_usize();
    let chunks: Vec<&[u8]> = data.chunks(width).collect();
    // One block needs no pool, so it runs on the caller thread.
    if chunks.len() < 2 {
        encode_blocks_serial(&chunks, block_size)
    } else {
        encode_blocks_parallel(&chunks, block_size)
    }
}

/// Encode each chunk on the caller thread with one reused state.
///
/// # Errors
///
/// Returns the codec fault of the first failing block in block order,
/// [`Error::OutOfMemory`] when a state or output allocation fails.
fn encode_blocks_serial(chunks: &[&[u8]], block_size: BlockSize) -> Result<Vec<Vec<u8>>> {
    let mut state = BlockEncoder::new(block_size)?;
    chunks
        .iter()
        .map(|chunk| encode_one(&mut state, chunk))
        .collect()
}

/// Encode each chunk range on the current pool with worker-local states.
///
/// Each worker owns one state built on the worker itself, so workers
/// share no lock and no state crosses threads. Each output holds one
/// block header plus payload. Workers take whole ranges, so uneven
/// blocks keep all workers busy. The outputs join in order, so the
/// bytes match the serial path.
///
/// # Errors
///
/// Returns the codec fault of the first failing block in block order,
/// [`Error::OutOfMemory`] when a state or output allocation fails.
fn encode_blocks_parallel(chunks: &[&[u8]], block_size: BlockSize) -> Result<Vec<Vec<u8>>> {
    let ranges = partition(chunks.len(), rayon::current_num_threads().max(1));
    // One state for each worker, built on the worker itself: no shared
    // lock, and the builds run together instead of stalling the caller.
    let groups: Vec<&[&[u8]]> = ranges.iter().map(|range| &chunks[range.clone()]).collect();
    let parts: Vec<Result<Vec<Vec<u8>>>> = groups
        .into_par_iter()
        .map(|group| {
            let mut state = BlockEncoder::new(block_size)?;
            let mut outs = Vec::with_capacity(group.len());
            for chunk in group {
                outs.push(encode_one(&mut state, chunk)?);
            }
            Ok(outs)
        })
        .collect();
    let mut out = Vec::with_capacity(chunks.len());
    for part in parts {
        out.extend(part?);
    }
    Ok(out)
}

/// Decode each block of `blocks` on the current pool into one plain vector.
///
/// The total plain size comes from the walked headers, so one output
/// vector holds every block. Each worker owns one state built on the
/// worker itself and decodes its ranges straight into disjoint output
/// slices: no shared lock, no per-block allocation, no extra copy.
/// The outputs join in order, so the bytes match the serial path.
///
/// # Errors
///
/// Returns the codec fault of the first failing block in block order,
/// [`Error::OutOfMemory`] when a state or the output allocation fails.
pub(crate) fn decode_blocks(blocks: &[FramedBlock<'_>], block_size: BlockSize) -> Result<Vec<u8>> {
    // One block needs no pool, so it runs on the caller thread.
    if blocks.len() < 2 {
        decode_blocks_serial(blocks, block_size)
    } else {
        decode_blocks_parallel(blocks, block_size)
    }
}

/// Decode each block on the caller thread into one plain vector.
///
/// # Errors
///
/// Returns the codec fault of the first failing block in block order,
/// [`Error::OutOfMemory`] when a state or the output allocation fails.
fn decode_blocks_serial(blocks: &[FramedBlock<'_>], block_size: BlockSize) -> Result<Vec<u8>> {
    let mut decoder = BlockDecoder::new(block_size)?;
    let mut out = Vec::new();
    for block in blocks {
        decoder.decode_block_into(block.payload, block.header.original_size, &mut out)?;
    }
    Ok(out)
}

/// Decode each block range on the current pool into disjoint slices.
///
/// The total plain size comes from the walked headers, so one output
/// vector holds every block. Each worker owns one state built on the
/// worker itself and decodes its ranges straight into disjoint output
/// slices: no shared lock, no per-block allocation, no extra copy.
/// The outputs join in order, so the bytes match the serial path.
///
/// # Errors
///
/// Returns the codec fault of the first failing block in block order,
/// [`Error::OutOfMemory`] when a state or the output allocation fails.
fn decode_blocks_parallel(blocks: &[FramedBlock<'_>], block_size: BlockSize) -> Result<Vec<u8>> {
    let offsets = block_end_offsets(blocks)?;
    let total = offsets[offsets.len() - 1];
    let mut out = filled(total, 0).ok_or(Error::OutOfMemory)?;
    let ranges = partition(blocks.len(), rayon::current_num_threads().max(1));
    // One state for each worker, built on the worker itself: no shared
    // lock, and the builds run together instead of stalling the caller.
    let groups: Vec<&[FramedBlock<'_>]> =
        ranges.iter().map(|range| &blocks[range.clone()]).collect();
    let mut regions: Vec<&mut [u8]> = Vec::with_capacity(ranges.len());
    let mut rest = out.as_mut_slice();
    for range in &ranges {
        let len = offsets[range.end] - offsets[range.start];
        let (head, tail) = rest.split_at_mut(len);
        regions.push(head);
        rest = tail;
    }
    groups
        .into_par_iter()
        .zip_eq(regions.into_par_iter())
        .map(|(group, mut region)| {
            let mut decoder = BlockDecoder::new(block_size)?;
            for block in group {
                let len =
                    usize::try_from(block.header.original_size).map_err(|_| Error::OutOfMemory)?;
                let tmp = std::mem::take(&mut region);
                if tmp.len() < len {
                    return Err(Error::OutputSizeMismatch);
                }
                let (head, tail) = tmp.split_at_mut(len);
                region = tail;
                decoder.decode_block_to(block.payload, block.header.original_size, head)?;
            }
            Ok(())
        })
        .collect::<Result<Vec<()>>>()?;
    Ok(out)
}

/// Walk the block headers into end offsets starting at 0.
///
/// # Errors
///
/// Returns [`Error::OutOfMemory`] when a size does not fit.
fn block_end_offsets(blocks: &[FramedBlock<'_>]) -> Result<Vec<usize>> {
    let mut offsets = Vec::with_capacity(blocks.len() + 1);
    offsets.push(0_usize);
    for block in blocks {
        let len = usize::try_from(block.header.original_size).map_err(|_| Error::OutOfMemory)?;
        offsets.push(
            offsets[offsets.len() - 1]
                .checked_add(len)
                .ok_or(Error::OutOfMemory)?,
        );
    }
    Ok(offsets)
}

/// Run `work` on the pool that `jobs` selects.
///
/// [`Jobs::Global`] selects the shared global pool. A count above
/// one builds a private pool for the call. The serial count
/// never reaches this function. Fewer than two `blocks` build no pool,
/// because `work` then runs on the caller thread.
///
/// # Errors
///
/// Returns [`Error::ThreadPoolFailed`] when the private pool fails to build,
/// otherwise the error of `work`.
pub(crate) fn run_in_pool<T: Send>(
    jobs: Jobs,
    blocks: usize,
    work: impl FnOnce() -> Result<T> + Send,
) -> Result<T> {
    if blocks < 2 {
        return work();
    }
    let (pool, _) = build_pool(jobs)?;
    run_on(pool.as_ref(), work)
}

/// Compress `reader` into `writer` under the streaming frame.
///
/// With a serial `jobs`, read one block at a time and write each block
/// when it is ready, so memory use stays near one block. If a read fails
/// then, the writer already holds the blocks before it. Other counts read
/// all input into memory, then call [`compress_slice`]. Use
/// [`compress_slice`] directly when the input is already in memory.
///
/// # Errors
///
/// Returns [`Error::Io`] when the read fails, otherwise the errors
/// of [`compress_slice`].
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use bzip3_rs::{BlockSize, Jobs, stream};
///
/// # fn main() -> Result<(), bzip3_rs::error::Error> {
/// let input = b"hello world ".repeat(1000);
///
/// let mut coded = Vec::new();
/// let written = stream::compress(Cursor::new(&input), &mut coded, BlockSize::MIN, Jobs::SERIAL)?;
/// assert_eq!(written, coded.len() as u64);
///
/// let mut plain = Vec::new();
/// stream::decompress(Cursor::new(&coded), &mut plain, Jobs::SERIAL)?;
/// assert_eq!(plain, input);
/// # Ok(())
/// # }
/// ```
pub fn compress<R: Read, W: Write>(
    mut reader: R,
    mut writer: W,
    block_size: BlockSize,
    jobs: Jobs,
) -> Result<u64> {
    if !jobs.is_serial() {
        return compress_slice(&read_fully(reader)?, writer, block_size, jobs);
    }
    let width = block_size.as_usize();
    let mut chunk = Vec::new();
    read_chunk(&mut reader, width, &mut chunk)?;
    // Build the state before the first write, so an allocation fault
    // leaves the writer empty.
    let mut state = if chunk.is_empty() {
        None
    } else {
        Some(BlockEncoder::new(block_size)?)
    };
    let mut total = write_counted(&mut writer, &StreamHeader::new(block_size).to_bytes())?;
    if let Some(state) = state.as_mut() {
        total += pump_reader_blocks(&mut reader, &mut writer, state, chunk, width)?;
    }
    writer.flush()?;
    Ok(total)
}

/// Read blocks from `reader` and write each framed block at once.
///
/// `chunk` holds the first block already read. Returns the framed
/// bytes written after the header.
///
/// # Errors
///
/// Returns [`Error::Io`] when the read fails, otherwise the errors
/// of [`compress_slice`].
fn pump_reader_blocks<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    state: &mut BlockEncoder,
    mut chunk: Vec<u8>,
    width: usize,
) -> Result<u64> {
    let mut total = 0;
    let mut block_bytes = Vec::new();
    loop {
        block_bytes.clear();
        append_framed_block(state, &chunk, &mut block_bytes)?;
        total += write_counted(&mut *writer, &block_bytes)?;
        if chunk.len() < width {
            break;
        }
        read_chunk(reader, width, &mut chunk)?;
        if chunk.is_empty() {
            break;
        }
    }
    Ok(total)
}

/// Compress `data` into `writer` under the streaming frame.
///
/// Output starts with the 9-byte header, then one block header plus
/// payload per input block. `jobs` selects the thread count. See
/// [`Jobs`]. Output bytes are identical at every count. Returns the
/// framed bytes written.
///
/// The serial path writes each block when it is ready. If a block
/// fails, the writer already holds the blocks before it. The parallel
/// path writes nothing when any block fails.
///
/// # Errors
///
/// Returns [`Error::OutOfMemory`] when a state fails to build or
/// [`Error::ThreadPoolFailed`] when the pool build fails,
/// [`Error::BlockTooLarge`] when a block exceeds the block size,
/// [`Error::BwtFailed`] when a transform fails, [`Error::Io`] when an
/// I/O call fails.
///
/// # Examples
///
/// ```
/// use bzip3_rs::{BlockSize, Jobs, stream};
///
/// # fn main() -> Result<(), bzip3_rs::error::Error> {
/// let input = b"hello world ".repeat(1000);
///
/// let mut serial = Vec::new();
/// stream::compress_slice(&input, &mut serial, BlockSize::MIN, Jobs::SERIAL)?;
///
/// let mut parallel = Vec::new();
/// stream::compress_slice(&input, &mut parallel, BlockSize::MIN, Jobs::from_count(2))?;
/// assert_eq!(serial, parallel);
/// assert!(serial.starts_with(&bzip3_rs::MAGIC_NUMBER));
/// # Ok(())
/// # }
/// ```
pub fn compress_slice<W: Write>(
    data: &[u8],
    mut writer: W,
    block_size: BlockSize,
    jobs: Jobs,
) -> Result<u64> {
    let total = if jobs.is_serial() {
        compress_slice_serial(data, &mut writer, block_size)?
    } else {
        compress_slice_parallel(data, &mut writer, block_size, jobs)?
    };
    writer.flush()?;
    Ok(total)
}

/// Compress `data` block by block on the caller thread.
///
/// Builds the state before the first write, so an allocation fault
/// leaves the writer empty. Each block goes to the writer when it is
/// ready, so the output never sits in memory whole. If a block fails,
/// the writer already holds the blocks before it.
///
/// # Errors
///
/// Returns [`Error::OutOfMemory`] when a state fails to build,
/// [`Error::BlockTooLarge`] when a block exceeds the block size,
/// [`Error::BwtFailed`] when a transform fails, [`Error::Io`] when an
/// I/O call fails.
fn compress_slice_serial<W: Write>(
    data: &[u8],
    writer: &mut W,
    block_size: BlockSize,
) -> Result<u64> {
    // Build the state before the first write, so an allocation
    // fault leaves the writer empty. Each block goes to the writer
    // when it is ready, so the output never sits in memory whole.
    let mut state = if data.is_empty() {
        None
    } else {
        Some(BlockEncoder::new(block_size)?)
    };
    let mut total = write_counted(&mut *writer, &StreamHeader::new(block_size).to_bytes())?;
    if let Some(state) = state.as_mut() {
        let width = block_size.as_usize();
        let mut block_bytes = Vec::new();
        for chunk in data.chunks(width) {
            block_bytes.clear();
            append_framed_block(state, chunk, &mut block_bytes)?;
            total += write_counted(&mut *writer, &block_bytes)?;
        }
    }
    Ok(total)
}

/// Compress `data` on the pool that `jobs` selects.
///
/// Output bytes are identical to the serial path. The path writes
/// nothing when any block fails.
///
/// # Errors
///
/// Returns [`Error::OutOfMemory`] when a state fails to build or
/// [`Error::ThreadPoolFailed`] when the pool build fails,
/// [`Error::BlockTooLarge`] when a block exceeds the block size,
/// [`Error::BwtFailed`] when a transform fails, [`Error::Io`] when an
/// I/O call fails.
fn compress_slice_parallel<W: Write>(
    data: &[u8],
    writer: &mut W,
    block_size: BlockSize,
    jobs: Jobs,
) -> Result<u64> {
    let groups = run_in_pool(jobs, data.len().div_ceil(block_size.as_usize()), || {
        encode_blocks(data, block_size)
    })?;
    let mut total = write_counted(&mut *writer, &StreamHeader::new(block_size).to_bytes())?;
    for group in &groups {
        total += write_counted(&mut *writer, group)?;
    }
    Ok(total)
}

/// Decompress `reader` into `writer` from the streaming frame.
///
/// Reads the 9-byte header, then decodes every block in order.
/// Trailing bytes that do not form a full valid block fail. `jobs`
/// follows the same rule as [`compress`]. Returns the plain bytes
/// written.
///
/// The serial path writes each block when it is ready. If a block or a
/// later header fails, the writer already holds the blocks before it.
/// The parallel path writes nothing when any block fails.
///
/// # Errors
///
/// Returns [`Error::TruncatedHeader`] on a short header or trailing bytes
/// below a full block header, [`Error::InvalidMagic`] on bad magic,
/// [`Error::InvalidHeader`] on bad sizes,
/// [`Error::InvalidBlockSize`] on a bad header block size,
/// [`Error::TruncatedData`] on a short payload,
/// [`Error::OutOfMemory`] when a state fails to build or
/// [`Error::ThreadPoolFailed`] when the pool build fails, the block
/// error when a payload fails to decode, [`Error::Io`] when an
/// I/O call fails.
///
/// # Examples
///
/// ```
/// use bzip3_rs::{BlockSize, Jobs, stream};
///
/// # fn main() -> Result<(), bzip3_rs::error::Error> {
/// let input = b"hello world ".repeat(1000);
/// let mut coded = Vec::new();
/// stream::compress_slice(&input, &mut coded, BlockSize::MIN, Jobs::SERIAL)?;
///
/// let mut plain = Vec::new();
/// let written = stream::decompress(coded.as_slice(), &mut plain, Jobs::Global)?;
/// assert_eq!(written, input.len() as u64);
/// assert_eq!(plain, input);
/// # Ok(())
/// # }
/// ```
pub fn decompress<R: Read, W: Write>(reader: R, mut writer: W, jobs: Jobs) -> Result<u64> {
    let input = read_fully(reader)?;
    let total = if jobs.is_serial() {
        decompress_serial(&input, &mut writer)?
    } else {
        decompress_parallel(&input, &mut writer, jobs)?
    };
    writer.flush()?;
    Ok(total)
}

/// Decode every block of `input` on the caller thread.
///
/// Decodes each block after its split and writes it at once. A fault
/// leaves the blocks before it in the writer.
///
/// # Errors
///
/// Returns [`Error::TruncatedHeader`] on a short header or trailing bytes
/// below a full block header, [`Error::InvalidMagic`] on bad magic,
/// [`Error::InvalidHeader`] on bad sizes,
/// [`Error::InvalidBlockSize`] on a bad header block size,
/// [`Error::TruncatedData`] on a short payload,
/// [`Error::OutOfMemory`] when a state fails to build, the block
/// error when a payload fails to decode, [`Error::Io`] when an
/// I/O call fails.
fn decompress_serial<W: Write>(input: &[u8], writer: &mut W) -> Result<u64> {
    // Decode each block after its split and write it at once. A
    // fault leaves the blocks before it in the writer.
    let blocks = StreamBlocks::new(input)?;
    let mut decoder = BlockDecoder::new(blocks.block_size())?;
    let mut plain = Vec::new();
    let mut total = 0;
    for block in blocks {
        let block = block?;
        plain.clear();
        decoder.decode_block_into(block.payload, block.header.original_size, &mut plain)?;
        total += write_counted(&mut *writer, &plain)?;
    }
    Ok(total)
}

/// Decode every block of `input` on the pool that `jobs` selects.
///
/// Writes nothing when any block fails.
///
/// # Errors
///
/// Returns the errors of [`decompress_serial`] plus
/// [`Error::ThreadPoolFailed`] when the pool build fails.
#[inline]
fn decompress_parallel<W: Write>(input: &[u8], writer: &mut W, jobs: Jobs) -> Result<u64> {
    let (block_size, blocks) = split_stream(input)?;
    let plain = run_in_pool(jobs, blocks.len(), || decode_blocks(&blocks, block_size))?;
    write_counted(&mut *writer, &plain)
}

#[cfg(test)]
mod tests {
    use super::compress;
    use super::decode_blocks;
    use super::decompress;
    use super::encode_blocks;
    use super::partition;
    use super::read_chunk;
    use super::read_fully;
    use super::run_in_pool;
    use super::write_counted;
    use crate::BlockSize;
    use crate::Jobs;
    use crate::error::Error;
    use crate::frame::split_stream;
    use std::io::Cursor;
    use std::io::Read;
    use std::io::Write;

    // Reader that fails each read.
    struct FailRead;

    impl Read for FailRead {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "pipe broke",
            ))
        }
    }

    // Writer that fails each write.
    struct FailWrite;

    impl Write for FailWrite {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "pipe broke",
            ))
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "pipe broke",
            ))
        }
    }

    #[test]
    fn partition_empty_len_returns_no_ranges() {
        assert_eq!(partition(0, 1), [] as [std::ops::Range<usize>; 0]);
        assert_eq!(partition(0, 0), [] as [std::ops::Range<usize>; 0]);
    }

    #[test]
    fn partition_single_item_returns_one_range() {
        assert_eq!(partition(1, 1), vec![0..1]);
    }

    #[test]
    fn partition_uneven_covers_all_items() {
        let ranges = partition(10, 3);
        let flat: Vec<usize> = ranges.iter().flat_map(std::clone::Clone::clone).collect();
        assert_eq!(flat, (0..10).collect::<Vec<usize>>());
        assert!(ranges.iter().all(|range| !range.is_empty()));
    }

    #[test]
    fn partition_more_parts_than_items_clamps_to_len() {
        assert_eq!(partition(3, 10), vec![0..1, 1..2, 2..3]);
        assert_eq!(partition(4, 2).len(), 2);
    }

    #[test]
    fn read_fully_reports_io_fault() {
        let result = read_fully(FailRead);
        assert!(matches!(result, Err(Error::Io(_))));
    }

    #[test]
    fn read_chunk_reports_io_fault() {
        let mut buf = Vec::new();
        let result = read_chunk(&mut FailRead, 8, &mut buf);
        assert!(matches!(result, Err(Error::Io(_))));
    }

    #[test]
    fn write_counted_reports_io_fault() {
        let mut dest = FailWrite;
        let result = write_counted(&mut dest, b"bytes");
        assert!(matches!(result, Err(Error::Io(_))));
        assert!(dest.flush().is_err());
    }

    #[test]
    fn read_chunk_reads_up_to_width() {
        let data = b"hello world";
        let mut source = Cursor::new(data);
        let mut buf = vec![9_u8; 3];
        read_chunk(&mut source, 5, &mut buf).expect("read runs");
        assert_eq!(buf, b"hello");
        read_chunk(&mut source, 64, &mut buf).expect("read runs");
        assert_eq!(buf, b" world");
    }

    #[test]
    fn write_counted_returns_byte_count() {
        let mut out = Vec::new();
        let count = write_counted(&mut out, b"bytes").expect("write runs");
        assert_eq!(count, 5);
        assert_eq!(out, b"bytes");
    }

    #[test]
    fn encode_blocks_empty_returns_no_blocks() {
        let blocks = encode_blocks(&[], BlockSize::MIN).expect("encode runs");
        assert_eq!(blocks, [] as [std::vec::Vec<u8>; 0]);
    }

    #[test]
    fn encode_blocks_single_block_runs_serial() {
        let data = b"hello world ".repeat(100);
        let blocks = encode_blocks(&data, BlockSize::MIN).expect("encode runs");
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn encode_blocks_many_blocks_decode_in_order() {
        let width = BlockSize::MIN.as_usize();
        let data: Vec<u8> = (0..width + 64)
            .map(|i| u8::try_from(i % 251).expect("value fits in u8"))
            .collect();
        let blocks = encode_blocks(&data, BlockSize::MIN).expect("encode runs");
        assert!(blocks.len() >= 2);
        // Wrap the blocks in a stream frame for the split helper.
        let mut framed = Vec::new();
        super::compress_slice(&data, &mut framed, BlockSize::MIN, Jobs::from_count(2))
            .expect("compress runs");
        let (block_size, split) = split_stream(&framed).expect("split runs");
        assert_eq!(block_size, BlockSize::MIN);
        let plain = decode_blocks(&split, block_size).expect("decode runs");
        assert_eq!(plain, data);
    }

    #[test]
    fn decode_blocks_single_block_runs_serial() {
        let data = b"hello world ".repeat(100);
        let mut framed = Vec::new();
        super::compress_slice(&data, &mut framed, BlockSize::MIN, Jobs::SERIAL)
            .expect("compress runs");
        let (block_size, split) = split_stream(&framed).expect("split runs");
        assert_eq!(split.len(), 1);
        let plain = decode_blocks(&split, block_size).expect("decode runs");
        assert_eq!(plain, data);
    }

    #[test]
    fn decode_blocks_parallel_checks_region_size() {
        let width = BlockSize::MIN.as_usize();
        let data: Vec<u8> = (0..2 * width + 8)
            .map(|i| u8::try_from(i % 251).expect("value fits in u8"))
            .collect();
        let mut framed = Vec::new();
        super::compress_slice(&data, &mut framed, BlockSize::MIN, Jobs::from_count(2))
            .expect("compress runs");
        let (block_size, split) = split_stream(&framed).expect("split runs");
        assert!(split.len() >= 2);
        let plain = decode_blocks(&split, block_size).expect("decode runs");
        assert_eq!(plain, data);
    }

    #[test]
    fn run_in_pool_short_circuits_below_two_blocks() {
        let zero = run_in_pool(Jobs::Global, 0, || Ok::<u32, Error>(11)).expect("work runs");
        assert_eq!(zero, 11);
        let one = run_in_pool(Jobs::from_count(2), 1, || Ok::<u32, Error>(12)).expect("work runs");
        assert_eq!(one, 12);
    }

    #[test]
    fn run_in_pool_uses_pool_for_many_blocks() {
        let value =
            run_in_pool(Jobs::from_count(2), 5, || Ok::<u32, Error>(13)).expect("work runs");
        assert_eq!(value, 13);
        let global = run_in_pool(Jobs::Global, 5, || Ok::<u32, Error>(14)).expect("work runs");
        assert_eq!(global, 14);
    }

    #[test]
    fn compress_serial_exact_width_writes_all_blocks() {
        let width = BlockSize::MIN.as_usize();
        let data: Vec<u8> = (0..width)
            .map(|i| u8::try_from(i % 251).expect("value fits in u8"))
            .collect();
        let mut coded = Vec::new();
        compress(Cursor::new(&data), &mut coded, BlockSize::MIN, Jobs::SERIAL)
            .expect("compress runs");
        let mut plain = Vec::new();
        decompress(Cursor::new(&coded), &mut plain, Jobs::SERIAL).expect("decompress runs");
        assert_eq!(plain, data);
    }

    #[test]
    fn compress_serial_empty_writes_header_only() {
        let mut coded = Vec::new();
        compress(Cursor::new(&[]), &mut coded, BlockSize::MIN, Jobs::SERIAL)
            .expect("compress runs");
        let mut plain = Vec::new();
        decompress(Cursor::new(&coded), &mut plain, Jobs::SERIAL).expect("decompress runs");
        assert_eq!(plain, [] as [u8; 0]);
    }
}
