//! `Read`-based streaming compression wrappers.
//!
//! Both types use the 9-byte streaming frame. The encoder emits one
//! block per full block of plain input. It yields byte-identical
//! output to the serial streaming encode for the same input. The
//! decoder reads the frame header at construction and decodes blocks
//! on demand. Codec faults surface as invalid-data errors. I/O
//! faults pass through unchanged.
//!
//! See [Frame formats](crate#frame-formats) for the streaming frame and
//! [Errors](crate#errors) for the way to get the codec error from an I/O error.

use std::fmt::Debug;
use std::fmt::Formatter;
use std::io::ErrorKind;
use std::io::Read;
use std::io::{self};

use crate::BlockSize;
use crate::Jobs;
use crate::batch::DecodeBatch;
use crate::batch::DecoderSet;
use crate::batch::EncoderSet;
use crate::block::BlockDecoder;
use crate::buffer::filled;
use crate::error::Error;
use crate::error::Result;
use crate::frame::BlockHeader;
use crate::frame::StreamHeader;
use crate::frame::parse_block_header;

/// Scratch bytes per inner read of the parallel decoder.
///
/// The parallel decoder reads what the source has ready, so it can
/// decode a full batch as soon as the bytes arrive.
const SCRATCH_LEN: usize = 64 * 1024;

/// Encoder that compresses a plain reader into the streaming frame.
///
/// Each `read` yields framed bytes: the 9-byte header first, then one
/// 8-byte block header plus payload per full input block. A short
/// final block closes the stream. Empty input yields the header only.
///
/// # Examples
///
/// ```
/// use std::io::{Cursor, Read};
/// use bzip3_rs::BlockSize;
/// use bzip3_rs::read::{Decoder, Encoder};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let input = b"hello world ".repeat(1000);
///
/// let mut coded = Vec::new();
/// Encoder::new(Cursor::new(&input), BlockSize::MIN)?.read_to_end(&mut coded)?;
///
/// let mut plain = Vec::new();
/// Decoder::new(Cursor::new(&coded))?.read_to_end(&mut plain)?;
/// assert_eq!(plain, input);
/// # Ok(())
/// # }
/// ```
pub struct Encoder<R> {
    /// Plain source, read in block order.
    inner: R,
    /// Encoder states reused across blocks.
    encoders: EncoderSet,
    /// Plain bytes awaiting a full block. In the parallel mode it holds up to one batch.
    pending: Vec<u8>,
    /// Framed bytes awaiting the caller. It starts with the frame header.
    coded: Vec<u8>,
    /// Read cursor into `coded`.
    pos: usize,
    /// True once the source hit EOF.
    inner_eof: bool,
}

impl<R: Debug> Debug for Encoder<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encoder")
            .field("inner", &self.inner)
            .field("block_size", &self.encoders.block_size())
            .finish_non_exhaustive()
    }
}

impl<R: Read> Encoder<R> {
    /// Wrap `inner` with an encoder for `block_size`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] when working memory allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Cursor, Read};
    /// use bzip3_rs::BlockSize;
    /// use bzip3_rs::read::Encoder;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut encoder = Encoder::new(Cursor::new(b"hello world"), BlockSize::MIN)?;
    /// assert_eq!(encoder.block_size(), BlockSize::MIN);
    ///
    /// let mut coded = Vec::new();
    /// encoder.read_to_end(&mut coded)?;
    /// assert!(coded.starts_with(&bzip3_rs::MAGIC_NUMBER));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn new(inner: R, block_size: BlockSize) -> Result<Self> {
        Self::with_jobs(inner, block_size, Jobs::SERIAL)
    }

    /// Wrap `inner` with an encoder that codes blocks in parallel.
    ///
    /// `jobs` selects the thread count. See [`Jobs`]. A serial `jobs`
    /// runs like [`Self::new`]. Output bytes are
    /// identical at every count. The encoder reads one block for each
    /// thread before it codes, so its memory grows with the thread
    /// count.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] when working memory allocation fails.
    /// Returns [`Error::ThreadPoolFailed`] when the pool build fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Cursor, Read};
    /// use bzip3_rs::{BlockSize, Jobs};
    /// use bzip3_rs::read::Encoder;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let input = b"hello world ".repeat(1000);
    ///
    /// let mut serial = Vec::new();
    /// Encoder::new(Cursor::new(&input), BlockSize::MIN)?.read_to_end(&mut serial)?;
    ///
    /// let mut parallel = Vec::new();
    /// let mut encoder = Encoder::with_jobs(Cursor::new(&input), BlockSize::MIN, Jobs::from_count(2))?;
    /// encoder.read_to_end(&mut parallel)?;
    /// assert_eq!(serial, parallel);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_jobs(inner: R, block_size: BlockSize, jobs: Jobs) -> Result<Self> {
        let width = block_size.as_usize();
        let mut pending = Vec::new();
        pending.try_reserve(width).map_err(|_| Error::OutOfMemory)?;
        Ok(Self {
            inner,
            encoders: EncoderSet::new(block_size, jobs)?,
            pending,
            coded: StreamHeader::new(block_size).to_bytes().to_vec(),
            pos: 0,
            inner_eof: false,
        })
    }

    /// Return the wrapped reader.
    #[must_use]
    #[inline]
    pub const fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Return the wrapped reader as a mutable reference.
    ///
    /// A direct read from it moves the source and corrupts the stream.
    #[must_use]
    #[inline]
    pub const fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Return the wrapped reader.
    ///
    /// The encoder loses the plain bytes that it read but did not code yet.
    /// It also loses the framed bytes that the caller did not read yet.
    #[must_use]
    #[inline]
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Return the configured block size.
    #[must_use]
    #[inline]
    pub const fn block_size(&self) -> BlockSize {
        self.encoders.block_size()
    }

    /// Fill `coded` with the next framed bytes.
    ///
    /// Reads up to one unit of plain bytes, then emits the blocks. At
    /// source EOF it emits the short final block, if any.
    fn produce(&mut self) -> io::Result<()> {
        if !self.inner_eof {
            let missing = self.encoders.input_len() - self.pending.len();
            let got = (&mut self.inner)
                .take(missing as u64)
                .read_to_end(&mut self.pending)?;
            self.inner_eof = got < missing;
        }
        if self.pending.is_empty() {
            return Ok(());
        }
        self.encoders
            .append_to(&self.pending, &mut self.coded)
            .map_err(Error::into_io)?;
        self.pending.clear();
        Ok(())
    }
}

impl<R: Read> Read for Encoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        while self.pos >= self.coded.len() {
            if self.inner_eof && self.pending.is_empty() {
                return Ok(0);
            }
            self.produce()?;
        }
        let n = (self.coded.len() - self.pos).min(buf.len());
        buf[..n].copy_from_slice(&self.coded[self.pos..self.pos + n]);
        self.pos += n;
        if self.pos >= self.coded.len() {
            self.coded.clear();
            self.pos = 0;
        }
        Ok(n)
    }
}

/// Read from `inner` until `tail` holds `len` bytes.
///
/// Return false when the source ends first.
///
/// # Errors
///
/// Returns [`Error::Io`] when the read fails.
#[inline]
fn read_to_len<R: Read>(inner: &mut R, tail: &mut Vec<u8>, len: usize) -> Result<bool> {
    let missing = len.saturating_sub(tail.len());
    inner.take(missing as u64).read_to_end(tail)?;
    Ok(tail.len() >= len)
}

/// Decode the next full block from `inner` into `plain`.
///
/// Read the exact bytes of the block, so `tail` is empty after each block.
/// Returns true when a block was decoded. Returns false at clean
/// EOF with no bytes left over.
///
/// # Errors
///
/// Returns the exact codec fault on bad sizes or payloads.
fn decode_next_block<R: Read>(
    decoder: &mut BlockDecoder,
    inner: &mut R,
    tail: &mut Vec<u8>,
    plain: &mut Vec<u8>,
    pos: &mut usize,
) -> Result<bool> {
    if !read_to_len(inner, tail, BlockHeader::LEN)? {
        return if tail.is_empty() {
            Ok(false)
        } else {
            Err(Error::TruncatedHeader)
        };
    }
    let header = parse_block_header(tail, decoder.block_size())?;
    let end = BlockHeader::LEN + header.compressed_size as usize;
    if !read_to_len(inner, tail, end)? {
        return Err(Error::TruncatedData);
    }
    // `read` calls this only after it drained `plain`.
    plain.clear();
    decoder.decode_block_into(&tail[BlockHeader::LEN..end], header.original_size, plain)?;
    tail.clear();
    *pos = 0;
    Ok(true)
}

/// Decode the next batch of full blocks from `tail` into `plain`.
///
/// Read from the source until the batch is full, a fault shows, or
/// the source ends. Return true when `plain` holds bytes. A fault
/// after those bytes waits in `pending_fault`.
///
/// # Errors
///
/// Returns the exact codec fault when no bytes come before it.
fn decode_batch<R: Read>(
    batch: &mut DecodeBatch,
    inner: &mut R,
    tail: &mut Vec<u8>,
    plain: &mut Vec<u8>,
    pos: &mut usize,
    scratch: &mut [u8],
    pending_fault: &mut Option<Error>,
) -> Result<bool> {
    let (mut end, mut fault) = loop {
        let (end, fault) = batch.queue_blocks(tail, 0)?;
        if fault.is_some() || batch.queued() == batch.max_blocks() {
            break (end, fault);
        }
        let n = inner.read(scratch)?;
        if n == 0 {
            let fault = if end == tail.len() {
                None
            } else if tail.len() - end < BlockHeader::LEN {
                Some(Error::TruncatedHeader)
            } else {
                Some(Error::TruncatedData)
            };
            break (end, fault);
        }
        tail.extend_from_slice(&scratch[..n]);
    };
    let queued = batch.queued();
    batch.decode_queued(tail);
    plain.clear();
    *pos = 0;
    for index in 0..queued {
        let start = batch.block_start(index);
        match batch.take_plain(index) {
            Ok(bytes) => plain.extend_from_slice(bytes),
            Err(block_fault) => {
                end = start;
                fault = Some(block_fault);
                break;
            }
        }
    }
    tail.drain(..end);
    match fault {
        Some(fault) if plain.is_empty() => Err(fault),
        Some(fault) => {
            *pending_fault = Some(fault);
            Ok(true)
        }
        None => Ok(!plain.is_empty()),
    }
}

/// Decoder that decompresses a streaming reader into plain bytes.
///
/// The constructor reads the 9-byte frame header at once, so header
/// faults surface with their exact variant. Block faults surface as
/// invalid-data errors on read. Trailing bytes that do not form a
/// full valid block fail the read. Clean EOF after a block yields
/// zero bytes.
///
/// # Examples
///
/// ```
/// use std::io::{Cursor, Read};
/// use bzip3_rs::{BlockSize, Jobs, stream};
/// use bzip3_rs::read::Decoder;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let input = b"hello world ".repeat(1000);
/// let mut coded = Vec::new();
/// stream::compress_slice(&input, &mut coded, BlockSize::MIN, Jobs::SERIAL)?;
///
/// let mut decoder = Decoder::new(Cursor::new(&coded))?;
/// assert_eq!(decoder.block_size(), BlockSize::MIN);
///
/// let mut plain = Vec::new();
/// decoder.read_to_end(&mut plain)?;
/// assert_eq!(plain, input);
/// # Ok(())
/// # }
/// ```
pub struct Decoder<R> {
    /// Framed source, read in block order.
    inner: R,
    /// Decoder states reused across blocks.
    decoders: DecoderSet,
    /// Plain bytes awaiting the caller. In the parallel mode it holds up to one batch.
    plain: Vec<u8>,
    /// Read cursor into `plain`.
    pos: usize,
    /// Framed bytes read but not yet decoded.
    tail: Vec<u8>,
    /// True once the source hit clean EOF.
    eof: bool,
    /// Scratch for one source read. Empty in the serial mode.
    scratch: Vec<u8>,
    /// Fault that follows the bytes in `plain`.
    pending_fault: Option<Error>,
}

impl<R: Debug> Debug for Decoder<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decoder")
            .field("inner", &self.inner)
            .field("block_size", &self.decoders.block_size())
            .finish_non_exhaustive()
    }
}

impl<R: Read> Decoder<R> {
    /// Wrap `inner` and read its frame header at once.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TruncatedHeader`] on a short header,
    /// [`Error::InvalidMagic`] on bad magic, [`Error::InvalidBlockSize`] on an invalid header block size or
    /// [`Error::OutOfMemory`] on a working memory failure,
    /// [`Error::Io`] on an I/O failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Cursor;
    /// use bzip3_rs::error::Error;
    /// use bzip3_rs::read::Decoder;
    ///
    /// // The constructor reads the header at once, so a header fault shows here.
    /// let result = Decoder::new(Cursor::new([0_u8; 9]));
    /// assert!(matches!(result, Err(Error::InvalidMagic)));
    /// ```
    #[inline]
    pub fn new(inner: R) -> Result<Self> {
        Self::with_jobs(inner, Jobs::SERIAL)
    }

    /// Wrap `inner`, read its frame header at once, and decode blocks in parallel.
    ///
    /// `jobs` selects the thread count. See [`Jobs`]. A serial `jobs`
    /// runs like [`Self::new`]. Output bytes and
    /// errors are identical at every count. The decoder reads one
    /// block for each thread before it decodes, so its memory grows
    /// with the thread count. A block fault shows after the bytes of
    /// the blocks before it.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::new`]. Returns [`Error::ThreadPoolFailed`]
    /// when the pool build fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Cursor, Read};
    /// use bzip3_rs::{BlockSize, Jobs, stream};
    /// use bzip3_rs::read::Decoder;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let input = b"hello world ".repeat(1000);
    /// let mut coded = Vec::new();
    /// stream::compress_slice(&input, &mut coded, BlockSize::MIN, Jobs::SERIAL)?;
    ///
    /// let mut plain = Vec::new();
    /// Decoder::with_jobs(Cursor::new(&coded), Jobs::from_count(2))?.read_to_end(&mut plain)?;
    /// assert_eq!(plain, input);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_jobs(mut inner: R, jobs: Jobs) -> Result<Self> {
        let mut header = [0_u8; StreamHeader::LEN];
        inner.read_exact(&mut header).map_err(|e| {
            if e.kind() == ErrorKind::UnexpectedEof {
                Error::TruncatedHeader
            } else {
                Error::Io(e)
            }
        })?;
        let block_size = StreamHeader::from_bytes(&header)?.block_size;
        let scratch = if jobs.is_serial() {
            Vec::new()
        } else {
            filled(SCRATCH_LEN, 0_u8).ok_or(Error::OutOfMemory)?
        };
        Ok(Self {
            inner,
            decoders: DecoderSet::new(block_size, jobs)?,
            plain: Vec::new(),
            pos: 0,
            tail: Vec::new(),
            eof: false,
            scratch,
            pending_fault: None,
        })
    }

    /// Return the wrapped reader.
    #[must_use]
    #[inline]
    pub const fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Return the wrapped reader as a mutable reference.
    ///
    /// A direct read from it moves the source and corrupts the stream.
    #[must_use]
    #[inline]
    pub const fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Return the wrapped reader.
    ///
    /// The decoder loses the framed bytes that it read but did not decode yet.
    /// It also loses the plain bytes that the caller did not read yet.
    #[must_use]
    #[inline]
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Return the block size from the frame header.
    #[must_use]
    #[inline]
    pub const fn block_size(&self) -> BlockSize {
        self.decoders.block_size()
    }

    /// Decode the next bytes from the source into `plain`.
    ///
    /// Return true when `plain` holds bytes. Return false at clean EOF
    /// with no bytes left over.
    ///
    /// # Errors
    ///
    /// Returns the exact codec fault when no bytes come before it.
    fn decode_more(&mut self) -> Result<bool> {
        match &mut self.decoders {
            DecoderSet::Serial(decoder) => decode_next_block(
                decoder,
                &mut self.inner,
                &mut self.tail,
                &mut self.plain,
                &mut self.pos,
            ),
            DecoderSet::Parallel(batch) => decode_batch(
                batch,
                &mut self.inner,
                &mut self.tail,
                &mut self.plain,
                &mut self.pos,
                &mut self.scratch,
                &mut self.pending_fault,
            ),
        }
    }

    /// Fill `plain` with the next decoded bytes.
    fn produce(&mut self) -> io::Result<bool> {
        if self.eof {
            return Ok(false);
        }
        if let Some(fault) = self.pending_fault.take() {
            return Err(fault.into_io());
        }
        match self.decode_more() {
            Ok(true) => Ok(true),
            Ok(false) => {
                self.eof = true;
                Ok(false)
            }
            Err(e) => Err(e.into_io()),
        }
    }
}

impl<R: Read> Read for Decoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        while self.pos >= self.plain.len() {
            if !self.produce()? {
                return Ok(0);
            }
        }
        let n = (self.plain.len() - self.pos).min(buf.len());
        buf[..n].copy_from_slice(&self.plain[self.pos..self.pos + n]);
        self.pos += n;
        if self.pos >= self.plain.len() {
            self.plain.clear();
            self.pos = 0;
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::Decoder;
    use super::Encoder;
    use crate::BlockSize;
    use crate::Jobs;
    use crate::error::Error;
    use std::io::Cursor;
    use std::io::ErrorKind;
    use std::io::Read;
    use std::io::{self};

    /// Source that fails every read.
    #[derive(Debug)]
    struct FailRead;

    impl Read for FailRead {
        fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::new(ErrorKind::BrokenPipe, "source fault"))
        }
    }

    /// Source that yields `data`, then fails every read.
    #[derive(Debug)]
    struct FailAfter {
        /// Bytes served before the fault.
        data: Vec<u8>,
        /// Read cursor into `data`.
        pos: usize,
    }

    impl Read for FailAfter {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.pos < self.data.len() {
                let take = (self.data.len() - self.pos).min(buf.len());
                buf[..take].copy_from_slice(&self.data[self.pos..self.pos + take]);
                self.pos += take;
                return Ok(take);
            }
            Err(io::Error::new(ErrorKind::BrokenPipe, "source fault"))
        }
    }

    /// Encode one framed stream for the decoder tests.
    fn coded_sample() -> (Vec<u8>, Vec<u8>) {
        let input = b"hello world ".repeat(1000);
        let mut coded = Vec::new();
        Encoder::new(Cursor::new(&input), BlockSize::MIN)
            .unwrap()
            .read_to_end(&mut coded)
            .unwrap();
        (input, coded)
    }

    #[test]
    fn encoder_accessors_expose_inner_and_block_size() {
        let mut encoder = Encoder::new(Cursor::new(b"hello world"), BlockSize::MIN).unwrap();
        assert_eq!(encoder.block_size(), BlockSize::MIN);
        assert_eq!(*encoder.get_ref().get_ref(), b"hello world");
        encoder.get_mut().set_position(0);
        assert_eq!(encoder.into_inner().into_inner(), b"hello world");
    }

    #[test]
    fn encoder_debug_names_inner_and_block_size() {
        let text = format!(
            "{:?}",
            Encoder::new(Cursor::new(b"hello"), BlockSize::MIN).unwrap()
        );
        assert!(text.contains("Encoder"));
        assert!(text.contains("block_size"));
    }

    #[test]
    fn encoder_empty_input_gives_header_only_and_clean_eof() {
        let mut encoder = Encoder::new(Cursor::new(b""), BlockSize::MIN).unwrap();
        let mut coded = Vec::new();
        encoder.read_to_end(&mut coded).unwrap();
        // Empty input yields the header only.
        assert_eq!(coded.len(), crate::StreamHeader::LEN);
        // Reads past EOF keep returning zero.
        let mut one = [0_u8; 8];
        assert_eq!(encoder.read(&mut one).unwrap(), 0);
        assert_eq!(encoder.read(&mut one).unwrap(), 0);
    }

    #[test]
    fn encoder_produce_drains_pending_after_eof() {
        let mut encoder = Encoder::new(Cursor::new(b"hello world"), BlockSize::MIN).unwrap();
        // Simulate a late plain tail: the source hit EOF with bytes still pending.
        encoder.inner_eof = true;
        encoder.pending.extend_from_slice(b"late tail");
        encoder.produce().unwrap();
        assert_eq!(encoder.pending, Vec::new());
    }

    #[test]
    fn encoder_forwards_inner_read_fault() {
        let mut encoder = Encoder::new(FailRead, BlockSize::MIN).unwrap();
        let fault = encoder.read_to_end(&mut Vec::new()).unwrap_err();
        assert_eq!(fault.kind(), ErrorKind::BrokenPipe);
    }

    #[test]
    fn encoder_parallel_output_matches_serial() {
        let input = b"hello world ".repeat(1000);
        let mut serial = Vec::new();
        Encoder::new(Cursor::new(&input), BlockSize::MIN)
            .unwrap()
            .read_to_end(&mut serial)
            .unwrap();
        let mut parallel =
            Encoder::with_jobs(Cursor::new(&input), BlockSize::MIN, Jobs::from_count(2)).unwrap();
        assert_eq!(parallel.block_size(), BlockSize::MIN);
        let mut coded = Vec::new();
        parallel.read_to_end(&mut coded).unwrap();
        assert_eq!(serial, coded);
    }

    #[test]
    fn decoder_with_jobs_maps_header_faults() {
        // A short source ends the header read.
        let fault = Decoder::with_jobs(Cursor::new([0_u8; 4]), Jobs::SERIAL).unwrap_err();
        assert!(matches!(fault, Error::TruncatedHeader));
        // Any other read fault passes through as I/O.
        let fault = Decoder::with_jobs(FailRead, Jobs::SERIAL).unwrap_err();
        assert!(matches!(fault, Error::Io(_)));
    }

    #[test]
    fn decoder_accessors_expose_inner_and_block_size() {
        let (_input, coded) = coded_sample();
        let mut decoder = Decoder::new(Cursor::new(coded.clone())).unwrap();
        assert_eq!(decoder.block_size(), BlockSize::MIN);
        assert_eq!(decoder.get_ref().get_ref(), &coded);
        let _ = decoder.get_mut();
        assert_eq!(decoder.into_inner().into_inner(), coded);
    }

    #[test]
    fn decoder_debug_names_inner_and_block_size() {
        let (_input, coded) = coded_sample();
        let text = format!("{:?}", Decoder::new(Cursor::new(&coded)).unwrap());
        assert!(text.contains("Decoder"));
        assert!(text.contains("block_size"));
    }

    #[test]
    fn decoder_read_returns_zero_after_clean_eof() {
        let (input, coded) = coded_sample();
        let mut decoder = Decoder::new(Cursor::new(&coded)).unwrap();
        let mut plain = Vec::new();
        decoder.read_to_end(&mut plain).unwrap();
        assert_eq!(plain, input);
        // Reads past EOF keep returning zero.
        let mut one = [0_u8; 4];
        assert_eq!(decoder.read(&mut one).unwrap(), 0);
        assert_eq!(decoder.read(&mut one).unwrap(), 0);
    }

    #[test]
    fn decoder_forwards_block_read_fault() {
        let (_input, coded) = coded_sample();
        let mut decoder = Decoder::new(FailAfter {
            data: coded,
            pos: 0,
        })
        .unwrap();
        let mut plain = Vec::new();
        let fault = decoder.read_to_end(&mut plain).unwrap_err();
        assert_eq!(fault.kind(), ErrorKind::BrokenPipe);
    }

    #[test]
    fn decoder_parallel_forwards_block_read_fault() {
        let (_input, coded) = coded_sample();
        let mut decoder = Decoder::with_jobs(
            FailAfter {
                data: coded,
                pos: 0,
            },
            Jobs::from_count(2),
        )
        .unwrap();
        let mut plain = Vec::new();
        let fault = decoder.read_to_end(&mut plain).unwrap_err();
        assert_eq!(fault.kind(), ErrorKind::BrokenPipe);
    }
}
