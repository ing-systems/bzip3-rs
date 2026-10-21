//! `Write`-based streaming compression wrappers.
//!
//! Both types use the 9-byte streaming frame. The encoder writes the
//! header at construction and emits one block per full block of plain
//! input. The decoder buffers framed input, emits plain output per
//! full block, and detects truncation in `finish`. Codec faults
//! surface as invalid-data errors on write and with their exact
//! variant in `finish`. I/O faults pass through unchanged.
//!
//! See [Frame formats](crate#frame-formats) for the streaming frame and
//! [Errors](crate#errors) for the way to get the codec error from an I/O error.

use std::fmt::Debug;
use std::fmt::Formatter;
use std::io::Write;
use std::io::{self};

use crate::BlockSize;
use crate::Jobs;
use crate::batch::DecoderSet;
use crate::batch::EncoderSet;
use crate::error::Error;
use crate::error::Result;
use crate::frame::BlockHeader;
use crate::frame::StreamHeader;

/// Encoder that compresses plain writes into the streaming frame.
///
/// The constructor writes the 9-byte header at once. Each `write`
/// buffers plain input and emits full blocks in order. `finish`
/// emits the short final block, if any, and returns the inner writer.
/// Empty input yields the header only.
///
/// Dropping the encoder without `finish` also emits the final block,
/// but it ignores the errors. Call `finish` to see them.
///
/// # Examples
///
/// ```
/// use std::io::Write;
/// use bzip3_rs::BlockSize;
/// use bzip3_rs::write::Encoder;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let input = b"hello world ".repeat(1000);
///
/// let mut encoder = Encoder::new(Vec::new(), BlockSize::MIN)?;
/// encoder.write_all(&input)?;
/// let coded = encoder.finish()?; // `finish` closes the stream and returns the writer.
///
/// let mut plain = Vec::new();
/// bzip3_rs::stream::decompress(coded.as_slice(), &mut plain, bzip3_rs::Jobs::SERIAL)?;
/// assert_eq!(plain, input);
/// # Ok(())
/// # }
/// ```
pub struct Encoder<W: Write> {
    /// Framed sink, written in block order. It is `None` after `finish`.
    inner: Option<W>,
    /// Encoder states reused across blocks.
    encoders: EncoderSet,
    /// Plain bytes awaiting a full block. Never longer than one block,
    /// or one batch in the parallel mode.
    pending: Vec<u8>,
}

impl<W: Write + Debug> Debug for Encoder<W> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encoder")
            .field("inner", &Self::writer_ref(self.inner.as_ref()))
            .field("block_size", &self.encoders.block_size())
            .finish_non_exhaustive()
    }
}

impl<W: Write> Encoder<W> {
    /// Wrap `inner` and write its frame header at once.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] when working memory allocation fails,
    /// [`Error::Io`] when the header write fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::{BlockSize, StreamHeader};
    /// use bzip3_rs::write::Encoder;
    ///
    /// # fn main() -> Result<(), bzip3_rs::error::Error> {
    /// let encoder = Encoder::new(Vec::new(), BlockSize::MIN)?;
    ///
    /// // The constructor wrote the 9-byte header already.
    /// let written = encoder.get_ref().map_or(0, Vec::len);
    /// assert_eq!(written, StreamHeader::LEN);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(inner: W, block_size: BlockSize) -> Result<Self> {
        Self::with_jobs(inner, block_size, Jobs::SERIAL)
    }

    /// Wrap `inner`, write its frame header at once, and code blocks in parallel.
    ///
    /// `jobs` selects the thread count. See [`Jobs`]. A serial `jobs`
    /// runs like [`Self::new`]. Output bytes are
    /// identical at every count. The encoder buffers one block for
    /// each thread, so its memory grows with the thread count.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] when working memory allocation fails,
    /// [`Error::ThreadPoolFailed`] when the pool build fails,
    /// [`Error::Io`] when the header write fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Write;
    /// use bzip3_rs::{BlockSize, Jobs};
    /// use bzip3_rs::write::Encoder;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let input = b"hello world ".repeat(1000);
    ///
    /// let mut serial = Encoder::new(Vec::new(), BlockSize::MIN)?;
    /// serial.write_all(&input)?;
    ///
    /// let mut parallel = Encoder::with_jobs(Vec::new(), BlockSize::MIN, Jobs::from_count(2))?;
    /// parallel.write_all(&input)?;
    ///
    /// assert_eq!(serial.finish()?, parallel.finish()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_jobs(mut inner: W, block_size: BlockSize, jobs: Jobs) -> Result<Self> {
        let width = block_size.as_usize();
        let mut pending = Vec::new();
        pending.try_reserve(width).map_err(|_| Error::OutOfMemory)?;
        let encoders = EncoderSet::new(block_size, jobs)?;
        inner.write_all(&StreamHeader::new(block_size).to_bytes())?;
        Ok(Self {
            inner: Some(inner),
            encoders,
            pending,
        })
    }

    /// Borrow the writer. It exists until `finish` takes it.
    const fn writer_ref(inner: Option<&W>) -> Option<&W> {
        inner
    }

    /// Borrow the writer mutably. It exists until `finish` takes it.
    const fn writer_mut(inner: Option<&mut W>) -> Option<&mut W> {
        inner
    }

    /// Take the writer out. It exists until `finish` takes it.
    const fn take_writer(&mut self) -> Option<W> {
        self.inner.take()
    }

    /// Return the wrapped writer.
    ///
    /// The result is `Some` for every caller, because `finish` takes the
    /// encoder by value.
    #[must_use]
    pub const fn get_ref(&self) -> Option<&W> {
        Self::writer_ref(self.inner.as_ref())
    }

    /// Return the wrapped writer as a mutable reference.
    ///
    /// A direct write to it puts foreign bytes in the stream.
    /// The result is `Some` for every caller, because `finish` takes the
    /// encoder by value.
    #[must_use]
    pub const fn get_mut(&mut self) -> Option<&mut W> {
        Self::writer_mut(self.inner.as_mut())
    }

    /// Return the configured block size.
    #[must_use]
    pub const fn block_size(&self) -> BlockSize {
        self.encoders.block_size()
    }

    /// Emit the short final block, flush, and return the inner writer.
    ///
    /// # Errors
    ///
    /// Returns the codec fault when the final block fails to encode,
    /// [`Error::Io`] when a write or flush fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Write;
    /// use bzip3_rs::{BlockSize, StreamHeader};
    /// use bzip3_rs::write::Encoder;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut encoder = Encoder::new(Vec::new(), BlockSize::MIN)?;
    /// encoder.write_all(b"hello world")?;
    ///
    /// let coded = encoder.finish()?;
    /// assert!(coded.len() > StreamHeader::LEN);
    /// # Ok(())
    /// # }
    /// ```
    pub fn finish(mut self) -> Result<W> {
        self.write_final_block()?;
        self.take_writer()
            .ok_or_else(|| Error::Io(io::Error::new(io::ErrorKind::BrokenPipe, "writer missing")))
    }

    /// Emit the short final block and flush. Do nothing after `finish`.
    ///
    /// The call empties `pending` first, so a second call never
    /// writes the same bytes again.
    fn write_final_block(&mut self) -> Result<()> {
        let Some(inner) = self.inner.as_mut() else {
            return Ok(());
        };
        let pending = std::mem::take(&mut self.pending);
        if !pending.is_empty() {
            self.encoders.write_to(&pending, inner)?;
        }
        inner.flush()?;
        Ok(())
    }
}

impl<W: Write> Drop for Encoder<W> {
    fn drop(&mut self) {
        // Errors are lost here. `finish` is the way to see them.
        let _ = self.write_final_block();
    }
}

impl<W: Write> Write for Encoder<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // One unit is one block, or one batch in the parallel mode.
        let width = self.encoders.input_len();
        let Some(inner) = Self::writer_mut(self.inner.as_mut()) else {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "writer missing"));
        };
        let mut rest = buf;
        // Fill the pending unit first. `pending` stays within one unit.
        if !self.pending.is_empty() {
            let take = (width - self.pending.len()).min(rest.len());
            self.pending.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if self.pending.len() < width {
                return Ok(buf.len());
            }
            self.encoders
                .write_to(&self.pending, inner)
                .map_err(Error::into_io)?;
            self.pending.clear();
        }
        // Encode full units straight from the caller bytes.
        while rest.len() >= width {
            let (chunk, tail) = rest.split_at(width);
            self.encoders
                .write_to(chunk, inner)
                .map_err(Error::into_io)?;
            rest = tail;
        }
        self.pending.extend_from_slice(rest);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let Some(inner) = Self::writer_mut(self.inner.as_mut()) else {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "writer missing"));
        };
        inner.flush()
    }
}

/// Decoder that decompresses framed writes into plain output.
///
/// Each `write` buffers framed input and emits plain output per full
/// block. `finish` rejects any partial tail and returns the inner
/// writer, so truncated streams surface instead of silent loss.
/// Empty framed input fails `finish` with a malformed header.
///
/// # Examples
///
/// ```
/// use std::io::Write;
/// use bzip3_rs::{BlockSize, Jobs, stream};
/// use bzip3_rs::write::Decoder;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let input = b"hello world ".repeat(1000);
/// let mut coded = Vec::new();
/// stream::compress_slice(&input, &mut coded, BlockSize::MIN, Jobs::SERIAL)?;
///
/// let mut decoder = Decoder::new(Vec::new());
/// decoder.write_all(&coded)?;
/// let plain = decoder.finish()?;
/// assert_eq!(plain, input);
/// # Ok(())
/// # }
/// ```
pub struct Decoder<W> {
    /// Plain sink, written in block order.
    inner: W,
    /// Decoder states, built once the header arrives.
    decoders: Option<DecoderSet>,
    /// Framed bytes read but not yet decoded.
    tail: Vec<u8>,
    /// Plain bytes of one block, reused across blocks.
    plain: Vec<u8>,
    /// Thread count of the decoder states.
    jobs: Jobs,
}

impl<W: Debug> Debug for Decoder<W> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decoder")
            .field("inner", &self.inner)
            .field(
                "block_size",
                &self.decoders.as_ref().map(DecoderSet::block_size),
            )
            .finish_non_exhaustive()
    }
}

impl<W: Write> Decoder<W> {
    /// Wrap `inner`. No input is read yet.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::write::Decoder;
    ///
    /// let decoder = Decoder::new(Vec::<u8>::new());
    /// assert!(decoder.block_size().is_none()); // No header arrived yet.
    /// ```
    #[must_use]
    pub const fn new(inner: W) -> Self {
        Self {
            inner,
            decoders: None,
            tail: Vec::new(),
            plain: Vec::new(),
            jobs: Jobs::SERIAL,
        }
    }

    /// Wrap `inner` and decode blocks in parallel. No input is read yet.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Write;
    /// use bzip3_rs::{BlockSize, Jobs, stream};
    /// use bzip3_rs::write::Decoder;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let input = b"hello world ".repeat(1000);
    /// let mut coded = Vec::new();
    /// stream::compress_slice(&input, &mut coded, BlockSize::MIN, Jobs::SERIAL)?;
    ///
    /// let mut decoder = Decoder::with_jobs(Vec::new(), Jobs::from_count(2));
    /// decoder.write_all(&coded)?;
    /// assert_eq!(decoder.finish()?, input);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// `jobs` selects the thread count. See [`Jobs`]. A serial `jobs`
    /// runs like [`Self::new`]. Output bytes and
    /// errors are identical at every count. The decoder decodes one
    /// block for each thread, so its memory grows with the thread
    /// count.
    #[must_use]
    pub const fn with_jobs(inner: W, jobs: Jobs) -> Self {
        let mut decoder = Self::new(inner);
        decoder.jobs = jobs;
        decoder
    }

    /// Return the wrapped writer.
    #[must_use]
    pub const fn get_ref(&self) -> &W {
        &self.inner
    }

    /// Return the wrapped writer as a mutable reference.
    ///
    /// A direct write to it puts foreign bytes in the stream.
    #[must_use]
    pub const fn get_mut(&mut self) -> &mut W {
        &mut self.inner
    }

    /// Return the block size from the frame header.
    ///
    /// Returns `None` until the header arrives.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Write;
    /// use bzip3_rs::{BlockSize, StreamHeader};
    /// use bzip3_rs::write::Decoder;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let header = StreamHeader::new(BlockSize::MIN).to_bytes();
    /// let mut decoder = Decoder::new(Vec::new());
    ///
    /// decoder.write_all(&header[..5])?;
    /// assert_eq!(decoder.block_size(), None);
    ///
    /// decoder.write_all(&header[5..])?;
    /// assert_eq!(decoder.block_size(), Some(BlockSize::MIN));
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn block_size(&self) -> Option<BlockSize> {
        self.decoders.as_ref().map(DecoderSet::block_size)
    }

    /// Build the decoder once the header bytes have arrived.
    ///
    /// # Errors
    ///
    /// Returns the exact header fault.
    fn read_header(&mut self) -> Result<()> {
        if self.decoders.is_none() && self.tail.len() >= StreamHeader::LEN {
            let size = StreamHeader::parse(&self.tail)?.block_size;
            self.decoders = Some(DecoderSet::new(size, self.jobs)?);
            self.tail.drain(..StreamHeader::LEN);
        }
        Ok(())
    }

    /// Decode every full block in `tail` into the inner writer.
    ///
    /// # Errors
    ///
    /// Returns the exact codec fault or the I/O fault.
    fn drain_blocks(&mut self) -> Result<()> {
        // Drain the decoded bytes once, so one large write costs
        // linear time.
        let Some(decoders) = self.decoders.as_mut() else {
            return Ok(());
        };
        let mut start = 0;
        let outcome =
            decoders.write_blocks(&self.tail, &mut start, &mut self.plain, &mut self.inner);
        self.tail.drain(..start);
        outcome
    }

    /// Name the fault of a partial tail.
    ///
    /// Runs only when bytes remain after all full blocks drained. A
    /// full block header then passed its checks already, so the
    /// payload is short.
    const fn partial_tail_error(&self) -> Error {
        if self.decoders.is_none() || self.tail.len() < BlockHeader::LEN {
            Error::TruncatedHeader
        } else {
            Error::TruncatedData
        }
    }

    /// Reject any partial tail, flush, and return the inner writer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TruncatedHeader`] when no header or only a
    /// partial block header remains, [`Error::InvalidMagic`] on bad magic,
    /// [`Error::InvalidBlockSize`] on a bad header block size,
    /// [`Error::TruncatedData`] when a full block header remains with a short payload,
    /// [`Error::OutOfMemory`] when a decoder state fails to build,
    /// [`Error::ThreadPoolFailed`] when the pool build fails,
    /// the block error when a payload fails to decode,
    /// [`Error::Io`] when the flush fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Write;
    /// use bzip3_rs::error::Error;
    /// use bzip3_rs::write::Decoder;
    ///
    /// # fn main() -> std::io::Result<()> {
    /// let mut decoder = Decoder::new(Vec::new());
    /// decoder.write_all(b"BZ3v1")?; // Only 5 bytes of the 9-byte header.
    ///
    /// let result = decoder.finish();
    /// assert!(matches!(result, Err(Error::TruncatedHeader)));
    /// # Ok(())
    /// # }
    /// ```
    pub fn finish(mut self) -> Result<W> {
        self.read_header()?;
        self.drain_blocks()?;
        if !self.tail.is_empty() {
            return Err(self.partial_tail_error());
        }
        if self.decoders.is_none() {
            return Err(Error::TruncatedHeader);
        }
        self.inner.flush()?;
        Ok(self.inner)
    }
}

impl<W: Write> Write for Decoder<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let outcome = match self.decoders.as_mut() {
            // No bytes wait in `tail`, so decode the whole blocks straight
            // from the caller bytes. Only a partial block gets copied.
            Some(decoders) if self.tail.is_empty() => {
                let mut start = 0;
                let outcome =
                    decoders.write_blocks(buf, &mut start, &mut self.plain, &mut self.inner);
                self.tail.extend_from_slice(&buf[start..]);
                outcome
            }
            _ => {
                self.tail.extend_from_slice(buf);
                self.read_header().and_then(|()| self.drain_blocks())
            }
        };
        outcome.map_err(Error::into_io)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::Decoder;
    use super::Encoder;
    use crate::BlockSize;
    use crate::Jobs;
    use crate::StreamHeader;
    use crate::error::Error;
    use std::io::ErrorKind;
    use std::io::Write;
    use std::io::{self};

    /// Sink that takes bytes but fails every flush.
    struct FailFlush;

    impl Write for FailFlush {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::new(ErrorKind::BrokenPipe, "sink fault"))
        }
    }

    /// Sink that fails every write.
    struct FailWrite;

    impl Write for FailWrite {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(ErrorKind::BrokenPipe, "sink fault"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// Encode one framed stream for the decoder tests.
    fn coded_sample() -> (Vec<u8>, Vec<u8>) {
        let input = b"hello world ".repeat(10_000);
        let mut coded = Vec::new();
        crate::stream::compress_slice(&input, &mut coded, BlockSize::MIN, Jobs::SERIAL).unwrap();
        (input, coded)
    }

    #[test]
    fn encoder_debug_names_inner_and_block_size() {
        let text = format!("{:?}", Encoder::new(Vec::new(), BlockSize::MIN).unwrap());
        assert!(text.contains("Encoder"));
        assert!(text.contains("block_size"));
        // A missing writer still formats.
        let mut encoder = Encoder::new(Vec::new(), BlockSize::MIN).unwrap();
        encoder.inner = None;
        let text = format!("{encoder:?}");
        assert!(text.contains("Encoder"));
    }

    #[test]
    fn encoder_accessors_expose_writer_and_block_size() {
        let mut encoder = Encoder::new(Vec::new(), BlockSize::MIN).unwrap();
        assert_eq!(encoder.block_size(), BlockSize::MIN);
        // The constructor wrote the header already.
        assert_eq!(encoder.get_ref().map(Vec::len), Some(StreamHeader::LEN));
        assert!(encoder.get_mut().is_some());
        encoder.write_all(b"hello world").unwrap();
        let coded = encoder.finish().unwrap();
        assert!(coded.len() > StreamHeader::LEN);
    }

    #[test]
    fn encoder_final_block_writes_once() {
        let mut encoder = Encoder::new(Vec::new(), BlockSize::MIN).unwrap();
        encoder.write_all(b"hello world").unwrap();
        encoder.write_final_block().unwrap();
        assert_eq!(encoder.pending, Vec::new());
        // A second call writes nothing new.
        encoder.write_final_block().unwrap();
        let coded = encoder.finish().unwrap();
        let mut plain = Vec::new();
        crate::stream::decompress(coded.as_slice(), &mut plain, Jobs::SERIAL).unwrap();
        assert_eq!(plain, b"hello world");
    }

    #[test]
    fn encoder_write_and_flush_fail_without_writer() {
        let mut encoder = Encoder::new(Vec::new(), BlockSize::MIN).unwrap();
        encoder.inner = None;
        let fault = encoder.write(b"hello").unwrap_err();
        assert_eq!(fault.kind(), ErrorKind::BrokenPipe);
        let fault = encoder.flush().unwrap_err();
        assert_eq!(fault.kind(), ErrorKind::BrokenPipe);
        // A missing writer is a no-op for the final block ...
        encoder.write_final_block().unwrap();
        // ... but `finish` reports it.
        let fault = encoder.finish().unwrap_err();
        assert_eq!(fault.to_string(), "writer missing");
    }

    #[test]
    fn encoder_flush_passes_inner_fault_through() {
        let mut encoder = Encoder::new(FailFlush, BlockSize::MIN).unwrap();
        let fault = encoder.flush().unwrap_err();
        assert_eq!(fault.kind(), ErrorKind::BrokenPipe);
        let mut encoder = Encoder::new(Vec::new(), BlockSize::MIN).unwrap();
        encoder.flush().unwrap();
    }

    #[test]
    fn encoder_header_write_fault_is_io() {
        // Both sink faults surface as I/O faults.
        let mut sink = FailWrite;
        assert!(sink.write(b"header").is_err());
        sink.flush().unwrap();
        assert!(matches!(
            Encoder::new(FailWrite, BlockSize::MIN),
            Err(Error::Io(_))
        ));
    }

    #[test]
    fn decoder_debug_names_inner_and_block_size() {
        let text = format!("{:?}", Decoder::new(Vec::<u8>::new()));
        assert!(text.contains("Decoder"));
        assert!(text.contains("block_size"));
        // A known header still formats.
        let mut decoder = Decoder::new(Vec::new());
        decoder
            .write_all(&StreamHeader::new(BlockSize::MIN).to_bytes())
            .unwrap();
        let text = format!("{decoder:?}");
        assert!(text.contains("Decoder"));
        assert!(text.contains("block_size"));
    }

    #[test]
    fn decoder_block_size_waits_for_full_header() {
        let mut decoder = Decoder::new(Vec::new());
        assert_eq!(decoder.block_size(), None);
        assert_eq!(decoder.get_ref().as_slice(), []);
        assert_eq!(decoder.get_mut().as_slice(), []);
        let header = StreamHeader::new(BlockSize::MIN).to_bytes();
        decoder.write_all(&header[..5]).unwrap();
        assert_eq!(decoder.block_size(), None);
        decoder.write_all(&header[5..]).unwrap();
        assert_eq!(decoder.block_size(), Some(BlockSize::MIN));
    }

    #[test]
    fn decoder_split_writes_match_single_write() {
        let (input, coded) = coded_sample();
        // One byte per write forces the buffered path every time.
        let mut split = Decoder::new(Vec::new());
        for byte in &coded {
            split.write_all(std::slice::from_ref(byte)).unwrap();
        }
        assert_eq!(split.finish().unwrap(), input);
        // One write takes the same bytes.
        let mut single = Decoder::new(Vec::new());
        single.write_all(&coded).unwrap();
        single.flush().unwrap();
        assert_eq!(single.finish().unwrap(), input);
    }

    #[test]
    fn decoder_straight_write_leaves_partial_tail() {
        let (input, coded) = coded_sample();
        let mut decoder = Decoder::new(Vec::new());
        decoder.write_all(&coded[..StreamHeader::LEN]).unwrap();
        // The decoders exist and no bytes wait, so full blocks decode
        // straight from the caller bytes and only the tail is copied.
        let cut = coded.len() - 7;
        decoder.write_all(&coded[StreamHeader::LEN..cut]).unwrap();
        // The last block is short, so its bytes wait in the tail.
        assert_ne!(decoder.tail, Vec::new());
        decoder.write_all(&coded[cut..]).unwrap();
        assert_eq!(decoder.finish().unwrap(), input);
    }

    #[test]
    fn decoder_partial_tail_names_truncation() {
        let (_input, coded) = coded_sample();
        // A short block header is a truncated header.
        let mut decoder = Decoder::new(Vec::new());
        decoder.write_all(&coded[..StreamHeader::LEN]).unwrap();
        decoder.write_all(&coded[StreamHeader::LEN..][..3]).unwrap();
        assert!(matches!(decoder.finish(), Err(Error::TruncatedHeader)));
        // A full block header with a short payload is truncated data.
        let mut decoder = Decoder::new(Vec::new());
        decoder.write_all(&coded[..StreamHeader::LEN]).unwrap();
        decoder
            .write_all(&coded[StreamHeader::LEN..][..8 + 5])
            .unwrap();
        assert!(matches!(decoder.finish(), Err(Error::TruncatedData)));
    }

    #[test]
    fn decoder_finish_rejects_empty_and_bad_magic() {
        assert!(matches!(
            Decoder::new(Vec::new()).finish(),
            Err(Error::TruncatedHeader)
        ));
        // A bad magic fails the write already ...
        let mut decoder = Decoder::new(Vec::new());
        assert!(decoder.write_all(b"BADMAGIC!").is_err());
        // ... and the finish names it.
        assert!(matches!(decoder.finish(), Err(Error::InvalidMagic)));
    }

    #[test]
    fn decoder_flush_passes_inner_fault_through() {
        let mut decoder = Decoder::new(Vec::new());
        decoder.flush().unwrap();
        let mut decoder = Decoder::new(FailFlush);
        let fault = decoder.flush().unwrap_err();
        assert_eq!(fault.kind(), ErrorKind::BrokenPipe);
    }

    #[test]
    fn decoder_finish_flush_fault_is_io() {
        let (_input, coded) = coded_sample();
        let mut decoder = Decoder::new(FailFlush);
        decoder.write_all(&coded).unwrap();
        assert!(matches!(decoder.finish(), Err(Error::Io(_))));
    }

    #[test]
    fn decoder_parallel_roundtrip_matches_serial() {
        let (input, coded) = coded_sample();
        let mut decoder = Decoder::with_jobs(Vec::new(), Jobs::from_count(2));
        assert_eq!(decoder.block_size(), None);
        decoder.write_all(&coded).unwrap();
        assert_eq!(decoder.block_size(), Some(BlockSize::MIN));
        assert_eq!(decoder.finish().unwrap(), input);
    }
}
