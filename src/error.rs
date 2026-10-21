//! Error type for the bzip3 port.
//!
//! Variants keep the C reference codes where they exist. New variants
//! have no reference code. Display strings diverge from the reference text
//! where the reference text misleads. This divergence is recorded here.

use std::fmt::Display;
use std::fmt::Formatter;
use std::io;

/// Fallible result with [`Error`] as the error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Every failure mode of the bzip3 port.
///
/// Ten variants keep the C reference codes. Four variants have
/// no reference code. `OutOfMemory`, `InvalidBlockSize`, and
/// `ThreadPoolFailed` split the old reference init error. `Io` is
/// produced only by the streaming wrappers when a read or write fails.
/// Some reference codes map to more than one variant. Use [`Error::code`]
/// to get the reference code.
///
/// The enum is `#[non_exhaustive]`. A `match` needs a wildcard arm.
///
/// # Examples
///
/// ```
/// use bzip3_rs::decompress;
/// use bzip3_rs::error::Error;
///
/// match decompress(&[0_u8; 13]) {
///     Ok(_) => unreachable!("zero bytes are not a frame"),
///     Err(Error::InvalidMagic) => {}
///     Err(other) => panic!("unexpected error: {other}"),
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Burrows-Wheeler transform failed (reference code -2).
    BwtFailed,
    /// Checksum of the decoded block does not match (reference code -3).
    ChecksumMismatch,
    /// RLE or LZP data is corrupt (reference code -3).
    CorruptStageData,
    /// Magic number is wrong (reference code -4).
    InvalidMagic,
    /// Header is shorter than its fixed size (reference code -4).
    TruncatedHeader,
    /// Header field is out of range or inconsistent (reference code -4).
    InvalidHeader,
    /// Data ends before the end of a payload (reference code -5).
    TruncatedData,
    /// Block too large (reference code -6).
    BlockTooLarge,
    /// Payload is larger than the decode buffer (reference code -8).
    BufferTooSmall,
    /// Output buffer length is not the original block size (reference code -8).
    OutputSizeMismatch,
    /// Working memory allocation failed. No reference code.
    OutOfMemory,
    /// Block size outside the valid range. No reference code.
    InvalidBlockSize,
    /// Private thread pool build failed. No reference code.
    ThreadPoolFailed,
    /// Read or write failure inside a streaming wrapper. No reference code.
    Io(io::Error),
}

impl Error {
    /// Map a codec error to an I/O error for streaming wrappers.
    ///
    /// I/O faults wrapped in [`Error::Io`] pass through
    /// unchanged. Codec faults become invalid-data errors that keep the
    /// original [`Error`] as their source.
    #[must_use]
    pub(crate) fn into_io(self) -> io::Error {
        match self {
            Self::Io(e) => e,
            other => io::Error::new(io::ErrorKind::InvalidData, other),
        }
    }

    /// Numeric reference code (-2 to -8).
    ///
    /// Returns `None` for variants with no reference code.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::error::Error;
    ///
    /// assert_eq!(Error::InvalidMagic.code(), Some(-4));
    /// assert_eq!(Error::InvalidBlockSize.code(), None);
    /// ```
    #[must_use]
    pub const fn code(&self) -> Option<i32> {
        match self {
            Self::BwtFailed => Some(-2),
            Self::ChecksumMismatch | Self::CorruptStageData => Some(-3),
            Self::InvalidMagic | Self::TruncatedHeader | Self::InvalidHeader => Some(-4),
            Self::TruncatedData => Some(-5),
            Self::BlockTooLarge => Some(-6),
            Self::BufferTooSmall | Self::OutputSizeMismatch => Some(-8),
            Self::OutOfMemory | Self::InvalidBlockSize | Self::ThreadPoolFailed | Self::Io(_) => {
                None
            }
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BwtFailed => write!(f, "Burrows-Wheeler transform failed"),
            Self::ChecksumMismatch => write!(f, "Checksum mismatch"),
            Self::CorruptStageData => write!(f, "Corrupt RLE or LZP data"),
            Self::InvalidMagic => write!(f, "Invalid magic number"),
            Self::TruncatedHeader => write!(f, "Truncated header"),
            Self::InvalidHeader => write!(f, "Invalid header"),
            Self::TruncatedData => write!(f, "Truncated data"),
            Self::BlockTooLarge => write!(f, "Block too large"),
            Self::BufferTooSmall => write!(f, "Decode buffer too small"),
            Self::OutputSizeMismatch => write!(f, "Output buffer size mismatch"),
            Self::OutOfMemory => write!(f, "Out of memory"),
            Self::InvalidBlockSize => write!(f, "Invalid block size"),
            Self::ThreadPoolFailed => write!(f, "Thread pool build failed"),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

impl From<io::Error> for Error {
    #[inline]
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::Error;
    use super::Result;
    use std::error::Error as StdError;
    use std::io;

    #[test]
    fn display_strings_match_port() {
        let cases: &[(Error, &str)] = &[
            (Error::BwtFailed, "Burrows-Wheeler transform failed"),
            (Error::ChecksumMismatch, "Checksum mismatch"),
            (Error::CorruptStageData, "Corrupt RLE or LZP data"),
            (Error::InvalidMagic, "Invalid magic number"),
            (Error::TruncatedHeader, "Truncated header"),
            (Error::InvalidHeader, "Invalid header"),
            (Error::TruncatedData, "Truncated data"),
            (Error::BlockTooLarge, "Block too large"),
            (Error::BufferTooSmall, "Decode buffer too small"),
            (Error::OutputSizeMismatch, "Output buffer size mismatch"),
            (Error::OutOfMemory, "Out of memory"),
            (Error::InvalidBlockSize, "Invalid block size"),
            (Error::ThreadPoolFailed, "Thread pool build failed"),
        ];
        for (e, want) in cases {
            assert_eq!(e.to_string(), *want);
        }
    }

    #[test]
    fn codes_match_reference() {
        assert_eq!(Error::BwtFailed.code(), Some(-2));
        assert_eq!(Error::ChecksumMismatch.code(), Some(-3));
        assert_eq!(Error::CorruptStageData.code(), Some(-3));
        assert_eq!(Error::InvalidMagic.code(), Some(-4));
        assert_eq!(Error::TruncatedHeader.code(), Some(-4));
        assert_eq!(Error::InvalidHeader.code(), Some(-4));
        assert_eq!(Error::TruncatedData.code(), Some(-5));
        assert_eq!(Error::BlockTooLarge.code(), Some(-6));
        assert_eq!(Error::BufferTooSmall.code(), Some(-8));
        assert_eq!(Error::OutputSizeMismatch.code(), Some(-8));
        assert_eq!(Error::OutOfMemory.code(), None);
        assert_eq!(Error::InvalidBlockSize.code(), None);
        assert_eq!(Error::ThreadPoolFailed.code(), None);
    }

    #[test]
    fn io_variant_carries_io_error() {
        let cause = io::Error::new(io::ErrorKind::BrokenPipe, "pipe broke");
        let e = Error::from(cause);
        assert!(matches!(e, Error::Io(_)));
        assert_eq!(e.code(), None);
        assert_eq!(e.to_string(), "pipe broke");
        assert!(e.source().is_some());
    }

    #[test]
    fn non_io_variants_have_no_source() {
        assert!(Error::ChecksumMismatch.source().is_none());
    }

    #[test]
    fn result_alias_uses_error() {
        let r: Result<u32> = Err(Error::TruncatedData);
        assert!(matches!(r, Err(Error::TruncatedData)));
    }

    struct HeaderThenFail {
        rest: Vec<u8>,
        text: &'static str,
    }

    impl io::Read for HeaderThenFail {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.rest.is_empty() {
                return Err(io::Error::new(io::ErrorKind::BrokenPipe, self.text));
            }
            let n = self.rest.len().min(buf.len());
            buf[..n].copy_from_slice(&self.rest[..n]);
            self.rest.drain(..n);
            Ok(n)
        }
    }

    struct FailWrite;

    impl io::Write for FailWrite {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "pipe broke"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn read_decoder_passes_through_io_faults() {
        use crate::BlockSize;
        use crate::StreamHeader;
        use std::io::Read as _;
        let header = StreamHeader::new(BlockSize::MIN).to_bytes().to_vec();
        let mut decoder = crate::read::Decoder::new(HeaderThenFail {
            rest: header,
            text: "pipe broke",
        })
        .expect("header reads fine");
        let mut out = Vec::new();
        let fault = decoder.read_to_end(&mut out).unwrap_err();
        assert_eq!(fault.kind(), io::ErrorKind::BrokenPipe);
        assert_eq!(fault.to_string(), "pipe broke");
    }

    #[test]
    fn read_decoder_maps_codec_faults_to_invalid_data() {
        use crate::BlockSize;
        use crate::StreamHeader;
        use std::io::Read as _;
        let mut coded = StreamHeader::new(BlockSize::MIN).to_bytes().to_vec();
        coded.extend_from_slice(&[1_u8, 0, 0, 0, 1, 0, 0, 0, 0xFF]);
        let mut decoder =
            crate::read::Decoder::new(std::io::Cursor::new(coded)).expect("header reads fine");
        let mut out = Vec::new();
        let fault = decoder.read_to_end(&mut out).unwrap_err();
        assert_eq!(fault.kind(), io::ErrorKind::InvalidData);
        assert!(fault.get_ref().is_some());
    }

    #[test]
    fn write_decoder_passes_through_io_faults() {
        use crate::BlockSize;
        use crate::Jobs;
        use std::io::Write as _;
        let input = b"hello world ".repeat(100);
        let mut coded = Vec::new();
        crate::stream::compress_slice(&input, &mut coded, BlockSize::MIN, Jobs::SERIAL).unwrap();
        let mut decoder = crate::write::Decoder::new(FailWrite);
        decoder.write_all(&coded[..9]).unwrap();
        decoder.flush().unwrap();
        let fault = decoder.write_all(&coded[9..]).unwrap_err();
        assert_eq!(fault.kind(), io::ErrorKind::BrokenPipe);
    }

    #[test]
    fn write_encoder_passes_through_io_faults() {
        use crate::BlockSize;
        use std::io::Write as _;
        struct FailAfterFirst {
            seen: usize,
        }
        impl io::Write for FailAfterFirst {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.seen += buf.len();
                if self.seen > 9 {
                    return Err(io::Error::new(io::ErrorKind::BrokenPipe, "pipe broke"));
                }
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut encoder =
            crate::write::Encoder::new(FailAfterFirst { seen: 0 }, BlockSize::MIN).unwrap();
        let chunk = vec![b'x'; BlockSize::MIN.as_usize()];
        let fault = encoder.write(&chunk).unwrap_err();
        assert_eq!(fault.kind(), io::ErrorKind::BrokenPipe);
    }
}
