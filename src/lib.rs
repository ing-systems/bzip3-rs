//! Pure-Rust port of the bzip3 compressor.
//!
//! This crate reads and writes the bzip3 format. It has no C code and no
//! unsafe code. Output is compatible with C bzip3 in both directions.
//!
//! Each block passes through four stages: run-length coding, LZP,
//! a Burrows-Wheeler transform, and an order-0 context-mixing coder.
//! The crate checks a CRC32 value for each block.
//!
//! # Example
//!
//! ```
//! use bzip3_rs::{BlockSize, compress, decompress};
//!
//! # fn main() -> Result<(), bzip3_rs::error::Error> {
//! let data = b"hello world ".repeat(1000);
//! let coded = compress(&data, BlockSize::DEFAULT)?;
//! let plain = decompress(&coded)?;
//! assert_eq!(plain, data);
//! # Ok(())
//! # }
//! ```
//!
//! # Where to start
//!
//! - [`compress`] and [`decompress`] process a whole buffer.
//! - [`compress_with_jobs`] and [`decompress_with_jobs`] process blocks in parallel. See [`Jobs`].
//! - Module [`stream`] processes [`std::io::Read`] and [`std::io::Write`] values.
//! - Modules [`mod@read`] and [`mod@write`] hold encoder and decoder adapters.
//! - [`BlockEncoder`] and [`BlockDecoder`] process one block at a time.
//!   [`BlockDecoder::recover_block_into`] keeps the data of a damaged block
//!   and returns a [`BlockStatus`] value.
//! - [`BlockSize`] holds the checked block size.
//!
//! # Frame formats
//!
//! The crate uses two frame formats. They do not mix.
//!
//! - The one-shot frame has a 13-byte header: magic, block size, and block count.
//!   [`compress`] and [`decompress`] use it.
//! - The streaming frame has a 9-byte header: magic and block size.
//!   Blocks follow until the input ends.
//!   Module [`stream`], modules [`mod@read`] and [`mod@write`], and the `bzip3` tool use it.
//!
//! Each decoder reads only its own frame. Use the decoder that matches the encoder.
//!
//! ```
//! use bzip3_rs::{BlockSize, decompress, stream};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut coded = Vec::new();
//! stream::compress_slice(b"hello world", &mut coded, BlockSize::MIN, bzip3_rs::Jobs::SERIAL)?;
//!
//! // `decompress` reads the one-shot frame, so it rejects a streaming frame.
//! assert!(decompress(&coded).is_err());
//! # Ok(())
//! # }
//! ```
//!
//! # Errors
//!
//! Functions, constructors, and `finish` methods return [`error::Result`].
//!
//! The `read` and `write` adapters implement [`std::io::Read`] and [`std::io::Write`].
//! Their `read` and `write` methods return [`std::io::Error`].
//! A codec fault has the kind [`std::io::ErrorKind::InvalidData`].
//! The [`error::Error`] value is the source of that I/O error.
//!
//! ```
//! use std::io::{Cursor, ErrorKind, Read};
//! use bzip3_rs::error::Error;
//! use bzip3_rs::{BlockSize, Jobs, read, stream};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut coded = Vec::new();
//! stream::compress_slice(&b"hello world ".repeat(1000), &mut coded, BlockSize::MIN, Jobs::SERIAL)?;
//! let middle = coded.len() / 2;
//! coded[middle] ^= 0xFF; // Damage one payload byte.
//!
//! let mut decoder = read::Decoder::new(Cursor::new(coded.as_slice()))?;
//! let fault = decoder.read_to_end(&mut Vec::new()).unwrap_err();
//! assert_eq!(fault.kind(), ErrorKind::InvalidData);
//! let codec = fault.get_ref().and_then(|source| source.downcast_ref::<Error>());
//! assert!(codec.is_some());
//! # Ok(())
//! # }
//! ```
//!
//! # Parallelism
//!
//! Every [`Jobs`] value gives the same output bytes.
//! More threads use more memory, because each thread holds one block.
//!
//! # Cargo features
//!
//! The library needs no feature. Feature `cli` builds the `bzip3` tool.

#![forbid(unsafe_code)]
// Decision on the review `deny(warnings)` claim: ticket 01 asks for
// a "deny-by-default lint set", which is exactly the clippy denies
// below. No crate-wide `deny(warnings)` follows on purpose: a
// library deny would break downstream builds on newer toolchains.
#![deny(clippy::all, clippy::pedantic)]
#![warn(clippy::nursery)]
#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

mod batch;
mod block;
mod block_size;
mod buffer;
mod cm;
mod crc32;
pub mod error;
mod frame;
mod jobs;
mod lzp;
pub mod read;
mod rle;
mod sais;
pub mod stream;
pub mod write;

/// Run-length stage access for benches. Not part of the public API.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod rle_bench {
    pub use crate::rle::decode_into;
    pub use crate::rle::encode_into;
}

/// Suffix-array stage access for benches. Not part of the public API.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod sais_bench {
    pub use crate::sais::BwtScratch;
    pub use crate::sais::UnbwtScratch;
    pub use crate::sais::bwt_with;
    pub use crate::sais::unbwt_with;
}

pub use block::BlockDecoder;
pub use block::BlockEncoder;
pub use block::BlockStatus;
pub use block_size::BlockSize;
pub use frame::BlockHeader;
pub use frame::StreamHeader;
pub use jobs::Jobs;
pub use jobs::default_workers;

/// Magic bytes opening every frame.
///
/// # Examples
///
/// ```
/// use bzip3_rs::{BlockSize, MAGIC_NUMBER, compress};
///
/// # fn main() -> Result<(), bzip3_rs::error::Error> {
/// let coded = compress(b"hello world", BlockSize::DEFAULT)?;
/// assert!(coded.starts_with(&MAGIC_NUMBER));
/// # Ok(())
/// # }
/// ```
pub const MAGIC_NUMBER: [u8; 5] = *b"BZ3v1";

/// Compress `data` into one-shot framed output.
///
/// The header records the effective block size. This is the requested size.
/// When the input is smaller, it is `compress_bound(data.len())` with a minimum of 65 KiB.
/// Every block runs the full encode pipeline through one reused
/// encoder state. The output grows as needed, so no capacity error occurs.
///
/// # Errors
///
/// Returns [`Error::InvalidBlockSize`](error::Error::InvalidBlockSize)
/// when the clamped size leaves the valid range.
/// Returns [`Error::OutOfMemory`](error::Error::OutOfMemory)
/// when the encoder state allocation fails.
/// Returns [`Error::BlockTooLarge`](error::Error::BlockTooLarge)
/// when a block exceeds the effective size.
/// Returns [`Error::BwtFailed`](error::Error::BwtFailed)
/// when a block transform fails.
///
/// # Examples
///
/// ```
/// use bzip3_rs::{BlockSize, compress, decompress};
///
/// # fn main() -> Result<(), bzip3_rs::error::Error> {
/// let data = b"hello world ".repeat(1000);
/// let coded = compress(&data, BlockSize::DEFAULT)?;
/// assert!(coded.len() < data.len());
/// assert_eq!(decompress(&coded)?, data);
/// # Ok(())
/// # }
/// ```
#[must_use = "compressed output must be used or stored"]
#[inline]
pub fn compress(data: &[u8], block_size: BlockSize) -> error::Result<Vec<u8>> {
    frame::encode_frame(data, block_size, Jobs::SERIAL)
}

/// Compress `data` into one-shot framed output on a thread pool.
///
/// Output bytes are identical to [`compress`] at every job count.
/// `jobs` selects the thread count. See [`Jobs`].
///
/// # Errors
///
/// Returns the errors of [`compress`]. Returns
/// [`Error::ThreadPoolFailed`](error::Error::ThreadPoolFailed) when the private pool
/// fails to build.
///
/// # Examples
///
/// ```
/// use bzip3_rs::{BlockSize, Jobs, compress, compress_with_jobs};
///
/// # fn main() -> Result<(), bzip3_rs::error::Error> {
/// let data = b"hello world ".repeat(1000);
/// let serial = compress(&data, BlockSize::DEFAULT)?;
/// let parallel = compress_with_jobs(&data, BlockSize::DEFAULT, Jobs::from_count(2))?;
/// assert_eq!(serial, parallel);
/// # Ok(())
/// # }
/// ```
#[must_use = "compressed output must be used or stored"]
#[inline]
pub fn compress_with_jobs(
    data: &[u8],
    block_size: BlockSize,
    jobs: Jobs,
) -> error::Result<Vec<u8>> {
    frame::encode_frame(data, block_size, jobs)
}

/// Decompress one-shot framed input.
///
/// Trailing bytes after the last block are ignored. The output grows
/// as needed, so no capacity error occurs.
///
/// # Errors
///
/// Returns [`error::Error::TruncatedHeader`] on a short header,
/// [`error::Error::InvalidMagic`] on bad magic,
/// [`error::Error::InvalidHeader`] on bad sizes,
/// [`error::Error::TruncatedData`] on a short payload,
/// [`error::Error::InvalidBlockSize`] on an invalid header block size, and the
/// block error when a payload fails to decode.
///
/// # Examples
///
/// ```
/// use bzip3_rs::error::Error;
/// use bzip3_rs::{BlockSize, compress, decompress};
///
/// # fn main() -> Result<(), Error> {
/// let data = b"hello world ".repeat(1000);
/// let coded = compress(&data, BlockSize::DEFAULT)?;
/// assert_eq!(decompress(&coded)?, data);
///
/// // Input with a wrong magic number fails.
/// let bad = [0_u8; 13];
/// assert!(matches!(decompress(&bad), Err(Error::InvalidMagic)));
/// # Ok(())
/// # }
/// ```
#[must_use = "decompressed output must be used or stored"]
#[inline]
pub fn decompress(data: &[u8]) -> error::Result<Vec<u8>> {
    frame::decode_frame(data, Jobs::SERIAL)
}

/// Decompress one-shot framed input on a thread pool.
///
/// Output bytes and errors are identical to [`decompress`] at every
/// job count. `jobs` follows the rule of [`compress_with_jobs`].
///
/// # Errors
///
/// Returns the errors of [`decompress`]. Returns
/// [`error::Error::ThreadPoolFailed`] when the private pool fails to build.
///
/// # Examples
///
/// ```
/// use bzip3_rs::{BlockSize, Jobs, compress, decompress_with_jobs};
///
/// # fn main() -> Result<(), bzip3_rs::error::Error> {
/// let data = b"hello world ".repeat(1000);
/// let coded = compress(&data, BlockSize::DEFAULT)?;
/// assert_eq!(decompress_with_jobs(&coded, Jobs::Global)?, data);
/// # Ok(())
/// # }
/// ```
#[must_use = "decompressed output must be used or stored"]
#[inline]
pub fn decompress_with_jobs(data: &[u8], jobs: Jobs) -> error::Result<Vec<u8>> {
    frame::decode_frame(data, jobs)
}

/// Estimate the largest compressed output for an input of `input_size` bytes.
///
/// Matches the reference bound: `input_size + input_size / 50 + 32`. Saturates
/// instead of wrapping on huge inputs, so the call never panics.
///
/// # Examples
///
/// ```
/// use bzip3_rs::{BlockSize, compress, compress_bound};
///
/// # fn main() -> Result<(), bzip3_rs::error::Error> {
/// let data = b"hello world ".repeat(1000);
/// let coded = compress(&data, BlockSize::DEFAULT)?;
/// assert!(coded.len() <= compress_bound(data.len()));
/// assert_eq!(compress_bound(1000), 1052);
/// # Ok(())
/// # }
/// ```
#[must_use]
#[inline]
pub const fn compress_bound(input_size: usize) -> usize {
    input_size
        .saturating_add(input_size / 50)
        .saturating_add(32)
}

#[cfg(test)]
mod tests {
    use super::compress_bound;

    #[test]
    fn compress_bound_adds_fiftieth_plus_32() {
        assert_eq!(compress_bound(0), 32);
        assert_eq!(compress_bound(1), 33);
        assert_eq!(compress_bound(50), 83);
        assert_eq!(compress_bound(100), 134);
        assert_eq!(compress_bound(1_000_000), 1_020_032);
    }

    #[test]
    fn compress_bound_saturates_on_huge_input() {
        assert_eq!(compress_bound(usize::MAX), usize::MAX);
    }
}

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
