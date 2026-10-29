//! Block codecs: shared header logic plus forward and inverse pipelines.
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

mod decode;
mod encode;
mod model;

pub use decode::BlockDecoder;
pub use decode::BlockStatus;
pub use encode::BlockEncoder;

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

/// Fields read from the fixed prefix of one block payload.
#[derive(Debug, Clone, Copy)]
pub struct PayloadPrefix {
    /// Claimed original size, at most the block size.
    pub original_len: usize,
    /// CRC32 stored in the payload.
    pub crc: u32,
    /// BWT index, or [`STORED_MARKER`] for a raw block.
    pub bwt_index: i32,
    /// Bound of the block size, the decode buffer size.
    pub limit: usize,
}
