//! Block model byte and shared header sizing.

use super::MODEL_LEN;
use super::PREFIX_LEN;

/// Model byte of one coded block.
///
/// Bit `0x02` selects the LZP stage. Bit `0x04` selects the RLE
/// stage. Other bits stay clear on encode. The decoder reads any
/// wire value but only these bits select stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model(pub u8);

impl Model {
    /// Bit set when the LZP stage applies.
    const LZP: u8 = 0x02;
    /// Bit set when the RLE stage applies.
    const RLE: u8 = 0x04;

    /// True when the LZP stage applies.
    #[inline]
    pub const fn has_lzp(self) -> bool {
        self.0 & Self::LZP != 0
    }

    /// True when the RLE stage applies.
    #[inline]
    pub const fn has_rle(self) -> bool {
        self.0 & Self::RLE != 0
    }

    /// Mark the LZP stage as applied.
    #[inline]
    pub const fn set_lzp(&mut self) {
        self.0 |= Self::LZP;
    }

    /// Mark the RLE stage as applied.
    #[inline]
    pub const fn set_rle(&mut self) {
        self.0 |= Self::RLE;
    }

    /// Full header length: prefix, model byte, kept stage sizes.
    pub const fn header_len(self) -> usize {
        let mut len = PREFIX_LEN + MODEL_LEN;
        if self.has_lzp() {
            len += 4;
        }
        if self.has_rle() {
            len += 4;
        }
        len
    }
}
