//! Range-checked block size for the bzip3 port.

use crate::error::Error;
use crate::error::Result;

/// Block size in bytes, checked against the valid range.
///
/// Sizes run 65 KiB through 511 MiB. The default is 16 MiB.
/// Level 1 selects `MIN`, level 14 selects `MAX`; levels 2-13 are
/// powers of two from 128 KiB to 256 MiB.
///
/// # Examples
///
/// ```
/// use bzip3_rs::BlockSize;
///
/// # fn main() -> Result<(), bzip3_rs::error::Error> {
/// let size = BlockSize::kib(128)?;
/// assert_eq!(size.get(), 131_072);
/// assert!(BlockSize::MIN < size && size < BlockSize::MAX);
/// assert_eq!(BlockSize::default(), BlockSize::DEFAULT);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct BlockSize(u32);

impl BlockSize {
    /// Smallest valid size: 65 KiB.
    pub const MIN: Self = Self(65 * 1024);
    /// Largest valid size: 511 MiB.
    pub const MAX: Self = Self(511 * 1024 * 1024);
    /// Default size: 16 MiB.
    pub const DEFAULT: Self = Self(16 * 1024 * 1024);

    /// Build from a byte count.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBlockSize`] when `size` is outside `MIN..=MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::BlockSize;
    ///
    /// assert!(BlockSize::new(1024 * 1024).is_ok());
    /// assert!(BlockSize::new(1024).is_err()); // Below `MIN`.
    /// assert!(BlockSize::new(u32::MAX).is_err()); // Above `MAX`.
    /// ```
    #[inline]
    pub const fn new(size: u32) -> Result<Self> {
        if size < Self::MIN.0 || size > Self::MAX.0 {
            return Err(Error::InvalidBlockSize);
        }
        Ok(Self(size))
    }

    /// Build from a kibibyte count.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBlockSize`] when the size is outside `MIN..=MAX`
    /// or the multiplication overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::BlockSize;
    ///
    /// # fn main() -> Result<(), bzip3_rs::error::Error> {
    /// assert_eq!(BlockSize::kib(65)?, BlockSize::MIN);
    /// assert!(BlockSize::kib(64).is_err());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub const fn kib(kibibytes: u32) -> Result<Self> {
        match kibibytes.checked_mul(1024) {
            Some(size) => Self::new(size),
            None => Err(Error::InvalidBlockSize),
        }
    }

    /// Build from a mebibyte count.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBlockSize`] when the size is outside `MIN..=MAX`
    /// or the multiplication overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::BlockSize;
    ///
    /// # fn main() -> Result<(), bzip3_rs::error::Error> {
    /// assert_eq!(BlockSize::mib(16)?, BlockSize::DEFAULT);
    /// assert!(BlockSize::mib(512).is_err());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub const fn mib(mebibytes: u32) -> Result<Self> {
        match mebibytes.checked_mul(1024 * 1024) {
            Some(size) => Self::new(size),
            None => Err(Error::InvalidBlockSize),
        }
    }

    /// Build from a compression level 1-14.
    ///
    /// Level 1 selects [`Self::MIN`] (65 KiB). Levels 2-13 are powers
    /// of two: 128 KiB, 256 KiB, 512 KiB, 1 MiB, 2 MiB, 4 MiB, 8 MiB,
    /// 16 MiB, 32 MiB, 64 MiB, 128 MiB, 256 MiB.
    /// Level 14 selects [`Self::MAX`] (511 MiB).
    /// Level 9 matches [`Self::DEFAULT`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBlockSize`] when `level` is outside 1-14.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::BlockSize;
    ///
    /// # fn main() -> Result<(), bzip3_rs::error::Error> {
    /// assert_eq!(BlockSize::from_level(1)?, BlockSize::MIN);
    /// assert_eq!(BlockSize::from_level(9)?, BlockSize::DEFAULT);
    /// assert_eq!(BlockSize::from_level(14)?, BlockSize::MAX);
    /// assert!(BlockSize::from_level(0).is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub const fn from_level(level: u8) -> Result<Self> {
        match level {
            1 => Self::kib(65),
            2 => Self::kib(128),
            3 => Self::kib(256),
            4 => Self::kib(512),
            5 => Self::mib(1),
            6 => Self::mib(2),
            7 => Self::mib(4),
            8 => Self::mib(8),
            9 => Self::mib(16),
            10 => Self::mib(32),
            11 => Self::mib(64),
            12 => Self::mib(128),
            13 => Self::mib(256),
            14 => Self::mib(511),
            _ => Err(Error::InvalidBlockSize),
        }
    }

    /// Return the size in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::BlockSize;
    ///
    /// assert_eq!(BlockSize::MIN.get(), 65 * 1024);
    /// ```
    #[must_use]
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Return the size in bytes as `usize`.
    ///
    /// On 16-bit targets the value truncates like `as`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::BlockSize;
    ///
    /// let buffer = vec![0_u8; BlockSize::MIN.as_usize()];
    /// assert_eq!(buffer.len(), 65 * 1024);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl Default for BlockSize {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::BlockSize;
    use crate::error::Error;

    #[test]
    fn limits_match_valid_range() {
        assert_eq!(BlockSize::MIN.get(), 65 * 1024);
        assert_eq!(BlockSize::MAX.get(), 511 * 1024 * 1024);
        assert_eq!(BlockSize::DEFAULT.get(), 16 * 1024 * 1024);
    }

    #[test]
    fn default_is_16_mib() {
        assert_eq!(BlockSize::default(), BlockSize::DEFAULT);
    }

    #[test]
    fn new_accepts_boundaries() {
        assert_eq!(BlockSize::new(65 * 1024).unwrap().get(), 65 * 1024);
        assert_eq!(
            BlockSize::new(511 * 1024 * 1024).unwrap().get(),
            511 * 1024 * 1024
        );
        assert_eq!(
            BlockSize::new(16 * 1024 * 1024).unwrap(),
            BlockSize::DEFAULT
        );
    }

    #[test]
    fn new_rejects_outside_range() {
        for size in [0, 1, 65 * 1024 - 1, 511 * 1024 * 1024 + 1, u32::MAX] {
            assert!(
                matches!(BlockSize::new(size), Err(Error::InvalidBlockSize)),
                "{size}"
            );
        }
    }

    #[test]
    fn kib_ctor_scales_and_checks() {
        assert_eq!(BlockSize::kib(65).unwrap().get(), 65 * 1024);
        assert_eq!(BlockSize::kib(16 * 1024).unwrap(), BlockSize::DEFAULT);
        assert!(matches!(BlockSize::kib(64), Err(Error::InvalidBlockSize)));
        assert!(matches!(BlockSize::kib(0), Err(Error::InvalidBlockSize)));
        assert!(matches!(
            BlockSize::kib(u32::MAX),
            Err(Error::InvalidBlockSize)
        ));
    }

    #[test]
    fn mib_ctor_scales_and_checks() {
        assert_eq!(BlockSize::mib(16).unwrap(), BlockSize::DEFAULT);
        assert_eq!(BlockSize::mib(1).unwrap().get(), 1024 * 1024);
        assert_eq!(BlockSize::mib(511).unwrap(), BlockSize::MAX);
        assert!(matches!(BlockSize::mib(0), Err(Error::InvalidBlockSize)));
        assert!(matches!(BlockSize::mib(512), Err(Error::InvalidBlockSize)));
        assert!(matches!(
            BlockSize::mib(u32::MAX),
            Err(Error::InvalidBlockSize)
        ));
    }

    #[test]
    fn level_maps_powers_of_two() {
        for (level, bytes) in [
            (1, 65 * 1024),
            (2, 128 * 1024),
            (3, 256 * 1024),
            (4, 512 * 1024),
            (5, 1024 * 1024),
            (6, 2 * 1024 * 1024),
            (7, 4 * 1024 * 1024),
            (8, 8 * 1024 * 1024),
            (9, 16 * 1024 * 1024),
            (10, 32 * 1024 * 1024),
            (11, 64 * 1024 * 1024),
            (12, 128 * 1024 * 1024),
            (13, 256 * 1024 * 1024),
            (14, 511 * 1024 * 1024),
        ] {
            assert_eq!(
                BlockSize::from_level(level).unwrap().get(),
                bytes,
                "level {level}"
            );
        }
        assert_eq!(BlockSize::from_level(1).unwrap(), BlockSize::MIN);
        assert_eq!(BlockSize::from_level(14).unwrap(), BlockSize::MAX);
        assert_eq!(BlockSize::from_level(9).unwrap(), BlockSize::DEFAULT);
        for level in [0, 15, u8::MAX] {
            assert!(
                matches!(BlockSize::from_level(level), Err(Error::InvalidBlockSize)),
                "level {level}"
            );
        }
    }
}
