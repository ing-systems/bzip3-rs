//! Thread count selection for the parallel codec calls.

use std::num::NonZeroUsize;

/// Thread count for a codec call.
///
/// Every count gives the same output bytes and the same errors.
/// More threads use more memory, because each thread holds one block.
///
/// One thread runs on the caller thread and builds no pool. A larger
/// count builds a private pool for the call. [`Jobs::Global`] uses the
/// global rayon pool. The global pool size follows the rayon default,
/// usually the logical core count. For a physical core default, use
/// [`default_workers`] with [`Jobs::from_count`].
///
/// # Examples
///
/// ```
/// use bzip3_rs::{BlockSize, Jobs, compress_with_jobs, default_workers};
///
/// # fn main() -> Result<(), bzip3_rs::error::Error> {
/// let data = b"hello world ".repeat(1000);
/// let jobs = Jobs::from_count(default_workers());
/// let coded = compress_with_jobs(&data, BlockSize::DEFAULT, jobs)?;
/// assert!(!coded.is_empty());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Jobs {
    /// Use the global rayon pool.
    Global,
    /// Use this many threads.
    Threads(NonZeroUsize),
}

impl Jobs {
    /// One thread, on the caller thread. This is the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::Jobs;
    ///
    /// assert!(Jobs::SERIAL.is_serial());
    /// assert_eq!(Jobs::default(), Jobs::SERIAL);
    /// ```
    pub const SERIAL: Self = Self::Threads(NonZeroUsize::MIN);

    /// Build from a thread count. Zero selects [`Jobs::Global`].
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::Jobs;
    ///
    /// assert_eq!(Jobs::from_count(0), Jobs::Global);
    /// assert_eq!(Jobs::from_count(1), Jobs::SERIAL);
    /// assert!(!Jobs::from_count(4).is_serial());
    /// ```
    #[must_use]
    #[inline]
    pub const fn from_count(count: usize) -> Self {
        match NonZeroUsize::new(count) {
            Some(threads) => Self::Threads(threads),
            None => Self::Global,
        }
    }

    /// Return true when the call runs on the caller thread.
    ///
    /// # Examples
    ///
    /// ```
    /// use bzip3_rs::Jobs;
    ///
    /// assert!(Jobs::from_count(1).is_serial());
    /// assert!(!Jobs::Global.is_serial());
    /// ```
    #[must_use]
    #[inline]
    pub const fn is_serial(self) -> bool {
        matches!(self, Self::Threads(threads) if threads.get() == 1)
    }
}

impl Default for Jobs {
    #[inline]
    fn default() -> Self {
        Self::SERIAL
    }
}

/// Return the default worker count.
///
/// Use the physical core count. Limit the count by the available
/// parallelism, so containers with a quota do not oversubscribe.
/// Return at least one.
///
/// # Examples
///
/// ```
/// use bzip3_rs::default_workers;
///
/// assert!(default_workers() >= 1);
/// ```
#[must_use]
#[inline]
pub fn default_workers() -> usize {
    let physical = num_cpus::get_physical();
    let logical = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    if physical == 0 {
        logical.max(1)
    } else {
        physical.min(logical).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::Jobs;
    use super::default_workers;

    #[test]
    fn zero_selects_global_pool() {
        assert_eq!(Jobs::from_count(0), Jobs::Global);
    }

    #[test]
    fn one_is_serial() {
        assert_eq!(Jobs::from_count(1), Jobs::SERIAL);
        assert!(Jobs::SERIAL.is_serial());
        assert!(!Jobs::Global.is_serial());
        assert!(!Jobs::from_count(2).is_serial());
    }

    #[test]
    fn default_is_serial() {
        assert_eq!(Jobs::default(), Jobs::SERIAL);
    }

    #[test]
    fn default_workers_use_physical_limit() {
        let logical = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        let physical = num_cpus::get_physical();
        let workers = default_workers();
        assert!(workers >= 1);
        assert!(workers <= logical);
        if physical > 0 {
            assert!(workers <= physical);
            assert_eq!(workers, physical.min(logical));
        }
    }

    #[test]
    fn default_workers_never_falls_below_one() {
        for _ in 0..8 {
            assert!(default_workers() >= 1);
        }
        let logical = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        assert!(default_workers() <= logical.max(1));
    }

    #[test]
    fn from_count_covers_edges() {
        assert_eq!(Jobs::from_count(0), Jobs::Global);
        assert_eq!(Jobs::from_count(1), Jobs::SERIAL);
        assert!(!Jobs::from_count(2).is_serial());
        assert!(!Jobs::from_count(usize::MAX).is_serial());
        assert_ne!(Jobs::from_count(usize::MAX), Jobs::Global);
        assert_ne!(Jobs::from_count(usize::MAX), Jobs::SERIAL);
        assert!(!Jobs::from_count(64).is_serial());
    }
}
