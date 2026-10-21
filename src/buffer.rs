//! Fallible buffer allocation shared by the codec stages.
//!
//! Each helper returns `None` when the allocation fails. The caller
//! maps `None` to its own error variant.

/// Clear `buf` and fill it with `len` copies of `value`.
#[inline]
pub fn refill<V: Copy>(buf: &mut Vec<V>, len: usize, value: V) -> Option<()> {
    buf.clear();
    buf.try_reserve(len).ok()?;
    buf.resize(len, value);
    Some(())
}

/// Set the length of `buf` to `len`.
///
/// Keep the old items. New items, if any, get `value`. Use it when
/// the caller overwrites every item before it reads one.
#[inline]
pub fn resize_keep<V: Copy>(buf: &mut Vec<V>, len: usize, value: V) -> Option<()> {
    buf.truncate(len);
    buf.try_reserve(len - buf.len()).ok()?;
    buf.resize(len, value);
    Some(())
}

/// Build a new vector of `len` copies of `value`.
#[inline]
pub fn filled<V: Copy>(len: usize, value: V) -> Option<Vec<V>> {
    let mut buf = Vec::new();
    buf.try_reserve_exact(len).ok()?;
    buf.resize(len, value);
    Some(buf)
}

#[cfg(test)]
mod tests {
    use super::filled;
    use super::refill;
    use super::resize_keep;

    #[test]
    fn resize_keep_keeps_old_content() {
        let mut buf = vec![7_u8; 3];
        resize_keep(&mut buf, 5, 1).unwrap();
        assert_eq!(buf, [7, 7, 7, 1, 1]);
        resize_keep(&mut buf, 2, 0).unwrap();
        assert_eq!(buf, [7, 7]);
    }

    #[test]
    fn refill_replaces_old_content() {
        let mut buf = vec![7_u8; 3];
        refill(&mut buf, 5, 1).unwrap();
        assert_eq!(buf, [1; 5]);
        refill(&mut buf, 2, 0).unwrap();
        assert_eq!(buf, [0; 2]);
    }

    #[test]
    fn filled_gives_exact_length() {
        assert_eq!(filled(4, 9_u8).unwrap(), [9; 4]);
        assert_eq!(filled::<u8>(0, 0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn huge_request_fails_without_abort() {
        assert!(filled::<u64>(usize::MAX / 8, 0).is_none());
    }
}
