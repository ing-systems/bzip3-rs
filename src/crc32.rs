//! CRC32 checksum used by block headers.
//!
//! Table-driven sum matching the reference `crc32sum`. Seed is 1.
//! The table is the reflected CRC-32C table. A `const` loop builds it.

/// Reflected CRC-32C polynomial.
const POLYNOMIAL: u32 = 0x82F6_3B78;

/// Lookup table for the checksum loop (reference values).
const TABLE: [u32; 256] = {
    let mut table = [0_u32; 256];
    let mut i = 0_u32;
    while i < 256 {
        let mut crc = i;
        let mut bit = 0;
        while bit < 8 {
            crc = (crc >> 1) ^ (POLYNOMIAL & (crc & 1).wrapping_neg());
            bit += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
};

/// Lookup tables for the 8-byte loop.
///
/// Table 0 is the reference table. Table `k` advances a byte by `k`
/// further zero bytes, so eight bytes combine with eight lookups.
static SLICES: [[u32; 256]; 8] = {
    let mut tables = [[0_u32; 256]; 8];
    tables[0] = TABLE;
    let mut k = 1;
    while k < 8 {
        let mut i = 0;
        while i < 256 {
            let prev = tables[k - 1][i];
            tables[k][i] = (prev >> 8) ^ TABLE[(prev & 0xFF) as usize];
            i += 1;
        }
        k += 1;
    }
    tables
};

/// Add `data` to the running checksum `crc`.
///
/// The reference calls this as `crc32sum(1, data, size)`. The loop
/// folds eight bytes per step and gives the same sum as the
/// reference byte loop.
#[must_use]
pub fn sum(mut crc: u32, data: &[u8]) -> u32 {
    let (blocks, rest) = data.as_chunks::<8>();
    for block in blocks {
        let low = u32::from_le_bytes([block[0], block[1], block[2], block[3]]) ^ crc;
        let high = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        crc = SLICES[7][(low & 0xFF) as usize]
            ^ SLICES[6][((low >> 8) & 0xFF) as usize]
            ^ SLICES[5][((low >> 16) & 0xFF) as usize]
            ^ SLICES[4][(low >> 24) as usize]
            ^ SLICES[3][(high & 0xFF) as usize]
            ^ SLICES[2][((high >> 8) & 0xFF) as usize]
            ^ SLICES[1][((high >> 16) & 0xFF) as usize]
            ^ SLICES[0][(high >> 24) as usize];
    }
    for byte in rest {
        let index = usize::from(*byte ^ crc.to_le_bytes()[0]);
        crc = TABLE[index] ^ (crc >> 8);
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::TABLE;
    use super::sum;

    #[test]
    fn table_has_256_reference_entries() {
        assert_eq!(TABLE.len(), 256);
        assert_eq!(TABLE[0], 0);
        assert_eq!(TABLE[1], 0xF26B_8303);
        assert_eq!(TABLE[2], 0xE13B_70F7);
        assert_eq!(TABLE[128], 0x82F6_3B78);
        assert_eq!(TABLE[255], 0xAD7D_5351);
    }

    /// The reference byte loop.
    fn sum_bytewise(mut crc: u32, data: &[u8]) -> u32 {
        for byte in data {
            crc = TABLE[usize::from(*byte ^ crc.to_le_bytes()[0])] ^ (crc >> 8);
        }
        crc
    }

    #[test]
    fn eight_byte_loop_matches_byte_loop() {
        let mut state = 0x1234_5678_u32;
        let data: Vec<u8> = (0..300)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                state.to_le_bytes()[2]
            })
            .collect();
        for len in 0..data.len() {
            for seed in [1, 0, 0xFFFF_FFFF, 0xDEAD_BEEF] {
                assert_eq!(
                    sum(seed, &data[..len]),
                    sum_bytewise(seed, &data[..len]),
                    "len {len} seed {seed:#x}"
                );
            }
        }
    }

    #[test]
    fn oracle_vectors_match() {
        assert_eq!(sum(1, b"A"), 0x417B_1DBC);
        assert_eq!(sum(1, &[b'x'; 63]), 0x7954_2176);
        assert_eq!(sum(1, b""), 1);
    }
}
