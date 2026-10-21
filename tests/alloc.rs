//! Allocation guard for the per-block hot paths.
//!
//! A counting global allocator tallies the allocations of the test
//! thread. Each test runs one warm-up pass, so the reused buffers reach
//! their final size. The measured pass must then allocate nothing.

use std::alloc::GlobalAlloc;
use std::alloc::Layout;
use std::alloc::System;
use std::cell::Cell;
use std::io::Cursor;
use std::io::Read;
use std::io::Write;

use bzip3_rs::BlockDecoder;
use bzip3_rs::BlockEncoder;
use bzip3_rs::BlockSize;
use bzip3_rs::Jobs;
use bzip3_rs::read::Decoder as ReadDecoder;
use bzip3_rs::read::Encoder as ReadEncoder;
use bzip3_rs::write::Decoder as WriteDecoder;
use bzip3_rs::write::Encoder as WriteEncoder;

thread_local! {
    /// Allocations made by this thread since the last reset.
    static COUNT: Cell<usize> = const { Cell::new(0) };
    /// Bytes requested by this thread since the start.
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

/// Add `size` to the byte count of this thread.
fn add_bytes(size: usize) {
    BYTES.with(|bytes| bytes.set(bytes.get() + size));
}

/// System allocator that counts every allocation of the calling thread.
struct Counting;

// SAFETY: every call forwards to `System` with the same arguments.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNT.with(|count| count.set(count.get() + 1));
        add_bytes(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        COUNT.with(|count| count.set(count.get() + 1));
        add_bytes(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        COUNT.with(|count| count.set(count.get() + 1));
        add_bytes(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Run `work` and return the allocations it made on this thread.
fn count_allocs(work: impl FnOnce()) -> usize {
    let before = COUNT.with(Cell::get);
    work();
    COUNT.with(Cell::get) - before
}

/// Run `work` and return the bytes it requested on this thread.
fn count_bytes(work: impl FnOnce()) -> usize {
    let before = BYTES.with(Cell::get);
    work();
    BYTES.with(Cell::get) - before
}

/// Make `len` pseudo-random bytes below `alphabet`.
fn noise(seed: u32, len: usize, alphabet: u32) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            u8::try_from((state >> 16) % alphabet).unwrap()
        })
        .collect()
}

/// Make one input for every stage route of the block encoder.
fn stage_inputs() -> Vec<Vec<u8>> {
    let phrase: Vec<u8> = (0..50_u8).collect();
    let runs: Vec<u8> = (0..200_u8)
        .flat_map(|byte| std::iter::repeat_n(byte, 300))
        .collect();
    let mut mixed = runs[..20_000].to_vec();
    mixed.extend(phrase.repeat(400));
    vec![
        vec![7; 40],
        runs,
        phrase.repeat(600),
        mixed,
        noise(1, 60_000, 256),
        noise(2, 65_000, 4),
    ]
}

/// Make a stream input of equal blocks, so every block codes alike.
fn equal_blocks(count: usize) -> Vec<u8> {
    noise(3, usize::try_from(BlockSize::MIN.get()).unwrap(), 20).repeat(count)
}

#[test]
fn encode_block_into_does_not_allocate() {
    let inputs = stage_inputs();
    let mut state = BlockEncoder::new(BlockSize::MIN).unwrap();
    let mut out = Vec::new();
    for input in &inputs {
        out.clear();
        state.encode_block_into(input, &mut out).unwrap();
    }
    let allocs = count_allocs(|| {
        for input in &inputs {
            out.clear();
            state.encode_block_into(input, &mut out).unwrap();
        }
    });
    assert_eq!(allocs, 0);
}

#[test]
fn decode_block_does_not_allocate() {
    let inputs = stage_inputs();
    let mut state = BlockEncoder::new(BlockSize::MIN).unwrap();
    let payloads: Vec<Vec<u8>> = inputs
        .iter()
        .map(|input| state.encode_block(input).unwrap())
        .collect();
    let mut decoder = BlockDecoder::new(BlockSize::MIN).unwrap();
    let mut to = vec![0_u8; 70_000];
    let mut into = Vec::new();
    let mut pass = |decoder: &mut BlockDecoder| {
        for (input, payload) in inputs.iter().zip(&payloads) {
            let size = u32::try_from(input.len()).unwrap();
            let slice = &mut to[..input.len()];
            decoder.decode_block_to(payload, size, slice).unwrap();
            assert_eq!(slice, input.as_slice());
            into.clear();
            decoder.decode_block_into(payload, size, &mut into).unwrap();
            assert_eq!(&into, input);
        }
    };
    pass(&mut decoder);
    let allocs = count_allocs(|| pass(&mut decoder));
    assert_eq!(allocs, 0);
}

#[test]
fn write_wrappers_do_not_allocate_per_block() {
    let plain = equal_blocks(12);
    let mut framed = Vec::new();
    let mut encoder = WriteEncoder::new(&mut framed, BlockSize::MIN).unwrap();
    encoder.write_all(&plain).unwrap();
    encoder.finish().unwrap();

    // Warm up on the first half, then measure the second half.
    let half = plain.len() / 2;
    let mut encoder = WriteEncoder::new(std::io::sink(), BlockSize::MIN).unwrap();
    for piece in plain[..half].chunks(10_000) {
        encoder.write_all(piece).unwrap();
    }
    let allocs = count_allocs(|| {
        for piece in plain[half..].chunks(10_000) {
            encoder.write_all(piece).unwrap();
        }
    });
    assert_eq!(allocs, 0, "write encoder");

    let half = framed.len() / 2;
    let mut decoder = WriteDecoder::new(std::io::sink());
    for piece in framed[..half].chunks(10_000) {
        decoder.write_all(piece).unwrap();
    }
    let allocs = count_allocs(|| {
        for piece in framed[half..].chunks(10_000) {
            decoder.write_all(piece).unwrap();
        }
    });
    assert_eq!(allocs, 0, "write decoder");
}

#[test]
fn read_wrappers_do_not_allocate_per_block() {
    let plain = equal_blocks(12);
    let mut buf = vec![0_u8; 10_000];
    let mut framed = Vec::new();
    let mut encoder = ReadEncoder::new(Cursor::new(&plain), BlockSize::MIN).unwrap();
    encoder.read_to_end(&mut framed).unwrap();

    let mut encoder = ReadEncoder::new(Cursor::new(&plain), BlockSize::MIN).unwrap();
    let mut total = 0;
    while total < plain.len() / 2 {
        total += encoder.read(&mut buf).unwrap();
    }
    let allocs = count_allocs(|| while encoder.read(&mut buf).unwrap() != 0 {});
    assert_eq!(allocs, 0, "read encoder");

    let mut decoder = ReadDecoder::new(Cursor::new(&framed)).unwrap();
    let mut total = 0;
    while total < plain.len() / 2 {
        total += decoder.read(&mut buf).unwrap();
    }
    let allocs = count_allocs(|| while decoder.read(&mut buf).unwrap() != 0 {});
    assert_eq!(allocs, 0, "read decoder");
}

#[test]
fn counter_sees_allocations() {
    let allocs = count_allocs(|| drop(std::hint::black_box(vec![1_u8; 100])));
    assert!(allocs > 0);
}

#[test]
fn parallel_adapters_build_no_serial_state() {
    let size = BlockSize::mib(16).unwrap();
    let width = size.get() as usize;
    // One serial state needs more than two blocks. One block for
    // `pending` stays in the encoders.
    let limit = width / 2 * 3;
    let header = WriteEncoder::new(Vec::new(), size)
        .unwrap()
        .finish()
        .unwrap();
    let framed = {
        let mut framed = header.clone();
        framed.extend_from_slice(&[0; 8]);
        framed
    };
    let jobs = [(Jobs::SERIAL, true), (Jobs::from_count(2), false)];
    for (count, serial) in jobs {
        let read_encoder = count_bytes(|| {
            drop(ReadEncoder::with_jobs(std::io::empty(), size, count).unwrap());
        });
        let write_encoder = count_bytes(|| {
            drop(WriteEncoder::with_jobs(std::io::sink(), size, count).unwrap());
        });
        let read_decoder = count_bytes(|| {
            drop(ReadDecoder::with_jobs(Cursor::new(&framed), count).unwrap());
        });
        let write_decoder = count_bytes(|| {
            let mut decoder = WriteDecoder::with_jobs(std::io::sink(), count);
            decoder.write_all(&header).unwrap();
        });
        for (name, bytes) in [
            ("read encoder", read_encoder),
            ("write encoder", write_encoder),
            ("read decoder", read_decoder),
        ] {
            assert_eq!(
                bytes > limit,
                serial,
                "{name} jobs {count:?}: {bytes} bytes"
            );
        }
        // The write decoder builds its state at the header. A batch builds none.
        assert_eq!(
            write_decoder > limit,
            serial,
            "write decoder jobs {count:?}: {write_decoder}"
        );
    }
}
