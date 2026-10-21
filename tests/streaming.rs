//! Streaming framing, wrappers, and parallel streams (ticket 10).
//!
//! Fixtures `stream-empty`, `stream-1byte`, and `stream-65b` come from
//! the pinned C oracle CLI (pinned at `fd3d71f`, default
//! 16 MiB block). Corpus inputs are rebuilt in code. Round-trips run
//! through all four wrappers and both `stream::` functions. The
//! trailing-bytes cases pin the recorded asymmetry: one-shot decode
//! ignores trailing bytes, streaming decode rejects them.

use std::io::Cursor;
use std::io::Read;
use std::io::Write;

use bzip3_rs::BlockDecoder;
use bzip3_rs::BlockEncoder;
use bzip3_rs::BlockHeader;
use bzip3_rs::BlockSize;
use bzip3_rs::Jobs;
use bzip3_rs::MAGIC_NUMBER;
use bzip3_rs::StreamHeader;
use bzip3_rs::decompress;
use bzip3_rs::error::Error;
use bzip3_rs::read::Decoder as ReadDecoder;
use bzip3_rs::read::Encoder as ReadEncoder;
use bzip3_rs::stream;
use bzip3_rs::write::Decoder as WriteDecoder;
use bzip3_rs::write::Encoder as WriteEncoder;

const EMPTY_BZ3: &[u8] = include_bytes!("vectors/stream-empty.bz3");
const ONE_BZ3: &[u8] = include_bytes!("vectors/stream-1byte.bz3");
const B65_BZ3: &[u8] = include_bytes!("vectors/stream-65b.bz3");

/// Smallest block size used by the tests (65 KiB).
fn small_size() -> BlockSize {
    BlockSize::MIN
}

/// Paragraph repeated to build the text inputs.
fn para() -> Vec<u8> {
    let mut text = Vec::new();
    text.extend_from_slice(b"This is the 100th Etext file presented by Project Gutenberg. ");
    text.extend_from_slice(b"Shakespeare, William. The complete works in one file. ");
    text.extend_from_slice(b"To be, or not to be, that is the question. ");
    text.extend_from_slice(b"All the world's a stage, and all the men and women merely players. ");
    text
}

/// Doubled 0..50 phrase repeated to 2000 bytes (LZP model input).
fn periodic_input() -> Vec<u8> {
    let phrase: Vec<u8> = (0..50_u8).collect();
    let doubled = [phrase.clone(), phrase].concat();
    doubled.repeat(20)
}

/// Fixed-seed filler bytes (plain coded model input).
fn random_input(len: usize) -> Vec<u8> {
    let mut state: u32 = 0xC0_FFEE;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            u8::try_from((state >> 16) & 0xFF).expect("masked byte fits")
        })
        .collect()
}

/// Corpus inputs covering every model path plus multi-block text.
fn corpus() -> Vec<(&'static str, Vec<u8>)> {
    let text = para().repeat(800);
    let text140k = text[..140_000].to_vec();
    vec![
        ("empty", Vec::new()),
        ("1byte", b"A".to_vec()),
        ("63b", vec![b'x'; 63]),
        ("64b", vec![b'y'; 64]),
        ("65b", vec![b'z'; 65]),
        ("runs2k", vec![b'A'; 2000]),
        ("periodic2k", periodic_input()),
        ("random4k", random_input(4096)),
        ("text140k", text140k),
    ]
}

/// Compress `input` through `stream::compress` at `jobs`.
fn framed(input: &[u8], size: BlockSize, jobs: Jobs) -> Vec<u8> {
    let mut out = Vec::new();
    let count = stream::compress(Cursor::new(input), &mut out, size, jobs).unwrap();
    assert_eq!(count, out.len() as u64, "count matches output");
    out
}

/// Decompress `code` through `stream::decompress` at `jobs`.
fn unframed(code: &[u8], jobs: Jobs) -> Vec<u8> {
    let mut out = Vec::new();
    let count = stream::decompress(Cursor::new(code), &mut out, jobs).unwrap();
    assert_eq!(count, out.len() as u64, "count matches output");
    out
}

#[test]
fn streaming_empty_is_nine_byte_header() {
    let code = framed(b"", BlockSize::DEFAULT, Jobs::SERIAL);
    assert_eq!(code, EMPTY_BZ3);
    assert_eq!(code.len(), 9);
    assert_eq!(&code[..5], b"BZ3v1");
    assert_eq!(
        u32::from_le_bytes(code[5..9].try_into().unwrap()),
        BlockSize::DEFAULT.get()
    );
    assert_eq!(unframed(&code, Jobs::SERIAL), b"");
}

#[test]
fn streaming_header_records_requested_size_verbatim() {
    // One-shot encode clamps small inputs to 65 KiB. Streaming keeps
    // the requested size, so the headers differ on small inputs.
    let size = BlockSize::mib(1).unwrap();
    let code = framed(b"A", size, Jobs::SERIAL);
    assert_eq!(code.len(), 26);
    assert_eq!(
        u32::from_le_bytes(code[5..9].try_into().unwrap()),
        size.get()
    );
    assert_eq!(&code[..5], b"BZ3v1");
    assert_eq!(unframed(&code, Jobs::SERIAL), b"A");
}

#[test]
fn oracle_cli_outputs_decode_byte_identically() {
    assert_eq!(unframed(EMPTY_BZ3, Jobs::SERIAL), b"");
    assert_eq!(unframed(ONE_BZ3, Jobs::SERIAL), b"A");
    assert_eq!(unframed(B65_BZ3, Jobs::SERIAL), vec![b'z'; 65]);
}

#[test]
fn tiny_stored_output_matches_oracle_bytes() {
    // Stored blocks are fully determined, so our bytes match the CLI.
    let code = framed(b"A", BlockSize::DEFAULT, Jobs::SERIAL);
    assert_eq!(code, ONE_BZ3);
}

#[test]
fn block_header_layout_is_sizes_then_payload() {
    let header = BlockHeader::new(9, 1);
    assert_eq!(BlockHeader::LEN, 8);
    let bytes = header.to_bytes();
    assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()), 9);
    assert_eq!(u32::from_le_bytes(bytes[4..].try_into().unwrap()), 1);
    assert_eq!(BlockHeader::from_bytes(&bytes), header);
    assert_eq!(MAGIC_NUMBER, *b"BZ3v1");
}

#[test]
fn multiblock_output_holds_three_blocks() {
    let text = para().repeat(800);
    let input = &text[..140_000];
    let code = framed(input, small_size(), Jobs::SERIAL);
    assert_eq!(&code[..5], b"BZ3v1");
    // Walk the frame: 9-byte header, then comp/orig/payload triples.
    let mut rest = &code[9..];
    let mut blocks = 0;
    let mut total = 0;
    while !rest.is_empty() {
        let comp = u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize;
        let orig = u32::from_le_bytes(rest[4..8].try_into().unwrap()) as usize;
        total += orig;
        rest = &rest[8 + comp..];
        blocks += 1;
    }
    assert_eq!(blocks, 3);
    assert_eq!(total, input.len());
    assert_eq!(unframed(&code, Jobs::SERIAL), input);
}

#[test]
fn trailing_garbage_is_rejected() {
    let mut code = framed(b"A", small_size(), Jobs::SERIAL);
    code.extend_from_slice(b"TRAIL");
    assert!(matches!(
        stream::decompress(Cursor::new(&code), &mut Vec::new(), Jobs::SERIAL),
        Err(Error::TruncatedHeader)
    ));
}

#[test]
fn trailing_garbage_is_rejected_on_coded_frames() {
    let text = para().repeat(800);
    let input = &text[..140_000];
    let mut code = framed(input, small_size(), Jobs::SERIAL);
    code.extend_from_slice(b"TRAIL");
    assert!(matches!(
        stream::decompress(Cursor::new(&code), &mut Vec::new(), Jobs::SERIAL),
        Err(Error::TruncatedHeader)
    ));
    // One-shot decode keeps ignoring trailing bytes: the asymmetry.
    let one_shot = bzip3_rs::compress(input, small_size()).unwrap();
    let mut trailed = one_shot.clone();
    trailed.extend_from_slice(b"TRAIL");
    assert_eq!(decompress(&trailed).unwrap(), input);
}

#[test]
fn cut_payload_is_truncated() {
    let code = framed(&vec![b'A'; 2000], small_size(), Jobs::SERIAL);
    let cut = &code[..code.len() - 10];
    assert!(matches!(
        stream::decompress(Cursor::new(cut), &mut Vec::new(), Jobs::SERIAL),
        Err(Error::TruncatedData)
    ));
}

#[test]
fn bad_stream_header_fails_exact_variant() {
    let mut bad = framed(b"A", small_size(), Jobs::SERIAL);
    bad[0] = b'X';
    assert!(matches!(
        stream::decompress(Cursor::new(&bad), &mut Vec::new(), Jobs::SERIAL),
        Err(Error::InvalidMagic)
    ));
    let short = &bad[..4];
    assert!(matches!(
        stream::decompress(Cursor::new(short), &mut Vec::new(), Jobs::SERIAL),
        Err(Error::TruncatedHeader)
    ));
    let mut bad = framed(b"A", small_size(), Jobs::SERIAL);
    bad[5..9].copy_from_slice(&(-1_i32).to_le_bytes());
    assert!(matches!(
        stream::decompress(Cursor::new(&bad), &mut Vec::new(), Jobs::SERIAL),
        Err(Error::InvalidBlockSize)
    ));
}

#[test]
fn read_wrappers_round_trip_corpus() {
    for (name, input) in corpus() {
        let mut encoder = ReadEncoder::new(Cursor::new(&input), small_size()).unwrap();
        assert_eq!(encoder.block_size(), small_size());
        let mut code = Vec::new();
        encoder.read_to_end(&mut code).unwrap();
        let mut decoder = ReadDecoder::new(Cursor::new(&code)).unwrap();
        assert_eq!(decoder.block_size(), small_size());
        let mut plain = Vec::new();
        decoder.read_to_end(&mut plain).unwrap();
        assert_eq!(plain, input, "read wrappers differ for {name}");
    }
}

#[test]
fn write_wrappers_round_trip_corpus() {
    for (name, input) in corpus() {
        let mut encoder = WriteEncoder::new(Vec::new(), small_size()).unwrap();
        assert_eq!(encoder.block_size(), small_size());
        encoder.write_all(&input).unwrap();
        let code = encoder.finish().unwrap();
        let mut decoder = WriteDecoder::new(Vec::new());
        decoder.write_all(&code).unwrap();
        let plain = decoder.finish().unwrap();
        assert_eq!(plain, input, "write wrappers differ for {name}");
    }
}

#[test]
fn wrapper_output_matches_serial_stream_bytes() {
    // Full-block buffering gives identical splits, so bytes match.
    for (name, input) in corpus() {
        let serial = framed(&input, small_size(), Jobs::SERIAL);
        let mut encoder = ReadEncoder::new(Cursor::new(&input), small_size()).unwrap();
        let mut via_read = Vec::new();
        encoder.read_to_end(&mut via_read).unwrap();
        assert_eq!(via_read, serial, "read encoder differs for {name}");
        let mut encoder = WriteEncoder::new(Vec::new(), small_size()).unwrap();
        encoder.write_all(&input).unwrap();
        let via_write = encoder.finish().unwrap();
        assert_eq!(via_write, serial, "write encoder differs for {name}");
    }
}

/// Writer that accepts a fixed number of bytes, then fails.
struct FailAfter {
    kept: Vec<u8>,
    room: usize,
}

impl Write for FailAfter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.room < buf.len() {
            return Err(std::io::Error::other("full"));
        }
        self.room -= buf.len();
        self.kept.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn serial_compress_slice_writes_block_by_block() {
    let input = para().repeat(800);
    let full = framed(&input, small_size(), Jobs::SERIAL);
    // Room for the header and the first block only.
    let first =
        9 + 8 + usize::try_from(u32::from_le_bytes(full[9..13].try_into().unwrap())).unwrap();
    let mut sink = FailAfter {
        kept: Vec::new(),
        room: first,
    };
    let result = stream::compress_slice(&input, &mut sink, small_size(), Jobs::SERIAL);
    assert!(matches!(result, Err(Error::Io(_))));
    assert_eq!(sink.kept, full[..first]);
}

#[test]
fn write_wrappers_take_many_blocks_in_one_write() {
    // One write holds 16 full blocks plus a short tail. The wrappers
    // must emit every block from that single call.
    let width = usize::try_from(small_size().get()).unwrap();
    let input: Vec<u8> = para().into_iter().cycle().take(16 * width + 100).collect();
    let serial = framed(&input, small_size(), Jobs::SERIAL);
    let mut encoder = WriteEncoder::new(Vec::new(), small_size()).unwrap();
    encoder.write_all(&input).unwrap();
    let code = encoder.finish().unwrap();
    assert_eq!(code, serial, "write encoder differs from serial stream");
    let mut decoder = WriteDecoder::new(Vec::new());
    decoder.write_all(&code).unwrap();
    assert_eq!(decoder.finish().unwrap(), input, "write decoder output");
    assert_eq!(
        unframed(&code, Jobs::SERIAL),
        input,
        "serial decoder output"
    );
}

#[test]
fn write_encoder_emits_header_eagerly() {
    let encoder = WriteEncoder::new(Vec::new(), small_size()).unwrap();
    assert_eq!(encoder.get_ref().as_ref().unwrap().len(), 9);
    assert_eq!(&encoder.get_ref().as_ref().unwrap()[..5], b"BZ3v1");
    let code = encoder.finish().unwrap();
    assert_eq!(code.len(), 9);
    assert_eq!(unframed(&code, Jobs::SERIAL), b"");
}

#[test]
fn write_decoder_finish_detects_truncation() {
    // Zero framed bytes: no header ever arrives.
    let decoder = WriteDecoder::new(Vec::new());
    assert!(matches!(decoder.finish(), Err(Error::TruncatedHeader)));
    // Partial header.
    let code = framed(b"A", small_size(), Jobs::SERIAL);
    let mut decoder = WriteDecoder::new(Vec::new());
    decoder.write_all(&code[..4]).unwrap();
    assert!(matches!(decoder.finish(), Err(Error::TruncatedHeader)));
    // Full header plus a partial block header.
    let mut decoder = WriteDecoder::new(Vec::new());
    decoder.write_all(&code[..9 + 4]).unwrap();
    assert!(matches!(decoder.finish(), Err(Error::TruncatedHeader)));
    // Full block header plus a short payload.
    let full = framed(&vec![b'A'; 2000], small_size(), Jobs::SERIAL);
    let mut decoder = WriteDecoder::new(Vec::new());
    decoder.write_all(&full[..full.len() - 10]).unwrap();
    assert!(matches!(decoder.finish(), Err(Error::TruncatedData)));
    // Valid stream passes and restores the input.
    let mut decoder = WriteDecoder::new(Vec::new());
    decoder.write_all(&full).unwrap();
    assert_eq!(decoder.finish().unwrap(), vec![b'A'; 2000]);
}

#[test]
fn read_decoder_constructor_names_header_faults() {
    let mut bad = framed(b"A", small_size(), Jobs::SERIAL);
    bad[0] = b'X';
    assert!(matches!(
        ReadDecoder::new(Cursor::new(&bad)),
        Err(Error::InvalidMagic)
    ));
    assert!(matches!(
        ReadDecoder::new(Cursor::new(Vec::new())),
        Err(Error::TruncatedHeader)
    ));
}

#[test]
fn read_decoder_trailing_fails_read() {
    let mut code = framed(b"A", small_size(), Jobs::SERIAL);
    code.extend_from_slice(b"TRAIL");
    let mut decoder = ReadDecoder::new(Cursor::new(&code)).unwrap();
    let mut plain = Vec::new();
    let err = decoder.read_to_end(&mut plain).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn parallel_compress_is_byte_identical() {
    for (name, input) in corpus() {
        let one = framed(&input, small_size(), Jobs::SERIAL);
        for jobs in [Jobs::Global, Jobs::from_count(2), Jobs::from_count(8)] {
            assert_eq!(
                framed(&input, small_size(), jobs),
                one,
                "{name} at {jobs:?}"
            );
        }
    }
}

#[test]
fn parallel_decompress_is_byte_identical() {
    for (name, input) in corpus() {
        let code = framed(&input, small_size(), Jobs::SERIAL);
        let one = unframed(&code, Jobs::SERIAL);
        assert_eq!(one, input, "serial decode differs for {name}");
        for jobs in [Jobs::Global, Jobs::from_count(2), Jobs::from_count(8)] {
            assert_eq!(unframed(&code, jobs), one, "{name} at {jobs:?}");
        }
    }
}

#[test]
fn stream_round_trips_at_many_threads() {
    let text = para().repeat(800);
    let input = &text[..140_000];
    for jobs in [
        Jobs::SERIAL,
        Jobs::Global,
        Jobs::from_count(3),
        Jobs::from_count(16),
    ] {
        let mut code = Vec::new();
        let written = stream::compress(Cursor::new(input), &mut code, small_size(), jobs).unwrap();
        assert_eq!(written, code.len() as u64);
        let mut plain = Vec::new();
        let read = stream::decompress(Cursor::new(&code), &mut plain, jobs).unwrap();
        assert_eq!(read, plain.len() as u64);
        assert_eq!(plain, input, "round-trip differs at {jobs:?}");
    }
}

#[test]
fn codec_state_is_send_and_thread_portable() {
    fn assert_send<T: Send>() {}
    assert_send::<BlockEncoder>();
    assert_send::<ReadEncoder<Cursor<Vec<u8>>>>();
    assert_send::<ReadDecoder<Cursor<Vec<u8>>>>();
    assert_send::<WriteEncoder<Vec<u8>>>();
    assert_send::<WriteDecoder<Vec<u8>>>();
    let input = vec![b'A'; 2000];
    let first = framed(&input, small_size(), Jobs::SERIAL);
    let moved = std::thread::spawn(move || {
        let mut state = BlockEncoder::new(small_size()).unwrap();
        state.encode_block(&input).unwrap()
    })
    .join()
    .unwrap();
    let mut state = BlockEncoder::new(small_size()).unwrap();
    assert_eq!(moved, state.encode_block(&vec![b'A'; 2000]).unwrap());
    assert_eq!(framed(&vec![b'A'; 2000], small_size(), Jobs::Global), first);
}

#[test]
fn adapters_give_access_to_the_inner_value() {
    let input = vec![b'A'; 2000];
    let code = framed(&input, small_size(), Jobs::SERIAL);

    let mut reader = ReadEncoder::new(Cursor::new(input.clone()), small_size()).unwrap();
    assert_eq!(reader.get_mut().position(), 0);
    assert_eq!(reader.into_inner().into_inner(), input);

    let mut reader = ReadDecoder::new(Cursor::new(code.clone())).unwrap();
    assert_eq!(reader.get_mut().position(), 9);
    assert_eq!(reader.into_inner().into_inner(), code);

    let mut writer = WriteEncoder::new(Vec::new(), small_size()).unwrap();
    assert_eq!(writer.get_mut().as_mut().unwrap().len(), 9);
    assert_eq!(writer.finish().unwrap().len(), 9);

    let mut writer = WriteDecoder::new(Vec::new());
    assert!(writer.get_mut().is_empty());
}

#[test]
fn write_decoder_knows_block_size_after_header() {
    let code = framed(b"A", small_size(), Jobs::SERIAL);
    let mut writer = WriteDecoder::new(Vec::new());
    assert_eq!(writer.block_size(), None);
    writer.write_all(&code[..9]).unwrap();
    assert_eq!(writer.block_size(), Some(small_size()));
}

#[test]
fn public_types_implement_debug() {
    let size = small_size();
    let text = format!("{:?}", BlockEncoder::new(size).unwrap());
    assert!(text.starts_with("BlockEncoder"), "{text}");
    let text = format!(
        "{:?}",
        ReadEncoder::new(Cursor::new(Vec::new()), size).unwrap()
    );
    assert!(text.contains("block_size"), "{text}");
    let text = format!("{:?}", WriteDecoder::new(Vec::<u8>::new()));
    assert!(text.contains("Decoder"), "{text}");
}

/// Write `input` through a shared sink and drop the encoder without `finish`.
fn drop_without_finish(input: &[u8], jobs: Jobs) -> Vec<u8> {
    struct Shared(std::rc::Rc<std::cell::RefCell<Vec<u8>>>);
    impl Write for Shared {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.borrow_mut().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let sink = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut encoder = WriteEncoder::with_jobs(Shared(sink.clone()), small_size(), jobs).unwrap();
    encoder.write_all(input).unwrap();
    drop(encoder);
    sink.take()
}

#[test]
fn encoder_drop_writes_the_final_block() {
    let input: Vec<u8> = (0..200_000_u32).map(|i| (i % 251) as u8).collect();
    let code = drop_without_finish(&input, Jobs::SERIAL);
    assert_eq!(unframed(&code, Jobs::SERIAL), input);
    assert_eq!(code, framed(&input, small_size(), Jobs::SERIAL));
}

#[test]
fn encoder_drop_writes_the_final_block_in_parallel_mode() {
    let input: Vec<u8> = (0..500_000_u32).map(|i| (i % 253) as u8).collect();
    let code = drop_without_finish(&input, Jobs::from_count(4));
    assert_eq!(unframed(&code, Jobs::SERIAL), input);
}

#[test]
fn encoder_drop_after_finish_writes_nothing_more() {
    let input = vec![b'A'; 2000];
    let mut encoder = WriteEncoder::new(Vec::new(), small_size()).unwrap();
    encoder.write_all(&input).unwrap();
    let code = encoder.finish().unwrap();
    assert_eq!(code, framed(&input, small_size(), Jobs::SERIAL));
}

#[test]
fn stream_header_layout_is_magic_then_block_size() {
    let header = StreamHeader::new(small_size());
    assert_eq!(StreamHeader::LEN, 9);
    let bytes = header.to_bytes();
    assert_eq!(bytes[..5], MAGIC_NUMBER);
    assert_eq!(
        u32::from_le_bytes(bytes[5..].try_into().unwrap()),
        small_size().get()
    );
    assert_eq!(StreamHeader::from_bytes(&bytes).unwrap(), header);
}

#[test]
fn stream_header_rejects_bad_bytes() {
    let mut bytes = StreamHeader::new(small_size()).to_bytes();
    bytes[5..].copy_from_slice(&1_u32.to_le_bytes());
    assert!(matches!(
        StreamHeader::from_bytes(&bytes),
        Err(Error::InvalidBlockSize)
    ));
    bytes[0] ^= 1;
    assert!(matches!(
        StreamHeader::from_bytes(&bytes),
        Err(Error::InvalidMagic)
    ));
}

#[test]
fn manual_blocks_decode_through_the_adapter() {
    let input: Vec<u8> = (0..150_000_u32).map(|i| (i % 249) as u8).collect();
    let mut code = StreamHeader::new(small_size()).to_bytes().to_vec();
    let mut encoder = BlockEncoder::new(small_size()).unwrap();
    for chunk in input.chunks(small_size().as_usize()) {
        let payload = encoder.encode_block(chunk).unwrap();
        code.extend_from_slice(
            &BlockHeader::for_block(payload.len(), chunk.len())
                .unwrap()
                .to_bytes(),
        );
        code.extend_from_slice(&payload);
    }
    let mut plain = Vec::new();
    ReadDecoder::new(Cursor::new(code))
        .unwrap()
        .read_to_end(&mut plain)
        .unwrap();
    assert_eq!(plain, input);
}

#[test]
fn adapter_output_decodes_by_hand() {
    let input: Vec<u8> = (0..150_000_u32).map(|i| (i % 247) as u8).collect();
    let mut writer = WriteEncoder::new(Vec::new(), small_size()).unwrap();
    writer.write_all(&input).unwrap();
    let code = writer.finish().unwrap();

    let (head, mut rest) = code.split_at(StreamHeader::LEN);
    let header = StreamHeader::from_bytes(head.try_into().unwrap()).unwrap();
    let mut decoder = BlockDecoder::new(header.block_size).unwrap();
    let mut plain = Vec::new();
    while !rest.is_empty() {
        let (block, tail) = rest.split_at(BlockHeader::LEN);
        let block = BlockHeader::from_bytes(block.try_into().unwrap());
        let (payload, tail) = tail.split_at(block.compressed_size as usize);
        decoder
            .decode_block_into(payload, block.original_size, &mut plain)
            .unwrap();
        rest = tail;
    }
    assert_eq!(plain, input);
}
