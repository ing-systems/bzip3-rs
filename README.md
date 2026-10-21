# bzip3-rs

[![crates.io](https://img.shields.io/crates/v/bzip3-rs.svg)](https://crates.io/crates/bzip3-rs)
[![docs.rs](https://img.shields.io/docsrs/bzip3-rs)](https://docs.rs/bzip3-rs)
[![CI](https://github.com/ing-systems/bzip3-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ing-systems/bzip3-rs/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/rustc-1.88%2B-blue.svg)](#minimum-rust-version)
[![License](https://img.shields.io/badge/license-LGPL--3.0%20AND%20Apache--2.0-blue.svg)](#license)

Pure-Rust port of the [bzip3](https://github.com/iczelia/bzip3) compressor.

- No C code. No FFI. No `unsafe` code (`#![forbid(unsafe_code)]`).
- Output is compatible with C bzip3 in both directions.
- Blocks run in parallel with [rayon](https://crates.io/crates/rayon).
- Each block carries a CRC32 check.
- The crate has a library API and a `bzip3` command-line tool.

Each block passes through four stages: run-length coding, LZP, a Burrows-Wheeler transform (suffix array), and an order-0 context-mixing coder.

## Install

Add the library to a project:

```sh
cargo add bzip3-rs
```

The library name in Rust code is `bzip3_rs`.

Install the command-line tool:

```sh
cargo install bzip3-rs --features cli
```

### Debian and Ubuntu

Download the `.deb` file for your architecture (`amd64` or `arm64`) from the [release page](https://github.com/ing-systems/bzip3-rs/releases).
Then install it:

```sh
sudo apt install ./bzip3-rs_*.deb
```

The package replaces the C `bzip3` package. It supports Debian 12 and later, and Ubuntu 22.04 and later.

## Library usage

Each example below is a complete program.

### One call

```rust
use bzip3_rs::{BlockSize, compress, decompress};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = b"hello world ".repeat(1000);

    let coded = compress(&data, BlockSize::DEFAULT)?;
    let plain = decompress(&coded)?;

    assert_eq!(plain, data);
    Ok(())
}
```

### Parallel blocks

`Jobs::SERIAL` runs on the calling thread. `Jobs::Global` uses the global rayon pool. `Jobs::from_count(4)` builds a private pool. Output bytes are the same for every job count.

```rust
use bzip3_rs::{BlockSize, Jobs, compress_with_jobs, decompress_with_jobs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = b"hello world ".repeat(1000);
    let jobs = Jobs::from_count(4);

    let coded = compress_with_jobs(&data, BlockSize::DEFAULT, jobs)?;
    let plain = decompress_with_jobs(&coded, jobs)?;

    assert_eq!(plain, data);
    Ok(())
}
```

### Streams

Module `stream` reads from a `Read` value and writes to a `Write` value. It uses the 9-byte streaming frame.

```rust
use std::io::Cursor;
use bzip3_rs::{BlockSize, Jobs, stream};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = b"hello world ".repeat(1000);
    let size = BlockSize::DEFAULT;
    let jobs = Jobs::SERIAL;

    let mut coded = Vec::new();
    stream::compress(Cursor::new(&input), &mut coded, size, jobs)?;

    let mut plain = Vec::new();
    stream::decompress(Cursor::new(coded.as_slice()), &mut plain, jobs)?;
    assert_eq!(plain, input);

    // Use `compress_slice` when the input is in memory.
    let mut coded_again = Vec::new();
    stream::compress_slice(&input, &mut coded_again, size, jobs)?;
    assert_eq!(coded_again, coded);
    Ok(())
}
```

### Read and Write adapters

```rust
use std::io::{Cursor, Read, Write};
use bzip3_rs::BlockSize;
use bzip3_rs::read::{Decoder as ReadDecoder, Encoder as ReadEncoder};
use bzip3_rs::write::{Decoder as WriteDecoder, Encoder as WriteEncoder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = b"hello world ".repeat(1000);
    let size = BlockSize::MIN;

    // Read adapters.
    let mut encoder = ReadEncoder::new(Cursor::new(&input), size)?;
    let mut coded = Vec::new();
    encoder.read_to_end(&mut coded)?;

    let mut decoder = ReadDecoder::new(Cursor::new(coded.as_slice()))?;
    let mut plain = Vec::new();
    decoder.read_to_end(&mut plain)?;
    assert_eq!(plain, input);

    // Write adapters. Call `finish` to close the stream and to see errors.
    let mut encoder = WriteEncoder::new(Vec::new(), size)?;
    encoder.write_all(&input)?;
    let coded = encoder.finish()?;

    let mut decoder = WriteDecoder::new(Vec::new());
    decoder.write_all(&coded)?;
    let plain = decoder.finish()?;
    assert_eq!(plain, input);
    Ok(())
}
```

`write::Encoder` also writes the last block when you drop it. A drop ignores errors, so call `finish`.

### Block by block

Use `BlockEncoder` and `BlockDecoder` to build the streaming frame yourself.

```rust
use bzip3_rs::{BlockDecoder, BlockEncoder, BlockHeader, BlockSize, StreamHeader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = b"hello world ".repeat(1000);
    let size = BlockSize::MIN;

    // Encode: the stream header, then a block header and a payload for each block.
    let mut coded = StreamHeader::new(size).to_bytes().to_vec();
    let mut encoder = BlockEncoder::new(size)?;
    for chunk in input.chunks(size.as_usize()) {
        let payload = encoder.encode_block(chunk)?;
        coded.extend_from_slice(&BlockHeader::for_block(payload.len(), chunk.len())?.to_bytes());
        coded.extend_from_slice(&payload);
    }

    // Decode the same bytes.
    let (head, mut rest) = coded.split_at(StreamHeader::LEN);
    let header = StreamHeader::from_bytes(head.try_into()?)?;
    let mut decoder = BlockDecoder::new(header.block_size)?;
    let mut plain = Vec::new();
    while !rest.is_empty() {
        let (block, tail) = rest.split_at(BlockHeader::LEN);
        let block = BlockHeader::from_bytes(block.try_into()?);
        let (payload, tail) = tail.split_at(block.compressed_size as usize);
        decoder.decode_block_into(payload, block.original_size, &mut plain)?;
        rest = tail;
    }
    assert_eq!(plain, input);
    Ok(())
}
```

`BlockDecoder::recover_block_into` decodes a damaged block. It always appends the original number of bytes. A checksum fault keeps the decoded bytes. Any other fault appends zero bytes. The call returns a `BlockStatus` value that tells which case happened.

### Helpers

```rust
use bzip3_rs::{BlockSize, MAGIC_NUMBER, compress_bound};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(MAGIC_NUMBER, *b"BZ3v1");
    assert_eq!(compress_bound(1000), 1052);

    let _bytes = BlockSize::new(16 * 1024 * 1024)?;
    let _kib = BlockSize::kib(65)?;
    let _mib = BlockSize::mib(16)?;
    let _level = BlockSize::from_level(9)?; // Level 9 is 16 MiB.
    Ok(())
}
```

### Block size

| Item                   | Value   |
| ---------------------- | ------- |
| `BlockSize::MIN`       | 65 KiB  |
| `BlockSize::DEFAULT`   | 16 MiB  |
| `BlockSize::MAX`       | 511 MiB |

`BlockSize::from_level` maps levels 1 to 14 to these sizes:

| Level | 1      | 2       | 3       | 4       | 5     | 6     | 7     |
| ----- | ------ | ------- | ------- | ------- | ----- | ----- | ----- |
| Size  | 65 KiB | 128 KiB | 256 KiB | 512 KiB | 1 MiB | 2 MiB | 4 MiB |

| Level | 8     | 9      | 10     | 11     | 12      | 13      | 14      |
| ----- | ----- | ------ | ------ | ------ | ------- | ------- | ------- |
| Size  | 8 MiB | 16 MiB | 32 MiB | 64 MiB | 128 MiB | 256 MiB | 511 MiB |

Full API documentation: <https://docs.rs/bzip3-rs>.

## Command-line tool

Build the tool with the `cli` feature. The `cargo install` command above does this.

```sh
bzip3 -e input.bin        # writes input.bin.bz3, removes input.bin
bzip3 -d input.bin.bz3    # writes input.bin, removes input.bin.bz3
bzip3 -t input.bin.bz3    # checks the file, writes nothing
bzip3 -r input.bin.bz3    # recovers a damaged file, keeps input.bin.bz3
bzip3 -c -d input.bin.bz3 # writes decoded data to standard output
```

To run from a checkout without install:

```sh
cargo run --release --features cli --bin bzip3 -- --help
```

### Modes

The last mode flag wins.

| Flag                      | Action                                |
| ------------------------- | ------------------------------------- |
| `-e`, `-z`, `--encode`    | Compress data (default).              |
| `-d`, `--decode`          | Decompress data.                      |
| `-t`, `--test`            | Check data. Write no output.          |
| `-r`, `--recover`         | Decompress damaged data. Write zeros for bad blocks. |
| `-h`, `--help`            | Show help.                            |
| `-V`, `--version`         | Show the version.                     |

### Flags

| Flag                  | Action                                                                         |
| --------------------- | ------------------------------------------------------------------------------ |
| `-c`, `--stdout`      | Write to standard output.                                                      |
| `-f`, `--force`       | Overwrite existing output.                                                     |
| `-k`, `--keep`        | Keep input files.                                                              |
| `-v`, `--verbose`     | Show byte counts.                                                              |
| `-q`, `--quiet`       | Hide non-essential warnings.                                                   |
| `-b N`, `--block=N`   | Set the block size. `N` is in MiB, or use `NK` / `NKiB` for KiB. Range: 65 KiB to 511 MiB. Default: 16 MiB. |
| `-j N`, `--jobs=N`    | Set the thread count, 0 to 64. Default: 0 (all cores).                         |
| `-1` to `-14`         | Set the block size by level. See the level table above.                        |
| `--fast`              | Same as `-1`.                                                                  |
| `--best`              | Same as `-14`.                                                                 |

### File rules

- Encode adds the suffix `.bz3`.
- Decode removes the suffix `.bz3`. If the name has no suffix, decode writes `<name>.out`.
- Encode refuses an input name that already ends with `.bz3`.
- After success, the tool removes the input file. It keeps the file with `-k`, `-c`, test mode, or standard input.
- Recover mode never removes the input file. It keeps its output after a failure.
- If `argv[0]` is `bunzip3` or `bz3cat`, the tool decodes. `bz3cat` also writes to standard output.

### Recover mode

Recover mode decodes every block that it can find.

- A block with a checksum fault keeps its decoded bytes. Any other damaged block gives zero bytes.
  The output keeps its size, so the offsets after a damaged block stay correct.
- A damaged block prints one line on standard error:
  `Writing invalid block <number> at input offset <offset>: <fault>. Output bytes <first> to <last> keep the decoded bytes.`
  The line ends with `are zero.` when the block gives zero bytes.
- A bad block header or missing data stops the run. The blocks before it stay in the output.
  The tool prints `Recovery stopped at block <number>, input offset <offset>. Output bytes written: <size>. Damaged blocks: <count>.`
  Bzip3 blocks have no sync marker, so the tool does not search for the next block.
- A bad block size in the frame header prints `Recovery mode: Proceeding.`.
  The tool then takes the block size from the block headers.
- The exit code is 2 when a block is damaged or the run stops early. A bad signature is a fatal fault.

### Helper scripts and man pages

The repository holds the scripts [`bz3diff`](https://github.com/ing-systems/bzip3-rs/blob/main/scripts/bz3diff), [`bz3grep`](https://github.com/ing-systems/bzip3-rs/blob/main/scripts/bz3grep), and [`bz3more`](https://github.com/ing-systems/bzip3-rs/blob/main/scripts/bz3more).
These scripts need the `bzip3` tool in `PATH`.
The directory [`man/man1`](https://github.com/ing-systems/bzip3-rs/tree/main/man/man1) holds the man pages.

## Compatibility with C bzip3

- This crate reads files from C bzip3. C bzip3 reads files from this crate.
- One-shot output uses a 13-byte frame header.
- Streaming output uses a 9-byte frame header.
- One-shot decode ignores bytes after the last block.
- Streaming decode rejects trailing bytes that do not form a full block.
- Streaming encode records the requested block size as given.
- One-shot encode clamps small inputs to 65 KiB.

### Differences from C bzip3

- The default job count is 0 (all cores). C bzip3 uses 1 thread.
- Output bytes are the same at every job count.
- Recover mode writes zero bytes for a block with a fault other than the checksum. C bzip3 writes the decode buffer, which holds partial data.
- Recover mode exits with code 2 when it finds damage. C bzip3 exits with code 0.
- Recover mode never removes the input file. C bzip3 removes it unless `-k` is set.
- Recover mode takes the block size from the block headers when the frame header has a bad block size. The reference uses 511 MiB.
- Batch remove paths are absent.

## Cargo features

| Feature           | Default | Purpose                                              |
| ----------------- | ------- | ---------------------------------------------------- |
| `cli`             | no      | Builds the `bzip3` tool. Adds the `clap` dependency. |
| `bench-internals` | no      | Exposes stage functions for benchmarks. Not public API. |

## Minimum Rust version

Rust 1.88 or later. The crate uses edition 2024.

## Development

```sh
cargo test
cargo test --features cli
cargo clippy --all-targets --features cli
cargo bench --bench end_to_end
cargo bench --bench stages
```

The workflow `perf-gate` compares the speed of the `bzip3` tool with the C bzip3 tool.
The file [`tests/corpus/README.md`](https://github.com/ing-systems/bzip3-rs/blob/main/tests/corpus/README.md) describes the test corpus.

## License

The crate has the license `LGPL-3.0-only AND Apache-2.0`.

- [`LICENSE`](https://github.com/ing-systems/bzip3-rs/blob/main/LICENSE) holds the LGPL-3.0 text.
- [`LICENSE-APACHE`](https://github.com/ing-systems/bzip3-rs/blob/main/LICENSE-APACHE) holds the Apache-2.0 text.

## Attribution

This crate is an independent rewrite in Rust of [bzip3](https://github.com/iczelia/bzip3) by Kamila Szewczyk.

- Original C code: Copyright 2022-2024 Kamila Szewczyk.
- BWT (libsais) and LZP code: Copyright 2021-2022 Ilya Grebnov.
