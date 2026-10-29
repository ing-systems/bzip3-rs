# Changelog

All notable changes to this project are in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Recover mode for the `bzip3` tool: `-r` and `--recover`. It writes every block that it can decode. Each damaged block prints its number, input offset and output range. See the README for the differences from C bzip3.
- `BlockDecoder::recover_block_into` and `BlockStatus`. The method keeps the data of a damaged block. The value tells if the bytes are intact, kept after a checksum fault, or zero.
- Debian and Ubuntu `.deb` packages for amd64 and arm64. A workflow builds them and attaches them to each release.

### Changed

- Faster decode on builds with `-C target-cpu=x86-64-v3` or `native`: 14-20% less time. The compiler no longer packs the entropy decoder bit loop into vector lanes. Speed on the default target does not change. Output does not change.

## [1.0.1] - 2026-10-21

### Fixed

- Fix Windows test build: gate non-UTF8 level-token fixtures by OS.

## [1.0.0] - 2026-10-21

### Added

- Pure-Rust compress and decompress for the bzip3 format. Output is compatible with C bzip3 in both directions.
- One-shot API: `compress`, `decompress`, `compress_with_jobs`, `decompress_with_jobs`.
- Streaming API: module `stream`, `read` and `write` adapters, `BlockEncoder`, `BlockDecoder`.
- Parallel block processing with rayon (`Jobs`).
- CRC32 check for each block.
- Command-line tool `bzip3` (feature `cli`), helper scripts, and man pages.
