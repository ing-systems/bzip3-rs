# Corpus oracle matrix (ticket 08)

Rust encoder output, verified by the pinned C oracle.

## Layout

- `gen_inputs.py`: write deterministic corpus inputs to a work dir.
- `encoder_main.rs`: source of the throwaway encoder crate (built in
  the work dir, outside this repo). One-shot `compress` at a requested
  block size, writes the framed output.
- `decode_check.c`: oracle driver. Decodes one Rust-produced file with
  `bz3_decompress` and byte-compares it against the raw input.
- `matrix.sh`: full run. Generates inputs, builds the encoder and the
  driver, encodes every corpus file at every matrix block size, and
  fails on the first oracle mismatch.

## Corpus

`empty` (0 B), `1byte`, `63b`/`64b`/`65b` (stored-raw boundary),
`runs` (RLE model), `periodic` (LZP-only model), `random` (plain
coded model, fixed seed), `text-200k` (multi-block at 65 KiB),
`text-6m` (multi-block at 1 MiB). Matrix block sizes: 65 KiB,
1 MiB, 16 MiB.

## Run

```sh
sh tests/corpus/matrix.sh
```

Environment overrides: `WORK` (default `/tmp/opencode/corpus-08`),
`ORACLE` (default `/tmp/opencode/bzip3-oracle`, pinned `fd3d71f`).
The script needs `cargo`, `cc`, and `python3`.
