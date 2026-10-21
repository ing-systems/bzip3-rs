#!/bin/bash
# Oracle cross-decode matrix for ticket 08.
#
# Encodes every corpus file at every matrix block size with the Rust
# one-shot API, then decodes each output with the pinned C oracle and
# byte-compares it against the input. Fails on the first mismatch.
#
# Overrides: WORK (default /tmp/opencode/corpus-08),
# ORACLE (default /tmp/opencode/bzip3-oracle, pinned fd3d71f).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="${WORK:-/tmp/opencode/corpus-08}"
ORACLE="${ORACLE:-/tmp/opencode/bzip3-oracle}"

mkdir -p "$WORK/inputs" "$WORK/ours"
python3 "$HERE/gen_inputs.py" "$WORK/inputs"

ENC="$WORK/encoder"
if [ ! -d "$ENC" ]; then
    cargo new --bin --name encoder "$ENC" >/dev/null
fi
cp "$HERE/encoder_main.rs" "$ENC/src/main.rs"
printf '[package]\nname = "encoder"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\nbzip3-rs = { path = "%s" }\n' "$REPO" > "$ENC/Cargo.toml"
cargo build --release --manifest-path "$ENC/Cargo.toml"

DRIVER="$WORK/decode_check"
cc -O2 -o "$DRIVER" "$HERE/decode_check.c" \
    -I"$ORACLE/include" -L"$ORACLE/build" -lbzip3 \
    "-Wl,-rpath,$ORACLE/build"

# Matrix block sizes: 65 KiB floor, 1 MiB, 16 MiB default.
SIZES="66560 1048576 16777216"
fails=0
for input in "$WORK"/inputs/*.bin; do
    name="$(basename "$input" .bin)"
    for size in $SIZES; do
        out="$WORK/ours/$name-$size.bz3"
        "$ENC/target/release/encoder" "$size" "$input" "$out"
        if ! "$DRIVER" "$input" "$out"; then
            fails=$((fails + 1))
        fi
    done
done
if [ "$fails" -ne 0 ]; then
    echo "MATRIX FAILED: $fails mismatch(es)"
    exit 1
fi
echo "MATRIX PASSED"
