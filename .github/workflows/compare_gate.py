#!/usr/bin/env python3
"""Same-machine throughput gate: ours vs the pinned C oracle.

Runs every gate case back to back on this machine with both CLIs at
``-j 1`` and fails when the Rust side is slower than the bar: geomean
of Rust/C time ratios below 1.0, or any single case below 0.95.

Time ratio per case = oracle seconds / ours seconds, so values above
1.0 mean Rust is faster. Runs interleave ours/oracle per rep (warmup
plus best of ``--reps``) to defeat thermal drift. Timed inputs are
large (32 MB text, 8 MB fixed-seed random) so codec time dominates
process startup and I/O noise; the small-file corpus still runs
through encode, decode, and byte-compare as correctness checks.
Decode cases reuse the timed encode outputs and byte-compare every
round trip, so the gate also checks correctness.

Usage:
    compare_gate.py --ours BIN --oracle-bin BIN --repo DIR [--reps N]
"""

import argparse
import math
import os
import platform
import statistics
import subprocess
import sys
import time

GEOMEAN_BAR = 1.0
CASE_BAR = 0.95
BLOCKS = (1, 16)
INPUTS = ("runs.bin", "periodic.bin", "random.bin", "text-200k.bin", "text-6m.bin")
# Timed inputs (MiB) generated deterministically by this script. Large
# enough that codec time dominates process startup and I/O noise.
TIMED_TEXT_MB = 32
TIMED_RAND_MB = 8
RAND_SEED = 0xC0FFEE
PARA = (
    b"This is the 100th Etext file presented by Project Gutenberg. "
    b"Shakespeare, William. The complete works in one file. "
    b"To be, or not to be, that is the question. "
    b"All the world's a stage, and all the men and women merely players. "
)


def write_timed_inputs(work):
    """Write the large deterministic timed inputs. Return paths."""
    import random as pyrandom

    text_path = os.path.join(work, "gate-text.bin")
    rand_path = os.path.join(work, "gate-rand.bin")
    if not os.path.exists(text_path):
        want = TIMED_TEXT_MB * 1024 * 1024
        with open(text_path, "wb") as handle:
            while handle.tell() < want:
                handle.write(PARA[: min(len(PARA), want - handle.tell())])
    if not os.path.exists(rand_path):
        rng = pyrandom.Random(RAND_SEED)
        want = TIMED_RAND_MB * 1024 * 1024
        with open(rand_path, "wb") as handle:
            chunk = 1 << 20
            left = want
            while left > 0:
                handle.write(rng.randbytes(min(chunk, left)))
                left -= chunk
    return (("gate-text", text_path), ("gate-rand", rand_path))


def run(cmd, stdin_path, stdout_path, timeout=600):
    """Run cmd with stdin/stdout redirected to files. Return seconds."""
    with open(stdin_path, "rb") as stdin, open(stdout_path, "wb") as stdout:
        start = time.perf_counter()
        proc = subprocess.run(cmd, stdin=stdin, stdout=stdout, timeout=timeout)
        secs = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return secs


def check_equal(first, second, label):
    """Fail when two files differ."""
    with open(first, "rb") as a, open(second, "rb") as b:
        if a.read() != b.read():
            raise RuntimeError(f"byte mismatch: {label}")


def time_pair(cmd_a, in_a, cmd_b, in_b, out_a, out_b, reps):
    """Interleave both commands per rep; return best seconds each."""
    run(cmd_a, in_a, out_a)
    run(cmd_b, in_b, out_b)
    best_a, best_b = None, None
    for _ in range(reps):
        secs_a = run(cmd_a, in_a, out_a)
        secs_b = run(cmd_b, in_b, out_b)
        best_a = secs_a if best_a is None else min(best_a, secs_a)
        best_b = secs_b if best_b is None else min(best_b, secs_b)
    return best_a, best_b


def cpu_model():
    """Return the CPU model string for the re-baseline record."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main():
    """Run the full gate matrix and enforce the bar."""
    parser = argparse.ArgumentParser(description="Oracle throughput gate.")
    parser.add_argument("--ours", required=True, help="Our release bzip3 binary.")
    parser.add_argument("--oracle-bin", required=True, help="Oracle bzip3 binary.")
    parser.add_argument("--repo", required=True, help="Repo root (corpus scripts).")
    parser.add_argument("--work", default="perf-gate-work", help="Scratch dir.")
    parser.add_argument("--reps", type=int, default=5, help="Timed reps per case.")
    args = parser.parse_args()

    inputs_dir = os.path.join(args.work, "inputs")
    os.makedirs(args.work, exist_ok=True)
    gen = os.path.join(args.repo, "tests", "corpus", "gen_inputs.py")
    subprocess.run([sys.executable, gen, inputs_dir], check=True)

    print(f"machine: {platform.system()} {platform.machine()} | {cpu_model()}")
    ratios = {}
    seconds = {}
    # Correctness sweep: full corpus both directions, byte-compared.
    for name in INPUTS:
        src = os.path.join(inputs_dir, name)
        for block in BLOCKS:
            tag = f"{name}/mib{block}"
            ours_enc = os.path.join(args.work, f"ours-{name}-{block}.bz3")
            orc_enc = os.path.join(args.work, f"orc-{name}-{block}.bz3")
            run([args.ours, "-e", "-b", str(block), "-j", "1", "-c"], src, ours_enc)
            run([args.oracle_bin, "-e", "-b", str(block), "-j", "1", "-c"], src, orc_enc)
            ours_out = os.path.join(args.work, f"ours-{name}-{block}.out")
            orc_out = os.path.join(args.work, f"orc-{name}-{block}.out")
            run([args.ours, "-d", "-c", "-j", "1"], ours_enc, ours_out)
            run([args.oracle_bin, "-d", "-c", "-j", "1"], orc_enc, orc_out)
            check_equal(src, ours_out, f"ours decode {tag}")
            check_equal(src, orc_out, f"oracle decode {tag}")
            print(f"check-only: {tag}")
    # Timed gate: large inputs both directions at both block sizes.
    for name, src in write_timed_inputs(args.work):
        for block in BLOCKS:
            tag = f"{name}/mib{block}"
            ours_enc = os.path.join(args.work, f"t-ours-{name}-{block}.bz3")
            orc_enc = os.path.join(args.work, f"t-orc-{name}-{block}.bz3")
            ours_secs, orc_secs = time_pair(
                [args.ours, "-e", "-b", str(block), "-j", "1", "-c"],
                src,
                [args.oracle_bin, "-e", "-b", str(block), "-j", "1", "-c"],
                src,
                ours_enc,
                orc_enc,
                args.reps,
            )
            ratios[f"encode/{tag}"] = orc_secs / ours_secs
            seconds[f"encode/{tag}"] = (ours_secs, orc_secs)
            ours_out = os.path.join(args.work, f"t-ours-{name}-{block}.out")
            orc_out = os.path.join(args.work, f"t-orc-{name}-{block}.out")
            ours_d, orc_d = time_pair(
                [args.ours, "-d", "-c", "-j", "1"],
                ours_enc,
                [args.oracle_bin, "-d", "-c", "-j", "1"],
                orc_enc,
                ours_out,
                orc_out,
                args.reps,
            )
            check_equal(src, ours_out, f"ours decode {tag}")
            check_equal(src, orc_out, f"oracle decode {tag}")
            ratios[f"decode/{tag}"] = orc_d / ours_d
            seconds[f"decode/{tag}"] = (ours_d, orc_d)

    worst = min(ratios.values())
    geomean = math.exp(statistics.fmean(math.log(r) for r in ratios.values()))
    for case in sorted(ratios):
        ours_s, orc_s = seconds[case]
        print(f"{case:32s} ratio {ratios[case]:.3f} "
              f"(ours {ours_s:.3f}s oracle {orc_s:.3f}s)")
    print(f"geomean {geomean:.3f} (bar {GEOMEAN_BAR}), worst {worst:.3f} "
          f"(bar {CASE_BAR})")
    failures = [c for c, r in ratios.items() if r < CASE_BAR]
    if failures:
        print(f"GATE FAILED: below per-case bar: {', '.join(sorted(failures))}")
        sys.exit(1)
    if geomean < GEOMEAN_BAR:
        print("GATE FAILED: geomean below bar")
        sys.exit(1)
    print("GATE PASSED")


if __name__ == "__main__":
    main()
