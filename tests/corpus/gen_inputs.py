"""Write deterministic corpus inputs for the ticket 08 oracle matrix."""

import os
import random
import sys

work = sys.argv[1]
os.makedirs(work, exist_ok=True)


def put(name, data):
    with open(os.path.join(work, name), "wb") as handle:
        handle.write(data)


put("empty.bin", b"")
put("1byte.bin", b"A")
put("63b.bin", b"x" * 63)
put("64b.bin", b"y" * 64)
put("65b.bin", b"z" * 65)
put("runs.bin", b"A" * 100_000)
phrase = bytes(range(50)) * 2
put("periodic.bin", phrase * 1000)
rng = random.Random(0xC0FFEE)
put("random.bin", bytes(rng.randrange(256) for _ in range(1_000_000)))
para = (
    b"This is the 100th Etext file presented by Project Gutenberg. "
    b"Shakespeare, William. The complete works in one file. "
    b"To be, or not to be, that is the question. "
    b"All the world's a stage, and all the men and women merely players. "
)
put("text-200k.bin", (para * 800)[:200_000])
put("text-6m.bin", (para * 30_000)[:6_000_000])
print("inputs written to", work)
