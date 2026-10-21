/* Oracle check driver: decode one Rust-produced one-shot file with
   bz3_decompress and byte-compare it against the raw input.
   Usage: decode_check <raw-input> <rust-output.bz3>. Exit 0 on match. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libbz3.h"

static uint8_t * slurp(const char * path, size_t * len) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        perror("fopen");
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    if (n < 0) {
        perror("ftell");
        exit(1);
    }
    fseek(f, 0, SEEK_SET);
    uint8_t * buf = malloc((size_t)n > 0 ? (size_t)n : 1);
    if (!buf) {
        fprintf(stderr, "oom\n");
        exit(1);
    }
    if (n > 0 && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "short read %s\n", path);
        exit(1);
    }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: decode_check <raw> <encoded>\n");
        return 2;
    }
    size_t raw_len = 0, enc_len = 0;
    uint8_t * raw = slurp(argv[1], &raw_len);
    uint8_t * enc = slurp(argv[2], &enc_len);

    size_t cap = raw_len + 64;
    uint8_t * out = malloc(cap);
    if (!out) {
        fprintf(stderr, "oom\n");
        return 1;
    }
    size_t out_size = cap;
    int rc = bz3_decompress(enc, out, enc_len, &out_size);
    int ok = rc == 0 && out_size == raw_len &&
        (raw_len == 0 || memcmp(out, raw, raw_len) == 0);
    printf(
        "oracle decode %s: rc=%d out=%zu want=%zu match=%d\n",
        argv[2],
        rc,
        out_size,
        raw_len,
        ok
    );
    free(raw);
    free(enc);
    free(out);
    return ok ? 0 : 1;
}
