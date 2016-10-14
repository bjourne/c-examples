// This example demonstrates how to improve the performance of
// memcpy() and memmove().
#include <stdlib.h>
#include <string.h>
#include "datatypes/common.h"

#define BUFFER_SIZE (100 << 10)

ptr buf = 0;
void *(*my_memmove)(void *, const void *, size_t) = NULL;

static void *
wrapped_memmove(void *dest, const void *src, size_t n) {
    if (dest == src)
        return dest;
    return memmove(dest, src, n);
}

static void
shuffle_memory() {
    for (int i = 0; i < 100000; i++) {
        ptr ofs_src = rand_n(BUFFER_SIZE);
        ptr ofs_dst = rand_n(BUFFER_SIZE);
        ptr src = buf + ofs_src;
        ptr dst = buf + ofs_dst;

        size_t size = rand_n(BUFFER_SIZE);
        size = MIN(size, BUFFER_SIZE - ofs_src);
        size = MIN(size, BUFFER_SIZE - ofs_dst);
        my_memmove((void *)dst, (void *)src, size);
    }
}

static void
unshuffle_memory() {
    for (int i = 0; i < 100000; i++) {
        ptr ofs = rand_n(BUFFER_SIZE);
        ptr dst = buf + ofs;
        ptr src = buf + ofs;

        size_t size = rand_n(BUFFER_SIZE);
        size = MIN(size, BUFFER_SIZE - ofs);
        my_memmove((void *)dst, (void *)src, size);
    }
}

int
main(int argc, char *argv[]) {
    buf = (ptr)malloc(BUFFER_SIZE);
    memset((void *)buf, 33, BUFFER_SIZE);
    srand(5678);
    my_memmove = &memmove;
    PRINT_RUN_INT("Shuffling using memmove()", shuffle_memory);
    my_memmove = &wrapped_memmove;
    PRINT_RUN_INT("Shuffling wrapped_memmove()", shuffle_memory);

    my_memmove = &memmove;
    PRINT_RUN_INT("Unshuffling using memmove()", unshuffle_memory);
    my_memmove = &wrapped_memmove;
    PRINT_RUN_INT("Unshuffling using wrapped_memmove()", unshuffle_memory);

    free((void *)buf);
    return 0;
}
