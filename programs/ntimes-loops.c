#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>

#define N_BLOCK_BITS 15
#define N_BLOCK (1 << N_BLOCK_BITS)
#define N_BLOCK_MASK (N_BLOCK - 1)

int
count_blocked(const char *s) {
    int r = 0;
    int tmp = 0;
    size_t n = strlen(s);
    for (size_t i = n & N_BLOCK_MASK; i--; ++s) {
        tmp += (*s == 's') - (*s == 'p');
    }
    r += tmp;

    for (n >>= N_BLOCK_BITS; n--;) {
        tmp = 0;
        for (int i = N_BLOCK; i--; ++s) {
            tmp += (*s == 's') - (*s == 'p');
        }
        r += tmp;
    }
    return r;
}


int
count_switches(const char *input) {
    int res = 0;
    while (true) {
        char c = *input++;
        switch (c) {
        case '\0':
            return res;
        case 's':
            res += 1;
            break;
        case 'p':
            res -= 1;
            break;
        default:
            break;
        }
    }
}

static
int to_add[256] = {
  ['s'] = 1,
  ['p'] = -1,
};

int
count_lookup(const char *input) {
    int res = 0;
    while (true) {
        char c = *input++;
        if (c == '\0') {
            return res;
        } else {
             res += to_add[(int) c];
        }
    }
}

int
count_naive(const char* input) {
    size_t len = strlen(input);
    int res = 0;
    for (size_t i = 0; i < len; ++i) {
        char c = input[i];
        res += c == 's';
        res -= c == 'p';
    }
    return res;
}


int
count_compl(const char* input) {
    int res = 0;
    while ((uintptr_t) input % sizeof(size_t)) {
        char c = *input++;
        res += c == 's';
        res -= c == 'p';
        if (c == 0) return res;
    }

    const size_t ONES = ((size_t) -1) / 255;  // 0x...01010101
    const size_t HIGH_BITS = ONES << 7;       // 0x...80808080
    const size_t SMASK = ONES * (size_t) 's'; // 0x...73737373
    const size_t PMASK = ONES * (size_t) 'p'; // 0x...70707070
    size_t s_accum = 0;
    size_t p_accum = 0;
    int iters = 0;
    while (1) {
        size_t w;
        memcpy(&w, input, sizeof(size_t));
        if ((w - ONES) & ~w & HIGH_BITS) break;
        input += sizeof(size_t);

        size_t s_high_bits = ((w ^ SMASK) - ONES) & ~(w ^ SMASK) & HIGH_BITS;
        size_t p_high_bits = ((w ^ PMASK) - ONES) & ~(w ^ PMASK) & HIGH_BITS;

        s_accum += s_high_bits >> 7;
        p_accum += p_high_bits >> 7;
        if (++iters >= 255 / sizeof(size_t)) {
            res += s_accum % 255;
            res -= p_accum % 255;
            iters = s_accum = p_accum = 0;
        }
    }
    res += s_accum % 255;
    res -= p_accum % 255;

    while (1) {
        char c = *input++;
        res += c == 's';
        res -= c == 'p';
        if (c == 0)
            break;
    }
    return res;
}

// Count of set bits in `plus` minus count of set bits in `minus`
// The result is in [ -32 .. +32 ] interval
static inline int
popcnt_diff(uint32_t p, uint32_t m) {
    p = __builtin_popcount(p);
    m = __builtin_popcount(m);
    return (int)p - (int)m;
}

// Horizontal sum of all 4 int64_t elements in the AVX2 vector
static inline int64_t
hadd_epi64( __m256i v32 )
{
    __m128i v = _mm256_extracti128_si256( v32, 1 );
    v = _mm_add_epi64( v, _mm256_castsi256_si128( v32 ) );
    const int64_t high = _mm_extract_epi64( v, 1 );
    const int64_t low = _mm_cvtsi128_si64( v );
    return high + low;
}




// Code from https://gist.github.com/Const-me/3ade77faad47f0fbb0538965ae7f8e04
int
count_avx2(const char *s) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i ch_s = _mm256_set1_epi8('s');
    const __m256i ch_p = _mm256_set1_epi8('p');

    // The pointer is aligned by 32 bytes, which serves two purposes:
    // we can use aligned loads, and most importantly loading 32 bytes
    // guarantees to not cross page boundary.  VMEM permissions are
    // defined for aligned 4kb pages, we can technically load within a
    // page without access violations, despite the language standard
    // says it's UB
    assert((uintptr_t)s % 32 == 0);

    __m256i cnt = _mm256_setzero_si256();
    while (true) {
        // Load 32 bytes from the pointer
        const __m256i v = _mm256_load_si256((const __m256i *)s);

        // Compare bytes for v == '\0'
        const __m256i z = _mm256_cmpeq_epi8( v, zero );
        // Compare bytes for equality with these two other markers
        __m256i cmp_p = _mm256_cmpeq_epi8(v, ch_s);
        __m256i cmp_m = _mm256_cmpeq_epi8(v, ch_p);

        const uint32_t bmp_zero = (uint32_t)_mm256_movemask_epi8( z );
        if (bmp_zero) {
            // At least one byte of the 32 was zero
            const int res = (int)( hadd_epi64(cnt) / 0xFF );

            uint32_t bmp_p = (uint32_t)_mm256_movemask_epi8(cmp_p);
            uint32_t bmp_m = (uint32_t)_mm256_movemask_epi8(cmp_m);

            // Clear higher bits in the two bitmaps which were after
            // the first found `\0`
            const uint32_t len = _tzcnt_u32(bmp_zero);
            bmp_p = _bzhi_u32(bmp_p, len );
            bmp_m = _bzhi_u32(bmp_m, len );

            // Produce the result
            return res + popcnt_diff(bmp_p, bmp_m);
        }

        // Increment the source pointer
        s += 32;

        // Compute horizontal sum of bytes within 8-byte lanes
        cmp_p = _mm256_sad_epu8(cmp_p, zero);
        cmp_m = _mm256_sad_epu8(cmp_m, zero);

        cmp_p = _mm256_sub_epi64(cmp_p, cmp_m);
        // Update the counter
        cnt = _mm256_add_epi64(cnt, cmp_p);
    }
}

int
count_avx2_2(const char *s) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i ch_s = _mm256_set1_epi8('s');
    const __m256i ch_p = _mm256_set1_epi8('p');

    assert((uintptr_t)s % 32 == 0);

    __m256i cnt0 = _mm256_setzero_si256();
    __m256i cnt1 = _mm256_setzero_si256();

    while (true) {
        // Load 64 bytes from the pointer
        const __m256i v0 = _mm256_load_si256((const __m256i *)s);
        const __m256i v1 = _mm256_load_si256((const __m256i *)(s + 32));
        s += 64;

        const __m256i z0 = _mm256_cmpeq_epi8(v0, zero);
        const __m256i z1 = _mm256_cmpeq_epi8(v1, zero);

        // Compare bytes for equality with these two other markers
        __m256i cmp_p0 = _mm256_cmpeq_epi8(v0, ch_s);
        __m256i cmp_m0 = _mm256_cmpeq_epi8(v0, ch_p);
        __m256i cmp_p1 = _mm256_cmpeq_epi8(v1, ch_s);
        __m256i cmp_m1 = _mm256_cmpeq_epi8(v1, ch_p);

        const uint32_t bmp_zero0 = (uint32_t)_mm256_movemask_epi8(z0);
        const uint32_t bmp_zero1 = (uint32_t)_mm256_movemask_epi8(z1);
        if (bmp_zero0 || bmp_zero1) {
            if (bmp_zero0) {
                const int res = (int)(hadd_epi64(cnt0) / 0xff) +
                    (int)(hadd_epi64(cnt1) / 0xff);
                uint32_t bmp_p0 = (uint32_t)_mm256_movemask_epi8(cmp_p0);
                uint32_t bmp_m0 = (uint32_t)_mm256_movemask_epi8(cmp_m0);

                const uint32_t len = _tzcnt_u32(bmp_zero0);
                bmp_p0 = _bzhi_u32(bmp_p0, len);
                bmp_m0 = _bzhi_u32(bmp_m0, len);

                return res + popcnt_diff(bmp_p0, bmp_m0);
            }
            cmp_p0 = _mm256_sad_epu8(cmp_p0, zero);
            cmp_m0 = _mm256_sad_epu8(cmp_m0, zero);
            cmp_p0 = _mm256_sub_epi64(cmp_p0, cmp_m0);
            cnt0 = _mm256_add_epi64(cnt0, cmp_p0);

            const int res = (int)(hadd_epi64(cnt0) / 0xff) +
                (int)(hadd_epi64(cnt1) / 0xff);

            uint32_t bmp_p1 = (uint32_t)_mm256_movemask_epi8(cmp_p1);
            uint32_t bmp_m1 = (uint32_t)_mm256_movemask_epi8(cmp_m1);

            const uint32_t len = _tzcnt_u32(bmp_zero1);
            bmp_p1 = _bzhi_u32(bmp_p1, len);
            bmp_m1 = _bzhi_u32(bmp_m1, len);

            return res + popcnt_diff(bmp_p1, bmp_m1);
        }

        cmp_p0 = _mm256_sad_epu8(cmp_p0, zero);
        cmp_m0 = _mm256_sad_epu8(cmp_m0, zero);
        cmp_p0 = _mm256_sub_epi64(cmp_p0, cmp_m0);
        cnt0 = _mm256_add_epi64(cnt0, cmp_p0);

        cmp_p1 = _mm256_sad_epu8(cmp_p1, zero);
        cmp_m1 = _mm256_sad_epu8(cmp_m1, zero);
        cmp_p1 = _mm256_sub_epi64(cmp_p1, cmp_m1);
        cnt1 = _mm256_add_epi64(cnt1, cmp_p1);
    }
}