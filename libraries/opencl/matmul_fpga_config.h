#ifndef _PE_SYSTOLIC_ARRAY_H_
#define _PE_SYSTOLIC_ARRAY_H_

// This is important but it is not enforced:
// PE_ROWS + PE_COLS <= ROWS_INTERLEAVED

// design space exploration of three vector sizes: float4, float8 and float16
#define DOT_PROD_VECTOR_SIZE     8

#define PE_ROWS                  2
#define PE_COLS                  2
#define ROWS_INTERLEAVED         32
#define COLUMNS_INTERLEAVED      32

#define MAT_A_BLOCK_WIDTH           (16 * DOT_PROD_VECTOR_SIZE)
#define MAT_A_BLOCK_HEIGHT          (ROWS_INTERLEAVED * PE_ROWS)

#define ACCUM_SHIFT_REG_SIZE        (ROWS_INTERLEAVED * COLUMNS_INTERLEAVED)
#define C_OUT_SHIFT_REG_SIZE        ACCUM_SHIFT_REG_SIZE

#define MAT_A_BLOCK_SIZE            (MAT_A_BLOCK_HEIGHT * MAT_A_BLOCK_WIDTH)
#define MAT_A_BLOCK_NUM_VECTORS     (MAT_A_BLOCK_SIZE   / DOT_PROD_VECTOR_SIZE)

#define MAT_B_BLOCK_HEIGHT          MAT_A_BLOCK_WIDTH
#define MAT_B_BLOCK_WIDTH           (COLUMNS_INTERLEAVED * PE_COLS)
#define MAT_B_BLOCK_SIZE            (MAT_B_BLOCK_HEIGHT  * MAT_B_BLOCK_WIDTH)
#define MAT_B_BLOCK_NUM_VECTORS     (MAT_B_BLOCK_SIZE    / DOT_PROD_VECTOR_SIZE)

#define MAT_C_BLOCK_HEIGHT          MAT_A_BLOCK_HEIGHT
#define MAT_C_BLOCK_WIDTH           MAT_B_BLOCK_WIDTH

#define VECTOR_FLOAT4_ZERO          (float4)(0.0f, 0.0f, 0.0f, 0.0f)
#define VECTOR_FLOAT8_ZERO          (float8)(VECTOR_FLOAT4_ZERO,VECTOR_FLOAT4_ZERO)
#define VECTOR_FLOAT16_ZERO         (float16)(VECTOR_FLOAT8_ZERO,VECTOR_FLOAT8_ZERO)

#define ROW_VECS                    (MAT_A_BLOCK_WIDTH / DOT_PROD_VECTOR_SIZE)
#define ROW_VECS_MASK               (ROW_VECS - 1)

#define SWAP_RANGE                  (ROWS_INTERLEAVED * COLUMNS_INTERLEAVED * ROW_VECS)
#define SWAP_RANGE_MASK             (SWAP_RANGE - 1)

#define RANGE                       (2 * SWAP_RANGE)
#define RANGE_MASK                  (RANGE - 1)

#ifndef HOST

#if DOT_PROD_VECTOR_SIZE==4
    typedef float4 vec_float_t;
    #define VECTOR_ZERO         VECTOR_FLOAT4_ZERO
#elif DOT_PROD_VECTOR_SIZE==8
    typedef float8 vec_float_t;
    #define VECTOR_ZERO         VECTOR_FLOAT8_ZERO
#elif DOT_PROD_VECTOR_SIZE==16
    typedef float16 vec_float_t;
    #define VECTOR_ZERO         VECTOR_FLOAT16_ZERO
#else
    #error Unsupported DOT_PROD_VECTOR_SIZE
#endif

struct vec_float_t_bool {
    vec_float_t data;
    bool  c;  // indicates a new row/column pair
};

#define LVEC 1

// The number of rows rounded up to the next power of 2
#if PE_ROWS <= 1
    #define BANKROWS 1
#elif PE_ROWS <= 2
    #define BANKROWS 2
#elif PE_ROWS <= 4
    #define BANKROWS 4
#elif PE_ROWS <= 8
    #define BANKROWS 8
#elif PE_ROWS <= 16
    #define BANKROWS 16
#elif PE_ROWS <= 32
    #define BANKROWS 32
#elif PE_ROWS <= 64
    #define BANKROWS 64
#elif PE_ROWS <= 128
    #define BANKROWS 128
#elif PE_ROWS <= 256
    #define BANKROWS 256
#else
    #error "PE_ROWS too large, BANKROWS cannot be defined"
#endif

// The number of columns rounded up to the next power of 2
#if PE_COLS <= 1
    #define BANKCOLS 1
#elif PE_COLS <= 2
    #define BANKCOLS 2
#elif PE_COLS <= 4
    #define BANKCOLS 4
#elif PE_COLS <= 8
    #define BANKCOLS 8
#elif PE_COLS <= 16
    #define BANKCOLS 16
#elif PE_COLS <= 32
    #define BANKCOLS 32
#elif PE_COLS <= 64
    #define BANKCOLS 64
#elif PE_COLS <= 128
    #define BANKCOLS 128
#elif PE_COLS <= 256
    #define BANKCOLS 256
#else
    #error "PE_COLS too large, BANKCOLS cannot be defined"
#endif

#endif


#endif // _PE_SYSTOLIC_ARRAY_H_
