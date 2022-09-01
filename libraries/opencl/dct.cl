#define BLOCKDIM_X      8
#define BLOCKDIM_Y      8
#define BLOCK_SIZE      8
#define BLOCK_SIZE_F    1
#define SIMD_LOC        1
#define SIMD_WI         1
#define COMP_U          1

#define    C_a 1.3870398453221474618216191915664f  //a = sqrt(2) * cos(1 * pi / 16)
#define    C_b 1.3065629648763765278566431734272f  //b = sqrt(2) * cos(2 * pi / 16)
#define    C_c 1.1758756024193587169744671046113f  //c = sqrt(2) * cos(3 * pi / 16)
#define    C_d 0.78569495838710218127789736765722f //d = sqrt(2) * cos(5 * pi / 16)
#define    C_e 0.54119610014619698439972320536639f //e = sqrt(2) * cos(6 * pi / 16)
#define    C_f 0.27589937928294301233595756366937f //f = sqrt(2) * cos(7 * pi / 16)
#define C_norm 0.35355339059327376220042218105242f //1 / sqrt(8)

inline void
dct8(float *D_I, float *D_O){
    float X07P = D_I[0] + D_I[7];
    float X16P = D_I[1] + D_I[6];
    float X25P = D_I[2] + D_I[5];
    float X34P = D_I[3] + D_I[4];

    float X07M = D_I[0] - D_I[7];
    float X61M = D_I[6] - D_I[1];
    float X25M = D_I[2] - D_I[5];
    float X43M = D_I[4] - D_I[3];

    float X07P34PP = X07P + X34P;
    float X07P34PM = X07P - X34P;
    float X16P25PP = X16P + X25P;
    float X16P25PM = X16P - X25P;

    D_O[0] = C_norm * (X07P34PP + X16P25PP);
    D_O[2] = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
    D_O[4] = C_norm * (X07P34PP - X16P25PP);
    D_O[6] = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

    D_O[1] = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
    D_O[3] = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
    D_O[5] = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
    D_O[7] = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}

__attribute__((num_simd_work_items(SIMD_WI)))
__attribute__((num_compute_units(COMP_U)))
__kernel __attribute__((reqd_work_group_size(BLOCKDIM_Y / BLOCK_SIZE,
                                             BLOCKDIM_X / SIMD_LOC, 1)))
void dct(
    __global float * restrict src,
    __global float * restrict dst,
    uint height,
    uint width
){
    __local float   transp[BLOCKDIM_Y][BLOCKDIM_X + 1];
    const uint      local_y = BLOCK_SIZE * get_local_id(0);
    const uint      local_x = get_local_id(1) * SIMD_LOC;

    const uint      modLocalX = local_x & (BLOCK_SIZE - 1);

    // The pixel we are computing dct for
    const uint      global_y = get_group_id(0) * BLOCKDIM_Y + local_y;
    const uint      global_x = get_group_id(1) * BLOCKDIM_X + local_x;


    if ((global_x - modLocalX + BLOCK_SIZE - 1 >= width) ||
        (global_y + BLOCK_SIZE - 1 >= height) )
        return;

    __local float *l_V = &transp[local_y +         0][local_x +         0];
    __local float *l_H = &transp[local_y + modLocalX][local_x  - modLocalX];

    dst[global_y * width + global_x] = 99;

    printf("test %d %d\n", global_y, global_x);


    src += global_y * width + global_x;
    dst += global_y * width + global_x;

    float D_0[BLOCK_SIZE];
    float D_1[BLOCK_SIZE];
    float D_2[BLOCK_SIZE];
    float D_3[BLOCK_SIZE];

    for(uint i = 0; i < BLOCK_SIZE; i++)
    {
        l_V[i * (BLOCKDIM_X + 1)] = src[i * width];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint i = 0; i < BLOCK_SIZE; i++)
    {
        D_0[i] = l_H[i];
    }

    for (uint i = 0; i < BLOCK_SIZE_F; i++)
    {
    	dct8(&D_0[i * 8], &D_1[i * 8]);
    }

    for(uint i = 0; i < BLOCK_SIZE; i++)
    {
        l_H[i] = D_1[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint i = 0; i < BLOCK_SIZE; i++)
    {
        D_2[i] = l_V[i * (BLOCKDIM_X + 1)];
    }

    for(uint i = 0; i < BLOCK_SIZE_F; i++)
    {
    	dct8(&D_2[i * 8], &D_3[i * 8]);
    }

    for(uint i = 0; i < BLOCK_SIZE; i++)
    {
        dst[i * width] = D_3[i];
    }
}
