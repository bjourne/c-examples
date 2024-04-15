__kernel void
kern_main(global unsigned int * restrict frame_in,
          global unsigned int * restrict frame_out,
          const int iterations,
          const unsigned int threshold) {

    int gx[3][3] = {
        {-1,-2,-1},
        {0,0,0},
        {1,2,1}
    };
    int gy[3][3] = {
        {-1,0,1},
        {-2,0,2},
        {-1,0,1}
    };

    int rows[2 * COLS + 3];

    int count = -(2 * COLS + 3);
    while (count != iterations) {
        #pragma unroll
        for (int i = COLS * 2 + 2; i > 0; --i) {
            rows[i] = rows[i - 1];
        }
        rows[0] = count >= 0 ? frame_in[count] : 0;

        int x_dir = 0;
        int y_dir = 0;

        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            #pragma unroll
            for (int j = 0; j < 3; ++j) {
                unsigned int pixel = rows[i * COLS + j];
                unsigned int b = pixel & 0xff;
                unsigned int g = (pixel >> 8) & 0xff;
                unsigned int r = (pixel >> 16) & 0xff;

                unsigned int luma = r * 66 + g * 129 + b * 25;
                luma = (luma + 128) >> 8;
                luma += 16;

                x_dir += luma * gx[i][j];
                y_dir += luma * gy[i][j];
            }
        }

        int temp = abs(x_dir) + abs(y_dir);
        unsigned int clamped;
        if (temp > threshold) {
            clamped = 0xffffff;
        } else {
            clamped = 0;
        }

        if (count >= 0) {
            frame_out[count] = clamped;
        }
        count++;
    }
}
