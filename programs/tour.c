#include <assert.h>
#include "linalg/linalg-io.h"

int
main(int argc, char *argv[]) {
    int n;
    vec2 *cities = v2_array_read(stdin, &n);
    for (int i = 0; i < n; i++) {
        printf("%d: %.2f %.2f\n", i, cities[i].x, cities[i].y);
    }
}
