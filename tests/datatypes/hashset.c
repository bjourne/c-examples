#include <assert.h>
#include <inttypes.h>
#include <time.h>
#include "datatypes/hashset.h"

int main(int argc, char *argv[]) {

    rand_init(0);
    hashset* hs = hs_init();

    for (int i = 0; i < 10; i++) {
        size_t el = rand_n(100);
        hs_add(hs, el);
    }

    HS_FOR_EACH_ITEM(hs, { printf("%" PRIuPTR "\n", p); });

    hs_free(hs);

    return 0;
}
