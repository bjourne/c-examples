// To determine if my rbtree is faster than std::multimap
#include <assert.h>
#include <iostream>
#include <map>
extern "C" {
#include "datatypes/common.h"
#include "datatypes/vector.h"
}


// 13.8
void
test_torture_comp() {
    std::multimap<uint64_t, uint64_t> map;
    uint64_t range = 10 * 1000 * 1000 * 1000LL;
    uint64_t count = 10 * 1000 * 1000;
    for (uint64_t i = 0; i < count; i++) {
        uint64_t key = rand_n(range);
        map.insert(std::make_pair(key, key));
    }
}

// 26.3
void
test_torture() {
    vector *v = v_init(32);
    std::multimap<uint64_t, uint64_t> map;
    uint64_t range = 10 * 1000 * 1000 * 1000LL;
    uint64_t count = 10 * 1000 * 1000;
    for (uint64_t i = 0; i < count; i++) {
        uint64_t key = rand_n(range);
        map.insert(std::make_pair(key, key));
        v_add(v, key);
    }
    for (uint64_t i = 0; i < count; i++) {
        uint64_t key = v->array[i];
        map.erase(key);
    }
    v_free(v);
    assert(map.size() == 0);
}


int
main(int argc, char *argv[]) {
    time_t seed = time(NULL);
    srand(seed);
    PRINT_RUN(test_torture);
}
