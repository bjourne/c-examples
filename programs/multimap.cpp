// To determine if my rbtree is faster than std::multimap
#include <assert.h>
#include <inttypes.h>
#include <iostream>
#include <map>
#include <vector>
extern "C" {
#include "datatypes/common.h"
#include "datatypes/vector.h"
}

// 13.8
void
test_torture_comp() {
    std::multimap<int, ptr> map;
    uint64_t count = 10 * 1000 * 1000;
    for (uint64_t i = 0; i < count; i++) {
        int key = rand();
        map.insert(std::make_pair(key, key));
    }
}

// 26.3
void
test_torture() {
    vector *v = v_init(32);
    std::multimap<int, ptr> map;
    uint64_t count = 10 * 1000 * 1000;
    for (uint64_t i = 0; i < count; i++) {
        int key = rand();
        map.insert(std::make_pair(key, key));
        v_add(v, key);
    }
    for (uint64_t i = 0; i < count; i++) {
        int key = (int)v->array[i];
        map.erase(key);
    }
    v_free(v);
    assert(map.size() == 0);
}

void
test_torture_2() {
    std::map<int, ptr> map;
    uint64_t count = 10 * 1000 * 1000;
    for (uint64_t i = 0; i < count; i++) {
        int key = rand();
        map.insert(std::make_pair(key, key));
    }
    printf("%" PRIu64 " elements in set\n", map.size());
}

void
test_torture_3() {
    std::map<int, ptr> map;
    std::vector<int> v;
    uint64_t count = 10 * 1000 * 1000;
    for (uint64_t i = 0; i < count; i++) {
        int key = rand();
        map.insert(std::make_pair(key, key));
        v.push_back(key);
    }
    for (uint64_t i = 0; i < count; i++) {
        map.erase(v[i]);
    }
    printf("%" PRIu64 " elements in set\n", map.size());
}

int
main(int argc, char *argv[]) {
    rand_init(0);
    PRINT_RUN(test_torture);
    PRINT_RUN(test_torture_comp);
    PRINT_RUN(test_torture_2);
    PRINT_RUN(test_torture_3);
}
