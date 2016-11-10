#include <inttypes.h>
#include <stdlib.h>
#include "collectors/common.h"

// Pointer methods
size_t
p_size(ptr p) {
    switch (P_GET_TYPE(p)) {
    case TYPE_INT:
    case TYPE_FLOAT:
    case TYPE_WRAPPER:
        return NPTRS(2);
    case TYPE_ARRAY: {
        size_t n_els = *SLOT_P(*SLOT_P(p, 0), 0);
        return NPTRS(2 + n_els);
    }
    default:
        error("Unknown type in p_size!\n");
        return 0;
    }
}

size_t
p_slot_count(ptr p) {
    switch (P_GET_TYPE(p)) {
    case TYPE_INT: return 0;
    case TYPE_FLOAT: return 0;
    case TYPE_WRAPPER: return 1;
    case TYPE_ARRAY: {
        size_t n_els = *SLOT_P(*SLOT_P(p, 0), 0);
        return 1 + n_els;
    }
    default:
        error("Unknown type in p_slot_count!\n");
        return 0;
    }
}

static
void p_print(int ind, size_t i, ptr p);

void p_print_slots(int ind, ptr *base, size_t n) {
    for (size_t i = 0; i < n; i++) {
        p_print(ind + 2, i, base[i]);
    }
}

static char *
type_name(unsigned int t) {
    switch (t) {
    case TYPE_INT: return "int";
    case TYPE_FLOAT: return "float";
    case TYPE_WRAPPER: return "wrapper";
    case TYPE_ARRAY: return "array";
    default: return "unknown";
    }
}

static
void p_print(int ind, size_t n, ptr p) {
    for (int x = 0; x < ind; x++) { putchar(' '); }
    if (p == 0) {
        printf("%2" PRId64 ": null\n", n);
        return;
    }
    unsigned int t = P_GET_TYPE(p);
    printf("%2" PRId64 ": %s @ 0x%" PRIx64 ": ", n, type_name(t), p);
    if (t == TYPE_FLOAT) {
        printf("%.3f", (double)*SLOT_P(p, 0));
    } else if (t == TYPE_INT) {
        printf("%d", (int)*SLOT_P(p, 0));
    }
    putchar('\n');
    size_t n_slots = p_slot_count(p);
    p_print_slots(ind + 2, SLOT_P(p, 0), n_slots);
}
