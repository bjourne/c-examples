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
        error("Unknown type!\n");
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
        error("Unknown type!\n");
        return 0;
    }
}
