// Copyright (C) 2023-2024 Björn A. Lindqvist <bjourne@gmail.com>
//
// This kernel demonstrates how work items in a workgroup can
// communicate with each other.
#define VECTOR_WIDTH    8
#include "libraries/opencl/utils.cl"
#define MAILBOX_SIZE    100

typedef struct {
    uint prio;
    uint recv;
    uint data;
    uint padding;
} mail;

typedef struct {
    uint n_mails;
    mail mails[MAILBOX_SIZE];
} mailbox;

void
mailbox_post(mailbox *h, mail mail) {
    uint i = h->n_mails++;
    while (i) {
        uint parent = (i - 1) / 2;
        if (h->mails[parent].prio <= mail.prio) {
            break;
        }
        h->mails[i] = h->mails[parent];
        i = parent;
    }
    h->mails[i] = mail;
}

mail
mailbox_take(mailbox *me) {
    me->n_mails--;
    mail temp = me->mails[me->n_mails];
    mail min = me->mails[0];
    uint i = 0;
    while (true) {
        uint swap = (i * 2) + 1;
        if (swap >= me->n_mails) {
            break;
        }
        uint other = swap + 1;
        if ((other < me->n_mails) &&
            (me->mails[other].prio <= me->mails[swap].prio)) {
            swap = other;
        }
        if (temp.prio <= me->mails[swap].prio) {
            break;
        }
        me->mails[i] = me->mails[swap];
        i = swap;
    }
    me->mails[i] = temp;
    return min;
}

#define N_WORK_ITEMS 4

__kernel void
run_heap(const uint T, const uint N, __global mail *mails) {
    uint me = get_local_id(0);
    ranges r = slice_work(N, N_WORK_ITEMS, me, 1);

    // boxes[i][j] contains messages from work item i to work item j.
    __local mailbox boxes[N_WORK_ITEMS][N_WORK_ITEMS];
    for (uint i = 0; i < N_WORK_ITEMS; i++) {
        for (uint j = 0; j < N_WORK_ITEMS; j++) {
            boxes[i][j].n_mails = 0;
        }
    }
    for (uint t = 0; t < T; t++) {
        for (uint i = 0; i < N_WORK_ITEMS; i++) {
            __local mailbox *box = &boxes[i][me];
            while(box->n_mails > 0 && box->mails[0].prio == t) {
                mail m = mailbox_take(box);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint n = r.c0; n < r.c1; n++) {
            mail m = mails[N * t + n];
            m.prio = t + m.prio;
            __local mailbox *box = &boxes[me][m.recv];
            mailbox_post(box, m);
            if (box->n_mails == MAILBOX_SIZE) {
                printf("Full at %d: %d\n", t, box->mails[0].prio);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (me == 0) {
        printf("== Remaining mails\n");
        for (uint i = 0; i < N_WORK_ITEMS; i++) {
            for (uint j = 0; j < N_WORK_ITEMS; j++) {
                __local mailbox *box = &boxes[i][j];
                uint prio = box->mails[0].prio;
                printf("%2d -> %2d = %5d, prio %d\n",
                       i, j,
                       box->n_mails,
                       prio);
            }
        }
    }
}
