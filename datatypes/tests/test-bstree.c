#include <assert.h>
#include <stdio.h>
#include <time.h>
#include "../bstree.h"

size_t rand_n(size_t n) {
    return rand() % n;
}

int main(int argc, char *argv[]) {

    srand(time(NULL));

    bstree *bst1 = bst_add(NULL, 123);
    bst1 = bst_remove(bst1, 123);
    assert(bst1 == NULL);

    bstree *bst2 = NULL;
    bst2 = bst_add(bst2, 10);
    bst2 = bst_add(bst2, 20);
    bst2 = bst_remove(bst2, 20);
    assert(bst2->left == NULL);
    assert(bst2->right == NULL);
    bst_free(bst2);

    bstree *bst3 = NULL;
    bst_remove(bst3, 30);

    bst3 = bst_add(bst3, 10);
    bst3 = bst_add(bst3, 20);
    bst3 = bst_add(bst3, 30);

    assert(bst3->data == 10);
    assert(bst3->right->data == 20);
    assert(bst3->right->right->data == 30);

    bst3 = bst_remove(bst3, 20);
    assert(bst3->right->data == 30);
    bst3 = bst_add(bst3, 2);
    assert(bst3->left->data == 2);
    assert(bst_min_node(bst3)->data == 2);
    assert(bst_max_node(bst3)->data == 30);

    bst3 = bst_remove(bst3, 10);
    assert(bst3->data == 30);
    bst_free(bst3);


    bstree *bst4 = NULL;
    bst4 = bst_add(bst4, 50);
    bst4 = bst_add(bst4, 40);
    bst4 = bst_add(bst4, 70);
    bst4 = bst_add(bst4, 60);
    bst4 = bst_add(bst4, 80);
    assert(bst_size(bst4) == 5);

    assert(bst_find(bst4, 80)->data == 80);

    bst4 = bst_remove(bst4, 50);
    bst_print_inorder(bst4);
    bst_free(bst4);

    return 0;
}
