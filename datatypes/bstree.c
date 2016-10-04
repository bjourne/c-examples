#include <stdio.h>
#include "bstree.h"

static bstree *
bst_init(ptr root) {
    bstree *bst = (bstree *)malloc(sizeof(bstree));
    bst->left = NULL;
    bst->right = NULL;
    bst->data = root;
    return bst;
}

void
bst_free(bstree *bst) {
    if (bst) {
        bst_free(bst->left);
        bst_free(bst->right);
        free(bst);
    }
}

bstree *
bst_add(bstree *bst, ptr data) {
    if (!bst) {
        return bst_init(data);
    }
    ptr this_data = bst->data;
    if (data < this_data) {
        bst->left = bst_add(bst->left, data);
    } else {
        bst->right = bst_add(bst->right, data);
    }
    return bst;
}

void
bst_print_inorder(bstree *bst) {
    if (bst) {
        bst_print_inorder(bst->left);
        printf("%lu\n", bst->data);
        bst_print_inorder(bst->right);
    }
}

bstree *
bst_find(bstree *bst, ptr data) {
    if (!bst)
        return bst;
    ptr this_data = bst->data;
    if (data < this_data) {
        return bst_find(bst->left, data);
    } else if (data > this_data) {
        return bst_find(bst->right, data);
    }
    return bst;
}

size_t
bst_size(bstree *bst) {
    if (!bst)
        return 0;
    return 1 + bst_size(bst->left) + bst_size(bst->right);
}

bstree *
bst_min_node(bstree *bst) {
    while (bst->left) {
        bst = bst->left;
    }
    return bst;
}

bstree *
bst_max_node(bstree *bst) {
    while (bst->right) {
        bst = bst->right;
    }
    return bst;
}

bstree *
bst_remove(bstree *bst, ptr data) {
    if (!bst)
        return bst;
    if (data < bst->data) {
        bst->left = bst_remove(bst->left, data);
    } else if (data > bst->data) {
        bst->right = bst_remove(bst->right, data);
    } else {
        // Found node to delete
        if (!bst->left) {
            bstree *right = bst->right;
            bst->right = NULL;
            bst_free(bst);
            bst = right;
        } else if (!bst->right) {
            bstree *left = bst->left;
            bst->left = NULL;
            bst_free(bst);
            bst = left;
        } else {
            // It has two children. Copy inorder successors value.
            bstree *inorder_succ = bst_min_node(bst->right);
            bst->data = inorder_succ->data;
            bst->right = bst_remove(bst->right, inorder_succ->data);
        }
    }
    return bst;
}
