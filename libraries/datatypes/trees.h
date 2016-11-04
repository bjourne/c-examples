#ifndef DATATYPES_TREES_H
#define DATATYPES_TREES_H

typedef enum {
    BST_LEFT = 0,
    BST_RIGHT = 1
} bstdir;

// Determines the direction of n in relation to p. n is allowed to be NULL.
#define BST_DIR_OF2(n, p)   ((n) == (p)->childs[BST_LEFT] ? BST_LEFT : BST_RIGHT)

// Determines if n is a left or right child. n can't be root because
// that node isn't a child.
#define BST_DIR_OF(n)       BST_DIR_OF2(n, (n->parent))



#endif
