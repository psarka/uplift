import numpy as np
cimport numpy as np

from sklearn.tree._tree cimport Tree

TREE_LEAF = -1
cdef SIZE_t _TREE_LEAF = TREE_LEAF

cdef class UpliftTree(Tree):

    def __cinit__(self, int n_features, np.ndarray[SIZE_t, ndim=1] n_classes,
                  int n_outputs, char* importances):

        self.importances = importances


    cpdef compute_feature_importances(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        cdef Node* left
        cdef Node* right
        cdef Node* nodes = self.nodes
        cdef Node* node = nodes
        cdef Node* end_node = node + self.node_count

        cdef double normalizer = 0.

        cdef double larger_impurity
        cdef double n_of_larger
        cdef double smaller_impurity
        cdef double n_of_smaller
        cdef double n

        cdef np.ndarray[np.float64_t, ndim=1] importances
        importances = np.zeros((self.n_features,))
        cdef DOUBLE_t* importance_data = <DOUBLE_t*>importances.data

        with nogil:
            while node != end_node:
                if node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    left = &nodes[node.left_child]
                    right = &nodes[node.right_child]

                    if left.impurity > right.impurity:
                        larger_impurity = left.impurity
                        n_of_larger = left.weighted_n_node_samples
                        smaller_impurity = right.impurity
                        n_of_smaller = right.weighted_n_node_samples
                    else:
                        larger_impurity = right.impurity
                        n_of_larger = right.weighted_n_node_samples
                        smaller_impurity = left.impurity
                        n_of_smaller = left.weighted_n_node_samples

                    n = node.weighted_n_node_samples

                    importance_data[node.feature] += (
                        n / 2 * node.impurity -
                        min(n_of_larger, n / 2) * larger_impurity -
                        (n / 2 - min(n_of_larger, n / 2) * smaller_impurity))

                node += 1

        importances /= nodes[0].weighted_n_node_samples

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                importances /= normalizer

        print("modified importances computed")
        return importances
