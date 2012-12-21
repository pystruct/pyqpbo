###############################################################
# Python bindings for QPBO algorithm by Vladimir Kolmogorov.
#
# Author: Andreas Mueller <amueller@ais.uni-bonn.de>
# License: BSD 3-clause
#
# More infos on my blog: peekaboo-vision.blogspot.com

import numpy as np
cimport numpy as np
from libcpp cimport bool
from time import time

np.import_array()

cdef extern from "stdlib.h":
    void srand(unsigned int seed) 

ctypedef int NodeId
ctypedef int EdgeId

cdef extern from "QPBO.h":
    cdef cppclass QPBO[REAL]:
        QPBO(int node_num_max, int edge_num_max)
        bool Save(char* filename, int format=0)
        bool Load(char* filename)
        void Reset()
        NodeId AddNode(int num)
        void AddUnaryTerm(NodeId i, REAL E0, REAL E1)
        EdgeId AddPairwiseTerm(NodeId i, NodeId j, REAL E00, REAL E01, REAL E10, REAL E11)
        void AddPairwiseTerm(EdgeId e, NodeId i, NodeId j, REAL E00, REAL E01, REAL E10, REAL E11)
        int GetLabel(NodeId i)
        void Solve()
        void ComputeWeakPersistencies()
        bool Improve()


def binary_graph(np.ndarray[np.int32_t, ndim=2, mode='c'] edges,
                 np.ndarray[np.int32_t, ndim=2, mode='c'] unary_cost,
                 np.ndarray[np.int32_t, ndim=2, mode='c'] pairwise_cost):
    """QPBO inference on a graph with binary variables.
    
    Pairwise potentials are the same for all edges.

    Parameters
    ----------
    edges : nd-array, shape=(n_edges, 2)
        Edge-list describing the graph. Edges are given
        using node-indices from 0 to n_nodes-1.

    unary_cost : nd-array, shape=(n_nodes, 2)
        Unary potential costs. Rows correspond to rows, columns
        to states.
    
    pairwise_cost : nd-array, shape=(2, 2)
        Symmetric pairwise potential.

    Returns
    -------
    result : nd-array, shape=(n_nodes,)
        Approximate MAP as inferred by QPBO.
        Values are 0, 1 and -1 for non-assigned nodes.

    """
    cdef int n_nodes = unary_cost.shape[0]
    if unary_cost.shape[1] != 2:
        raise ValueError("unary_cost must be of shape (n_nodes, 2).")
    if edges.shape[1] != 2:
        raise ValueError("edges must be of shape (n_edges, 2).")
    if pairwise_cost.shape[0] != pairwise_cost.shape[1]:
        raise ValueError("pairwise_cost must be square matrix.")
    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")
    cdef int n_edges = edges.shape[0] 
    # create qpbo object
    cdef QPBO[int] * q = new QPBO[int](n_nodes, n_edges)
    q.AddNode(n_nodes)
    cdef int* data_ptr = <int*> unary_cost.data
    # add unary terms
    for i in xrange(n_nodes):
        q.AddUnaryTerm(i, data_ptr[2 * i], data_ptr[2 * i + 1])
    # add pairwise terms
    # we have global terms
    cdef int e00 = pairwise_cost[0, 0]
    cdef int e10 = pairwise_cost[1, 0]
    cdef int e01 = pairwise_cost[0, 1]
    cdef int e11 = pairwise_cost[1, 1]

    for e in edges:
        q.AddPairwiseTerm(e[0], e[1], e00, e10, e01, e11)

    q.Solve()
    q.ComputeWeakPersistencies()

    # get result
    cdef np.npy_intp result_shape[1]
    result_shape[0] = n_nodes
    cdef np.ndarray[np.int32_t, ndim=1] result = np.PyArray_SimpleNew(1, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(n_nodes):
        result_ptr[i] = q.GetLabel(i)

    del q
    return result


def binary_grid(np.ndarray[np.int32_t, ndim=3, mode='c'] unary_cost,
                np.ndarray[np.int32_t, ndim=2, mode='c'] pairwise_cost,
                verbose=0):
    """QPBO inference on a 2d grid with binary variables.
    
    Pairwise potentials are the same for all edges.

    Parameters
    ----------
    unary_cost : nd-array, shape=(height, width, 2)
        Unary potential costs. First two dimensions correspond to
        position in the grid, the last dimension corresponds to states.

    pairwise_cost : nd-array, shape=(2, 2)
        Symmetric pairwise potential.

    Returns
    -------
    result : nd-array, shape=(height, width)
        Approximate MAP as inferred by QPBO.
        Values are 0, 1 and -1 for non-assigned nodes.

    """
    cdef int h = unary_cost.shape[0]
    cdef int w = unary_cost.shape[1]
    if verbose > 0:
        print("w: %d, h: %d" % (w, h))
    if unary_cost.shape[2] != 2:
        raise ValueError("unary_cost must be of shape (h, w, 2).")
    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")
    cdef int n_nodes = w * h
    cdef int n_edges = (w - 1) * h + (h - 1) * w
    # create qpbo object
    cdef QPBO[int] * q = new QPBO[int](n_nodes, n_edges)
    q.AddNode(n_nodes)
    cdef int* data_ptr = <int*> unary_cost.data
    # add unary terms
    for i in xrange(n_nodes):
        q.AddUnaryTerm(i, data_ptr[2 * i], data_ptr[2 * i + 1])
    # add pairwise terms
    # we have global terms
    cdef int e00 = pairwise_cost[0, 0]
    cdef int e10 = pairwise_cost[1, 0]
    cdef int e01 = pairwise_cost[0, 1]
    cdef int e11 = pairwise_cost[1, 1]

    for i in xrange(h):
        for j in xrange(w):
            node_id = i * w + j
            if i < h - 1:
                #down
                q.AddPairwiseTerm(node_id, node_id + w, e00, e10, e01, e11)
            if j < w - 1:
                #right
                q.AddPairwiseTerm(node_id, node_id + 1, e00, e10, e01, e11)

    q.Solve()
    q.ComputeWeakPersistencies()

    # get result
    cdef np.npy_intp result_shape[2]
    result_shape[0] = h
    result_shape[1] = w
    cdef np.ndarray[np.int32_t, ndim=2] result = np.PyArray_SimpleNew(2, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(n_nodes):
        result_ptr[i] = q.GetLabel(i)

    del q
    return result


def binary_grid_VH(np.ndarray[np.int32_t, ndim=3, mode='c'] unary_cost,
                   np.ndarray[np.int32_t, ndim=2, mode='c'] pairwise_cost,
                   np.ndarray[np.int32_t, ndim=2, mode='c'] V,
                   np.ndarray[np.int32_t, ndim=2, mode='c'] H):
    """QPBO inference on a 2d grid with binary variables.
    
    Pairwise potentials can be multiplicatively modified for each edge.
    This allows for effects like respecting edges in images.

    Parameters
    ----------
    unary_cost : nd-array, shape=(height, width, 2)
        Unary potential costs. First two dimensions correspond to
        position in the grid, the last dimension corresponds to states.

    pairwise_cost : nd-array, shape=(2, 2)
        Symmetric pairwise potential.

    V : nd-array, shape=(height - 1, width)
        Multiplicative modification of pairwise costs for vertical edges.

    H : nd-array, shape=(height, width - 1)
        Multiplicative modification of pairwise costs for horizontal edges.

    Returns
    -------
    result : nd-array, shape=(height, width)
        Approximate MAP as inferred by QPBO.
        Values are 0, 1 and -1 for non-assigned nodes.

    """

    cdef int h = unary_cost.shape[0]
    cdef int w = unary_cost.shape[1]
    print("w: %d, h: %d" % (w, h))
    if unary_cost.shape[2] != 2:
        raise ValueError("unary_cost must be of shape (h, w, 2).")
    if V.shape[0] != h - 1 or V.shape[1] != w:
        raise ValueError("V must be of shape (h-1, w), got (%d, %d)." % (V.shape[0], V.shape[1]))
    if H.shape[0] != h or H.shape[1] != w - 1:
        raise ValueError("H must be of shape (h, w-1), got (%d, %d)." % (H.shape[0], H.shape[1]))

    cdef int n_nodes = w * h
    cdef int n_edges = (w - 1) * h + (h - 1) * w
    # create qpbo object
    cdef QPBO[int] * q = new QPBO[int](n_nodes, n_edges)
    q.AddNode(n_nodes)
    cdef int* data_ptr = <int*> unary_cost.data
    # add unary terms
    for i in xrange(n_nodes):
        q.AddUnaryTerm(i, data_ptr[2 * i], data_ptr[2 * i + 1])
    # add pairwise terms
    # we have global terms
    cdef int e00 = pairwise_cost[0, 0]
    cdef int e10 = pairwise_cost[1, 0]
    cdef int e01 = pairwise_cost[0, 1]
    cdef int e11 = pairwise_cost[1, 1]
    cdef int vert_cost
    cdef int horz_cost

    for i in xrange(h):
        for j in xrange(w):
            node_id = i * w + j
            if i < h - 1:
                #down
                vert_cost = V[i, j]
                print("vert cost: %d" % vert_cost)
                q.AddPairwiseTerm(node_id, node_id + w, vert_cost * e00, vert_cost * e01, vert_cost * e10, vert_cost * e11)
            if j < w - 1:
                #right
                horz_cost = H[i, j]
                print("horz cost: %d" % horz_cost)
                q.AddPairwiseTerm(node_id, node_id + 1, horz_cost * e00, horz_cost * e01, horz_cost * e10, horz_cost * e11)

    q.Solve()

    # get result
    cdef np.npy_intp result_shape[2]
    result_shape[0] = h
    result_shape[1] = w
    cdef np.ndarray[np.int32_t, ndim=2] result = np.PyArray_SimpleNew(2, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(n_nodes):
        result_ptr[i] = q.GetLabel(i)
    return result


def alpha_expansion_grid(np.ndarray[np.int32_t, ndim=3, mode='c'] unary_cost,
                         np.ndarray[np.int32_t, ndim=2, mode='c']
                         pairwise_cost, int n_iter=3, verbose=0,
                         random_seed=None):
    """Alpha expansion using QPBO inference on a 2d grid.
    
    Pairwise potentials are the same for all edges.
    Alpha expansion is very efficient but inference is only approximate and
    none of the persistence properties of QPBO are preserved.

    Parameters
    ----------
    unary_cost : nd-array, shape=(height, width, n_states)
        Unary potential costs. First two dimensions correspond to
        position in the grid, the last dimension corresponds to states.

    pairwise_cost : nd-array, shape=(n_states, n_states)
        Symmetric pairwise potential.

    n_iter : int, default=3
        Number of expansion iterations (how often to go over labels).

    verbose : int, default=0
        Verbosity.

    random_seed: int or None
        If int, a fixed random seed is used for reproducable results.

    Returns
    -------
    result : nd-array, shape=(height, width)
        Approximate MAP as inferred by QPBO.
        Values are 0, 1 and -1 for non-assigned nodes.

    """
    cdef int h = unary_cost.shape[0]
    cdef int w = unary_cost.shape[1]
    cdef int n_labels =  unary_cost.shape[2]
    cdef int n_edges = (w - 1) * h + (h - 1) * w
    cdef int n_nodes = w * h
    cdef np.ndarray[np.int32_t, ndim=2] x
    cdef int old_label
    cdef int label
    cdef int changes
    cdef int node_id
    cdef int node_label
    cdef int e00, e01, e10, e11

    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")
    if random_seed is None:
        rnd_state = np.random.mtrand.RandomState()
        srand(time())
    else:
        rnd_state = np.random.mtrand.RandomState(random_seed)
        srand(random_seed)

    # initial guess
    x = np.zeros((h, w), dtype=np.int32)
    if verbose > 0:
        print("x shape: (%d, %d)" %(x.shape[0], x.shape[1]))
        print("x size: %d" %x.size)
    cdef int* x_ptr = <int*> x.data
    cdef int* x_ptr_current

    cdef int* data_ptr = <int*> unary_cost.data
    cdef int* data_ptr_current

    # create qpbo object
    cdef QPBO[int] * q = new QPBO[int](n_nodes, n_edges)
    #cdef int* data_ptr = <int*> unary_cost.data
    for n in xrange(n_iter):
        if verbose > 0:
            print("iteration: %d" % n)
        changes = 0
        for alpha in rnd_state.permutation(n_labels):
            q.AddNode(n_nodes)
            for i in xrange(h):
                for j in xrange(w):
                    node_id = i * w + j
                    # first state is "keep x", second is "switch to alpha"
                    # TODO: what if state is already alpha? Need to collapse?
                    x_ptr_current = x_ptr + node_id
                    node_label = x_ptr_current[0]
                    data_ptr_current = data_ptr + n_labels * node_id
                    if alpha == node_label:
                        q.AddUnaryTerm(node_id, data_ptr_current[node_label], 100000)
                    else:
                        q.AddUnaryTerm(node_id, data_ptr_current[node_label], data_ptr_current[alpha])
                    e01 = pairwise_cost[node_label, alpha]
                    e11 = pairwise_cost[alpha, alpha]
                    if i < h - 1:
                        #down
                        e00 = pairwise_cost[node_label, x_ptr_current[w]]
                        e10 = pairwise_cost[alpha, x_ptr_current[w]]

                        q.AddPairwiseTerm(node_id, node_id + w, e00, e01, e10, e11)
                    if j < w - 1:
                        #right
                        e00 = pairwise_cost[node_label, x_ptr_current[1]]
                        e10 = pairwise_cost[alpha, x_ptr_current[1]]
                        q.AddPairwiseTerm(node_id, node_id + 1, e00, e01, e10, e11)
            q.Solve()
            q.ComputeWeakPersistencies()
            improve = True
            while improve:
                improve = q.Improve()

            for i in xrange(n_nodes):
                old_label = x_ptr[i]
                label = q.GetLabel(i)
                if label == 1:
                    x_ptr[i] = alpha
                    changes += 1
                if label < 0:
                    print("LABEL <0 !!!")
            q.Reset()
        if verbose > 0:
            print("alpha: %d, changes: %d" % (alpha, changes))
        if changes == 0:
            break
    del q
    return x


def alpha_expansion_graph(np.ndarray[np.int32_t, ndim=2, mode='c'] edges,
                          np.ndarray[np.int32_t, ndim=2, mode='c'] unary_cost,
                          np.ndarray[np.int32_t, ndim=2, mode='c']
                          pairwise_cost, int n_iter=5, verbose=False,
                          random_seed=None):
    """Alpha expansion using QPBO inference on general graph.
    
    Pairwise potentials are the same for all edges.

    Alpha expansion is very efficient but inference is only approximate and
    none of the persistence properties of QPBO are preserved.

    Parameters
    ----------
    edges : nd-array, shape=(n_edges, 2)
        Edge-list describing the graph. Edges are given
        using node-indices from 0 to n_nodes-1.

    unary_cost : nd-array, shape=(n_nodes, n_states)
        Unary potential costs.

    pairwise_cost : nd-array, shape=(n_states, n_states)
        Symmetric pairwise potential.

    n_iter : int, default=5
        Number of expansion iterations (how often to go over labels).

    verbose : int, default=0
        Verbosity.

    random_seed: int or None
        If int, a fixed random seed is used for reproducable results.

    Returns
    -------
    result : nd-array, shape=(n_nodes,)
        Approximate MAP as inferred by QPBO.
        Values are 0, 1 and -1 for non-assigned nodes.

    """

    cdef int n_nodes = unary_cost.shape[0]
    cdef int n_labels =  unary_cost.shape[1]
    cdef int n_edges = edges.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] x
    cdef int old_label
    cdef int label
    cdef int changes
    cdef int e00, e01, e10, e11
    cdef int edge0, edge1

    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")
    if random_seed is None:
        rnd_state = np.random.mtrand.RandomState()
        srand(time())
    else:
        rnd_state = np.random.mtrand.RandomState(random_seed)
        srand(random_seed)

    # initial guess
    x = np.zeros(n_nodes, dtype=np.int32)
    cdef int* edge_ptr = <int*> edges.data
    cdef int* x_ptr = <int*> x.data
    cdef int* x_ptr_current

    cdef int* data_ptr = <int*> unary_cost.data
    cdef int* data_ptr_current

    # create qpbo object
    cdef QPBO[int] * q = new QPBO[int](n_nodes, n_edges)
    #cdef int* data_ptr = <int*> unary_cost.data
    for n in xrange(n_iter):
        if verbose > 0:
            print("iteration: %d" % n)
        changes = 0
        for alpha in rnd_state.permutation(n_labels):
            q.AddNode(n_nodes)
            for i in xrange(n_nodes):
                # first state is "keep x", second is "switch to alpha"
                # TODO: what if state is already alpha? Need to collapse?
                if alpha == x[i]:
                    q.AddUnaryTerm(i, unary_cost[i, x_ptr[i]], 100000)
                else:
                    q.AddUnaryTerm(i, unary_cost[i, x_ptr[i]], unary_cost[i, alpha])
            for e in xrange(n_edges):
                edge0 = edge_ptr[2 * e]
                edge1 = edge_ptr[2 * e + 1]
                #down
                e00 = pairwise_cost[x_ptr[edge0], x_ptr[edge1]]
                e01 = pairwise_cost[x_ptr[edge0], alpha]
                e10 = pairwise_cost[alpha, x_ptr[edge1]]
                e11 = pairwise_cost[alpha, alpha]
                q.AddPairwiseTerm(edge0, edge1, e00, e01, e10, e11)

            q.Solve()
            q.ComputeWeakPersistencies()
            improve = True
            while improve:
                improve = q.Improve()

            for i in xrange(n_nodes):
                old_label = x_ptr[i]
                label = q.GetLabel(i)
                if label == 1:
                    x_ptr[i] = alpha
                    changes += 1
                if label < 0:
                    print("LABEL <0 !!!")
            # compute energy:
            q.Reset()
        if verbose > 0:
            print("changes: %d" % changes)
        if changes == 0:
            break
    del q
    return x
