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
        np.ndarray[np.int32_t, ndim=2, mode='c'] data_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] smoothness_cost):
    cdef int n_nodes = data_cost.shape[0]
    if data_cost.shape[1] != 2:
        raise ValueError("data_cost must be of shape (n_nodes, 2).")
    if edges.shape[1] != 2:
        raise ValueError("data_cost must be of shape (n_edges, 2).")
    if smoothness_cost.shape[0] != smoothness_cost.shape[1]:
        raise ValueError("smoothness_cost must be square matrix.")
    cdef int n_edges = edges.shape[0] 
    # create qpbo object
    cdef QPBO[int] * q = new QPBO[int](n_nodes, n_edges)
    q.AddNode(n_nodes)
    cdef int* data_ptr = <int*> data_cost.data
    # add unary terms
    for i in xrange(n_nodes):
        q.AddUnaryTerm(i, data_ptr[2 * i], data_ptr[2 * i + 1])
    # add pairwise terms
    # we have global terms
    cdef int e00 = smoothness_cost[0, 0]
    cdef int e10 = smoothness_cost[1, 0]
    cdef int e01 = smoothness_cost[0, 1]
    cdef int e11 = smoothness_cost[1, 1]

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


def binary_grid(np.ndarray[np.int32_t, ndim=3, mode='c'] data_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] smoothness_cost, verbose=0):
    cdef int h = data_cost.shape[0]
    cdef int w = data_cost.shape[1]
    if verbose > 0:
        print("w: %d, h: %d" % (w, h))
    if data_cost.shape[2] != 2:
        raise ValueError("data_cost must be of shape (h, w, 2).")
    cdef int n_nodes = w * h
    cdef int n_edges = (w - 1) * h + (h - 1) * w
    # create qpbo object
    cdef QPBO[int] * q = new QPBO[int](n_nodes, n_edges)
    q.AddNode(n_nodes)
    cdef int* data_ptr = <int*> data_cost.data
    # add unary terms
    for i in xrange(n_nodes):
        q.AddUnaryTerm(i, data_ptr[2 * i], data_ptr[2 * i + 1])
    # add pairwise terms
    # we have global terms
    cdef int e00 = smoothness_cost[0, 0]
    cdef int e10 = smoothness_cost[1, 0]
    cdef int e01 = smoothness_cost[0, 1]
    cdef int e11 = smoothness_cost[1, 1]

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


def binary_grid_VH(np.ndarray[np.int32_t, ndim=3, mode='c'] data_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] smoothness_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] V,
        np.ndarray[np.int32_t, ndim=2, mode='c'] H):

    cdef int h = data_cost.shape[0]
    cdef int w = data_cost.shape[1]
    print("w: %d, h: %d" % (w, h))
    if data_cost.shape[2] != 2:
        raise ValueError("data_cost must be of shape (h, w, 2).")
    if V.shape[0] != h - 1 or V.shape[1] != w:
        raise ValueError("V must be of shape (h-1, w), got (%d, %d)." % (V.shape[0], V.shape[1]))
    if H.shape[0] != h or H.shape[1] != w - 1:
        raise ValueError("H must be of shape (h, w-1), got (%d, %d)." % (H.shape[0], H.shape[1]))

    cdef int n_nodes = w * h
    cdef int n_edges = (w - 1) * h + (h - 1) * w
    # create qpbo object
    cdef QPBO[int] * q = new QPBO[int](n_nodes, n_edges)
    q.AddNode(n_nodes)
    cdef int* data_ptr = <int*> data_cost.data
    # add unary terms
    for i in xrange(n_nodes):
        q.AddUnaryTerm(i, data_ptr[2 * i], data_ptr[2 * i + 1])
    # add pairwise terms
    # we have global terms
    cdef int e00 = smoothness_cost[0, 0]
    cdef int e10 = smoothness_cost[1, 0]
    cdef int e01 = smoothness_cost[0, 1]
    cdef int e11 = smoothness_cost[1, 1]
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


def alpha_expansion_grid(np.ndarray[np.int32_t, ndim=3, mode='c'] data_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] smoothness_cost, int n_iter=3, verbose=0, random_seed=None):
    cdef int h = data_cost.shape[0]
    cdef int w = data_cost.shape[1]
    cdef int n_labels =  data_cost.shape[2]
    cdef int n_edges = (w - 1) * h + (h - 1) * w
    cdef int n_nodes = w * h
    cdef np.ndarray[np.int32_t, ndim=2] x
    cdef int old_label
    cdef int label
    cdef int changes
    cdef int node_id
    cdef int node_label
    cdef int e00, e01, e10, e11

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

    cdef int* data_ptr = <int*> data_cost.data
    cdef int* data_ptr_current

    # create qpbo object
    cdef QPBO[int] * q = new QPBO[int](n_nodes, n_edges)
    #cdef int* data_ptr = <int*> data_cost.data
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
                    e01 = smoothness_cost[node_label, alpha]
                    e11 = smoothness_cost[alpha, alpha]
                    if i < h - 1:
                        #down
                        e00 = smoothness_cost[node_label, x_ptr_current[w]]
                        e10 = smoothness_cost[alpha, x_ptr_current[w]]

                        q.AddPairwiseTerm(node_id, node_id + w, e00, e01, e10, e11)
                    if j < w - 1:
                        #right
                        e00 = smoothness_cost[node_label, x_ptr_current[1]]
                        e10 = smoothness_cost[alpha, x_ptr_current[1]]
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
        np.ndarray[np.int32_t, ndim=2, mode='c'] data_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] smoothness_cost, int n_iter=5, verbose=False, random_seed=None):

    cdef int n_nodes = data_cost.shape[0]
    cdef int n_labels =  data_cost.shape[1]
    cdef int n_edges = edges.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] x
    cdef int old_label
    cdef int label
    cdef int changes
    cdef int e00, e01, e10, e11
    cdef int edge0, edge1

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

    cdef int* data_ptr = <int*> data_cost.data
    cdef int* data_ptr_current

    # create qpbo object
    cdef QPBO[int] * q = new QPBO[int](n_nodes, n_edges)
    #cdef int* data_ptr = <int*> data_cost.data
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
                    q.AddUnaryTerm(i, data_cost[i, x_ptr[i]], 100000)
                else:
                    q.AddUnaryTerm(i, data_cost[i, x_ptr[i]], data_cost[i, alpha])
            for e in xrange(n_edges):
                edge0 = edge_ptr[2 * e]
                edge1 = edge_ptr[2 * e + 1]
                #down
                e00 = smoothness_cost[x_ptr[edge0], x_ptr[edge1]]
                e01 = smoothness_cost[x_ptr[edge0], alpha]
                e10 = smoothness_cost[alpha, x_ptr[edge1]]
                e11 = smoothness_cost[alpha, alpha]
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
