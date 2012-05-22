import numpy as np
cimport numpy as np
from libcpp cimport bool

np.import_array()

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

def binary_grid(np.ndarray[np.int32_t, ndim=3, mode='c'] data_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] smoothness_cost):
    cdef int h = data_cost.shape[0]
    cdef int w = data_cost.shape[1]
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
            if i > 0:
                #up
                q.AddPairwiseTerm(node_id, node_id - w, e00, e10, e01, e11)
            if j > 0:
                #left
                q.AddPairwiseTerm(node_id, node_id - 1, e00, e10, e01, e11)

    # solve
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

