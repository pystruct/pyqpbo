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
            if i < h - 1:
                #down
                q.AddPairwiseTerm(node_id, node_id + w, e00, e10, e01, e11)
            if j < w - 1:
                #right
                q.AddPairwiseTerm(node_id, node_id + 1, e00, e10, e01, e11)

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
                q.AddPairwiseTerm(node_id, node_id + w, vert_cost * e00, vert_cost * e10, vert_cost * e01, vert_cost * e11)
            if j < w - 1:
                #right
                horz_cost = H[i, j]
                print("horz cost: %d" % horz_cost)
                q.AddPairwiseTerm(node_id, node_id + 1, horz_cost * e00, horz_cost * e10, horz_cost * e01, horz_cost * e11)

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
        np.ndarray[np.int32_t, ndim=2, mode='c'] smoothness_cost):
    cdef int h = data_cost.shape[0]
    cdef int w = data_cost.shape[1]
    cdef int n_labels =  data_cost.shape[2]
    cdef int n_edges = (w - 1) * h + (h - 1) * w
    cdef int n_nodes = w * h
    cdef np.ndarray[np.int32_t, ndim=2] x
    cdef int old_label

    cdef n_iter = 1

    # initial guess
    #x = np.argmin(data_cost, axis=2)
    x = np.zeros((h, w), dtype=np.int32)
    print("x shape: (%d, %d)" %(x.shape[0], x.shape[1]))
    print("x size: %d" %x.size)
    cdef int* x_ptr = <int*> x.data

    # create qpbo object
    cdef QPBO[int] * q = new QPBO[int](n_nodes, n_edges)
    #cdef int* data_ptr = <int*> data_cost.data

    for n in xrange(n_iter):
        print("iteration: %d" % n)
        for alpha in [0, 1]:
            q.AddNode(n_nodes)
            print("alpha: %d" % alpha)
            for i in xrange(h):
                for j in xrange(w):
                    node_id = i * w + j
                    # first state is "keep x", second is "switch to alpha"
                    # TODO: what if state is already alpha?
                    q.AddUnaryTerm(node_id, data_cost[i, j, x[i, j]], data_cost[i, j, alpha])
                    print("added node %d, x[i,j]=%d, cost: %d %d" % (node_id, x[i,j], data_cost[i,j,x[i,j]], data_cost[i,j,alpha]))
                    #if i < h - 1:
                        ##down
                        #print(x[i, j])
                        #print(alpha)
                        #pair = smoothness_cost[[x[i, j], alpha], :][:, [x[i + 1, j], alpha]]
                        #print(pair)
                        #q.AddPairwiseTerm(node_id, node_id + w, pair[0, 0], pair[1, 0], pair[0, 1], pair[1, 1])
                    #if j < w - 1:
                        ##right
                        #pair = smoothness_cost[[x[i, j], alpha], :][:, [x[i, j + 1], alpha]]
                        #print(pair)
                        #q.AddPairwiseTerm(node_id, node_id + 1, pair[0, 0], pair[1, 0], pair[0, 1], pair[1, 1])
            q.Solve()
            for i in xrange(n_nodes):
                old_label = x_ptr[i]
                if q.GetLabel(i) == 1:
                    x_ptr[i] = alpha
                print("node: %d, old label: %d, new label: %d, cut: %d" % (i, old_label, x_ptr[i], q.GetLabel(i)))
            q.Reset()
            print("fin")
    return x
