import numpy as np
from scipy.sparse import csr_matrix


def save_sparse_csr(filename, array):
    array_sparse = csr_matrix(array)
    np.savez(filename, data=array_sparse.data, indices=array_sparse.indices, indptr=array_sparse.indptr, shape=array_sparse.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape']).toarray()

def load_sparse_csr_npz(filename):
    with np.load(filename) as Data:
        data = Data['data']
        indices = Data['indices']
        indptr = Data['indptr']
        shape = Data['shape']
        return csr_matrix((data, indices, indptr), shape=shape).toarray()