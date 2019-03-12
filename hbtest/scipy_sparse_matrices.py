import numpy as np
from scipy.sparse import csc_matrix

np.random.seed = 0
mat = csc_matrix(np.random.rand(10, 12) > 0.7, dtype=int)
mat[1, 0] = 2  # add some variety to the matrix
mat[0, 1] = 3
print(mat.A)
print('shape:', mat.shape, mat.data.shape, mat.indptr.shape, mat.indices.shape)
print(mat.indptr)
print(mat.indptr[:-1])
print(mat.indices[mat.indptr[:-1]])

# print(mat.indices)


from scipy.sparse import csr_matrix

docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world"]]
indptr = [0]
indices = []
data = []
vocabulary = {}
for d in docs:
    for term in d:
        index = vocabulary.setdefault(term, len(vocabulary))
        indices.append(index)
        data.append(1)
    indptr.append(len(indices))

curr = csr_matrix((data, indices, indptr), dtype=int).toarray()
print(curr)
print(data, indices, indptr)
print(curr.data.shape)
