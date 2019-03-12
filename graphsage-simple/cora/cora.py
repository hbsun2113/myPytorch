from __future__ import absolute_import

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os, sys
class CoraDataset(object):
    def __init__(self):
        self.name = 'cora'
        self.dir = '.'
        self.zip_file_path='{}/{}.zip'.format(self.dir, self.name)
        # download(_get_dgl_url(_urls[self.name]), path=self.zip_file_path)
        # extract_archive(self.zip_file_path, '{}/{}'.format(self.dir, self.name))
        self._load()

    def _load(self):
        idx_features_labels = np.genfromtxt("{}/cora/cora.content".
                                            format(self.dir),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1],
                                 dtype=np.float32)
        labels = _encode_onehot(idx_features_labels[:, -1])
        self.num_labels = labels.shape[1]

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/cora/cora.cites".format(self.dir),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]),
                             (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        self.graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())

        features = _normalize(features)
        self.features = np.array(features.todense())
        self.labels = np.where(labels)[1]

        self.train_mask = _sample_mask(range(140), labels.shape[0])
        self.val_mask = _sample_mask(range(200, 500), labels.shape[0])
        self.test_mask = _sample_mask(range(500, 1500), labels.shape[0])
		
		

cora = CoraDataset()
print('Finish')