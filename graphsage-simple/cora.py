from __future__ import absolute_import

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os, sys


def _encode_onehot(labels):
    classes = set(labels)
    print('1classes', classes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    print('1classes_dict', classes_dict)
    classes = list(sorted(set(labels)))
    print('2classes', classes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    print('2classes_dict', classes_dict)

    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def _normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = np.inf
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


class CoraDataset(object):
    def __init__(self):
        self.name = 'cora'
        self.dir = '.'
        self.zip_file_path = '{}/{}.zip'.format(self.dir, self.name)
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

        remove = {}
        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/cora/cora.cites".format(self.dir),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        for i in range(edges.shape[0]):
            if edges[i, 0] == edges[i, 1]:
                print(i, edges[i, 0])
            tt = (edges[i, 0], edges[i, 1])
            remove[tt] = 1
            # tr = (edges[i, 1], edges[i, 0])
            # if remove.__contains__(tr):
                # print(tt, tr)


        print('length', len(remove))

        adj = sp.coo_matrix((np.ones(edges.shape[0]),
                             (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # print(1, adj.sum())  # 5429
        # print(2, adj.T.sum())  # 5429
        #
        # print(3, (adj + adj.T).sum())
        # print(4, adj.T > adj)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # 只能是>， 因为存在相互引用的情况。否则应该都是>=
        # print(5, adj.sum())  # 10556 ,所以是存在5429*2-10556=302个相互引用的情况
        self.graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())

        features = _normalize(features)
        self.features = np.array(features.todense())
        self.labels = np.where(labels)[1]

        self.train_mask = _sample_mask(range(140), labels.shape[0])
        self.val_mask = _sample_mask(range(200, 500), labels.shape[0])
        self.test_mask = _sample_mask(range(500, 1500), labels.shape[0])


cora = CoraDataset()
print('Finish')
