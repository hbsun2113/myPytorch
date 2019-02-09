"""
.. _model-rgcn:

Relational Graph Convolutional Network Tutorial
================================================

**Author:** Lingfan Yu, Mufei Li, Zheng Zhang

The vanilla Graph Convolutional Network (GCN)
(`paper <https://arxiv.org/pdf/1609.02907.pdf>`_,
`DGL tutorial <http://doc.dgl.ai/tutorials/index.html>`_) exploits
structural information of the dataset (i.e. the graph connectivity) to
improve the extraction of node representations. Graph edges are left as
untyped.

A knowledge graph is made up by a collection of triples of the form
(subject, relation, object). Edges thus encode important information and
have their own embeddings to be learned. Furthermore, there may exist
multiple edges among any given pair.

A recent model Relational-GCN (R-GCN) from the paper
`Modeling Relational Data with Graph Convolutional
Networks <https://arxiv.org/pdf/1703.06103.pdf>`_ is one effort to
generalize GCN to handle different relations between entities in knowledge
base. This tutorial shows how to implement R-GCN with DGL.
"""
###############################################################################
# R-GCN: a brief introduction
# ---------------------------
# In *statistical relational learning* (SRL), there are two fundamental
# tasks:
#
# - **Entity classification**, i.e., assign types and categorical
#   properties to entities.
# - **Link prediction**, i.e., recover missing triples.
#
# In both cases, missing information are expected to be recovered from
# neighborhood structure of the graph. Here is the example from the R-GCN
# paper:
#
# "Knowing that Mikhail Baryshnikov was educated at the Vaganova Academy
# implies both that Mikhail Baryshnikov should have the label person, and
# that the triple (Mikhail Baryshnikov, lived in, Russia) must belong to the
# knowledge graph."
#
# R-GCN solves these two problems using a common graph convolutional network
# extended with multi-edge encoding to compute embedding of the entities, but
# with different downstream processing:
#
# - Entity classification is done by attaching a softmax classifier at the
#   final embedding of an entity (node). Training is through loss of standard
#   cross-entropy.
# - Link prediction is done by reconstructing an edge with an autoencoder
#   architecture, using a parameterized score function. Training uses negative
#   sampling.
#
# This tutorial will focus on the first task to show how to generate entity
# representation. `Complete
# code <https://github.com/dmlc/dgl/tree/rgcn/examples/pytorch/rgcn>`_
# for both tasks can be found in DGL's github repository.
#
# Key ideas of R-GCN
# -------------------
# Recall that in GCN, the hidden representation for each node :math:`i` at
# :math:`(l+1)^{th}` layer is computed by:
#
# .. math:: h_i^{l+1} = \sigma\left(\sum_{j\in N_i}\frac{1}{c_i} W^{(l)} h_j^{(l)}\right)~~~~~~~~~~(1)\\
#
# where :math:`c_i` is a normalization constant.
#
# The key difference between R-GCN and GCN is that in R-GCN, edges can
# represent different relations. In GCN, weight :math:`W^{(l)}` in equation
# :math:`(1)` is shared by all edges in layer :math:`l`. In contrast, in
# R-GCN, different edge types use different weights and only edges of the
# same relation type :math:`r` are associated with the same projection weight
# :math:`W_r^{(l)}`.
#
# So the hidden representation of entities in :math:`(l+1)^{th}` layer in
# R-GCN can be formulated as the following equation:
#
# .. math:: h_i^{l+1} = \sigma\left(W_0^{(l)}h_i^{(l)}+\sum_{r\in R}\sum_{j\in N_i^r}\frac{1}{c_{i,r}}W_r^{(l)}h_j^{(l)}\right)~~~~~~~~~~(2)\\
#
# where :math:`N_i^r` denotes the set of neighbor indices of node :math:`i`
# under relation :math:`r\in R` and :math:`c_{i,r}` is a normalization
# constant. In entity classification, the R-GCN paper uses
# :math:`c_{i,r}=|N_i^r|`.
#
# The problem of applying the above equation directly is rapid growth of
# number of parameters, especially with highly multi-relational data. In
# order to reduce model parameter size and prevent overfitting, the original
# paper proposes to use basis decomposition:
#
# .. math:: W_r^{(l)}=\sum\limits_{b=1}^B a_{rb}^{(l)}V_b^{(l)}~~~~~~~~~~(3)\\
#
# Therefore, the weight :math:`W_r^{(l)}` is a linear combination of basis
# transformation :math:`V_b^{(l)}` with coefficients :math:`a_{rb}^{(l)}`.
# The number of bases :math:`B` is much smaller than the number of relations
# in the knowledge base.
#
# .. note::
#    Another weight regularization, block-decomposition, is implemented in
#    the `link prediction <link-prediction_>`_.
#
# Implement R-GCN in DGL
# ----------------------
#
# An R-GCN model is composed of several R-GCN layers. The first R-GCN layer
# also serves as input layer and takes in features (e.g. description texts)
# associated with node entity and project to hidden space. In this tutorial,
# we only use entity id as entity feature.
#
# R-GCN Layers
# ~~~~~~~~~~~~
#
# For each node, an R-GCN layer performs the following steps:
#
# - Compute outgoing message using node representation and weight matrix
#   associated with the edge type (message function)
# - Aggregate incoming messages and generate new node representations (reduce
#   and apply function)
#
# The following is the definition of an R-GCN hidden layer.
#
# .. note::
#    Each relation type is associated with a different weight. Therefore,
#    the full weight matrix has three dimensions: relation, input_feature,
#    output_feature.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['rel_type'] * self.in_feat + edges.src['id']
                return {'msg': embed[index] * edges.data['norm']}
        else:
            def message_func(edges):
                w = weight[edges.data['rel_type']]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


###############################################################################
# Define full R-GCN model
# ~~~~~~~~~~~~~~~~~~~~~~~

class Model(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels,
                 num_bases=-1, num_hidden_layers=1):
        super(Model, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        features = torch.arange(self.num_nodes)
        return features

    def build_input_layer(self):
        return RGCNLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True)

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                         activation=partial(F.softmax, dim=1))

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        for layer in self.layers:
            layer(g)
        return g.ndata.pop('h')

###############################################################################
# Handle dataset
# ~~~~~~~~~~~~~~~~
# In this tutorial, we use AIFB dataset from R-GCN paper:

# load graph data
from dgl.contrib.data import load_data
import numpy as np
data = load_data(dataset='aifb')
num_nodes = data.num_nodes
num_rels = data.num_rels
num_classes = data.num_classes
labels = data.labels
train_idx = data.train_idx
# split training and validation set
val_idx = train_idx[:len(train_idx) // 5]
train_idx = train_idx[len(train_idx) // 5:]

# edge type and normalization factor
edge_type = torch.from_numpy(data.edge_type)
edge_norm = torch.from_numpy(data.edge_norm).unsqueeze(1)

labels = torch.from_numpy(labels).view(-1)

###############################################################################
# Create graph and model
# ~~~~~~~~~~~~~~~~~~~~~~~

# configurations
n_hidden = 16 # number of hidden units
n_bases = -1 # use number of relations as number of bases
n_hidden_layers = 0 # use 1 input layer, 1 output layer, no hidden layer
n_epochs = 25 # epochs to train
lr = 0.01 # learning rate
l2norm = 0 # L2 norm coefficient

# create graph
g = DGLGraph()
g.add_nodes(num_nodes)
g.add_edges(data.edge_src, data.edge_dst)
g.edata.update({'rel_type': edge_type, 'norm': edge_norm})

# create model
model = Model(len(g),
              n_hidden,
              num_classes,
              num_rels,
              num_bases=n_bases,
              num_hidden_layers=n_hidden_layers)

###############################################################################
# Training loop
# ~~~~~~~~~~~~~~~~

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

print("start training...")
model.train()
for epoch in range(n_epochs):
    optimizer.zero_grad()
    logits = model.forward(g)
    loss = F.cross_entropy(logits[train_idx], labels[train_idx])
    loss.backward()

    optimizer.step()

    train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx])
    train_acc = train_acc.item() / len(train_idx)
    val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
    val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx])
    val_acc = val_acc.item() / len(val_idx)
    print("Epoch {:05d} | ".format(epoch) +
          "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
              train_acc, loss.item()) +
          "Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(
              val_acc, val_loss.item()))

###############################################################################
# .. _link-prediction:
#
# The second task: Link prediction
# --------------------------------
# So far, we have seen how to use DGL to implement entity classification with
# R-GCN model. In the knowledge base setting, representation generated by
# R-GCN can be further used to uncover potential relations between nodes. In
# R-GCN paper, authors feed the entity representations generated by R-GCN
# into the `DistMult <https://arxiv.org/pdf/1412.6575.pdf>`_ prediction model
# to predict possible relations.
#
# The implementation is similar to the above but with an extra DistMult layer
# stacked on top of the R-GCN layers. You may find the complete
# implementation of link prediction with R-GCN in our `example
# code <https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn/link_predict.py>`_.
