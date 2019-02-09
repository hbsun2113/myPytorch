"""
.. _model-capsule:

Capsule Network Tutorial
===========================

**Author**: Jinjing Zhou, `Jake Zhao <https://cs.nyu.edu/~jakezhao/>`_, Zheng Zhang, Jinyang Li

It is perhaps a little surprising that some of the more classical models
can also be described in terms of graphs, offering a different
perspective. This tutorial describes how this can be done for the
`capsule network <http://arxiv.org/abs/1710.09829>`__.
"""
#######################################################################################
# Key ideas of Capsule
# --------------------
#
# The Capsule model offers two key ideas.
#
# **Richer representation** In classic convolutional networks, a scalar
# value represents the activation of a given feature. By contrast, a
# capsule outputs a vector. The vector's length represents the probability
# of a feature being present. The vector's orientation represents the
# various properties of the feature (such as pose, deformation, texture
# etc.).
#
# |image0|
#
# **Dynamic routing** The output of a capsule is preferentially sent to
# certain parents in the layer above based on how well the capsule's
# prediction agrees with that of a parent. Such dynamic
# "routing-by-agreement" generalizes the static routing of max-pooling.
#
# During training, routing is done iteratively; each iteration adjusts
# "routing weights" between capsules based on their observed agreements,
# in a manner similar to a k-means algorithm or `competitive
# learning <https://en.wikipedia.org/wiki/Competitive_learning>`__.
#
# In this tutorial, we show how capsule's dynamic routing algorithm can be
# naturally expressed as a graph algorithm. Our implementation is adapted
# from `Cedric
# Chee <https://github.com/cedrickchee/capsule-net-pytorch>`__, replacing
# only the routing layer. Our version achieves similar speed and accuracy.
#
# Model Implementation
# ----------------------
# Step 1: Setup and Graph Initialization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The connectivity between two layers of capsules form a directed,
# bipartite graph, as shown in the Figure below.
#
# |image1|
#
# Each node :math:`j` is associated with feature :math:`v_j`,
# representing its capsule’s output. Each edge is associated with
# features :math:`b_{ij}` and :math:`\hat{u}_{j|i}`. :math:`b_{ij}`
# determines routing weights, and :math:`\hat{u}_{j|i}` represents the
# prediction of capsule :math:`i` for :math:`j`.
#
# Here's how we set up the graph and initialize node and edge features.

import torch.nn as nn
import torch as th
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import dgl


def init_graph(in_nodes, out_nodes, f_size):
    g = dgl.DGLGraph()
    all_nodes = in_nodes + out_nodes
    g.add_nodes(all_nodes)

    in_indx = list(range(in_nodes))
    out_indx = list(range(in_nodes, in_nodes + out_nodes))
    # add edges use edge broadcasting
    for u in in_indx:
        g.add_edges(u, out_indx)

    # init states
    g.ndata['v'] = th.zeros(all_nodes, f_size)
    g.edata['b'] = th.zeros(in_nodes * out_nodes, 1)
    return g


#########################################################################################
# Step 2: Define message passing functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This is the pseudo code for Capsule's routing algorithm as given in the
# paper:
#
# |image2|
# We implement pseudo code lines 4-7 in the class `DGLRoutingLayer` as the following steps:
#
# 1. Calculate coupling coefficients:
#
#    -  Coefficients are the softmax over all out-edge of in-capsules:
#       :math:`\textbf{c}_{i,j} = \text{softmax}(\textbf{b}_{i,j})`.
#
# 2. Calculate weighted sum over all in-capsules:
#
#    -  Output of a capsule is equal to the weighted sum of its in-capsules
#       :math:`s_j=\sum_i c_{ij}\hat{u}_{j|i}`
#
# 3. Squash outputs:
#
#    -  Squash the length of a capsule's output vector to range (0,1), so it can represent the probability (of some feature being present).
#    -  :math:`v_j=\text{squash}(s_j)=\frac{||s_j||^2}{1+||s_j||^2}\frac{s_j}{||s_j||}`
#
# 4. Update weights by the amount of agreement:
#
#    -  The scalar product :math:`\hat{u}_{j|i}\cdot v_j` can be considered as how well capsule :math:`i` agrees with :math:`j`. It is used to update
#       :math:`b_{ij}=b_{ij}+\hat{u}_{j|i}\cdot v_j`

class DGLRoutingLayer(nn.Module):
    def __init__(self, in_nodes, out_nodes, f_size):
        super(DGLRoutingLayer, self).__init__()
        self.g = init_graph(in_nodes, out_nodes, f_size)
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.in_indx = list(range(in_nodes))
        self.out_indx = list(range(in_nodes, in_nodes + out_nodes))

    def forward(self, u_hat, routing_num=1):
        self.g.edata['u_hat'] = u_hat

        # step 2 (line 5)
        def cap_message(edges):
            return {'m': edges.data['c'] * edges.data['u_hat']}

        self.g.register_message_func(cap_message)

        def cap_reduce(nodes):
            return {'s': th.sum(nodes.mailbox['m'], dim=1)}

        self.g.register_reduce_func(cap_reduce)

        for r in range(routing_num):
            # step 1 (line 4): normalize over out edges
            edges_b = self.g.edata['b'].view(self.in_nodes, self.out_nodes)
            self.g.edata['c'] = F.softmax(edges_b, dim=1).view(-1, 1)

            # Execute step 1 & 2
            self.g.update_all()

            # step 3 (line 6)
            self.g.nodes[self.out_indx].data['v'] = self.squash(self.g.nodes[self.out_indx].data['s'], dim=1)

            # step 4 (line 7)
            v = th.cat([self.g.nodes[self.out_indx].data['v']] * self.in_nodes, dim=0)
            self.g.edata['b'] = self.g.edata['b'] + (self.g.edata['u_hat'] * v).sum(dim=1, keepdim=True)

    @staticmethod
    def squash(s, dim=1):
        sq = th.sum(s ** 2, dim=dim, keepdim=True)
        s_norm = th.sqrt(sq)
        s = (sq / (1.0 + sq)) * (s / s_norm)
        return s


############################################################################################################
# Step 3: Testing
# ~~~~~~~~~~~~~~~
#
# Let's make a simple 20x10 capsule layer:
in_nodes = 20
out_nodes = 10
f_size = 4
u_hat = th.randn(in_nodes * out_nodes, f_size)
routing = DGLRoutingLayer(in_nodes, out_nodes, f_size)

############################################################################################################
# We can visualize a capsule network's behavior by monitoring the entropy
# of coupling coefficients. They should start high and then drop, as the
# weights gradually concentrate on fewer edges:
entropy_list = []
dist_list = []

for i in range(10):
    routing(u_hat)
    dist_matrix = routing.g.edata['c'].view(in_nodes, out_nodes)
    entropy = (-dist_matrix * th.log(dist_matrix)).sum(dim=1)
    entropy_list.append(entropy.data.numpy())
    dist_list.append(dist_matrix.data.numpy())

stds = np.std(entropy_list, axis=1)
means = np.mean(entropy_list, axis=1)
plt.errorbar(np.arange(len(entropy_list)), means, stds, marker='o')
plt.ylabel("Entropy of Weight Distribution")
plt.xlabel("Number of Routing")
plt.xticks(np.arange(len(entropy_list)))
plt.close()
############################################################################################################
# |image3|
#
# Alternatively, we can also watch the evolution of histograms:

import seaborn as sns
import matplotlib.animation as animation

fig = plt.figure(dpi=150)
fig.clf()
ax = fig.subplots()


def dist_animate(i):
    ax.cla()
    sns.distplot(dist_list[i].reshape(-1), kde=False, ax=ax)
    ax.set_xlabel("Weight Distribution Histogram")
    ax.set_title("Routing: %d" % (i))


ani = animation.FuncAnimation(fig, dist_animate, frames=len(entropy_list), interval=500)
plt.close()

############################################################################################################
# |image4|
#
# Or monitor the how lower level capsules gradually attach to one of the
# higher level ones:
import networkx as nx
from networkx.algorithms import bipartite

g = routing.g.to_networkx()
X, Y = bipartite.sets(g)
height_in = 10
height_out = height_in * 0.8
height_in_y = np.linspace(0, height_in, in_nodes)
height_out_y = np.linspace((height_in - height_out) / 2, height_out, out_nodes)
pos = dict()

fig2 = plt.figure(figsize=(8, 3), dpi=150)
fig2.clf()
ax = fig2.subplots()
pos.update((n, (i, 1)) for i, n in zip(height_in_y, X))  # put nodes from X at x=1
pos.update((n, (i, 2)) for i, n in zip(height_out_y, Y))  # put nodes from Y at x=2


def weight_animate(i):
    ax.cla()
    ax.axis('off')
    ax.set_title("Routing: %d  " % i)
    dm = dist_list[i]
    nx.draw_networkx_nodes(g, pos, nodelist=range(in_nodes), node_color='r', node_size=100, ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=range(in_nodes, in_nodes + out_nodes), node_color='b', node_size=100, ax=ax)
    for edge in g.edges():
        nx.draw_networkx_edges(g, pos, edgelist=[edge], width=dm[edge[0], edge[1] - in_nodes] * 1.5, ax=ax)


ani2 = animation.FuncAnimation(fig2, weight_animate, frames=len(dist_list), interval=500)
plt.close()

############################################################################################################
# |image5|
#
# The full code of this visualization is provided at
# `link <https://github.com/dmlc/dgl/blob/master/examples/pytorch/capsule/simple_routing.py>`__; the complete
# code that trains on MNIST is at `link <https://github.com/dmlc/dgl/tree/tutorial/examples/pytorch/capsule>`__.
#
# .. |image0| image:: https://i.imgur.com/55Ovkdh.png
# .. |image1| image:: https://i.imgur.com/9tc6GLl.png
# .. |image2| image:: https://i.imgur.com/mv1W9Rv.png
# .. |image3| image:: https://i.imgur.com/dMvu7p3.png
# .. |image4| image:: https://github.com/VoVAllen/DGL_Capsule/raw/master/routing_dist.gif
# .. |image5| image:: https://github.com/VoVAllen/DGL_Capsule/raw/master/routing_vis.gif

