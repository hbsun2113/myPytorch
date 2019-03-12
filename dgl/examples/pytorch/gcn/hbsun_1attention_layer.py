"""
Semi-Supervised Classification with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1609.02907
Code: https://github.com/tkipf/gcn
GCN with SPMV specialization.
"""
import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import copy
import random
import matplotlib.pyplot as plt


def draw(acc, loss):
    plt.figure()
    x = range(0, len(acc))
    plt.plot(x, acc, label='acc')
    plt.plot(x, loss, label='loss')
    plt.legend()
    plt.savefig("hbsun_1attention_layer.png")


def gcn_msg(edge):
    msg = edge.src['h'] * edge.src['norm']
    # print('debug', edge, edge.src['h'].shape)  # 13264=10556+2708
    # print('debug', edge.edges()[0])
    return {'m': msg}


def gcn_reduce(node):
    accum = torch.sum(node.mailbox['m'], 1) * node.data['norm']
    return {'h': accum}


class MLP(nn.Module):
    def __init__(self, in_feats, out_feats, dropout, activation, bias=True):  # (F, F)
        super(MLP, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)  # (F, F)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, nei):
        nei = self.dropout(nei)  # (B, N, F)
        nei = self.linear(nei)
        if self.activation:
            nei = self.activation(nei)
        return nei


class NodeApplyModule(nn.Module):
    def __init__(self, out_feats, activation=None, bias=True):
        super(NodeApplyModule, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, nodes):
        h = nodes.data['h']
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return {'h': h}


class GCNLayer(nn.Module):
    def __init__(self,
                 g1,
                 g2,
                 in_feats,
                 hid_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.g1 = g1
        self.g2 = g2
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_feats, hid_feats, bias=bias)
        self.linear1 = nn.Linear(in_feats, hid_feats, bias=bias)
        self.linear2 = nn.Linear(in_feats, hid_feats, bias=bias)
        self.fc = nn.Linear(hid_feats, out_feats, bias=bias)
        self.al = nn.Linear(hid_feats, 1)
        self.ar = nn.Linear(hid_feats, 1)
        self.activation = activation
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linear1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linear2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.al.weight, gain=1.414)
        nn.init.xavier_normal_(self.ar.weight, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(0.2)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            bias_stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-bias_stdv, bias_stdv)

    def forward(self, h):
        h = self.dropout(h)
        h = self.linear(h)
        self.g1.ndata['h'] = h
        self.g2.ndata['h'] = h
        self.g1.update_all(gcn_msg, gcn_reduce)
        self.g2.update_all(gcn_msg, gcn_reduce)
        h1 = self.g1.ndata.pop('h')
        h2 = self.g2.ndata.pop('h')
        ai = self.al(h)
        aj1 = self.ar(h1)
        aj2 = self.ar(h2)
        a1 = self.leaky_relu(ai + aj1)
        a1 = torch.exp(a1).clamp(-10, 10)
        a2 = self.leaky_relu(ai + aj2)
        a2 = torch.exp(a2).clamp(-10, 10)
        alpha1 = a1/(a1+a2)
        alpha2 = a2/(a1+a2)
        h = alpha1*h1+alpha2*h2

        h = self.fc(h)
        return h



    # def forward1(self, h):
    #     h1 = self.dropout(h)
    #     h2 = self.dropout(h)
    #     # self_h = h
    #     self.g1.ndata['h'] = self.linear1(h1)
    #     self.g2.ndata['h'] = self.linear2(h2)
    #     self.g1.update_all(gcn_msg, gcn_reduce)
    #     self.g2.update_all(gcn_msg, gcn_reduce)
    #     h1 = self.g1.ndata.pop('h')
    #     h1 = self.activation(h1)
    #     h2 = self.g2.ndata.pop('h')
    #     h2 = self.activation(h2)
    #     h = torch.cat((h1, h2), dim=1)
    #     h = self.dropout(h)
    #     h = self.fc(h)
    #     return h


class GCN(nn.Module):
    def __init__(self,
                 g1,
                 g2,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(g1, g2, in_feats, n_hidden, n_classes, activation, dropout))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    random.seed(args.syn_seed)
    np.random.seed(args.syn_seed)
    torch.manual_seed(args.syn_seed)

    # load and preprocess dataset
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)  # 2708个节点
    train_mask = torch.ByteTensor(data.train_mask)
    val_mask = torch.ByteTensor(data.val_mask)
    test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.sum().item(),
           val_mask.sum().item(),
           test_mask.sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        torch.cuda.manual_seed_all(args.syn_seed)

    # graph preprocess and calculate normalization factor
    g1 = DGLGraph(data.graph)
    n_edges = g1.number_of_edges()
    # add self loop
    g1.add_edges(g1.nodes(), g1.nodes())
    # normalization
    degs1 = g1.in_degrees().float()
    norm1 = torch.pow(degs1, -0.5)
    norm1[torch.isinf(norm1)] = 0
    if cuda:
        norm1 = norm1.cuda()
    g1.ndata['norm'] = norm1.unsqueeze(1)
    print('g1.number_of_edges=', g1.number_of_edges())

    src = g1.edges()[0]
    des = g1.edges()[1]
    edge_dict1 = {}
    for e in range(0, src.shape[0]):
        edge_dict1.setdefault(src[e].item(), set()).add(des[e].item())
    # print(len(edge_dict1[1344]), edge_dict1[1344])

    # edge_dict2 = copy.deepcopy(edge_dict1)
    # for e in range(0, src.shape[0]):
    #     edge_dict2[src[e].item()].update(edge_dict1[des[e].item()])
    edge_dict2 = {}
    for e in range(0, src.shape[0]):
        edge_dict2.setdefault(src[e].item(), set()).update(edge_dict1[des[e].item()])
    # print(len(edge_dict2[1344]), edge_dict2[1344])

    for key in edge_dict2.keys():
        edge_dict2[key].difference_update(edge_dict1[key])
        # edge_dict2[key].add(key)
    # print(len(edge_dict2[1344]), edge_dict2[1344])

    g2 = DGLGraph(data.graph)
    g2.clear()
    g2.add_nodes(g1.number_of_nodes())
    for key in edge_dict2.keys():
        src = key
        des_set = edge_dict2[key]
        for des in des_set:
            g2.add_edge(src, des)
    degs2 = g2.in_degrees().float()
    norm2 = torch.pow(degs2, -0.5)
    norm2[torch.isinf(norm2)] = 0
    if cuda:
        norm2 = norm2.cuda()
    g2.ndata['norm'] = norm2.unsqueeze(1)
    print('g2.number_of_edges=', g2.number_of_edges())

    # create GCN model
    model = GCN(g1, g2,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    train_loss = []
    val_acc = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)
        train_loss.append(loss.item())
        acc = evaluate(model, features, labels, val_mask)
        val_acc.append(acc)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            acc, n_edges / np.mean(dur) / 1000))
    draw(val_acc, train_loss)
    print()
    acc = evaluate(model, features, labels, test_mask)
    # print("Test Accuracy {:.4f}".format(acc))
    print("Train Loss {:.4f} | Val Acc {:.4f} | Test Acc {:.4f}".format(np.mean(train_loss), np.mean(val_acc), acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    args = parser.parse_args()
    print(args)

    main(args)
