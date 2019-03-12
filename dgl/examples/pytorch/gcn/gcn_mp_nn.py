"""GCN using basic message passing

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import random
import matplotlib.pyplot as plt


def draw(acc, loss):
    plt.figure()
    x = range(0, len(acc))
    plt.plot(x, acc, label='acc')
    plt.plot(x, loss, label='loss')
    plt.legend()
    plt.savefig("gcn_mp_nn.png")


def draw1():
    x = np.linspace(0, 2 * np.pi, 100)
    y1, y2 = np.sin(x), np.cos(x)

    plt.plot(x, y1)
    plt.plot(x, y2)

    plt.title('line chart')
    plt.xlabel('x')
    plt.ylabel('y')

    # plt.show()
    plt.savefig("test")
    exit()


def gcn_msg(edge):
    msg = edge.src['h'] * edge.src['norm']
    return {'m': msg}


def gcn_reduce(node):
    accum = torch.sum(node.mailbox['m'], 1) * node.data['norm']
    return {'h': accum}

# 与传统的不同，我们是先加上bias再对neighbor进行平均。 之前是先平均再加上bias。
class GCNLayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)  # Test Accuracy 0.8230
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))  # Test Accuracy 0.8120
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        # nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        self.reset_parameters()

    def reset_parameters(self):
        print(self.linear.weight.size(), self.linear.bias.size())
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            bias_stdv = 1. / math.sqrt(self.linear.bias.size(0))
            self.linear.bias.data.uniform_(-bias_stdv, bias_stdv)

    def reset_parameters1(self):
        print('go into rest1')
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            bias_stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-bias_stdv, bias_stdv)

    def forward(self, h):
        h = self.dropout(h)

        self.g.ndata['h'] = self.linear(h)
        # h = torch.mm(h, self.weight)
        # if self.bias is not None:
        #     h = h + self.bias
        # self.g.ndata['h'] = h

        self.g.update_all(gcn_msg, gcn_reduce)
        h = self.g.ndata.pop('h')
        if self.activation:
            h = self.activation(h)
        return h


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(g, in_feats, n_hidden, activation, dropout))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(g, n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.append(GCNLayer(g, n_hidden, n_classes, None, dropout))

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
    labels = torch.LongTensor(data.labels)
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
    g = DGLGraph(data.graph)
    n_edges = g.number_of_edges()
    # add self loop
    g.add_edges(g.nodes(), g.nodes())
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
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
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
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
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()
    print(args)
    main(args)

#xavier_uniform_
# cora           0.8230
# citeseer       0.7160
# pubmed         0.7920


