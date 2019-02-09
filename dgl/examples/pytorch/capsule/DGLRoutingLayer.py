import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl


class DGLRoutingLayer(nn.Module):
    def __init__(self, in_nodes, out_nodes, f_size, batch_size=0, device='cpu'):
        super(DGLRoutingLayer, self).__init__()
        self.batch_size = batch_size
        self.g = init_graph(in_nodes, out_nodes, f_size, device=device)
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.in_indx = list(range(in_nodes))
        self.out_indx = list(range(in_nodes, in_nodes + out_nodes))
        self.device = device

    def forward(self, u_hat, routing_num=1):
        self.g.edata['u_hat'] = u_hat
        batch_size = self.batch_size

        # step 2 (line 5)
        def cap_message(edges):
            if batch_size:
                return {'m': edges.data['c'].unsqueeze(1) * edges.data['u_hat']}
            else:
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
            if self.batch_size:
                self.g.nodes[self.out_indx].data['v'] = squash(self.g.nodes[self.out_indx].data['s'], dim=2)
            else:
                self.g.nodes[self.out_indx].data['v'] = squash(self.g.nodes[self.out_indx].data['s'], dim=1)

            # step 4 (line 7)
            v = th.cat([self.g.nodes[self.out_indx].data['v']] * self.in_nodes, dim=0)
            if self.batch_size:
                self.g.edata['b'] = self.g.edata['b'] + (self.g.edata['u_hat'] * v).mean(dim=1).sum(dim=1, keepdim=True)
            else:
                self.g.edata['b'] = self.g.edata['b'] + (self.g.edata['u_hat'] * v).sum(dim=1, keepdim=True)


def squash(s, dim=1):
    sq = th.sum(s ** 2, dim=dim, keepdim=True)
    s_norm = th.sqrt(sq)
    s = (sq / (1.0 + sq)) * (s / s_norm)
    return s


def init_graph(in_nodes, out_nodes, f_size, device='cpu'):
    g = dgl.DGLGraph()
    g.set_n_initializer(dgl.frame.zero_initializer)
    all_nodes = in_nodes + out_nodes
    g.add_nodes(all_nodes)
    in_indx = list(range(in_nodes))
    out_indx = list(range(in_nodes, in_nodes + out_nodes))
    # add edges use edge broadcasting
    for u in in_indx:
        g.add_edges(u, out_indx)

    g.edata['b'] = th.zeros(in_nodes * out_nodes, 1).to(device)
    return g
