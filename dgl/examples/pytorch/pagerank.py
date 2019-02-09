import networkx as nx
import torch
import dgl
import dgl.function as fn

N = 100
g = nx.nx.erdos_renyi_graph(N, 0.05)
g = dgl.DGLGraph(g)

DAMP = 0.85
K = 10

def compute_pagerank(g):
    g.ndata['pv'] = torch.ones(N) / N
    degrees = g.out_degrees(g.nodes()).type(torch.float32)
    for k in range(K):
        g.ndata['pv'] = g.ndata['pv'] / degrees
        g.update_all(message_func=fn.copy_src(src='pv', out='m'),
                     reduce_func=fn.sum(msg='m', out='pv'))
        g.ndata['pv'] = (1 - DAMP) / N + DAMP * g.ndata['pv']
    return g.ndata['pv']

pv = compute_pagerank(g)
print(pv)
