from dgl import DGLError
from dgl.utils import toindex
from dgl.graph_index import create_graph_index

def test_node_subgraph():
    gi = create_graph_index()
    gi.add_nodes(4)
    gi.add_edge(0, 1)
    gi.add_edge(0, 2)
    gi.add_edge(0, 2)
    gi.add_edge(0, 3)

    sub2par_nodemap = [2, 0, 3]
    sgi = gi.node_subgraph(toindex(sub2par_nodemap))

    for s, d, e in zip(*sgi.edges()):
        assert sgi.induced_edges[e] in gi.edge_id(
                sgi.induced_nodes[s], sgi.induced_nodes[d])

def test_edge_subgraph():
    gi = create_graph_index()
    gi.add_nodes(4)
    gi.add_edge(0, 1)
    gi.add_edge(0, 1)
    gi.add_edge(0, 2)
    gi.add_edge(2, 3)

    sub2par_edgemap = [3, 2]
    sgi = gi.edge_subgraph(toindex(sub2par_edgemap))

    for s, d, e in zip(*sgi.edges()):
        assert sgi.induced_edges[e] in gi.edge_id(
                sgi.induced_nodes[s], sgi.induced_nodes[d])


if __name__ == '__main__':
    test_node_subgraph()
    test_edge_subgraph()
