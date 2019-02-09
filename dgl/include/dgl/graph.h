/*!
 *  Copyright (c) 2018 by Contributors
 * \file dgl/graph.h
 * \brief DGL graph index class.
 */
#ifndef DGL_GRAPH_H_
#define DGL_GRAPH_H_

#include <vector>
#include <string>
#include <cstdint>
#include <utility>
#include <tuple>

#include "graph_interface.h"

namespace dgl {

class Graph;
class GraphOp;

/*!
 * \brief Base dgl graph index class.
 *
 * DGL's graph is directed. Vertices are integers enumerated from zero.
 *
 * Removal of vertices/edges is not allowed. Instead, the graph can only be "cleared"
 * by removing all the vertices and edges.
 *
 * When calling functions supporing multiple edges (e.g. AddEdges, HasEdges),
 * the input edges are represented by two id arrays for source and destination
 * vertex ids. In the general case, the two arrays should have the same length.
 * If the length of src id array is one, it represents one-many connections.
 * If the length of dst id array is one, it represents many-one connections.
 */
class Graph: public GraphInterface {
 public:
  /*! \brief default constructor */
  explicit Graph(bool multigraph = false) : is_multigraph_(multigraph) {}

  /*! \brief construct a graph from the coo format. */
  Graph(IdArray src_ids, IdArray dst_ids, IdArray edge_ids, size_t num_nodes,
      bool multigraph = false);

  /*! \brief default copy constructor */
  Graph(const Graph& other) = default;

#ifndef _MSC_VER
  /*! \brief default move constructor */
  Graph(Graph&& other) = default;
#else
  Graph(Graph&& other) {
    adjlist_ = other.adjlist_;
    reverse_adjlist_ = other.reverse_adjlist_;
    all_edges_src_ = other.all_edges_src_;
    all_edges_dst_ = other.all_edges_dst_;
    read_only_ = other.read_only_;
    is_multigraph_ = other.is_multigraph_;
    num_edges_ = other.num_edges_;
    other.Clear();
  }
#endif  // _MSC_VER

  /*! \brief default assign constructor */
  Graph& operator=(const Graph& other) = default;

  /*! \brief default destructor */
  ~Graph() = default;

  /*!
   * \brief Add vertices to the graph.
   * \note Since vertices are integers enumerated from zero, only the number of
   *       vertices to be added needs to be specified.
   * \param num_vertices The number of vertices to be added.
   */
  void AddVertices(uint64_t num_vertices);

  /*!
   * \brief Add one edge to the graph.
   * \param src The source vertex.
   * \param dst The destination vertex.
   */
  void AddEdge(dgl_id_t src, dgl_id_t dst);

  /*!
   * \brief Add edges to the graph.
   * \param src_ids The source vertex id array.
   * \param dst_ids The destination vertex id array.
   */
  void AddEdges(IdArray src_ids, IdArray dst_ids);

  /*!
   * \brief Clear the graph. Remove all vertices/edges.
   */
  void Clear() {
    adjlist_.clear();
    reverse_adjlist_.clear();
    all_edges_src_.clear();
    all_edges_dst_.clear();
    read_only_ = false;
    num_edges_ = 0;
  }

  /*!
   * \note not const since we have caches
   * \return whether the graph is a multigraph
   */
  bool IsMultigraph() const {
    return is_multigraph_;
  }

  /*!
   * \return whether the graph is read-only
   */
  virtual bool IsReadonly() const {
    return false;
  }

  /*! \return the number of vertices in the graph.*/
  uint64_t NumVertices() const {
    return adjlist_.size();
  }

  /*! \return the number of edges in the graph.*/
  uint64_t NumEdges() const {
    return num_edges_;
  }

  /*! \return true if the given vertex is in the graph.*/
  bool HasVertex(dgl_id_t vid) const {
    return vid < NumVertices();
  }

  /*! \return a 0-1 array indicating whether the given vertices are in the graph.*/
  BoolArray HasVertices(IdArray vids) const;

  /*! \return true if the given edge is in the graph.*/
  bool HasEdgeBetween(dgl_id_t src, dgl_id_t dst) const;

  /*! \return a 0-1 array indicating whether the given edges are in the graph.*/
  BoolArray HasEdgesBetween(IdArray src_ids, IdArray dst_ids) const;

  /*!
   * \brief Find the predecessors of a vertex.
   * \param vid The vertex id.
   * \param radius The radius of the neighborhood. Default is immediate neighbor (radius=1).
   * \return the predecessor id array.
   */
  IdArray Predecessors(dgl_id_t vid, uint64_t radius = 1) const;

  /*!
   * \brief Find the successors of a vertex.
   * \param vid The vertex id.
   * \param radius The radius of the neighborhood. Default is immediate neighbor (radius=1).
   * \return the successor id array.
   */
  IdArray Successors(dgl_id_t vid, uint64_t radius = 1) const;

  /*!
   * \brief Get all edge ids between the two given endpoints
   * \note Edges are associated with an integer id start from zero.
   *       The id is assigned when the edge is being added to the graph.
   * \param src The source vertex.
   * \param dst The destination vertex.
   * \return the edge id array.
   */
  IdArray EdgeId(dgl_id_t src, dgl_id_t dst) const;

  /*!
   * \brief Get all edge ids between the given endpoint pairs.
   * \note Edges are associated with an integer id start from zero.
   *       The id is assigned when the edge is being added to the graph.
   *       If duplicate pairs exist, the returned edge IDs will also duplicate.
   *       The order of returned edge IDs will follow the order of src-dst pairs
   *       first, and ties are broken by the order of edge ID.
   * \return EdgeArray containing all edges between all pairs.
   */
  EdgeArray EdgeIds(IdArray src, IdArray dst) const;

  /*!
   * \brief Find the edge ID and return the pair of endpoints
   * \param eid The edge ID
   * \return a pair whose first element is the source and the second the destination.
   */
  std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_id_t eid) const {
    return std::make_pair(all_edges_src_[eid], all_edges_dst_[eid]);
  }

  /*!
   * \brief Find the edge IDs and return their source and target node IDs.
   * \param eids The edge ID array.
   * \return EdgeArray containing all edges with id in eid.  The order is preserved.
   */
  EdgeArray FindEdges(IdArray eids) const;

  /*!
   * \brief Get the in edges of the vertex.
   * \note The returned dst id array is filled with vid.
   * \param vid The vertex id.
   * \return the edges
   */
  EdgeArray InEdges(dgl_id_t vid) const;

  /*!
   * \brief Get the in edges of the vertices.
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray InEdges(IdArray vids) const;

  /*!
   * \brief Get the out edges of the vertex.
   * \note The returned src id array is filled with vid.
   * \param vid The vertex id.
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray OutEdges(dgl_id_t vid) const;

  /*!
   * \brief Get the out edges of the vertices.
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray OutEdges(IdArray vids) const;

  /*!
   * \brief Get all the edges in the graph.
   * \note If sorted is true, the returned edges list is sorted by their src and
   *       dst ids. Otherwise, they are in their edge id order.
   * \param sorted Whether the returned edge list is sorted by their src and dst ids
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray Edges(const std::string &order = "") const;

  /*!
   * \brief Get the in degree of the given vertex.
   * \param vid The vertex id.
   * \return the in degree
   */
  uint64_t InDegree(dgl_id_t vid) const {
    CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
    return reverse_adjlist_[vid].succ.size();
  }

  /*!
   * \brief Get the in degrees of the given vertices.
   * \param vid The vertex id array.
   * \return the in degree array
   */
  DegreeArray InDegrees(IdArray vids) const;

  /*!
   * \brief Get the out degree of the given vertex.
   * \param vid The vertex id.
   * \return the out degree
   */
  uint64_t OutDegree(dgl_id_t vid) const {
    CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
    return adjlist_[vid].succ.size();
  }

  /*!
   * \brief Get the out degrees of the given vertices.
   * \param vid The vertex id array.
   * \return the out degree array
   */
  DegreeArray OutDegrees(IdArray vids) const;

  /*!
   * \brief Construct the induced subgraph of the given vertices.
   *
   * The induced subgraph is a subgraph formed by specifying a set of vertices V' and then
   * selecting all of the edges from the original graph that connect two vertices in V'.
   *
   * Vertices and edges in the original graph will be "reindexed" to local index. The local
   * index of the vertices preserve the order of the given id array, while the local index
   * of the edges preserve the index order in the original graph. Vertices not in the
   * original graph are ignored.
   *
   * The result subgraph is read-only.
   *
   * \param vids The vertices in the subgraph.
   * \return the induced subgraph
   */
  Subgraph VertexSubgraph(IdArray vids) const;

  /*!
   * \brief Construct the induced edge subgraph of the given edges.
   *
   * The induced edges subgraph is a subgraph formed by specifying a set of edges E' and then
   * selecting all of the nodes from the original graph that are endpoints in E'.
   *
   * Vertices and edges in the original graph will be "reindexed" to local index. The local
   * index of the edges preserve the order of the given id array, while the local index
   * of the vertices preserve the index order in the original graph. Edges not in the
   * original graph are ignored.
   *
   * The result subgraph is read-only.
   *
   * \param eids The edges in the subgraph.
   * \return the induced edge subgraph
   */
  Subgraph EdgeSubgraph(IdArray eids) const;

  /*!
   * \brief Return a new graph with all the edges reversed.
   *
   * The returned graph preserves the vertex and edge index in the original graph.
   *
   * \return the reversed graph
   */
  GraphPtr Reverse() const;

  /*!
   * \brief Return the successor vector
   * \param vid The vertex id.
   * \return the successor vector
   */
  DGLIdIters SuccVec(dgl_id_t vid) const {
    return DGLIdIters(adjlist_[vid].succ.begin(), adjlist_[vid].succ.end());
  }

  /*!
   * \brief Return the out edge id vector
   * \param vid The vertex id.
   * \return the out edge id vector
   */
  DGLIdIters OutEdgeVec(dgl_id_t vid) const {
    return DGLIdIters(adjlist_[vid].edge_id.begin(), adjlist_[vid].edge_id.end());
  }

  /*!
   * \brief Return the predecessor vector
   * \param vid The vertex id.
   * \return the predecessor vector
   */
  DGLIdIters PredVec(dgl_id_t vid) const {
    return DGLIdIters(reverse_adjlist_[vid].succ.begin(), reverse_adjlist_[vid].succ.end());
  }

  /*!
   * \brief Return the in edge id vector
   * \param vid The vertex id.
   * \return the in edge id vector
   */
  DGLIdIters InEdgeVec(dgl_id_t vid) const {
    return DGLIdIters(reverse_adjlist_[vid].edge_id.begin(),
                      reverse_adjlist_[vid].edge_id.end());
  }

  /*!
   * \brief Reset the data in the graph and move its data to the returned graph object.
   * \return a raw pointer to the graph object.
   */
  virtual GraphInterface *Reset() {
    Graph* gptr = new Graph();
    *gptr = std::move(*this);
    return gptr;
  }

  /*!
   * \brief Get the adjacency matrix of the graph.
   *
   * By default, a row of returned adjacency matrix represents the destination
   * of an edge and the column represents the source.
   * \param transpose A flag to transpose the returned adjacency matrix.
   * \param fmt the format of the returned adjacency matrix.
   * \return a vector of three IdArray.
   */
  virtual std::vector<IdArray> GetAdj(bool transpose, const std::string &fmt) const;

  /*!
   * \brief Sample a subgraph from the seed vertices with neighbor sampling.
   * The neighbors are sampled with a uniform distribution.
   * \return a subgraph
   */
  virtual SampledSubgraph NeighborUniformSample(IdArray seeds, const std::string &neigh_type,
                                                int num_hops, int expand_factor) const {
    LOG(FATAL) << "NeighborUniformSample isn't supported in mutable graph";
    return SampledSubgraph();
  }

 protected:
  friend class GraphOp;
  /*! \brief Internal edge list type */
  struct EdgeList {
    /*! \brief successor vertex list */
    std::vector<dgl_id_t> succ;
    /*! \brief predecessor vertex list */
    std::vector<dgl_id_t> edge_id;
  };
  typedef std::vector<EdgeList> AdjacencyList;

  /*! \brief adjacency list using vector storage */
  AdjacencyList adjlist_;
  /*! \brief reverse adjacency list using vector storage */
  AdjacencyList reverse_adjlist_;

  /*! \brief all edges' src endpoints in their edge id order */
  std::vector<dgl_id_t> all_edges_src_;
  /*! \brief all edges' dst endpoints in their edge id order */
  std::vector<dgl_id_t> all_edges_dst_;

  /*! \brief read only flag */
  bool read_only_ = false;
  /*!
   * \brief Whether if this is a multigraph.
   *
   * When a multiedge is added, this flag switches to true.
   */
  bool is_multigraph_ = false;
  /*! \brief number of edges */
  uint64_t num_edges_ = 0;
};

}  // namespace dgl

#endif  // DGL_GRAPH_H_
