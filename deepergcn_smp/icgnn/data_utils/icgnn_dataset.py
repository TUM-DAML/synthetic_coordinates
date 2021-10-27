"""
ICGNN version of the TUDataset
Includes the original graphs in the TUDataset
https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/tu_dataset.html#TUDataset

along with VERSE embeddings, linegraph, distances between nodes
and angles between edges
"""

import copy
from functools import partial

print = partial(print, flush=True)

import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T

from ..models.embeddings import create_embedding
from ..models.basis import get_dist_basis, get_cos_basis


def graph_to_icgnn(graph, emb_dim, num_dist_basis=4, num_cos_basis=4):
    embedding = torch.Tensor(create_embedding(graph, emb_dim, "verse"))

    graph.emb = embedding

    emb_diff = torch.index_select(embedding, dim=0, index=graph.edge_index[0])
    emb_diff -= torch.index_select(embedding, dim=0, index=graph.edge_index[1])
    emb_dist = torch.norm(emb_diff, p=2, dim=1)

    # TODO: retain the original edge_attr, dont overwrite it!
    graph.edge_attr = get_dist_basis(emb_dist, num_dist_basis)

    graph_copy = Data(
        edge_index=copy.deepcopy(graph.edge_index), num_nodes=graph.num_nodes
    )
    transform = T.Compose([T.LineGraph(force_directed=True), T.Constant()])
    linegraph = transform(graph_copy)

    noloop_mask = ~(
        graph.edge_index[0, linegraph.edge_index[0]]
        == graph.edge_index[1, linegraph.edge_index[1]]
    )
    line_idx_noloop = graph.edge_index.new_empty((2, noloop_mask.sum()))
    line_idx_noloop[0] = torch.masked_select(linegraph.edge_index[0], noloop_mask)
    line_idx_noloop[1] = torch.masked_select(linegraph.edge_index[1], noloop_mask)
    linegraph.edge_index = line_idx_noloop

    emb_diff_ji = torch.index_select(
        -emb_diff, dim=0, index=linegraph.edge_index[0]
    )  # Ri <- Rj
    emb_diff_jk = torch.index_select(
        emb_diff, dim=0, index=linegraph.edge_index[1]
    )  # Rj -> Rk
    emb_dist_ji = torch.norm(emb_diff_ji, p=2, dim=1)
    emb_dist_jk = torch.norm(emb_diff_jk, p=2, dim=1)
    emb_angle = torch.acos(
        (emb_diff_ji * emb_diff_jk).sum(-1) / (emb_dist_ji * emb_dist_jk)
    )

    linegraph.edge_attr = get_cos_basis(emb_angle, num_cos_basis)

    return graph, linegraph


class ICGNN_Data(Data):
    def __init__(
        self,
        x_g=None,
        edge_index_g=None,
        edge_attr_g=None,
        x_emb=None,
        x_lg=None,
        edge_index_lg=None,
        edge_attr_lg=None,
        y=None,
    ):
        """
        *_g: properties of the graph
        *_lg: properties of the linegraph

        See tutorial here:
        https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html#pairs-of-graphs
        """
        super().__init__()

        self.edge_index_g = edge_index_g
        self.x_g = x_g
        self.x_emb = x_emb
        self.edge_attr_g = edge_attr_g

        self.edge_index_lg = edge_index_lg
        self.x_lg = x_lg
        self.edge_attr_lg = edge_attr_lg

        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_g":
            return self.x_g.size(0)
        if key == "edge_index_lg":
            return self.x_lg.size(0)
        else:
            return super().__inc__(key, value)